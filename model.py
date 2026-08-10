#Here, we will have a GroqModel (openai/gpt-oss-20b) as the base llm for the agent.
#
# this is winnow's brain. it does not know anything about datahub on its own, it just
# decides which tool from datahub_tools.py to call and then explains what came back.
# the loop is: ask the model -> model asks for a tool -> we run it -> feed the result
# back -> repeat until the model stops asking and writes its answer.

import json
import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

import agent_context_tools as kit
import datahub_tools as dt

# read GROQ_API_KEY out of .env. that file is gitignored so the key never gets committed.
load_dotenv()

# the windows console defaults to cp1252, which cannot encode most of what an llm
# writes: curly quotes, en dashes, non breaking hyphens, emoji. without this a finished
# answer dies on print with UnicodeEncodeError after all the work is already done.
# errors="replace" means an odd character degrades to '?' instead of losing the answer.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # already utf-8, or redirected somewhere that cannot be reconfigured

# gpt-oss-20b is the fast one to develop against. openai/gpt-oss-120b is on the same
# key and is noticeably better at multi step tool use, so it is the one to demo with.
# override without touching code: set WINNOW_MODEL in .env or the environment.
MODEL = os.getenv("WINNOW_MODEL", "openai/gpt-oss-120b")

# how many tool round trips we allow before we force the model to answer. this stops a
# confused model from calling tools forever and burning the api quota.
MAX_STEPS = 12

# groq's free tier caps tokens per MINUTE, not per request (8000 TPM on gpt-oss-120b).
# the whole message history including every past tool result is resent on every step,
# so a long audit blows the cap around step 6 unless results are kept small. these two
# limits are what keep a broad sweep inside the budget.
MAX_TOOL_RESULT_CHARS = 3000
MAX_ROWS_PER_TOOL = 25

# a rate limit is only a wait if the request FITS in the window. a single request
# bigger than the whole per-minute allowance can never succeed no matter how long we
# wait, so the real defence is keeping the request small (see HISTORY_TOKEN_BUDGET).
# retries only help with genuine burst throttling.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_WAIT_SECONDS = 20

# how much of the conversation we are willing to resend. the tool schemas and the
# model's own reasoning also count toward the 8000 TPM ceiling, so the history gets
# well under half of it. older tool rounds are dropped once this is exceeded.
HISTORY_TOKEN_BUDGET = 3200

# gpt-oss supports "low" / "medium" / "high". leave as None to use the groq default.
REASONING_EFFORT = None

SYSTEM_PROMPT = """You are Winnow, a data catalog hygiene agent working against a live DataHub instance.

You find and explain metadata problems: missing descriptions, stale datasets, broken
lineage, duplicate assets, unused dashboards, missing tags, and orphaned datasets that
nothing consumes.

How to work:
- Always call tools to get real data. Never invent dataset names, URNs, or counts.
- Start with catalog_summary if you do not yet know what is in the catalog.
- Prefer a couple of targeted tool calls over many broad ones.
- When something looks like a cleanup candidate, check downstream consumers before
  saying it is safe to remove. Zero consumers means nothing breaks; anything above
  zero means removing it would break a real asset.

How to answer:
- Lead with the finding and the number that supports it.
- Reference specific datasets by their readable name, with the URN available if asked.
- Be honest about weak evidence. A dataset with no recorded timestamp is unknown age,
  not confirmed stale, and you should say so.
- If a tool returns nothing, say the catalog is clean on that dimension. Do not pad.

Fixing things:
- update_description, add_tags and add_owners write to the real catalog.
- If a tool result comes back with "dry_run": true, the change did NOT happen. Report it
  as a proposal ("would set X"), never as done. Say the user can re-run with --apply.
- Before writing a description, look at the dataset first. A description you inferred
  from the table name alone is a guess; say so rather than stating it as fact."""


# the json schemas the model reads to decide what it can call. these names must match
# the keys in TOOL_IMPLS below exactly.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "catalog_summary",
            "description": "Counts of every entity type and a breakdown of datasets by platform. Use this first to orient.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_missing_descriptions",
            "description": "Datasets with no description in either the editable or ingested properties. Sorted so the ones with the most downstream consumers come first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "description": "Optional platform filter, e.g. snowflake, dbt, looker."},
                    "limit": {"type": "integer", "description": "Max rows to return. Default 25."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_missing_tags",
            "description": "Datasets carrying no tags at all, so tag-based search cannot find them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_stale_datasets",
            "description": "Datasets not written to within the given number of days. Also returns datasets with no timestamp at all, whose age is unknown rather than confirmed stale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Staleness threshold in days. Default 90."},
                    "platform": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_broken_lineage",
            "description": "Datasets whose declared upstream sources no longer exist, meaning the lineage graph points at deleted assets.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicate_assets",
            "description": "Assets sharing a table name. Separates genuine same-platform duplicates from same-name-across-platforms, which are usually legitimate mirrors.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_unused_dashboards",
            "description": "Dashboards with no charts and no datasets attached, so they display nothing.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_orphan_datasets",
            "description": "Datasets with zero downstream consumers. Nothing reads them, so removing them breaks nothing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": "Full detail on one dataset: description, tags, owners, last modified, downstream consumer count.",
            "parameters": {
                "type": "object",
                "properties": {"urn": {"type": "string", "description": "The full dataset URN."}},
                "required": ["urn"],
            },
        },
    },
]

# maps the tool name the model uses to the real python function in datahub_tools.
TOOL_IMPLS = {
    "catalog_summary": dt.catalog_summary,
    "find_missing_descriptions": dt.find_missing_descriptions,
    "find_missing_tags": dt.find_missing_tags,
    "find_stale_datasets": dt.find_stale_datasets,
    "find_broken_lineage": dt.find_broken_lineage,
    "find_duplicate_assets": dt.find_duplicate_assets,
    "find_unused_dashboards": dt.find_unused_dashboards,
    "find_orphan_datasets": dt.find_orphan_datasets,
    "describe_dataset": dt.describe_dataset,
}

# count our own tools before merging, purely so the startup banner can say how many
# came from where.
DETECTION_TOOL_COUNT = len(TOOL_IMPLS)

# now fold in the agent context kit. our own tools above decide what counts as a
# problem, the kit's tools below do broad search and write the fixes back.
TOOLS = TOOLS + kit.KIT_TOOLS
TOOL_IMPLS.update(kit.KIT_IMPLS)


def confirm_write(action: str, details: Dict) -> bool:
    """show one proposed change and wait for a yes before it happens.

    this runs mid-answer, right when the model asks for the write, so you see the
    exact change rather than a summary after the fact.
    """
    print("\n" + "-" * 66)
    print(f"  PROPOSED CHANGE: {action}")
    for key, value in details.items():
        if value is None:
            continue
        shown = value if len(str(value)) < 300 else str(value)[:300] + "..."
        print(f"    {key}: {shown}")
    print("-" * 66)

    # piped stdin cannot answer, and silently treating that as yes would be the worst
    # possible default, so decline instead.
    if not sys.stdin.isatty():
        print("  declined automatically (no interactive terminal). use --yes to allow.")
        return False

    try:
        answer = input("  apply this change? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  declined.")
        return False
    return answer in {"y", "yes"}


def build_client() -> Groq:
    """make the groq client, failing loudly if the key is missing rather than at call time."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            "Put it in Datahub-agentic-project/.env as:  GROQ_API_KEY=gsk_...\n"
            "(that file is gitignored, so the key stays out of the repo)"
        )
    return Groq(api_key=key)


def run_tool(graph, name: str, arguments: str) -> str:
    """run one tool the model asked for and hand the result back as json.

    a tool blowing up is not fatal. we return the error text as the tool result so the
    model can see what went wrong and try a different approach, instead of the whole
    run dying on one bad call.
    """
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name}"})

    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"could not parse arguments: {e}"})

    # the model will happily ask for limit=100. every row it gets back is re-sent on
    # every later request in the conversation, so one greedy call can push the whole
    # run over the rate limit. clamp it.
    if isinstance(kwargs.get("limit"), int):
        kwargs["limit"] = min(kwargs["limit"], MAX_ROWS_PER_TOOL)
    if isinstance(kwargs.get("num_results"), int):
        kwargs["num_results"] = min(kwargs["num_results"], MAX_ROWS_PER_TOOL)

    try:
        result = fn(graph, **kwargs)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"})

    payload = json.dumps(result, default=str)

    # hard cap on how much one tool result can add to the conversation. the history
    # grows with every step and is resent in full each time, so an untruncated result
    # is paid for again on every subsequent request.
    if len(payload) > MAX_TOOL_RESULT_CHARS:
        payload = payload[:MAX_TOOL_RESULT_CHARS] + (
            f"... TRUNCATED at {MAX_TOOL_RESULT_CHARS} of {len(payload)} characters. "
            f"Results are capped at {MAX_ROWS_PER_TOOL} rows per call and calling again "
            "with a bigger limit will NOT return more. Use a platform filter to narrow "
            "the question, or work with what you have."
        )
    return payload


def complete_with_retry(client: Groq, verbose: bool, **kwargs):
    """one call to groq, waiting out rate limits instead of dying on them.

    groq's free tier limits tokens per minute, so a long conversation trips it partway
    through even though no single request is oversized. the window refills, so the right
    response is to wait rather than throw away the run.
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            is_rate_limit = "rate_limit" in str(e) or "tokens per minute" in str(e)
            if not is_rate_limit or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            if verbose:
                print(
                    f"   [rate limit] waiting {RATE_LIMIT_WAIT_SECONDS}s for the token "
                    f"window to refill (attempt {attempt + 1}/{RATE_LIMIT_RETRIES})"
                )
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    raise RuntimeError("unreachable")


def _call_key(name: str, arguments: str) -> str:
    """identity of a tool call for repeat detection.

    limit and num_results are deliberately ignored. they are clamped to
    MAX_ROWS_PER_TOOL anyway, so find_orphan_datasets(limit=10) and (limit=50) return
    the same rows. without this, a model that nudges the limit each time slips past
    the repeat check and loops until it runs out of steps, which is exactly what
    happened before.
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        args = {"_raw": arguments}
    if isinstance(args, dict):
        args = {k: v for k, v in args.items() if k not in {"limit", "num_results"}}
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _est_tokens(obj) -> int:
    """rough token count. four characters per token is close enough to budget with,
    and it costs nothing compared to calling a tokenizer."""
    return len(json.dumps(obj, default=str)) // 4


def _fit_to_budget(system: Dict, user: Dict, rounds: List[Dict]) -> List[Dict]:
    """build the request, keeping only as much history as the budget allows.

    a round is one assistant turn plus the tool results it asked for. those must be
    kept or dropped together: a tool message whose assistant tool_call has been
    removed refers to an id that no longer exists and the api rejects it.

    newest rounds are kept, because the model needs its most recent results to keep
    working. dropped ones are replaced with a note so it knows the gap is there
    rather than silently forgetting what it already did.
    """
    budget = HISTORY_TOKEN_BUDGET - _est_tokens(system) - _est_tokens(user)

    kept, used = [], 0
    for rnd in reversed(rounds):
        size = _est_tokens(rnd["assistant"]) + sum(_est_tokens(t) for t in rnd["tools"])
        if used + size > budget and kept:
            break
        kept.append(rnd)
        used += size
    kept.reverse()

    messages = [system, user]
    dropped = len(rounds) - len(kept)
    if dropped:
        called = [t["name"] for r in rounds[:dropped] for t in r["tools"]]
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"[{dropped} earlier tool round(s) dropped to stay within the token "
                    f"limit. Already called: {', '.join(sorted(set(called)))}. "
                    "Do not call these again with the same arguments; summarise what you "
                    "have and answer.]"
                ),
            }
        )
    for rnd in kept:
        messages.append(rnd["assistant"])
        messages.extend(rnd["tools"])
    return messages


def ask(client: Groq, graph, question: str, verbose: bool = True) -> str:
    """one full question -> answer cycle, running as many tool calls as the model wants."""
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": question}
    rounds: List[Dict] = []

    # remembers every (tool, args) already run so a repeat costs nothing
    seen: Dict[str, str] = {}

    for _ in range(MAX_STEPS):
        kwargs = dict(
            model=MODEL,
            messages=_fit_to_budget(system, user, rounds),
            tools=TOOLS,
            tool_choice="auto",
        )
        if REASONING_EFFORT:
            kwargs["reasoning_effort"] = REASONING_EFFORT

        response = complete_with_retry(client, verbose, **kwargs)
        message = response.choices[0].message

        # no tool calls means the model is done thinking and this is the actual answer
        if not message.tool_calls:
            return message.content or "(the model returned an empty response)"

        # the assistant turn has to be stored with the tool results it triggered,
        # otherwise the tool_call_id references point at nothing
        assistant = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        }

        tool_msgs = []
        learned_something = False
        for tc in message.tool_calls:
            key = _call_key(tc.function.name, tc.function.arguments)
            if key in seen:
                # asking the same thing twice cannot produce a new answer, and paying
                # for the payload again is what pushed us over the limit last time
                if verbose:
                    print(f"   [tool] {tc.function.name} (repeat, using earlier result)")
                result = json.dumps(
                    {
                        "repeat_call": True,
                        "note": "identical call already made, the result has not changed. "
                        "Use what you already have and move on.",
                    }
                )
            else:
                if verbose:
                    print(f"   [tool] {tc.function.name}({tc.function.arguments})")
                result = run_tool(graph, tc.function.name, tc.function.arguments)
                seen[key] = result
                learned_something = True

            tool_msgs.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result}
            )

        rounds.append({"assistant": assistant, "tools": tool_msgs})

        # a whole round of nothing but repeats means the model is circling, not
        # investigating. more steps will not help, so stop gathering and go answer.
        if not learned_something:
            if verbose:
                print("   [no new information this round] moving to the answer")
            break

    # out of steps. rather than throw away everything the run discovered, make one
    # final call with tools switched off so the model has to write an answer from what
    # it already gathered. a partial audit is worth far more than a placeholder.
    if verbose:
        print("   [step limit reached] asking for a final answer from what was gathered")

    messages = _fit_to_budget(system, user, rounds)
    messages.append(
        {
            "role": "user",
            "content": (
                "Stop calling tools. Using only the results you already have, give your "
                "audit now. Say plainly which areas you could not finish checking."
            ),
        }
    )
    final = complete_with_retry(
        client, verbose, model=MODEL, messages=messages, tool_choice="none", tools=TOOLS
    )
    return final.choices[0].message.content or "(no answer produced)"


def main() -> None:
    # flags can go anywhere in the args, everything left over is the question.
    flags = {"--apply", "--yes"}
    args = [a for a in sys.argv[1:] if a not in flags]
    apply_writes = "--apply" in sys.argv[1:]
    skip_confirm = "--yes" in sys.argv[1:]

    kit.set_apply(apply_writes)
    # with --apply you get asked before each write. --yes skips the asking, which is
    # for scripted runs where nobody is sitting at the terminal to answer.
    kit.set_confirm(None if skip_confirm else confirm_write)

    print("Winnow starting up...")

    # connect to datahub first. no point burning api calls if the catalog is unreachable.
    try:
        graph = dt.connect()
    except Exception as e:
        print(f"Could not reach DataHub at {dt.GMS_SERVER}: {type(e).__name__}: {e}")
        print("Is the quickstart stack running? Start it with:")
        print("  datahub docker quickstart")
        print("If something else already owns port 9200, use the bundled override:")
        print("  datahub docker quickstart --quickstart-compose-file datahub-local.yml")
        print(f"Pointing somewhere other than {dt.GMS_SERVER}? Set DATAHUB_GMS_URL.")
        raise SystemExit(1)

    # hand the same connection to the agent context kit, which reads it from a
    # contextvar rather than taking it as an argument.
    kit.bind(graph)

    print(f"Connected to DataHub at {dt.GMS_SERVER}")
    client = build_client()
    print(f"Using {MODEL} via Groq")
    print(f"Tools: {len(TOOLS)} ({DETECTION_TOOL_COUNT} detection, {len(kit.KIT_IMPLS)} agent context kit)")
    if apply_writes and skip_confirm:
        print("WRITE MODE, NO PROMPTS: " + ", ".join(sorted(kit.MUTATION_TOOLS)) + " will change the catalog immediately.")
    elif apply_writes:
        print("Write mode: you will be asked to approve each change before it happens.")
    else:
        print("Dry run: writes are reported, not made. Pass --apply to approve changes as they come up.")
    print()

    # a question on the command line runs once and exits, which is handy for scripting.
    if args:
        question = " ".join(args)
        print(f"> {question}")
        print("\n" + ask(client, graph, question))
        return

    # otherwise sit in a loop and take questions
    print("Ask about the catalog. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        print()
        print(ask(client, graph, question))
        print()


if __name__ == "__main__":
    main()

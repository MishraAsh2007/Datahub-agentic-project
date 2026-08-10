#Here, we will have a GroqModel (openai/gpt-oss-20b) as the base llm for the agent.
#
# this is winnow's brain. it does not know anything about datahub on its own, it just
# decides which tool from datahub_tools.py to call and then explains what came back.
# the loop is: ask the model -> model asks for a tool -> we run it -> feed the result
# back -> repeat until the model stops asking and writes its answer.

import json
import os
import sys
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

import datahub_tools as dt

# read GROQ_API_KEY out of .env. that file is gitignored so the key never gets committed.
load_dotenv()

MODEL = "openai/gpt-oss-20b"

# how many tool round trips we allow before we force the model to answer. this stops a
# confused model from calling tools forever and burning the api quota.
MAX_STEPS = 12

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
- If a tool returns nothing, say the catalog is clean on that dimension. Do not pad."""


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

    try:
        result = fn(graph, **kwargs)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"})

    return json.dumps(result, default=str)


def ask(client: Groq, graph, question: str, verbose: bool = True) -> str:
    """one full question -> answer cycle, running as many tool calls as the model wants."""
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        kwargs = dict(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        if REASONING_EFFORT:
            kwargs["reasoning_effort"] = REASONING_EFFORT

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # no tool calls means the model is done thinking and this is the actual answer
        if not message.tool_calls:
            return message.content or "(the model returned an empty response)"

        # the assistant turn has to go back into the history before the tool results,
        # otherwise the tool_call_id references point at nothing
        messages.append(
            {
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
        )

        for tc in message.tool_calls:
            if verbose:
                print(f"   [tool] {tc.function.name}({tc.function.arguments})")
            result = run_tool(graph, tc.function.name, tc.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result}
            )

    return "(hit the tool call limit without settling on an answer)"


def main() -> None:
    print("Winnow starting up...")

    # connect to datahub first. no point burning api calls if the catalog is unreachable.
    try:
        graph = dt.connect()
    except Exception as e:
        print(f"Could not reach DataHub at {dt.GMS_SERVER}: {type(e).__name__}: {e}")
        print("Is the quickstart stack running? Start it with:")
        print(r"  datahub docker quickstart --quickstart-compose-file C:\Users\ash7m\DataHub\datahub-local.yml")
        raise SystemExit(1)

    print(f"Connected to DataHub at {dt.GMS_SERVER}")
    client = build_client()
    print(f"Using {MODEL} via Groq\n")

    # a question on the command line runs once and exits, which is handy for scripting.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
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

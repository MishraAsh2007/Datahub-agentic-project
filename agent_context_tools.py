# the official datahub agent context kit, wired into winnow.
#
# datahub_tools.py is our own detection logic: opinionated audits that decide what
# counts as a problem. this file is the other half, the sanctioned kit that can
# actually read broadly and WRITE fixes back into the catalog.
#
# the kit's tools do not take a graph argument. they pull a DataHubClient out of a
# contextvar, so bind() has to run once at startup before any tool is called.
#
# every wrapper below still takes `graph` as its first argument even though the kit
# ignores it. that keeps one uniform dispatcher signature in model.py so kit tools and
# our own tools can live in the same registry.

from typing import Dict, List, Optional

import datahub_agent_context.context as dh_context
from datahub.sdk.main_client import DataHubClient

from datahub_agent_context.mcp_tools.descriptions import (
    update_description as _update_description,
)
from datahub_agent_context.mcp_tools.entities import get_entities as _get_entities
from datahub_agent_context.mcp_tools.lineage import get_lineage as _get_lineage
from datahub_agent_context.mcp_tools.owners import add_owners as _add_owners
from datahub_agent_context.mcp_tools.search import search as _search
from datahub_agent_context.mcp_tools.tags import add_tags as _add_tags

# writes are OFF unless main() explicitly turns them on with --apply. an agent looping
# over the whole catalog with write access is how a demo quietly becomes a cleanup job,
# so the default is to describe the change and not make it.
APPLY = False

# with --apply on, this gets asked before every single write. it takes the action name
# and the details and returns True to go ahead. leaving it as None means writes happen
# unprompted, which is what --yes does.
#
# this is the difference between "writes are allowed" and "this specific write is
# allowed". --apply alone only grants the first, and an agent can call a write tool
# many times inside one answer, so the per-change gate is what keeps you in the loop.
CONFIRM = None


def bind(graph) -> None:
    """point the kit at the same datahub connection our own tools already use.

    the kit reads its client from a contextvar rather than a parameter, so this has to
    happen once before any kit tool runs. set_client returns a reset token which we
    ignore, because the binding should last for the whole process.
    """
    dh_context.set_client(DataHubClient(graph=graph))


def set_apply(enabled: bool) -> None:
    """flip write mode on. called from model.py when --apply is passed."""
    global APPLY
    APPLY = enabled


def set_confirm(callback) -> None:
    """install the per-write approval prompt. pass None to write without asking."""
    global CONFIRM
    CONFIRM = callback


def _gate(action: str, **details):
    """decide whether one specific write is allowed to happen.

    returns None to mean "go ahead", or a result dict to return to the model instead
    of writing. three outcomes: dry run (writes off), declined (you said no), or
    approved (None, caller proceeds).
    """
    if not APPLY:
        return _dry_run(action, **details)

    if CONFIRM is not None and not CONFIRM(action, details):
        return {
            "dry_run": False,
            "applied": False,
            "declined": True,
            "would_do": action,
            "details": details,
            "note": "the user declined this change. do not retry it, move on.",
        }

    return None  # approved


def _as_tag_urn(tag: str) -> str:
    """the model will often say 'PII' when the api wants 'urn:li:tag:PII'."""
    return tag if tag.startswith("urn:li:tag:") else f"urn:li:tag:{tag}"


def _as_user_urn(owner: str) -> str:
    """same idea for owners: 'datahub' -> 'urn:li:corpuser:datahub'."""
    return owner if owner.startswith("urn:li:") else f"urn:li:corpuser:{owner}"


def _dry_run(action: str, **details) -> Dict:
    """what a write returns when APPLY is off. the model sees this and reports the
    proposed change instead of pretending it happened."""
    return {
        "dry_run": True,
        "applied": False,
        "would_do": action,
        "details": details,
        "note": "write mode is off. re-run with --apply to actually make this change.",
    }


# ---------------------------------------------------------------------------
# reads. these go through the kit rather than our own helpers so we get its richer
# search syntax, multi hop lineage and batch entity fetch.
# ---------------------------------------------------------------------------


def search(graph, query: str = "*", filter: Optional[str] = None, num_results: int = 10) -> Dict:
    return _search(query=query, filter=filter, num_results=num_results)


def get_lineage(graph, urn: str, upstream: bool = False, max_hops: int = 1, max_results: int = 30) -> Dict:
    """lineage in either direction. upstream=False means consumers, the thing that
    decides whether an asset is safe to remove."""
    return _get_lineage(urn=urn, upstream=upstream, max_hops=max_hops, max_results=max_results)


def get_entities(graph, urns: List[str]) -> List[Dict]:
    return _get_entities(urns=urns)


# ---------------------------------------------------------------------------
# writes. each one is gated on APPLY.
# ---------------------------------------------------------------------------


def update_description(
    graph, entity_urn: str, description: str, operation: str = "replace", column_path: Optional[str] = None
) -> Dict:
    blocked = _gate(
        "update_description",
        entity_urn=entity_urn,
        operation=operation,
        description=description,
        column_path=column_path,
    )
    if blocked is not None:
        return blocked
    return _update_description(
        entity_urn=entity_urn,
        operation=operation,
        description=description,
        column_path=column_path,
    )


def add_tags(graph, tags: List[str], entity_urns: List[str]) -> Dict:
    tag_urns = [_as_tag_urn(t) for t in tags]
    blocked = _gate("add_tags", tags=tag_urns, entity_urns=entity_urns)
    if blocked is not None:
        return blocked
    return _add_tags(tag_urns=tag_urns, entity_urns=entity_urns)


def add_owners(graph, owners: List[str], entity_urns: List[str]) -> Dict:
    owner_urns = [_as_user_urn(o) for o in owners]
    blocked = _gate("add_owners", owners=owner_urns, entity_urns=entity_urns)
    if blocked is not None:
        return blocked
    return _add_owners(owner_urns=owner_urns, entity_urns=entity_urns, ownership_type=None)


# ---------------------------------------------------------------------------
# the schemas the model reads. deliberately a curated subset of the kit, not all
# ~22 tools: every schema here is re-sent on every request, so the menu is kept to
# what winnow actually needs.
# ---------------------------------------------------------------------------

KIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the DataHub catalog with keyword syntax. Use for open-ended lookup when you do not already have a URN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword query. '*' matches everything."},
                    "filter": {"type": "string", "description": "Optional DataHub filter expression."},
                    "num_results": {"type": "integer", "description": "Max results. Default 10."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lineage",
            "description": "Lineage for one entity. upstream=false gives downstream consumers, which is what decides whether an asset is safe to remove. upstream=true gives its sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urn": {"type": "string", "description": "The entity URN."},
                    "upstream": {"type": "boolean", "description": "false for consumers (default), true for sources."},
                    "max_hops": {"type": "integer", "description": "How many hops to walk. Default 1."},
                },
                "required": ["urn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entities",
            "description": "Fetch full metadata for a batch of URNs at once. Cheaper than one call per entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urns": {"type": "array", "items": {"type": "string"}, "description": "URNs to fetch."},
                },
                "required": ["urns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_description",
            "description": "Write a description onto a dataset or column. Use this to fix documentation gaps you found. In dry-run mode it reports the change without making it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_urn": {"type": "string", "description": "URN of the entity to document."},
                    "description": {"type": "string", "description": "The description to write. Be specific and factual about what the data holds."},
                    "operation": {"type": "string", "enum": ["replace", "append", "remove"], "description": "Default replace."},
                    "column_path": {"type": "string", "description": "Set only when documenting a single column."},
                },
                "required": ["entity_urn", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_tags",
            "description": "Attach tags to one or more entities. Plain names are fine, they get turned into tag URNs. In dry-run mode it reports the change without making it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tag names, e.g. ['PII','deprecated']."},
                    "entity_urns": {"type": "array", "items": {"type": "string"}, "description": "Entities to tag."},
                },
                "required": ["tags", "entity_urns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_owners",
            "description": "Assign owners to entities. Plain usernames are fine. In dry-run mode it reports the change without making it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owners": {"type": "array", "items": {"type": "string"}, "description": "Usernames, e.g. ['datahub']."},
                    "entity_urns": {"type": "array", "items": {"type": "string"}, "description": "Entities to assign."},
                },
                "required": ["owners", "entity_urns"],
            },
        },
    },
]

KIT_IMPLS = {
    "search": search,
    "get_lineage": get_lineage,
    "get_entities": get_entities,
    "update_description": update_description,
    "add_tags": add_tags,
    "add_owners": add_owners,
}

# the write tools, so model.py can warn clearly about what --apply unlocks
MUTATION_TOOLS = {"update_description", "add_tags", "add_owners"}

# this is the hands of the agent. every function here is one thing winnow can actually
# do against the datahub knowledge graph. model.py is the brain that decides which of
# these to call, this file is what does the real work.
#
# every api call below was checked against acryl-datahub 1.7.0 before being written here.

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig
from datahub.ingestion.graph.openapi import RelationshipDirection
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    EditableDatasetPropertiesClass,
    GlobalTagsClass,
    OperationClass,
    OwnershipClass,
    UpstreamLineageClass,
)

# the gms metadata api. this is the port the agent talks to, NOT 9002 which is the web ui.
GMS_SERVER = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")

# "who depends on me" is an INCOMING DownstreamOf edge: the consumer declares itself
# downstream of us, so the arrow points at us. OUTGOING would give our own upstreams.
_DOWNSTREAM = RelationshipDirection.INCOMING
_UPSTREAM = RelationshipDirection.OUTGOING


def connect(server: str = GMS_SERVER) -> DataHubGraph:
    """open a connection to datahub and prove it actually works.

    careful: test_connection() returns None on success and RAISES on failure, so
    `if graph.test_connection():` is always falsy and silently swallows a healthy
    server. we let the exception be the failure signal instead.
    """
    graph = DataHubGraph(DataHubGraphConfig(server=server))
    graph.test_connection()  # raises if the server is unreachable
    return graph


def _urn_platform(urn: str) -> str:
    """pull 'snowflake' out of urn:li:dataset:(urn:li:dataPlatform:snowflake,name,PROD)."""
    try:
        return urn.split("dataPlatform:")[1].split(",")[0]
    except IndexError:
        return "unknown"


def _urn_name(urn: str) -> str:
    """pull the dotted table name out of a dataset urn."""
    try:
        return urn.split(",")[1]
    except IndexError:
        return urn


def _short_name(urn: str) -> str:
    """just the final segment, e.g. 'order_details'. used for duplicate detection."""
    return _urn_name(urn).split(".")[-1].lower()


def _description_of(graph: DataHubGraph, urn: str) -> Optional[str]:
    """a dataset can be described in two places and either one counts as documented.

    the editable aspect is what someone types into the ui, the properties aspect is
    what the ingestion source supplied. we treat either as 'has a description'.
    """
    editable = graph.get_aspect(urn, EditableDatasetPropertiesClass)
    if editable and editable.description and editable.description.strip():
        return editable.description
    props = graph.get_aspect(urn, DatasetPropertiesClass)
    if props and props.description and props.description.strip():
        return props.description
    return None


def _tags_of(graph: DataHubGraph, urn: str) -> List[str]:
    """return plain tag names, stripping the urn:li:tag: prefix off each one."""
    tags = graph.get_aspect(urn, GlobalTagsClass)
    if not tags or not tags.tags:
        return []
    return [t.tag.split(":")[-1] for t in tags.tags]


def _last_touched_ms(graph: DataHubGraph, urn: str) -> Optional[int]:
    """best available 'when did this last change' timestamp, in epoch millis.

    an Operation aspect is the strongest signal because it means something actually
    wrote to the table. we fall back to the lastModified on dataset properties, which
    only tells us when the metadata changed, so it is weaker evidence.
    """
    op = graph.get_aspect(urn, OperationClass)
    if op and op.lastUpdatedTimestamp:
        return op.lastUpdatedTimestamp
    props = graph.get_aspect(urn, DatasetPropertiesClass)
    if props and props.lastModified and props.lastModified.time:
        return props.lastModified.time
    return None


def list_datasets(
    graph: DataHubGraph, platform: Optional[str] = None, limit: int = 200
) -> List[str]:
    """every dataset urn, optionally narrowed to one platform like 'snowflake'.

    note platform is a real parameter here. it is not a 'platform:snowflake' search
    string, that syntax does nothing.
    """
    urns = graph.get_urns_by_filter(entity_types=["dataset"], platform=platform)
    return list(urns)[:limit]


def downstream_count(graph: DataHubGraph, urn: str) -> int:
    """how many assets depend on this one. zero means nothing would break if it went away."""
    return len(list(graph.get_related_entities(urn, ["DownstreamOf"], _DOWNSTREAM)))


# ---------------------------------------------------------------------------
# the actual audits. each returns plain dicts/lists so the llm can read them.
# ---------------------------------------------------------------------------


def find_missing_descriptions(
    graph: DataHubGraph, platform: Optional[str] = None, limit: int = 25
) -> List[Dict]:
    """datasets nobody has documented. the most common and most fixable metadata gap."""
    out = []
    for urn in list_datasets(graph, platform):
        if _description_of(graph, urn) is None:
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "downstream_consumers": downstream_count(graph, urn),
                }
            )
        if len(out) >= limit:
            break
    # something with consumers and no docs hurts more than an orphan nobody reads
    out.sort(key=lambda d: -d["downstream_consumers"])
    return out


def find_missing_tags(
    graph: DataHubGraph, platform: Optional[str] = None, limit: int = 25
) -> List[Dict]:
    """datasets with no tags at all, so they cannot be found by any tag based search."""
    out = []
    for urn in list_datasets(graph, platform):
        if not _tags_of(graph, urn):
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "has_description": _description_of(graph, urn) is not None,
                }
            )
        if len(out) >= limit:
            break
    return out


def find_stale_datasets(
    graph: DataHubGraph,
    days: int = 90,
    platform: Optional[str] = None,
    limit: int = 25,
) -> List[Dict]:
    """datasets that have not been written to in `days`, plus anything with no timestamp
    at all (unknown age is its own kind of problem worth surfacing)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    out = []
    for urn in list_datasets(graph, platform):
        ts = _last_touched_ms(graph, urn)
        if ts is None:
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "last_modified": None,
                    "days_stale": None,
                    "note": "no timestamp recorded, age unknown",
                }
            )
        elif ts < cutoff_ms:
            age = (datetime.now(timezone.utc) - datetime.fromtimestamp(ts / 1000, timezone.utc)).days
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "last_modified": datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat(),
                    "days_stale": age,
                    "downstream_consumers": downstream_count(graph, urn),
                }
            )
        if len(out) >= limit:
            break
    return out


def find_broken_lineage(graph: DataHubGraph, limit: int = 25) -> List[Dict]:
    """lineage edges pointing at assets that do not exist.

    this is how a lineage graph rots: an upstream table gets deleted but the edge
    declaring the dependency stays behind, so the graph claims a source that is gone.
    """
    out = []
    for urn in list_datasets(graph):
        lineage = graph.get_aspect(urn, UpstreamLineageClass)
        if not lineage or not lineage.upstreams:
            continue
        dangling = []
        for up in lineage.upstreams:
            try:
                if not graph.exists(up.dataset):
                    dangling.append(up.dataset)
            except Exception:
                dangling.append(up.dataset)
        if dangling:
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "missing_upstreams": dangling,
                    "broken_edge_count": len(dangling),
                }
            )
        if len(out) >= limit:
            break
    return out


def find_duplicate_assets(graph: DataHubGraph, limit: int = 25) -> List[Dict]:
    """assets sharing a table name.

    same name on the SAME platform is a genuine duplicate worth investigating. the
    same name across platforms is usually just the same table modelled twice (dbt
    building a snowflake table, a looker view on top of it), so those are reported
    separately rather than treated as a problem.
    """
    by_name = defaultdict(list)
    for urn in list_datasets(graph):
        by_name[_short_name(urn)].append(urn)

    same_platform, cross_platform = [], []
    for name, urns in by_name.items():
        if len(urns) < 2:
            continue
        platforms = [_urn_platform(u) for u in urns]
        entry = {"name": name, "count": len(urns), "platforms": platforms, "urns": urns}
        if len(set(platforms)) < len(platforms):
            same_platform.append(entry)
        else:
            cross_platform.append(entry)

    same_platform.sort(key=lambda d: -d["count"])
    cross_platform.sort(key=lambda d: -d["count"])
    return [
        {
            "likely_duplicates_same_platform": same_platform[:limit],
            "same_name_across_platforms": cross_platform[:limit],
            "note": "cross-platform matches are usually legitimate mirrors, not duplicates",
        }
    ]


def find_unused_dashboards(graph: DataHubGraph, limit: int = 25) -> List[Dict]:
    """dashboards wired to nothing.

    a dashboard with no charts and no datasets feeding it is not showing anyone
    anything, which makes it a safe cleanup candidate.
    """
    from datahub.metadata.schema_classes import DashboardInfoClass

    out = []
    for urn in graph.get_urns_by_filter(entity_types=["dashboard"]):
        info = graph.get_aspect(urn, DashboardInfoClass)
        if info is None:
            continue
        chart_count = len(info.charts or [])
        dataset_count = len(info.datasets or [])
        if chart_count == 0 and dataset_count == 0:
            out.append(
                {
                    "urn": urn,
                    "title": info.title,
                    "chart_count": chart_count,
                    "dataset_count": dataset_count,
                    "reason": "no charts and no datasets attached",
                }
            )
        if len(out) >= limit:
            break
    return out


def find_orphan_datasets(
    graph: DataHubGraph, platform: Optional[str] = None, limit: int = 25
) -> List[Dict]:
    """datasets with zero downstream consumers. this is the original winnow job:
    nothing reads them, so deleting them breaks nothing."""
    out = []
    for urn in list_datasets(graph, platform):
        consumers = downstream_count(graph, urn)
        if consumers == 0:
            out.append(
                {
                    "urn": urn,
                    "name": _urn_name(urn),
                    "platform": _urn_platform(urn),
                    "downstream_consumers": 0,
                    "has_description": _description_of(graph, urn) is not None,
                }
            )
        if len(out) >= limit:
            break
    return out


def describe_dataset(graph: DataHubGraph, urn: str) -> Dict:
    """everything we know about one dataset, for when the agent wants to drill in."""
    ownership = graph.get_aspect(urn, OwnershipClass)
    ts = _last_touched_ms(graph, urn)
    return {
        "urn": urn,
        "name": _urn_name(urn),
        "platform": _urn_platform(urn),
        "description": _description_of(graph, urn),
        "tags": _tags_of(graph, urn),
        "owners": [o.owner.split(":")[-1] for o in (ownership.owners if ownership else [])],
        "last_modified": datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat() if ts else None,
        "downstream_consumers": downstream_count(graph, urn),
    }


def catalog_summary(graph: DataHubGraph) -> Dict:
    """counts by entity type and platform, so the agent can orient before drilling in."""
    summary = {}
    for et in ["dataset", "dashboard", "chart", "dataFlow", "dataJob", "container", "domain"]:
        try:
            summary[et] = len(list(graph.get_urns_by_filter(entity_types=[et])))
        except Exception as e:
            summary[et] = f"error: {type(e).__name__}"

    platforms = defaultdict(int)
    for urn in list_datasets(graph, limit=10000):
        platforms[_urn_platform(urn)] += 1

    return {"entity_counts": summary, "dataset_platforms": dict(platforms)}

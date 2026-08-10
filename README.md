# Winnow — a data catalog hygiene agent for DataHub

Winnow audits a DataHub catalog for metadata rot and can fix part of it. It finds
undocumented datasets, orphans nothing consumes, stale tables, broken lineage,
duplicate assets, unused dashboards and untagged data — then writes descriptions,
tags and owners back through DataHub's official
[Agent Context Kit](https://docs.datahub.com/docs/dev-guides/agent-context/agent-context),
asking for your approval before each change.

The LLM is `openai/gpt-oss-120b` served by [Groq](https://groq.com).

## How it fits together

| File | Role |
|---|---|
| `model.py` | The agent. Groq client, tool schemas, the tool-calling loop, the approval prompt. Start here. |
| `datahub_tools.py` | Detection. Nine opinionated audits that decide what counts as a problem. |
| `agent_context_tools.py` | Remediation. A curated subset of the DataHub Agent Context Kit, gated behind approval. |
| `winnow_brain.py` | The original prototype. Superseded by the two modules above; kept for history. |

Detection is deliberately hand-written rather than delegated to generic search. A
query like "undocumented **and** has downstream consumers, worst first" encodes
judgement that a keyword search cannot express. The Kit supplies what would be
expensive to rebuild: broad search, multi-hop lineage, and the write operations.

## Setup

### 1. DataHub

```bash
pip install acryl-datahub
datahub docker quickstart
```

Needs Docker Desktop running, and pulls several GB on first run. When it finishes the
UI is at http://localhost:9002 (`datahub` / `datahub`) and the metadata API — what
this project talks to — is at http://localhost:8080.

Load the sample catalog:

```bash
datahub init          # accept the defaults for a local instance
datahub datapack load showcase-ecommerce
```

That gives you 67 datasets across Snowflake, dbt, Postgres, S3, Tableau, Power BI and
Looker, with real lineage between them.

### 2. Python

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows;  source venv/bin/activate elsewhere

pip install -r requirements.txt
pip install --no-deps datahub-agent-context
```

**Both install steps are required, in that order.** `datahub-agent-context` hard-pins
`acryl-datahub==1.6.0.6`; installing it normally downgrades your working 1.7.0 and
breaks every call in this project. `--no-deps` keeps 1.7.0 and `requirements.txt`
supplies the kit's transitive dependencies explicitly. The kit was verified to run
against 1.7.0.

Expect pip to warn about the version conflict. That warning is the intended outcome.

### 3. Your Groq key

```bash
copy .env.example .env           # cp on macOS/Linux
```

Then put your own key in it. Free keys from https://console.groq.com/keys work; this
was built on one. `.env` is gitignored.

## Usage

```bash
python model.py "which datasets are missing descriptions?"   # one-shot
python model.py                                              # interactive
```

Read-only by default: proposed changes are reported, never made.

### Making changes

```bash
python model.py --apply "add a description to order_history"
```

`--apply` arms the write tools. It does **not** hand the agent free rein — every
individual write pauses and waits for you:

```
------------------------------------------------------------------
  PROPOSED CHANGE: update_description
    entity_urn: urn:li:dataset:(urn:li:dataPlatform:snowflake,...order_history,PROD)
    operation: replace
    description: Historical order snapshot table
------------------------------------------------------------------
  apply this change? [y/N]
```

Anything but `y` declines, and the agent is told not to retry it. A bare Enter is a
decline. With no interactive terminal — piped input, CI — writes decline
automatically rather than treating silence as consent.

`--yes` skips the prompts entirely. Only for unattended runs, and it means the agent
can rewrite many datasets without asking.

## Troubleshooting

**`port is already allocated` on 9200 during quickstart.** Something else owns that
port — another Elasticsearch or OpenSearch, or a security stack like Wazuh whose
indexer is an OpenSearch fork. `datahub-local.yml` here is a stock quickstart compose
file with the search service moved to host port 9250:

```bash
datahub docker quickstart --quickstart-compose-file datahub-local.yml
```

Note the CLI's `--elastic-port` flag does *not* fix this: the compose file hardcodes
`published: '9200'` and never reads the variable that flag sets.

**`KeyError: 'Did not find a registered class for c'` when loading the datapack.**
Windows only. DataHub resolves a filesystem backend by URL-parsing the path, so
`C:\Users\...` parses with scheme `"c"` — your drive letter — and no `"c"` backend
exists. `datapack_load_win.py` patches it at runtime without modifying the installed
package:

```bash
python datapack_load_win.py load showcase-ecommerce
```

**`Request too large ... tokens per minute (TPM): Limit 8000`.** Groq's free tier
budgets tokens per *minute*, and the whole conversation is resent on every step. The
agent trims its own history to stay inside that, but a very broad question can still
trip it. Ask something narrower, or upgrade the Groq tier.

**`Python versions above 3.11 are not actively tested`.** A notice from the DataHub
CLI about their test matrix, not a real incompatibility. 3.12 works; this was built
on it.

**`UnicodeEncodeError` on Windows.** Fixed in `model.py`, which reconfigures stdout to
UTF-8 at startup. If you see it elsewhere, `set PYTHONIOENCODING=utf-8`.

## Notes on accuracy

The agent is instructed never to invent URNs or counts, and to distinguish weak
evidence from strong. Some specifics worth knowing when reading its output:

- A dataset with no recorded timestamp is reported as **unknown age**, not as
  confirmed stale.
- Same-name assets on *different* platforms are usually legitimate mirrors — a dbt
  model materialised in Snowflake and exposed via Looker — so they are reported
  separately from same-platform duplicates.
- "Zero downstream consumers" means nothing in the catalog reads it. Ad-hoc queries
  and external jobs are invisible to DataHub, so it is a strong signal, not proof.
- Self-referencing lineage edges are excluded from consumer counts. A job that reads
  and rewrites its own table would otherwise look like a consumer and hide a
  genuine orphan.
- Results are capped per call, but every audit reports the true total separately from
  the rows shown, so counts stay correct even when the list is truncated.

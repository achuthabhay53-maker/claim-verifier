# Claim Evidence Finder

Type a research claim. The app finds papers on PubMed and Semantic Scholar, then checks whether each paper
**really** supports the claim, how **directly**, and how **strongly**.

It is built as an evidence-validation tool, not a paper summarizer. For each paper it answers three questions:

1. Does the paper actually discuss the claim?
2. Does it support or contradict it?
3. How strong and direct is that evidence?

## How it works

```
claim ──► 1. Search ──► 2. Validate ──► 3. Second search (automatic, only if needed)
```

**1. Search** ([find_papers.py](find_papers.py))
- Medical topics search **PubMed only**. All other topics search **PubMed + Semantic Scholar**.
  The topic is auto-detected from keywords and can be overridden in the sidebar.
- Results are merged and de-duplicated (by DOI, then PMID, then title).
- Citation counts are added with a Semantic Scholar lookup by DOI/PMID (PubMed has no citation counts).

**2. Validate** ([evidence.py](evidence.py))
- Claude splits the claim into **intervention, comparator, outcome, population** and decides which
  **outcome level** the claim needs (see below).
- For every paper, Claude fills in a structured checklist: study design, sample size, intervention,
  comparator, the outcomes actually measured, limitations and key quotes.
- Rules written in code then downgrade over-generous labels. Every downgrade shows its reason on the page.
- Each quote is checked to appear **word for word** in the abstract. Invented quotes are flagged.

**3. Second search**
If no on-topic paper answers the claim at the outcome level it needs, the app runs one more search aimed
at that outcome (for example complications or survival), checks only the new papers and merges them in.

### The outcome ladder

A claim about a higher level cannot be proven by evidence from a lower one. Better simulator scores do not
show that patients survive more often.

| Level | What was measured |
|------:|-------------------|
| 1 | Knowledge, satisfaction, confidence, cognitive load |
| 2 | Performance inside the training tool or simulator |
| 3 | Skill assessed on a standard task (for example OSATS) |
| 4 | Performance or errors in real operations / practice |
| 5 | Patient outcomes (complications, survival, safety) |

### Labels

| Label | Meaning |
|-------|---------|
| `SUPPORTS` | Direct evidence for the claim at the level it needs |
| `PARTIALLY_SUPPORTS` | Related but not an exact match (a variant intervention, a different comparator) |
| `CONTRADICTS` | Evidence against the claim |
| `MIXED` | Findings point both ways |
| `INSUFFICIENT_EVIDENCE` | On topic, but did not measure what the claim needs |
| `IRRELEVANT` | About a different intervention |

Strength is `strong`, `moderate`, `weak` or `indirect`, based on study design, sample size and directness.

## Setup

Requires Python 3.13 (older 3.x versions probably work, but only 3.13 was used).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then fill in your keys
```

`.env` (never committed):

```
S2_API_KEY=...          # Semantic Scholar key; free, avoids rate limiting
ANTHROPIC_API_KEY=...   # used for claim analysis and evidence validation
```

Both keys are optional to start: without `S2_API_KEY` searches still work but are rate-limited, and without
`ANTHROPIC_API_KEY` you can search but not validate. You can also paste the Anthropic key in the sidebar.

## Run

Double-click [run.bat](run.bat), or:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Use the `.venv` interpreter every time. A different Python can have different package versions, which causes
confusing errors.

Then:
1. Enter your claim (optionally a custom search query, which can be a PubMed boolean query).
2. Click **1. Search papers**.
3. Click **2. Validate evidence for the claim**.
4. Read the headline result, the table (downloadable as CSV) and the per-paper details.

## Tests

The default suite is offline: no network and no API keys.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

Smoke tests against the real services are skipped unless you opt in:

```powershell
$env:RUN_LIVE = "1"; .venv\Scripts\python.exe -m pytest tests/test_live.py
```

## Logs

Everything is logged to `logs/research.log` (rotating, API keys never logged) and to the console. The page
has an **Activity log** panel at the bottom. The log shows every HTTP attempt, retries, routing decisions,
and the full traceback for any failure.

## GitHub automation (Gemini)

Both workflows use [`google-github-actions/run-gemini-cli`](https://github.com/google-github-actions/run-gemini-cli)
and need one repository secret named `GEMINI_API_KEY` (Settings > Secrets and variables > Actions; get a key
from Google AI Studio).

| Workflow | What it does |
|----------|--------------|
| [gemini-scheduled.yml](.github/workflows/gemini-scheduled.yml) | Weekdays 06:00 UTC (and on demand): runs the tests, has Gemini explain the result, opens or updates an issue if they fail |
| [gemini-chat.yml](.github/workflows/gemini-chat.yml) | Comment `@gemini-cli <question>` on an issue or pull request and Gemini replies. Only owners, members and collaborators can trigger it |

## Project layout

```
app.py              Streamlit page
find_papers.py      Search, retries, parsing, merging, citation lookup, logging
evidence.py         Claim breakdown, paper checklist, rules, quote check, summary, second search
tests/              Offline tests (+ opt-in live tests)
.github/workflows/  Gemini automation
```

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `unexpected keyword argument ...` right after code changes | Streamlit is still running old code. Stop it (Ctrl+C) and start it again |
| `Client.__init__() got an unexpected keyword argument 'proxies'` | Old `anthropic` package. Run `pip install -U anthropic` **inside the `.venv`** |
| `Your credit balance is too low` | The Anthropic account needs credits (Plans & Billing in the Anthropic console) |
| Semantic Scholar `HTTP 429` in the log | Rate limit. The app retries and continues; set `S2_API_KEY` to reduce it |
| Citation counts show `n/a` | The lookup was rate limited or the paper has no DOI/PMID match. Searching still works |
| Every paper shows `ERROR` | Open the Activity log; the traceback says why (key, credits, network) |

## Limitations

- Papers are judged from **abstracts only**. Abstracts often omit sample sizes, bias and secondary outcomes.
  Reading full texts (for example via Unpaywall or CORE) is the main next step.
- Each search looks at the top results per source, so "insufficient evidence" means *none of the papers
  found*, not *no such evidence exists*.
- Outcome levels are assigned by the model and can vary slightly between runs. The final label is also
  shaped by the rules in code, which makes it more stable than the raw model answer.
- The medical-topic detection is a keyword list, a rough guess. Use the sidebar override when it is wrong.
- This tool supports a human reviewer. It does not replace one.

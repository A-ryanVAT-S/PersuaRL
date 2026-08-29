# InsureDial

A persuasive **motor insurance** dialogue dataset: 1,931 multi-turn conversations
between a user and an insurance agent, annotated across four dimensions
(persuasion strategy, user intent, key domain terms, sentiment).

Built with a semi-automated, human-in-the-loop pipeline: 50 seed dialogues written
by trained annotators, expanded with GPT-4o under a prompt selected by expert
rating, then reviewed and filtered by human annotators. Initial labels came from
Gemini-2.0-Flash and were verified and corrected by five trained annotators
(Appendix B of the paper).

---

## Layout

```
data/
├── insuredial/
│   ├── dialogues.csv                    the corpus
│   └── expert_outputs/
│       ├── engagement.csv               persuasion strategy per turn
│       ├── intent.csv                   user intent per turn
│       ├── keyterm.csv                  domain terms per turn
│       └── sentiment.csv                emotional tone per turn
└── processed/                           built by scripts/prepare_data.sh (gitignored)
    ├── experts/<expert>.csv             (utterance, label, reason) for expert SFT
    └── classifiers/<dimension>.csv      (utterance, label) for the R1/R2 classifiers
```

`processed/` is derived, never edited by hand, and not versioned. Rebuild it with:

```bash
bash scripts/prepare_data.sh
```

---

## `insuredial/dialogues.csv`

| column | type | description |
|---|---|---|
| `conversation_id` | int | dialogue identifier, 1–1931 |
| `turn_no` | int | turn index within the dialogue (**odd numbers only** — see below) |
| `user_utterance` | str | the user's message |
| `new_agent_reply` | str | the reference agent response (the SFT / RL target) |

13,383 rows. `turn_no` runs 1, 3, 5, … because the original transcript numbered
user and agent turns separately; each row already pairs a user turn with the
agent turn that followed it. Nothing is missing — a row *is* one exchange.

| statistic | train | validation | test |
|---|---|---|---|
| dialogues | 1,545 | 97 | 289 |
| utterances | 21,134 | 1,462 | 4,170 |
| avg. turns / dialogue | 6.84 | 7.54 | 7.21 |
| avg. words / user turn | 14.38 | 17.21 | 17.19 |
| avg. words / agent turn | 53.78 | 59.74 | 56.94 |

> **Split on `conversation_id`, never on rows.** Turns in one dialogue share a
> persona, a vehicle and phrasing; a row-level split leaks the test set into
> training and inflates every lexical metric. `persuarl.data.splits` enforces
> this and every config goes through it.

---

## `insuredial/expert_outputs/*.csv`

Each expert's answer for every turn, precomputed once. Training and inference
look answers up by `(conversation_id, turn_no)` — the answers do not depend on
the Selector or the Generator, so recomputing them inside the RL loop would burn
GPU hours on identical results.

| column | description |
|---|---|
| `conversation_id`, `turn_no` | join key into `dialogues.csv` |
| `utterance` | the user turn the answer describes |
| `<expert>_answer` | the expert's rendered output |

Answer formats, one per expert:

| expert | rendered answer | label space |
|---|---|---|
| `engagement` | `The persuasion strategy is {label} and the reason is {reason}` | logical, emotional, credibility, personal, persona, default |
| `intent` | `Label: {label}` / `Reason: {reason}` — but the shipped file deviates, see [Known issues](#known-issues-in-the-shipped-files) | 6 intents (see below) |
| `keyterm` | `The keyterm extracted are {terms}` | open vocabulary |
| `sentiment` | `The sentiment is {label}` | positive, neutral, negative |

`prepare_data.sh` parses these back into `(label, reason)` columns using the
per-expert patterns in `src/persuarl/experts/registry.py`. Rows that do not parse
are reported and dropped, not guessed at.

### Label spaces

**Engagement strategy** (Appendix C.2.1)

| label | definition |
|---|---|
| Logical | factual reasoning, feature comparisons, cost–benefit arguments |
| Credibility | trustworthiness, reliability, reputation of the insurer |
| Emotional | reassurance, security, peace of mind |
| Personal | addresses the user's explicitly stated needs or concerns |
| Persona | aligns with the user's lifestyle, habits or preferences |
| Default | neutral, informative, no explicit persuasive framing |

**User intent** (Appendix C.2.2): `Request_Insurance_Quote`,
`Ask_Coverage_Details`, `Express_Concern`, `Request_Additional_Info`,
`Confirm_Interest`, `Ask_Price_or_Premium`.

**Sentiment**: `positive`, `neutral`, `negative`.

**Key terms** — open vocabulary. Common ones: comprehensive coverage, third-party
liability, roadside assistance, zero depreciation, deductibles, policy renewal,
personal accident cover, IDV. Vehicle-specific terms ("2024 Tesla Model 3", "EV")
are also extracted.

---

## Known issues in the shipped files

**`intent.csv` was replaced (previously it duplicated `engagement.csv`).** The
file shipped earlier was an export mistake: its `intent_answer` column held
persuasion-strategy prose ("The persuasion strategy is Logical appeal…") whose
body was byte-identical to `engagement_answer` for all 13,383 rows. It has been
replaced with genuine intent annotations. The current file covers **13,383 turns
across all 1,931 conversations**, joins cleanly onto `dialogues.csv` with zero
orphan rows, and uses all six intent labels:

| label | rows |
|---|---|
| `Ask_Price_or_Premium` | 2,793 |
| `Express_Concern` | 2,777 |
| `Ask_Coverage_Details` | 2,336 |
| `Confirm_Interest` | 2,197 |
| `Request_Insurance_Quote` | 2,001 |
| `Request_Additional_Info` | 1,279 |

**The replacement uses a different schema, and the loader does not yet read it.**
Unlike the other three expert CSVs, it carries the label in its own `label`
column and leaves `intent_answer` holding *only* the reason prose:

```
conversation_id,turn_no,speaker,utterance,new_agent_reply,label,intent_answer
1,1,User,"Hi, I'm looking to get motor insurance…",…,Request_Insurance_Quote,"The user is expressing a desire to initiate the process of obtaining motor insurance…"
```

`load_expert_outputs` returns only `[conversation_id, turn_no, intent_answer]`,
so the `label` column is dropped on load; `parse_answer("intent", …)` is then
handed a bare reason with no label in it. The `INTENT.answer_pattern` in
`src/persuarl/experts/registry.py` matches `The intent is X and the reason is Y`
or `Label: X\nReason: Y`, and matches **0 of 13,383** rows in this format. Until
that is reconciled, the downstream effects are unchanged from before:

- **Intent expert SFT** — `prepare_data.sh` reports all 13,383 intent rows as
  unparseable and drops them, so `processed/experts/intent.csv` comes out empty.
- **The R2 intent classifier** has no training data, so
  `processed/classifiers/intent.csv` will be empty or tiny.
- **The R2 reward** then contributes 0. Training still runs — the reward code
  degrades gracefully — but the intent-consistency term is inert and the D.3.1
  ablation for R2 is not meaningful.

Two ways to reconcile it. Either fold the label back into the answer column so
the shipped file matches the documented `Label:` / `Reason:` format:

```bash
python - <<'EOF'
import pandas as pd
p = "data/insuredial/expert_outputs/intent.csv"
d = pd.read_csv(p)
d["intent_answer"] = ("Label: " + d["label"].astype(str).str.strip()
                      + "\nReason: " + d["intent_answer"].fillna("").astype(str).str.strip())
d.drop(columns=["label"]).to_csv(p, index=False)
EOF
```

…or teach `load_expert_outputs` / `build_expert_training_file` to prefer an
explicit `label` column when the file provides one. Regenerating from your own
intent expert also produces the canonical format:

```bash
bash scripts/experts/run_expert_inference.sh --expert intent
```

**22 rows** have a label but an empty `intent_answer` (missing reason). They
carry a usable classifier target but no SFT reason text.

**Coverage.** `sentiment.csv` contains 497 rows whose `(conversation_id,
turn_no)` pair does not exist in `dialogues.csv` — mostly even turn numbers, i.e.
agent turns that were annotated but never paired into an exchange. The inner join
in `merge_expert_outputs` drops them, which is the correct behaviour: a route can
then never reference an answer for a turn the model is not being trained on.

The practical effect is that the four-expert merge yields **12,886 turns across
all 1,931 conversations**, not 13,383. Engagement and keyterm cover all 13,383;
sentiment is the binding constraint. No conversation is lost.

`engagement.csv` and `intent.csv` also carry extra `speaker` and
`new_agent_reply` columns (which is why they are ~15 MB and ~10 MB against
keyterm's 2 MB); both are ignored on load.

---

## Versioning these files

The corpus and the four expert CSVs total roughly 40 MB. That is comfortably
within plain Git's limits (GitHub warns per-file above 50 MB and blocks above
100 MB), so these files are tracked normally -- no Git LFS, and therefore no LFS
bandwidth quota burned every time someone clones.

If you regenerate the expert outputs at a much larger scale and individual files
approach 50 MB, switch them to LFS:

```bash
git lfs install
git lfs track "data/insuredial/**/*.csv"
git add .gitattributes
```

---

## Out-of-domain evaluation

The paper's cross-domain results use **DEAL** (Priya et al., 2024), a travel
negotiation dataset. It is not redistributed here. To evaluate on it, convert it
to the `dialogues.csv` schema above, annotate it with
`scripts/experts/run_expert_inference.sh`, and point
`configs/inference/persuarl.yaml` at the result:

```bash
python -m persuarl.cli.run_pipeline --config configs/inference/persuarl.yaml \
    --set data.dialogues_path=data/deal/dialogues.csv \
    --set output_path=results/persuarl_deal.csv
```

---

## Ethics and intended use

InsureDial is a research artefact for studying persuasive dialogue as **decision
support**, not decision enforcement. Responses are meant to align with user
intent, sentiment and context without coercive or repetitive pressure. Because
inaccuracies in the insurance domain can cause financial harm, the corpus was
built with domain experts verifying model-generated dialogues.

The dialogues are synthetic, generated by GPT-4o from human-written seeds and
filtered by humans. They may carry synthetic artefacts and do not fully capture
real user behaviour — treat performance on InsureDial as evidence about the
method, not as a deployment readiness signal. Company names appearing in the
dialogues are incidental to the generation process and are not endorsements.

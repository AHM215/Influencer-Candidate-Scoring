# Influencer / Boutique Candidate Scoring

Scores and ranks prospective Boutique owners so Celebrity Management can decide where to
spend scarce onboarding capacity, **before** that capacity is committed.

Input: a public Instagram profile (live capture, or a saved snapshot).
Output: a ranked shortlist, and for each candidate a one-page report carrying a **Fit**
score, a **Success Propensity**, the signals driving both, and an **onboard / hold / pass**
recommendation with the reason stated in numbers a CM lead can argue with.

```
$ score shortlist

 1. The ideal Boutique Owner              fit  90.3  propensity  64%  -> ONBOARD
 2. Layla (representative Candidate)      fit  87.8  propensity  59%  -> ONBOARD
 3. Large account, wrong category         fit  36.4  propensity  63%  -> HOLD
 4. Right person, no selling track record fit  78.7  propensity  17%  -> HOLD
 5. Bought engagement                     fit  80.8  propensity  12%  -> HOLD
 6. Strong Candidate, safety concern      fit  88.8  propensity  64%  -> PASS
```

---

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

score shortlist                      # rank everything, write out/index.html
                                     # (a committed copy lives in sample/)
score candidate layla.beauty.kw      # one Candidate Report
score validate                       # the validation harness
pytest                               # 70 tests, fully offline

score capture <handle>               # live public capture (rate limited, see below)
score extract <handle>               # record the LLM extraction (needs OPENAI_API_KEY)
```

No credentials are required for anything except `score extract`. Copy `.env.example` to
`.env` if you want to re-record extractions.

---

## The central decision: two scores, not one

The brief asks for "a fit/success score". That slash hides two different things, and
collapsing them is how this kind of system becomes unfalsifiable.

- **Fit** is a property of the candidate *as they are today* — category, audience
  geography, language, scale, brand safety. It is checkable right now, with no historical
  outcome data of any kind.
- **Success Propensity** is a claim about *a future that has not happened* — the chance
  they clear the Performance Bar in their first 90 days.

Blended into one number, you cannot say what the score predicts, so you cannot validate
it, and "hold" degrades to "the number landed in the middle". Kept apart, the
recommendation becomes an explicit policy over the pair, and every branch is a sentence
someone can disagree with:

| | Propensity < 0.5× base | < 1× base | 1–2× base | ≥ 2× base |
|---|---|---|---|---|
| **Brand safety flag** | pass | pass | pass | pass |
| **Fit < 50** | pass | pass | hold | hold |
| **Fit ≥ 50** | pass | **hold** | hold | onboard |

The bolded cell is deliberate. A candidate who fits well but has no *evidence* of selling
scores near the base rate through having nothing to show — and "pass" means don't come
back. Absent evidence is not negative evidence, so they are held and re-checked. Below
half the base rate the evidence is no longer absent but bad, and the pass applies to them
too. *(This asymmetry was found by an archetype regression, not by design — see Validation.)*

---

## How Success Propensity is computed

**No model is trained.** There are no real onboarding outcomes to train on, so a
classifier fitted to the synthetic cohort would recover our own generator, and any
reported accuracy would measure the pipeline rather than the hypothesis.

Instead: start from a stated base rate, and let each signal move the log-odds by an
individually argued likelihood ratio.

```
P(clears bar) = 20% base rate, then × 1.84 (strong commercial evidence)
                                × 1.48 (content style suits selling)
                                × 0.71 (posts only 1.2×/week)  ...
```

Every number is a claim you can dispute on domain grounds rather than an artefact of
fitting, explanations fall out for free because log-odds contributions are additive, and
— the decisive argument — **these likelihood ratios are exactly the quantities you
re-estimate when real outcomes arrive.** The prototype is the production model's
skeleton, not a throwaway.

Constraints that keep it honest: every likelihood ratio is capped to 0.5×–2.0×, the
posterior is clipped to [2%, 85%], and correlated signals are grouped (below). A flawless
candidate reaches ~76%, never certainty.

---

## The twelve signals

| Signal | Construct | Provenance | Why it earns a place |
|---|---|---|---|
| category alignment | Fit | inferred | Beauty/fragrance/fashion is what a Boutique sells |
| GCC audience share | Fit | **inferred** | The most important fit signal, and not observable — see Limitations |
| language fit | Fit | inferred | Arabic/English mix for a GCC storefront |
| audience scale | Fit | observed | Log-scaled; below 5k there is no audience to sell to |
| brand safety | Fit | inferred | A **veto**, not a weight |
| engagement rate | Propensity | observed | Closest free proxy for an audience that acts |
| comment/like ratio | Propensity | observed | Comments cost more than likes; conversation converts |
| engagement consistency | Propensity | observed | Reliable audience vs occasional virality |
| posting cadence | Propensity | observed | Non-monotonic: too little goes stale, too much is filler |
| authenticity plausibility | Propensity | inferred | Is the engagement plausible for this follower tier |
| commercial evidence | Propensity | inferred | **Strongest signal**: they have already asked this audience to buy |
| selling content style | Propensity | inferred | Tutorials/hauls/reviews sell natively; pure aesthetic does not |

Fit weights: category 35, GCC audience 30, scale 20, language 15. Brand safety is a hard
veto — a serious flag is not 20 points off, it is a different answer.

Every value carries a **provenance tag** — `observed`, `inferred`, or `mocked` — through
the scorer and onto every line of the report. Nothing inferred can be mistaken for
something measured.

---

## Validation methodology

This is the part the brief weights first, so it is worth being precise about what can and
cannot be validated here.

**What cannot.** There are no real outcomes. Any accuracy figure is measured against
labels we invented, so it validates the *pipeline*, not the *hypothesis*. This system
reports such figures under an explicit health warning and does not lead with them.

**What can.** Three things, and they are the actual harness:

1. **Behavioural properties over the whole cohort** — monotonicity (more of a good signal
   never lowers the score), the fit floor invariant, the brand-safety veto, one
   engagement signal counted per candidate, determinism.
2. **Archetype regressions** — five hand-built reference candidates whose recommendation
   is known in advance: the ideal creator, the large account in the wrong vertical, the
   engagement farm, the brand-safety case, the promising-but-unproven creator. If a weight
   change flips one, the change needs an argument.
3. **Signal ablation** — mute each signal and measure how far the ranking moves. This is
   the closest honest answer to "which signals actually predict success": it shows which
   signals move *our model*, and says plainly that whether they move reality is untested.

```
posting_cadence            rank correlation 0.959
commercial_evidence        rank correlation 0.966
engagement_consistency     rank correlation 0.968
authenticity_plausibility  rank correlation 0.973
selling_content_style      rank correlation 0.985
engagement_rate            rank correlation 0.998
comment_like_ratio         rank correlation 0.999
```

### The harness found two real bugs

Both are worth stating, because a validation suite that never fails is decoration.

**The fit-blind pass threshold.** `archetype_promising_unproven` — right category, right
audience, no selling history — came out `pass`. The propensity rule ignored fit entirely,
so landing two points under the base rate passed a well-fitting candidate on a knife-edge.
Fixed by the asymmetry in the policy table above.

**The engagement family picked the wrong member.** The grouping rule originally took the
*most informative* member by |log LR|. But engagement rate has the widest curve, so it
structurally dominates that comparison — and an engagement farm with inflated likes, no
comments and wild variance was credited for its bought likes, precisely the case the rule
exists to defend against. It now takes the **weakest** member: real engagement corroborates
itself across all three indicators, bought engagement inflates one and leaves the others
thin, so disagreement within the family is itself the evidence. The engagement-farm
archetype dropped from 29% to 12% propensity on that change alone.

### The synthetic cohort, and why it disagrees on purpose

400 candidates from a documented causal process, with a realistic confounder (larger
accounts show systematically lower engagement), label noise, and a 20% base rate.

The generator's outcome function is **deliberately not** the scorer's function. It weights
commercial evidence and audience fit more heavily and engagement less than the scorer's
priors do. If the two matched, the scorer would recover the labels perfectly and prove
nothing. Because they diverge, mediocre agreement is the designed result and disagreement
is *diagnosable*: it shows which of our stated priors would cost us most if reality
disagreed with them.

Reported under that health warning: **AUC 0.710**.

Two honest findings from the diagnostics:

- **The scorer is systematically over-confident.** In the 40–60% band it predicts 48.6%
  where the cohort delivers 29.7%. Against a world we invented that is not proof of
  miscalibration, but it does suggest the likelihood ratios are collectively too generous
  and would want tightening against real data.
- **The fitted-logistic cross-check flags `engagement_rate` and `authenticity_plausibility`
  as disagreeing in sign.** The diagnosis is that the generator makes those two interact
  multiplicatively (only *genuine* engagement converts) while the cross-check fits them
  additively — so the flip is a limitation of the linear diagnostic, not evidence against
  the prior. An earlier version of this finding was real, though: the generator originally
  let bought engagement convert, which is false by construction, and the cross-check caught
  it.

---

## Design decisions

The code refers to these by number.

**ADR-0001 — Fit and Success Propensity are separate scores.** Rejected: one blended
0-100 (unvalidatable, "hold" means nothing); predicting attributed sales directly (a
fitted revenue distribution over mocked outcomes is validating our own noise model).

**ADR-0002 — Argued likelihood ratios, not a classifier trained on the cohort.** Rejected:
logistic regression on synthetic labels (retained only as a harness cross-check); a second
0-100 rubric for propensity (looks like a probability without being one).

**ADR-0003 — Correlated engagement signals are grouped, and only the weakest applies.**
Multiplying three measures of one property triple-counts the evidence. Taking the weakest
makes disagreement inside the family costly, which is the signature of manipulation.
Summing or multiplying the family back together would reintroduce the double-counting.

**ADR-0004 — The cohort generator's outcome function deliberately differs from the
scorer's.** Aligning them destroys the only informative thing the harness measures. The
divergence must not be "fixed".

---

## Architecture

```
src/candidate_scoring/
  domain.py          types; the vocabulary the whole system speaks
  config.py          every tunable number, each with the argument for it
  signals/
    capture.py       adapter boundary: live profile -> snapshot
    derive.py        quantitative signals; no model, no network
    qualitative.py   LLM extraction -> structured signals
  scoring/
    fit.py  propensity.py  policy.py  explain.py
  cohort/            the synthetic cohort
  validation/        the harness and its statistics
  report/            HTML + JSON rendering
```

Files in, files out. No service, no database, no auth.

**The LLM does extraction only, never scoring.** It reads bios and captions and returns
structured values; a deterministic scorer turns those into numbers. Handing a profile to a
model and asking for a score would be unauditable and uncalibratable — it fails the exact
criterion this brief weights first. Extractions are recorded to fixtures, so scoring, the
harness and the entire test suite run offline with no API key. The provider's structured
outputs enforce the schema, and the prompt is generated from the same Pydantic model so
prompt and schema cannot drift. Tolerant parsing and one retry sit behind that, because a
refusal or a truncated reply still arrives as something unparseable and a Signal must
never be silently defaulted.

**Testing**: 70 tests. Unit tests for the scoring maths, property tests for the
invariants, archetype regressions, extraction-parsing tests over the shapes a chat model
actually returns, and report tests asserting that mocked signals are labelled and that an
onboard report still shows its weakest points.

---

## Sample report

`sample/` holds a committed run: open `sample/index.html` for the shortlist, and click
through to any candidate. `sample/<handle>.json` is the same content machine-readable,
including the likelihood-ratio contributions the engagement grouping rule suppressed.
Regenerate with `score shortlist --out sample`.

## On the demo candidate

`layla.beauty.kw` is a **representative** profile, not a real person, and every report
labels it `mocked`. The brief permits "one real or representative candidate".

The capture adapter is real and was verified against live Instagram data (a public profile
returning follower counts and 12 recent posts with engagement). Unauthenticated capture is
rate-limited per IP and started returning 429 within minutes, which is exactly why
snapshots are captured once to fixtures rather than scraped in the critical path. Run
`score capture <handle>` to see it work.

There is also a deliberate choice here: publishing a repository that scores a real, named
individual on **brand safety** and prints "pass" is a bad look regardless of the data being
public. Negative cases are carried as synthetic archetypes instead.

---

## What I deliberately cut

In order of how much it costs:

1. **Real historical onboarding outcomes** → synthetic cohort. The load-bearing
   limitation; everything downstream inherits it.
2. **A trained classifier** → argued likelihood ratios (ADR-0002).
3. **Follower growth rate** → a snapshot is one point in time. The single most valuable
   signal I could not get.
4. **Audience age and gender** → paid-API only; category alignment proxies much of it.
5. **Live scraping at demo time** → fixtures, so a demo cannot break mid-run.
6. **Video content itself** (TikTok/Reels) → captions and metadata only.
7. **Any service, database or auth** → files in, files out.
8. **Brand-overlap analysis** — which brands already work with a candidate.

**What I would do first with real access**, in order: re-estimate the likelihood ratios
against real outcomes; buy audience demographics so GCC share stops being inferred; take
repeat snapshots to recover growth rate.

---

## Limitations

- **The base rate is assumed, not measured.** 20% is a stated prior. It is the one number
  a CM lead could supply from memory, it lives in one place in `config.py`, and the
  policy is expressed as a *multiple* of it so changing it does not silently invalidate
  the thresholds.
- **GCC audience share is inferred from language and content, never measured.** A public
  profile does not expose audience geography. This is the most important fit signal and
  the weakest evidence in the system.
- **The performance bar is a placeholder.** "Top 20% of first-90-day cohort performance"
  is a guess at what success means. CM owns this definition, and everything downstream
  inherits it.
- **Ranking is more reliable than the absolute numbers.** Use this to decide who to call
  first, not to forecast revenue.
- **Engagement counts are visible; conversion is not.** Nothing here observes whether an
  audience has ever bought anything. `commercial_evidence` infers it from affiliate links
  and discount codes, which is a proxy for a proxy.

---

## Libraries and APIs used

`openai` (LLM extraction), `pydantic` (extraction schema and validation), `numpy`
(cohort generation and harness statistics), `typer` (CLI), `jinja2` (report rendering);
`pytest` and `ruff` for development. Statistics used by the harness — AUC, Spearman,
logistic regression — are implemented in `validation/metrics.py` rather than pulled from
scipy/sklearn, deliberately: a training framework in the dependency list would invite
exactly the confusion ADR-0002 exists to prevent.

Instagram's public `web_profile_info` endpoint is read unauthenticated for capture. No
paid APIs, no credentials in the repository.

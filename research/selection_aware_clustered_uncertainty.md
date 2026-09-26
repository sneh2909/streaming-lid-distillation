# Selection-aware clustered uncertainty for external checkpoint validation

Research cutoff: **2026-09-26**. Local artifacts, primary statistical and
speech-evaluation sources, web search, and live Hugging Face Hub/API metadata
were checked on that date. This note addresses backlog **METHOD-44** only. It
does not repair the external runner's source/manifest provenance bugs
(BUG-16/17), replace METHOD-43's absolute eligibility gates, or report an
external-validation result.

## Bottom line

The current external-validation confidence interval cannot support checkpoint
promotion, for two independent reasons.

1. Each FLEURS recording contributes 1 s, 2 s, and full-duration predictions.
   The code resamples those three views independently inside 21
   language-by-prefix cells. They are repeated measurements of one audio clip,
   not three independent utterances. The bootstrap must draw an audio ID once
   and carry all its prefix views—and all checkpoint predictions—with it.
2. The code first uses the same FLEURS outcomes to screen and choose among up
   to five checkpoints, then computes an ordinary fixed-comparison percentile
   interval for the winner versus step 1600. That interval ignores the
   selection operation and is optimistically selected.

An exact synthetic fixture exposes the first error. With 7 languages, 100
clips per language, and per-clip paired values consisting of 53 `+1` and 47
`-1` values repeated identically at all three prefixes, the true sample gain
is 6 percentage points. At the runner's seed 7 and 10,000 replicates:

| Resampling rule | 95% percentile interval | Bootstrap SD |
|---|---:|---:|
| Current independent language/prefix draws | `[1.7143, 10.3810]` pp | `2.1998` pp |
| One draw per utterance/language, all prefixes together | `[-1.4286, 13.4286]` pp | `3.8021` pp |

The current standard deviation is `0.57858` times the clustered value, almost
exactly `1/sqrt(3)`. It turns a deliberately non-significant fixture into an
interval excluding zero. This fixture says nothing about the unrun real
comparison; it proves the implementation can be anti-conservative.

The clean contract is:

- use FLEURS **validation** to rank or select, and label uncertainty from that
  same set as descriptive selection stability;
- freeze the selected checkpoint, comparator, metric, thresholds, and policy;
- make any confirmatory claim once on untouched data, using paired clustered
  resampling; and
- if untouched confirmation is unavailable, use a simultaneous
  multiplicity-adjusted bound over the complete frozen candidate family, not
  the winner's ordinary interval. Its transfer to this language-stratified,
  repeated-prefix, parallel-prompt design is **unverified** until a coverage
  simulation passes.

Expected model or accuracy gain: **none**. The expected gain is honest
uncertainty and a promotion decision that cannot be won by breaking repeated
measurements or selecting on the reported interval.

## 1. Exact repository audit

The read-only audit is bound to:

| Artifact | SHA-256 at audit time |
|---|---|
| live external driver after concurrent non-bootstrap edits | `6707d18f36eb115923ed7ff3c27b9e9111b6ae256704401bb94fccab73c8b196` |
| exact `paired_bootstrap_gain` source segment | `653c5f0ba3d9fb39b69c65ade8b0d58bd8b5a0f61b499e1fef86140a1d632585` |
| exact `external_select` source segment | `e19e93008c586c08c35de214619a079cf8da543fad4d0c17bc13efa96d895985` |
| local 700-row FLEURS manifest file | `5a3335cda8b679ca4a6de8f084bff23c8fa82c18e1670e8bfdc6da751b246614` |
| manifest's canonical record digest | `ef0ec36427f742074b1bc88bd42f1b7e5b93b43bb390f9382f5030f3d8d2a637` |
| frozen trajectory result | `c4533c0b4aecd667289dff1df38f0a7b79cd98df359450da817b47706c4a4379` |

The driver changed concurrently during this audit to add teacher memory/cache
handling and partial-slice summaries. Its bootstrap and selector source
segments remained the reviewed implementations; their separate hashes bind the
method finding. The whole-file hash is a read-time binding, not a claim of a
published experiment revision. There is no external `results.json` or
`REPORT.md`. No real external metric, interval, selected step, or verdict
exists to interpret.

### 1.1 What the point metric measures

For each checkpoint, `summarize_predictions()` computes language-macro
accuracy separately at 1 s, 2 s, and full duration, then averages those three
numbers. In the current selected cohort, all 700 recordings are at least
2.70 s long, so every recording occurs in every one of the three selection
views. The point estimator intentionally gives each language and prefix equal
weight. That is a defensible declared composite; correlation changes its
sampling uncertainty, not its chosen point weighting.

Let

```text
d[i,s,p] = 1(checkpoint s is correct for clip i at prefix p)
           - 1(step 1600 is correct for clip i at prefix p)
```

for prefix `p in {1 s, 2 s, full}`. The paired gain for checkpoint `s` is

```text
G[s] = mean over 7 languages and 3 prefixes of
       mean d[i,s,p] over clips i in that language.
```

All checkpoints predict the same clips, so the data must remain paired across
checkpoint as well as prefix. One bootstrap weight for clip `i` must apply to
every `d[i,s,p]`.

### 1.2 How the current interval breaks that unit

`paired_bootstrap_gain()` loops over every language and then every selection
prefix. For each of the 21 cells it independently generates a new
`(10,000, 100)` index matrix. It therefore allows one recording's 1 s view to
be drawn several times while its 2 s and full views are absent, and vice
versa. Averaging those independently resampled cells behaves as though there
were 2,100 independent utterances rather than 700 audio clips with three
views.

The correct minimum repair is a stratified cluster bootstrap:

```text
for replicate b:
    for each fixed deployment language l:
        draw the 100 audio clip IDs in l with replacement once
        use those same integer weights for every prefix and checkpoint
    recompute every per-prefix, per-language, composite, and selector metric
```

The seven languages are fixed target classes, not a random sample of
languages. Resample utterances **within** language; do not bootstrap the seven
language labels unless the estimand is explicitly changed to unseen-language
generalization.

### 1.3 The hierarchy is deeper than prefix repetition

The pinned FLEURS paper says each underlying sentence was recorded up to three
times by three different native speakers, and the corpus is n-way parallel
across languages. The local 700-row selection contains:

| Observable unit | Count |
|---|---:|
| Language-prefix correctness values in the primary composite | 2,100 |
| Unique audio files / `clip_id`s | 700 |
| Unique `(language, dataset_row_id)` prompt groups | 535 |
| Unique parallel `dataset_row_id` prompts across all languages | 150 |

Within a language, the 100 selected recordings cover only 72--79 unique
prompts:

| Language | Audio rows | Unique prompts | Prompt multiplicity 1 / 2 / 3 |
|---|---:|---:|---:|
| English | 100 | 79 | 61 / 15 / 3 |
| Hindi | 100 | 78 | 59 / 16 / 3 |
| Marathi | 100 | 76 | 53 / 22 / 1 |
| Bengali | 100 | 79 | 58 / 21 / 0 |
| Tamil | 100 | 72 | 48 / 20 / 4 |
| Telugu | 100 | 76 | 54 / 20 / 2 |
| Gujarati | 100 | 75 | 54 / 17 / 4 |

Thus an audio-level cluster fixes the certain prefix dependence but does not
capture possible shared-text difficulty. Two predeclared sensitivity analyses
are useful:

1. resample unique `(language, dataset_row_id)` clusters within language and
   carry all recordings/views/models in each cluster; and
2. resample the 150 global `dataset_row_id` prompt clusters, carry every
   selected language/recording/view/model in a prompt, and recompute each
   language denominator before macro-averaging.

The second analysis preserves cross-language parallel-content dependence but
has only 150 observable prompt clusters and unequal cluster composition. It is
a sensitivity analysis, not a proved uniquely correct design.

The released Hugging Face schema exposes `id`, audio/path/length, transcripts,
gender, language ID/name, and language group, but no speaker ID. The paper
states that speakers are disjoint between train and dev/test; it does not make
speaker identities available in the Hub table. Audio- and prompt-clustered
intervals therefore cannot address unknown repeated-speaker dependence. Any
claim that all 700 rows are speaker-independent is **unverified**.

## 2. Selection changes the inferential target

Three different questions must not share one field named `ci95`.

### Fixed-checkpoint contrast

If checkpoint `s` was frozen without using FLEURS outcomes, a paired cluster
interval for `G[s]` estimates its gain over step 1600 on the frozen FLEURS
population represented by the sample. For example, step 970 could be declared
the sole external challenger because it was selected on the separate synthetic
development trajectory. In that design, FLEURS must not reopen selection
among steps 0, 10, 800, and 1600.

### Performance of the selection rule

A procedure that uses FLEURS to choose among five checkpoints is itself the
object being evaluated. An outer split or resampling trial must rerun the
complete eligibility and selection operation, then evaluate the chosen step on
observations not used for that trial's selection. This estimates a **selection
procedure**, whose chosen checkpoint can differ across trials; it is not an
interval for the one checkpoint selected from all 700 rows.

### Performance of the data-selected checkpoint

The current code selects a checkpoint using all 700 FLEURS rows and then asks
for uncertainty about that same winner. Ordinary percentile bootstrap for the
now-fixed winner ignores the maximization/screening that produced it. Even if
all four challenger-versus-final nulls were independent, selecting any lower
tail that excludes zero from four nominal two-sided 95% intervals would have a
family false-positive probability of `1 - 0.975^4 = 9.63%`; the real
checkpoint contrasts are correlated and the actual value is **unverified**.

Rerunning selection inside every bootstrap replicate is necessary for a
selection-stability diagnostic, but a naive percentile interval of those
replicate winners is not automatically a valid post-selection confidence
interval. Model selection is discontinuous, and the selected parameter itself
changes across replicates. Use untouched confirmation or a method with an
explicit simultaneous/post-selection coverage guarantee.

## 3. What the literature supports

### Cluster the observational unit

[Field and Welsh](https://doi.org/10.1111/j.1467-9868.2007.00593.x)
(JRSS B, first published **2007-05-22**) analyze bootstrap methods for
clustered one-way data and show that consistency depends on resampling in a
way that matches the cluster model. Their theory is not for a
language-stratified, crossed-prompt LID composite, so it supports the principle
rather than proving this exact interval.

The speech-specific result is aligned.
[Liu and Peng](https://www.isca-archive.org/interspeech_2020/liu20c_interspeech.html)
(Interspeech, **2020-10**) show that ordinary utterance resampling can
underestimate uncertainty when speech errors are dependent, and propose
resampling blocks. Their endpoint is ASR WER and their example dependence is
speaker/session structure; they do not study repeated LID prefixes or adaptive
checkpoint selection.

### Selection and evaluation cannot be silently reused

[Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)
(JMLR, **2010-07**) show that optimizing a noisy finite-sample model-selection
criterion can overfit that criterion and bias subsequent performance
evaluation; they recommend treating selection as part of the procedure in
each evaluation trial. [Varma and Simon](https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873/)
(BMC Bioinformatics, **2006**) likewise demonstrate optimistic error estimates
when the same resampling result tunes and assesses a classifier, and motivate
independent validation or an outer loop. These studies do not set an LID gain
threshold or solve clustered speech inference.

[Efron](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207812/)
(JASA, online **2013-07-25**, issue **2014**) explicitly notes that model
selection introduces discontinuities and develops bootstrap accuracy methods
that include the known selection procedure. It is a warning against treating
the selected estimator as smooth; it is not a drop-in algorithm for this
metric.

The most directly relevant recent method found by the cutoff is
[Rink and Brannath](https://doi.org/10.1007/s10994-024-06632-w)
(Machine Learning, published **2025-02-14**). They frame evaluation-set model
selection as simultaneous inference and combine bootstrap tilting with a
maxT-type multiplicity correction to give a lower bound for a data-selected
model. Their paper allows weighted performance measures and provides code, but
its reported coverage simulations draw independent evaluation observations.
Applying it to paired gains, seven fixed language strata, repeated prefixes,
and parallel prompt clusters requires a new coverage simulation; that
extension is **unverified**. The reference implementation was archived in
[March 2026 under GPL-3.0](https://github.com/pascalrink/mabt), so it should be
treated as a research reference, not copied into this repository without a
license decision.

No primary source found in the dated search directly validates one method for
this exact combination of fixed-language macro weighting, audio/prompt
clusters, paired multi-model contrasts, hard eligibility gates, and adaptive
checkpoint selection. That negative search result is non-exhaustive.

## 4. Proposed evaluation contract

### Stage 0 — do not spend external data on locally ineligible states

METHOD-43 already shows that all 161 retained states fail absolute local
class-coverage, familiar-fit, and switch-evaluability gates. Under that
contract, the correct present behavior is:

```text
external_loader_called = false
external_selection_performed = false
uncertainty_status = not_applicable_no_locally_eligible_candidate
```

The stages below are for a future trajectory with at least one eligible state.

### Stage 1 — use validation for selection and stability

Before model outcomes are read, freeze:

- candidate step and model-state hashes, including the fixed final comparator;
- the exact 700-row manifest or a new full-validation manifest;
- primary composite and all absolute gates;
- prefix eligibility, language weighting, and tie order;
- audio and prompt cluster keys;
- bootstrap seed/replicate count and invalid-replicate policy; and
- whether the estimand is one fixed contrast or an adaptive selection rule.

For an adaptive selector, draw clip weights once per language and reuse the
same weight tensor for every checkpoint and prefix. In every replicate,
recompute all external metrics, eligibility, and the complete selector. Report:

- selection frequency for every step;
- `no_eligible_checkpoint` frequency;
- every gate's pass frequency;
- selected-step transition/tie frequencies; and
- the distribution of selected-minus-final gain, labelled
  `descriptive_selection_stability`, **not** a confidence interval.

If duplicating a prefix view changes this stability only because it creates an
extra independent draw, the implementation is wrong.

### Stage 2 — confirm once on untouched observations

After selection, freeze the single checkpoint and evaluate it once against
step 1600 on an untouched confirmation cohort. The pinned FLEURS test split is
one possible monolingual cohort, but it should remain locked until local
absolute gates, checkpoint, policy, and analysis code are final. It cannot
confirm code-switch behavior, so a separately pinned natural switch set is
still required for a global streaming-LID promotion.

Use a paired cluster lower confidence bound. If the product claim is “gain is
at least 2 percentage points,” require the lower bound itself to be at least
`+2 pp`. The current combination—point estimate at least `+2 pp` but lower
bound merely above zero—only supports a positive effect, not a confidently
material two-point effect.

Report audio-clustered inference as the minimum correction and the observable
prompt-cluster analyses as sensitivities. If a promotion conclusion changes
under the predeclared prompt sensitivity, return `uncertainty_not_robust`.
Hidden-speaker robustness remains unavailable on FLEURS.

### Stage 3 — if there is no untouched confirmation set

Do not use the selected winner's ordinary 95% interval. Two auditable options
are:

1. **Conservative family-wise bounds.** With four frozen challenger-versus-
   final contrasts, compute one-sided audio-cluster bounds at
   `alpha/4 = 0.0125` (Bonferroni) and allow any data-dependent selection only
   from that family. This is simple and conservative if each clustered bound
   has its claimed coverage. A shared-resample maxT bound can exploit candidate
   correlation but is more complex.
2. **Multiplicity-adjusted bootstrap tilting.** Adapt the Rink--Brannath MABT
   construction to the weighted paired composite, using one shared cluster
   weight per audio/prompt across all candidates. Before use, simulate coverage
   under null ties, correlated prefixes, correlated candidates, class
   imbalance/collapse, prompt effects, and missing views. Require at least the
   nominal one-sided 95% family coverage within declared Monte Carlo error.

Both options are **unverified for this project** until implemented and tested.
The second may be less conservative, but sophistication is not evidence. If
neither is validated, publish point estimates and selection frequencies only,
set `post_selection_interval_valid=false`, and prohibit the “CI excludes zero”
promotion gate.

## 5. Result schema

A future result should distinguish the sampling design, estimand, and claim:

```json
{
  "uncertainty_schema": 2,
  "primary_estimand": "fixed_checkpoint_gain_or_selection_rule",
  "candidate_steps": [],
  "final_comparator_step": 1600,
  "candidate_family_size_excluding_final": 0,
  "metric": "mean_7_languages_x_1s_2s_full_accuracy",
  "strata": "language",
  "primary_cluster_key": "clip_id",
  "weights_shared_across": ["prefix", "checkpoint"],
  "replicates": 10000,
  "seed": 7,
  "bootstrap_weight_digest": "...",
  "point_estimates": {},
  "fixed_contrast_intervals": {},
  "selection_stability": {
    "step_frequencies": {},
    "no_eligible_frequency": null,
    "descriptive_selected_gain_quantiles": []
  },
  "prompt_cluster_sensitivity": {
    "language_prompt_clusters": 535,
    "global_prompt_clusters": 150,
    "conclusion_stable": null
  },
  "speaker_cluster_status": "unavailable_no_speaker_id",
  "post_selection_method": "none_or_named_validated_method",
  "post_selection_interval_valid": false,
  "confirmation_data_role": "validation_or_locked_test",
  "locked_test_read": false
}
```

The artifact must also retain integer correct/support tables for every
language, prefix, checkpoint, and resample input cohort. A rounded macro score
is not sufficient to reconstruct or audit the interval.

## 6. Deterministic acceptance tests

1. **Reviewer stress fixture:** reproduce the exact 6 pp point estimate,
   current `[1.7143, 10.3810]` pp interval, and corrected
   `[-1.4286, 13.4286]` pp interval at seed 7 / 10,000 replicates.
2. **Perfect prefix duplication:** duplicate an existing prefix any number of
   times. The declared point weighting may change only if configured, but the
   cluster draw count and effective independent audio count must not increase.
3. **Shared model pairing:** swap or duplicate checkpoint columns; every
   replicate must use identical clip weights across all checkpoint contrasts.
4. **Language stratification:** an imbalanced synthetic cohort must retain the
   declared equal language macro weighting in every replicate.
5. **Missing view:** remove a prefix from selected clips and require
   eligibility-aware denominators; never synthesize a prediction or redraw a
   different audio set for that prefix.
6. **Selection replay:** construct a replicate that changes the best external
   candidate and prove the complete gates and tie order rerun, rather than
   holding the original winner fixed.
7. **Selection null:** four equal-quality challenger fixtures must show that
   the unadjusted selected interval fails the predeclared family-coverage
   simulation while the adopted simultaneous method meets it.
8. **Candidate order/tie invariance:** permuting candidate input order cannot
   change point estimates, bounds, or the frozen deterministic tie result.
9. **Prompt hierarchy:** the current manifest must reconstruct exactly 700
   clips, 535 language-prompt clusters, and 150 global prompt clusters; changed
   IDs invalidate the resampling identity.
10. **No speaker inference:** absence of a speaker field must serialize
    `unavailable_no_speaker_id`, never `speaker_independent=true`.
11. **Fail-before-load:** a METHOD-43 `no_eligible_checkpoint` fixture must
    prove no Hub/cache loader, prediction scorer, or locked-test reader ran.
12. **Role separation:** validation selection must leave
    `post_selection_interval_valid=false`; only the separately bound
    confirmation path may set a confirmatory result.

## 7. Dated Hugging Face audit

The external source is Hugging Face dataset ID
[`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
pinned to `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. On **2026-09-26**,
the live Hub API still resolved `main` to that full revision, reported
`lastModified=2026-05-15T09:35:34Z`, 1,004 siblings, public/ungated access, and
`license: cc-by-4.0`
([dated API response](https://huggingface.co/api/datasets/google/fleurs/revision/70bb2e84b976b7e960aa89f1c648e09c59f894dd)).
The [pinned card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md?code=true)
exposes 16 kHz audio and no speaker-ID feature.

With `main` equal to the pin,
[Hub dataset-server metadata](https://datasets-server.huggingface.co/size?dataset=google%2Ffleurs)
returned:

| Config | Validation | Locked test |
|---|---:|---:|
| `en_us` | 394 | 647 |
| `hi_in` | 239 | 418 |
| `mr_in` | 443 | 1,015 |
| `bn_in` | 402 | 920 |
| `ta_in` | 377 | 591 |
| `te_in` | 311 | 472 |
| `gu_in` | 432 | 1,000 |
| **Total** | **2,598** | **5,063** |

The local validation cohort deterministically takes 100 rows per language
from those 2,598 rows. Bootstrap statements are conditional on that exact
frozen 700-row cohort unless the full validation population is scored. The
hash ranking is outcome-blind, but treating its selected rows as an exact
probability sample of every future call domain is **unverified**.

The [FLEURS paper](https://arxiv.org/abs/2205.12446)
(submitted **2022-05-25**, later IEEE SLT) describes 102 languages,
approximately 12 hours per language, up to three native-speaker recordings per
sentence, 16 kHz audio, and speaker-disjoint train versus dev/test splits. It
also describes FLEURS as monolingual Speech-LangID evaluation. None of this
makes it a natural Hinglish switch benchmark or a telephony benchmark.

## Recommendation

Replace the current independent cell bootstrap before any external result is
published, but do not mistake that repair for a post-selection solution. For a
future locally eligible trajectory:

1. select on clustered FLEURS validation and report selection frequencies as
   descriptive evidence;
2. freeze one checkpoint and confirm it on untouched, speaker-audited data;
3. use shared paired cluster weights and require the lower bound—not just the
   point estimate—to clear the material gain threshold; and
4. if the same evaluation set must select and assess, adopt a simultaneous
   family-wise method only after project-shaped coverage simulations pass.

For the retained trajectory today, METHOD-43 remains decisive: there is no
eligible checkpoint, so external uncertainty is not applicable and the locked
test must remain unread.

# Voice–language mutual-information percentage audit

**Topic:** backlog POLISH-11: correct and harden the crossed-voice mutual-information wording
**Evidence cutoff / access date:** 2026-09-26
**Scope:** one reporting/arithmetic question. No audio was generated, no model was run, and no accuracy gain is claimed.

## Bottom line

The review correction is right. The augmented manifest value `1.559641623` bits is **55.56% of** the original `2.807354922` bits, so it represents a **44.44% reduction (55.56% remaining)**. Calling the value “1.5596 bits (44.44%)” after “reduces ... to” mislabeled the removed fraction as the remaining fraction.

The number also has a design assumption that should be stated: the 56-row crossed pool consists of eight new voice-family IDs, each appearing once under all seven labels, and those IDs are disjoint from the 14 current IDs. It is not a universal result of appending any 56 multilingual rows. If voices, languages, counts, or sampling weights differ, recompute from the exact contingency table.

Mutual information here establishes dependence between **manifest identifiers**. It does not prove acoustic speaker identity, causal shortcut use by the model, or an accuracy effect. All three remain **unverified** until the matched crossed experiment is run.

## Definition and exact local reconstruction

For empirical language `L` and voice-family identifier `V`, compute

```text
I(L;V) = H(L) - H(L|V)
       = sum(l,v) p(l,v) log2[p(l,v) / (p(l)p(v))].
```

[Shannon's original paper](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf) (published July/October 1948; accessed 2026-09-26) defines entropy as `-sum p log p`, shows that the uniform distribution over `n` choices has entropy `log n`, and defines conditional entropy as the remaining uncertainty when another variable is known. Those identities are sufficient for the calculation below. The percentages used here are explicitly relative to the current manifest value; they should not be named a generic normalized-mutual-information statistic.

The read-only reconstruction used the 70 single-segment training rows in `data/generated/manifest.jsonl`, exact-file SHA-256 `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c`. It found seven languages with ten rows each and 14 voices with five rows each; every voice occurs under exactly one language.

| Construction | `H(L)` | `H(L|V)` | `I(L;V)` | Original MI remaining | Original MI removed |
|---|---:|---:|---:|---:|---:|
| Current 70 rows | `log2(7)` | `0` | `2.807354922` bits | `100%` | `0%` |
| Append 56 disjoint crossed rows (`8 voices × 7 languages`) | `log2(7)` | `(56/126) log2(7)` | `(70/126) log2(7) = 1.559641623` bits | `55.5556%` | `44.4444%` |
| Sample current and disjoint crossed pools with 50:50 mass | `log2(7)` | `0.5 log2(7)` | `0.5 log2(7) = 1.403677461` bits | `50%` | `50%` |
| Proposed 98-row label-confounded arm | `log2(7)` | `0` | `2.807354922` bits | `100%` | `0%` |
| Proposed 98-row fully crossed arm | `log2(7)` | `log2(7)` | `0` bits | `0%` | `100%` |

The appended-pool simplification works because both component pools have a uniform language marginal and disjoint voice support. Conditioned on an old voice, language remains deterministic; conditioned on a new crossed voice, it is uniform over seven labels. With shared voice IDs or incomplete cells, that weighted formula generally does not apply.

## Correct reporting contract

Every reported dependence value should carry:

- the exact integer `language × voice_family` contingency table or its canonical hash;
- row-selection and sampling weights, including whether pool supports are disjoint;
- `mi_bits`, its logarithm base, and the empirical estimator definition;
- a named reference value when a percentage is shown;
- separate `fraction_of_reference_remaining = I_new / I_reference` and `fraction_of_reference_reduced = 1 - I_new / I_reference` fields.

For this comparison, the correct serialized values are:

```text
reference_mi_bits                  = 2.807354922057604
augmented_mi_bits                  = 1.559641623365335
fraction_of_reference_remaining    = 0.5555555555555554
fraction_of_reference_reduced      = 0.4444444444444446
```

A release check should reject prose or JSON that attaches `44.44%` to `1.5596 bits` without the word `reduction`, or calls `44.44%` the amount remaining. It should also reject a cached value after any row, voice-family mapping, split, or sampler weight changes.

## Dated Hugging Face Hub check

The Hub check is intentionally narrow because this iteration corrects reporting, not candidate selection. Direct Hub API queries on 2026-09-26 confirmed that the two most relevant crossed-design resources remain at the revisions already pinned in the parent memo:

- [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts/tree/7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca): current `main` revision `7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca`, last modified 2025-09-24, Apache-2.0 metadata, gated. A web-search result surfaced historical commit `ac57c4f605486314efefe8654c47a1692fa7b675` from 2024-10-28; it is not the API-resolved current revision. This reinforces using full revisions rather than search-result aliases.
- [`sarvamai/sarvam-dub-benchmark-set`](https://huggingface.co/datasets/sarvamai/sarvam-dub-benchmark-set/tree/ad489da29596f95ae527c7947fa123e75a4d7a0a): revision `ad489da29596f95ae527c7947fa123e75a4d7a0a`, last modified 2026-02-06, license metadata `other`. Its card describes 64 reference speakers across 11 target languages and 704 prompt rows, making it a useful crossed-design precedent, not a ready synthesized training corpus.

Both pinned URLs returned HTTP 200 on 2026-09-26. Neither card defines or validates this project's manifest-level MI percentage convention. No Hub evidence changes the arithmetic or licenses an acoustic-identity claim.

## Testable proposal

Add a pure contingency-table scorer and four fixtures matching the table above. It must recompute from rows rather than accept a stored scalar, be invariant to row order, bind sampler weights, and fail if `remaining + reduced != 1` within `1e-12`. The documentation test should require the exact phrase `44.44% reduction (55.56% remaining)` for the 56-row example.

Expected model or accuracy gain: **none**. The value is an experimental-design audit; whether fully crossed training improves held-out-profile macro-F1 is **unverified**.

## Sources

- C. E. Shannon, [*A Mathematical Theory of Communication*](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf), Bell System Technical Journal 27, July/October 1948 (accessed 2026-09-26).
- Hugging Face Hub, [`ai4bharat/indic-parler-tts` pinned tree](https://huggingface.co/ai4bharat/indic-parler-tts/tree/7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca) and [model API](https://huggingface.co/api/models/ai4bharat/indic-parler-tts) (accessed 2026-09-26).
- Hugging Face Hub, [`sarvamai/sarvam-dub-benchmark-set` pinned tree](https://huggingface.co/datasets/sarvamai/sarvam-dub-benchmark-set/tree/ad489da29596f95ae527c7947fa123e75a4d7a0a) and [dataset API](https://huggingface.co/api/datasets/sarvamai/sarvam-dub-benchmark-set) (accessed 2026-09-26).
- Local review finding, [`review/changes-2026-09-26-f03269e.md`](../review/changes-2026-09-26-f03269e.md), lines 59–61 and 84 (2026-09-26).

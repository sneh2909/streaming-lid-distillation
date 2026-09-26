# Phase 2 live-delta review — post-`ea58c4a` worktree

Review cutoff: 2026-09-26 14:36:08 +05:30. The committed baseline and HEAD are both `ea58c4aeda6b960ef4d72a7a1380b2211294b25b`; `git log ea58c4a..HEAD` and `git diff ea58c4a..HEAD` are empty. I reviewed the new live research bytes in addition to checking every experiment report for a delta. The prior `review/changes-2026-09-26-ea58c4a.md` is preserved rather than overwritten.

## Summary

- The edit to `research/crossed_voice_profile_diversification.md` correctly fixes both presentation defects in the parent memo: `1.559641623` bits is 55.56% remaining / 44.44% reduced, and the bound manifest digest now includes its missing final `c`.
- The new `research/voice_language_mutual_information_audit.md` is numerically and methodologically sound. It states the disjoint-support and uniform-language assumptions, distinguishes empirical identifier dependence from acoustic identity or causal model use, and labels the proposed accuracy effect unverified.
- The already-reviewed `research/kd_class_collapse_teacher_bias.md` remains byte-identical at SHA-256 `7d9742c508ab9f079ee48dd19e7a6d8146d9ea7b1b4f118130f0ad5a6ec6c6e0`.
- No file under `experiments/` changed, and no experiment report/result was added. All eight reports are byte-identical to their previously reviewed versions, so their previously checked conclusions remain unchanged.
- No new BUG, METHOD, DEFEND, or MINOR finding was produced. `.loop/inbox/review.md` was therefore left unchanged.

## Findings

No new findings.

## Changed research audit

The reviewed worktree hashes are:

- `research/crossed_voice_profile_diversification.md`: `75bc6b27cb9e9656293643af47df282dba48f52219b992bf5b08e8296dc0764b`;
- `research/voice_language_mutual_information_audit.md`: `e57a7ecde0573380c27b4ff56d5f1e59284e94577837175c58aa8eae39952bdc`.

An independent read-only reconstruction of `data/generated/manifest.jsonl` reproduced its SHA-256 `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c`, 70 single-segment training records, ten records for each of seven languages, and 14 `speaker_id` values with five records each. Every speaker ID occurs under exactly one label, hence

```text
H(L) = I(L;V) = log2(7) = 2.807354922057604 bits
H(L|V) = 0
```

For eight new voice IDs each appearing once in all seven languages, with support disjoint from the old 14 IDs, the new voices contribute 56 of 126 uniformly conditioned rows. Therefore

```text
I_new = (70 / 126) log2(7) = 1.5596416233653356 bits
I_new / I_old = 0.5555555555555556
1 - I_new / I_old = 0.4444444444444444
```

The memo's displayed serialized values differ only by final-digit floating-point rounding. Its 50:50-pool value `0.5 log2(7) = 1.403677461` and the zero-MI fully crossed construction also follow directly. The cited [Shannon paper](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf) supports the entropy identities; the pinned [Indic Parler-TTS card](https://huggingface.co/ai4bharat/indic-parler-tts/blob/7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca/README.md) and [Sarvam benchmark card](https://huggingface.co/datasets/sarvamai/sarvam-dub-benchmark-set) remain consistent with the narrowly scoped metadata claims. None of those external sources is used as evidence for an accuracy gain.

## Experiment report audit

`git diff ea58c4a..HEAD -- experiments`, the live tracked diff under `experiments/`, and the untracked inventory under `experiments/` are all empty. Current report hashes are:

| Report | SHA-256 | Previously checked conclusion |
|---|---|---|
| `checkpoint-trajectory/REPORT.md` | `294ce6160f554a844b475cb8202208dbd525d6b4b2b5c916dee9392cc74d57bc` | Retain the trajectory/selection machinery, but do not promote step 970 without absolute quality and external gates. |
| `cross-view-kd/REPORT.md` | `ac8a7683751fd51fb19e069d067fcded66cf56bd9237d779dbbb812a3cc06577` | Supports a controlled follow-up, not a switch-capable/global adoption claim. |
| `target-type-ablation/REPORT.md` | `a40e8b87310e95d955e8974834578f9c9acfb337524490c33dee9168adf7f79a` | Reject the tested replacement. |
| `teacher-anchor-hop-audit/REPORT.md` | `f4405b22d8315ee6818d908a3ee4cdfaaaf012eec644ca170acc7a65c3fc2bc5` | Reject the tested hop change. |
| `teacher-bakeoff/REPORT.md` | `b186dc1fa1df5134f802f52cdd46577a4145985d88bbfecb2c0606ca17d88abf` | Reject the tested teacher replacement under its stated gates. |
| `teacher-window-audit/REPORT.md` | `c539820485716d2887be9267d41532efa668254c1a31553c33fceb1250942421` | Reject the tested window change. |
| `telephony-robustness/REPORT.md` | `6b79c8cb6d5c384dd2e3659da467e594ae67587bc6cce0a9964676920d5f78d5` | Reject under the reported degradation gates. |
| `voice-text-factorial/REPORT.md` | `55bac1bb5261c23d7134dcb9eae935fb66ce0b8c59fca84dc92029512c685c54` | Supports a large profile association and failure of voice-only sufficiency, not a sole-cause claim. |

Those report/result pairs and conclusions were independently recomputed in the preceding reviewer iterations. Because both HEAD and every live experiment byte are unchanged, there is no new number or conclusion to re-adjudicate in this delta. `experiments/checkpoint-external-validation/` still has no `REPORT.md` or `results.json`; no external-validation conclusion exists to review.

## Verified correct

- The parent memo's new relative link resolves to the new audit file, and its corrected manifest digest matches the live manifest exactly.
- The MI derivation uses empirical row probabilities and explicitly binds the assumptions under which the weighted conditional-entropy shortcut is valid. It does not present the ratio as generic normalized mutual information.
- The proposed crossed experiment remains framed as an unverified intervention with a predeclared usefulness threshold, not as a forecast.
- `git diff --check` passes for the live tracked change. No code, test, model, result JSON, or experiment report changed, so the previously recorded 51/51 test run was not repeated.

## Interview questions

1. **Why is 44.44% the reduction rather than the amount remaining?** The augmented value is `(70/126)` of the original, or 55.56%; the removed fraction is `1 - 70/126 = 44.44%`.
2. **Why does disjoint voice support matter?** It lets old deterministic voices and new uniformly crossed voices contribute separate conditional-entropy terms. Reusing a voice ID across pools changes `p(language | voice)` and invalidates the simple weighted formula.
3. **What does maximum manifest MI prove?** Only that the recorded voice identifier perfectly predicts the recorded label in this sample. It does not prove acoustic identity, that the student uses the shortcut, or that crossing voices will improve accuracy.
4. **Why was training/test execution not repeated?** This delta changes research prose only. Source, tests, checkpoints, results, and experiment artifacts are byte-identical to the last reviewed tree.

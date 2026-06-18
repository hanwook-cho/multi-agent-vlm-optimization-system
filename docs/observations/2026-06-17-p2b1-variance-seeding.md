# Observation: Seeded Variance Test — MMBench Ability Is Robust, POPE Is a Coin-Flip (P2-B1)

**Date:** 2026-06-17
**Context:** Construction runs had no RNG seeding (the from-scratch projector initialised differently every run). Two lever runs came back degenerate in ways that were directionally impossible (more grounding collapsing POPE; more capacity making it degenerate), implicating uncontrolled init variance. We fixed seeding and re-ran the exact 50/50 "milestone" config (`151cf686`) at three seeds to measure the noise floor and test whether the milestone reproduces.
**Verdict:** **The milestone was partly a favorable draw.** Across identical configs, **MMBench is robust** (all runs above floor) but **POPE balanced-accuracy is wildly variable** (5–62, only half above floor). So "one student clears *both* POPE and MMBench" is **not a reliable property** — the MMBench half is, the POPE half is a coin-flip at this mix/capacity.

---

## Result (50/50 config `151cf686`, same spec, 4 runs)

| Run | POPE bal-acc | MMBench | RealWorldQA |
|---|---:|---:|---:|
| original (unseeded) | 55.0 ✓ | 0.62 | 0.50 |
| seed 1 | 62.3 ✓ | 0.70 | 0.40 |
| seed 2 | 13.3 ✗ | 0.59 | 0.36 |
| seed 3 | 5.0 ✗ | 0.59 | 0.42 |
| **summary** | **5–62, 2/4 above floor** | **0.59–0.70, 4/4 above floor (mean ~0.63)** | ~0.36–0.50 (around floor) |

## Reading

- **MMBench ability is reproducible.** Every run clears the 0.50 floor (0.59–0.70). The distribution-matching finding (train on ScienceQA → get MMBench) holds across seeds — this is a real, stable result. It also corroborates the single-skill ScienceQA run (`b2feb6b1`, 0.65): MMBench from matched data is genuine, not a lucky draw.
- **POPE is unstable.** Balanced-accuracy ranges 5–62 across *identical* configs; only 2 of 4 inits land above floor. POPE is a balanced yes/no eval, and the model's yes/no calibration is highly sensitive to init — especially at the 50/50 mix where grounding (1386 rows) is diluted by ScienceQA. Whether a run lands on a usable grounding calibration is largely luck at 0.5B.
- **Therefore the "both floors, one student" milestone is downgraded.** The system can *reliably* deliver MMBench via matched data; holding POPE *simultaneously* is variance-limited, not solved. Last turn's `151cf686` (POPE 55 / MMBench 0.62) was a favorable POPE draw.

## Why this matters (methodology)

This is the most important process lesson of P2-B1: **single-run results at n=100 with a from-scratch projector are not trustworthy for POPE.** The two "lever" negatives (ratio 60/40, rank 32) that triggered this were almost certainly init variance, not lever effects — they are *not* recorded as findings. Going forward, any POPE-dependent claim needs **multi-seed reporting (median/min across ≥3 seeds)**, not a single run. The seeding fix (`seed_everything`, `--seed`) makes this possible and makes every run reproducible.

The same caveat retroactively applies to earlier single-run numbers (`d3423bc0` POPE 68.3, `b2feb6b1` MMBench 0.65) — MMBench appears robust (corroborated here), but the POPE peak (68.3) may itself be a high draw and should be re-measured across seeds before being treated as the student's POPE level.

## What the system does next

1. **Stabilise POPE in the mix** — shift toward grounding (e.g. 65–70% grounding) *and* report median-of-3-seeds, to find a ratio where POPE is *reliably* above floor rather than coin-flip. Optimise the worst-case (min across seeds), not a lucky max.
2. **Capacity** — if POPE stays unstable even grounding-heavy, the 0.5B LM likely can't hold both calibrations robustly; a larger student is the lever (a budget/Phase-3 question).
3. **All future construction claims are multi-seed.** No more single-run milestones.

## Caveats

Internal-only numbers (n=100 slice, non-official protocol — memory `benchmark-eval-internal-only`). Here the *point* is the cross-seed spread; the single-run absolute values are exactly what this observation shows you cannot trust for POPE.

## Artifacts

- Ledgers: `artifacts/experiment_ledger/construction_151cf686f7b8{,_s1,_s2,_s3}.json`.
- Seeding fix: `runners/build_student.py` `seed_everything`; `services/construction_loop.py --seed`.

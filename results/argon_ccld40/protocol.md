# Argon Round4 CCLD — protocol recorded before execution

Date2026-10-15. Original proposal and code are archived verbatim in
proposal_original.txt and proposal_original.py. Authors/persona labels in that
proposal are unverified; no real researcher's endorsement is implied.

Exactly40 frozen duration-prefiltered videos, five per task/label, <=45s.
Manifest SHA2560349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459.
Use the original raw AV decoder caches; all cyclic answer rotations; no resampling,
replacement, question-specific routing or gold-based parameter calibration.
This is repeatedly used development data, not a held-out test.

Run all eight original mathematical conditions C0–C7. C0 native; C1 original
M9 windowed thought; C2 clean sync/anti recurrent delta; C3 native minus
visual-blind delta; C4 same-norm dimension-shuffled C3; C5 global thought+C3;
C6 global thought+native/sync visual-blind mixture; C7 C6+late audio damping.
Scales/layers/top48/temperature/seed match the supplied code.

Necessary implementation adaptations: frozen manifest replaces random sampling
and <=50s runtime skips; native original file replaces full-cut reencoding;
baseline letter-only prompt and identical spatial budget retained; decoder-only
replay reuses encoded media; only last-token lm_head; use_cache=False; bool
causal triangle replaces quadratic int64 indices; per-device per-layer masks
replace shared mask mutation. Never delete old output or silently skip an error.
Native/decoder replay tolerance<=.12 and finite outputs required. C7 adds-2.5
on question-to-audio keys only from zero-indexed block16 onward; pre16 remains
native. All conditions preserve original causal exclusions.

Interpretation caveats to evaluate explicitly:
- C1 uses speech_window. C2 uses hard_sync/anti_sync. C6/C7 retain hard_sync.
  Only C3/C4/C5 are window independent in their relevant inference path.
- C2 also removes old foreground/desmear, so it is not a clean clock-only
  ablation of historical M8. C5 uses global thought at.20, whereas C1 uses
  windowed thought at.22. The suite is informative, not fully factorial.
- Hidden-state subtraction is an intervention contrast, not an exact causal
  decomposition or a proof that nuisance factors cancel or all evidence survives.
- Visual-blind branches are blocked in every decoder layer to remove indirect
  visual-to-audio-to-question paths. The sensory encoders remain unchanged.
- C3/C5 still privilege visual-specific corrections; C7 explicitly damps audio.
  Success here cannot establish safety for audio-led/reverse/arbitrary questions.
- A fixed shuffle is a useful direction control but not a distribution over
  independent random directions. C3 must outperform it before semantic claims.

Report correct/wrong in every cell, TF prediction counts, paired rescues and
regressions, exact paired p (unadjusted), replay error and original M9 agreement.
No CVPR suitability claim based solely on40 development examples.

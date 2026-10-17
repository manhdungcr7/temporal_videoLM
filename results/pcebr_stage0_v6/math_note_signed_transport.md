# Signed Wasserstein term in the M3 routing scalar

This is an algebraic observation about the submitted source, not an experimental modification.

For two normalized probability distributions with CDFs F1,F2 and common enclosing support [a,b], define

W+ = integral max(F1−F2,0) dt / (b−a),
W− = integral max(F2−F1,0) dt / (b−a).

Then

W+−W− = integral(F1−F2) dt / (b−a) = (E[T2]−E[T1])/(b−a).

The first equality follows pointwise from max(d,0)−max(−d,0)=d. The second follows by integrating a CDF on bounded support: integral F(t)dt = b−E[T].

In the source, compute_wasserstein_synchrony_relational_features multiplies all Phi coordinates by joint_gate. The M3 builder then uses only `exp_phi[3] - exp_phi[4]` as its new transport contribution to rel_scalar. For the uniformly spaced interval bank, this equals joint_gate times the normalized difference in mean slot centers. Floating-point arithmetic, epsilon normalization, and clamps can introduce small numerical deviations.

Consequences:

- Computing W+ and W− exactly is mathematically correct, but subtracting them discards transport magnitude beyond signed mean lag. Two distributions with equal means can still have a nonzero W++W−.
- M2 linear probes see W+ and W− as separate coordinates, so this compression criticism does not apply to their entire feature vector.
- M3 still has other distribution-dependent ingredients: tanh precedence margin, synchrony/p_anchor profiles and per-token innovation. This does not prove that all of M3 is a mean-time model.
- The orthogonal content/order vector is tested and used by legacy tangent routing, but the new CAOR profile consumes aud_info scalars/probabilities rather than injecting that vector. Its fixed forward/reverse cosine is therefore not evidence that the actual CAOR output correction carries the tested vector geometry.

If the scientific goal is to retain distributional temporal information in the effect on the LLM, a useful next ablation is to compare a mean-lag-only controller with a controller retaining transport magnitude or the lag distribution. Such a comparison is proposed here; it has not been used to tune or replace this mainrun.

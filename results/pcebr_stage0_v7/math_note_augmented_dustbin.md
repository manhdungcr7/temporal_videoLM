# Augmented-dustbin construction: an analytic consequence, 2026-10-16

This is a derivation from the supplied algorithm, not a claim that Stage-0 QA has improved. Actual-input numerical checks are saved separately after mainrun.

Let p,q be the prescribed real-event marginals, and B = exp(u + M + v) the converged first-stage Sinkhorn coupling, with B1=p and B^T1=q. Let z be the dustbin score and a=clamp(joint_gate*sigmoid(-z)). The supplied augmented kernel is

```
K_aug = [ B          exp(z) p ]
        [ exp(z) q^T exp(2z)  ]
```

Its prescribed augmented marginals are [a p,1-a] and [a q,1-a]. Multiplying its real rows/columns by a and its dustbin row/column by (1-a)exp(-z) gives a feasible diagonal scaling:

```
J_aug = [ a^2 B       a(1-a) p ]
        [ a(1-a) q^T (1-a)^2  ]
```

Consequently at convergence M_act=a^2, P_null=1-a^2, and J_cond=B. The z contributions to augmented kernel border scores are row/column potentials removable by Sinkhorn scaling; z still affects the prescribed marginal a. Null probability does not independently assess the pair interaction matrix at fixed p,q,z,joint_gate. The construction is equivalent to a conditional coupling plus this scalar presence gate; merely extracting both from one J_aug does not make pair compatibility supply new null evidence. Finite iteration and floating point errors may introduce small departures, so they must be measured rather than silently assumed zero.

Separately, W_plus, W_minus and Bhattacharyya features in Psi are computed from J_cond's marginals. Under fixed p,q these channels and the mean lag are invariant to the coupling. Pair interactions can change the RBF lag summaries, joint synchrony masses and joint lag dispersion. They cannot change marginal Wasserstein transport solely by changing the pairing. A finite vector of 14 summaries is not an injective representation of an arbitrary K1 x K2 joint distribution.

The failed original M0 assertion compares Psi[13] against 0.3. In this code Psi[13] is std_J(lag)-std_uniform(lag), not std_J(lag). For the supplied bimodal toy, both raw standard deviations are0.4, so Psi[13]=0.0. The synchronous toy has about-0.4; the two feature vectors do remain distinct. This failure concerns the asserted expectation, and must not be reported as an original M0 pass.

Actual QA input audit adds a numerical limit: all8 true-query cases have joint_gate0 and Psi0. Because act_target is floored at1e−7 while M_act is floored at1e−12, the real block mass≈1e−14 divided by1e−12 returns J_cond whose sum≈.01. It is not a conditional probability distribution in this regime. The positive-mass derivation above remains valid for the intended converged construction; the implementation's extra mass clamp changes returned normalization. See deep_mechanism_audit.json and independent_cpu_math.json; zero-gate QA inputs do not validate nontrivial positive-gate null reasoning.

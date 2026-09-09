# What we are measuring, and why the mathematics licenses it

**Written 2026-09-10 by the Chief AI Research Scientist for the Research Director, to be given to a
Claude instance he promised to inform of the experiments' particulars and, later, their results.**
This is the particulars. It describes every measurement the programme makes, the mathematics each
rests on, and what each is and is not entitled to conclude. It reports no result about a model:
the device runs began today, and results follow when they exist. Where a number appears below it
is either a calibration of an instrument, labelled as such, or a count of work.

## 1. The question and the setting

The programme studies small open language models executing multi-step tool-using tasks, and asks
two things of them: whether fine-tuning changes what such a model represents or only how it
behaves, and whether the model carries a *state* of the task it is doing, in a form that can be
identified, substituted, and read. The second question is the one the mathematics below is built
for, and it is asked in a way that cannot be answered by finding a feature that correlates with a
word.

**The environment** is a workspace simulator: a flat set of files, six tools (`list_files`,
`read_file`, `search_files`, `calculate`, `replace_text`, `finish`), twelve generated task
families with expert trajectories, and injected faults applied by the environment before a call
is validated, so a transient failure or a false observation is a property of the world and not of
the harness. Every episode is a transcript of thoughts, actions and observations.

**The models** are Google's Gemma 3 instruction-tuned checkpoints at 4B and 12B parameters. Gemma
is the constraint, not the target: it is what the laptop could hold and what has public sparse
dictionaries at every layer. Larger models are expected once the rented card allows them, and every
rule below is written so that nothing names a layer index, a width or a family class.

**The compute** moved today from a laptop running Apple's MLX to a rented NVIDIA RTX PRO 6000
(96 GB) running PyTorch with CUDA. That migration is itself measured, in §7, because every number
from the card is compared with a number the laptop made, and the comparison is only meaningful if
the two paths agree where they should.

## 2. The objects

At layer $\ell$ and position $t$ of an episode the residual stream is $h_{\ell,t}\in\mathbb R^d$.
Two instruments read it.

### 2.1 The Jacobian lens

For an ordered pair of positions with the later one $u$ at a target layer $r$, the local map is
the derivative $K_\ell(\omega)=\partial h_{r,u}/\partial h_{\ell,t}$, where $\omega$ names the
context and the pair. The lens is its expectation over a declared law $\nu$ of contexts and pairs:

$$
J_\ell^{\nu}=\mathbb E_{\omega\sim\nu}[K_\ell(\omega)].
$$

Averaging is part of the definition. The released estimator sums over targets within a context
and averages over contexts, while the paper defines an expectation over pairs; the two differ by a
positive factor, $(T_0+1)/2$ for uniform eligible pairs, which rescales every map and changes no
ranking. We record which convention a lens was fitted under, and we fitted our own lenses at every
layer on the running model rather than reading a hosted one at one depth.

The lens turns a residual into a score over the vocabulary through the model's own readout $W$
and final-norm gain $g$:

$$
Lx = W\big((J_\ell x)\odot g\big).
$$

The per-vector RMS normaliser is a positive scalar and is dropped, so $L$ is linear; ranks and
overlaps are unaffected and score values are not logits. Pulling the readout rows back through
the lens gives one direction per vocabulary entry, $\tilde j_v=J_\ell^\top w_v$, normalised to
$j_v$ with length $\lambda_v$, so that $L=\Lambda B^\top$ with $B$ the matrix of unit
directions. $B$ is the *J dictionary*: what the model's future computation, as the lens sees it,
can read out of this layer.

### 2.2 The sparse dictionary

A JumpReLU sparse autoencoder gives a code $z=E(h)$ and a reconstruction $\hat h=b+Dz$ with unit
decoder rows $d_i$, and the residual of that reconstruction is an exact identity, not an
approximation:

$$
\varepsilon(h):=h-b-DE(h),\qquad h=b+Dz+\varepsilon(h).
$$

A dictionary is not an identifiable coordinate basis: $Dz=(DS)(S^{-1}z)$ for any invertible $S$,
so nothing below treats a feature as a coordinate. We use Google's Gemma Scope 2 residual
dictionaries at every layer of the 4B and the 12B, width 16,384, fetched by exact filename and
verified against the hub's declared digest for each file, because the deep-dive and every-layer
suites ship files identical in size and header at the same hook.

## 3. Stage A: the exact bridge between the two instruments

Because $L$ is linear, the two decompositions compose exactly:

$$
Lh = Lb + \sum_i z_i\,(Ld_i) + L\varepsilon .
$$

Feature $i$'s signed contribution to the score on token $v$ is $z_i\,(Ld_i)_v$, and everything the
dictionary did not explain sits in $L\varepsilon$, bounded by
$\|L\|_{\mathrm{op}}\|\varepsilon\|_2$. The code asserts this identity on the emitted token before
it reports anything, so a wrong orientation or convention cannot produce a ranked table.

**A1**, the readout of a dictionary through the lens, computes for every feature the top-$k$
tokens of $Ld_i$ at the layer the dictionary's hook names. $LD^\top$ is vocabulary by width, 17 GB
at this model, and is never formed; $U=DJ^\top$ is formed once and multiplied against the readout a
chunk of features at a time, and a test enforces this at the matrix multiply rather than by
reading the code. Three checks accompany every A1: the same quantity computed two ways, composed
and as a direct product, agreeing to float32; a **negative control** that reads the same decoder
through the lens of another layer and must share little of the top-$k$, and is shown to fail by
passing the same layer twice; and a **convention discriminator** that compares our readout,
with and without the gain, against the token lists the dictionary itself ships, with the raw arm
recorded before the other is looked at.

*Instrument calibration, not a finding about the model:* on the 4B at the dictionary's layer, the
shipped token lists match the gain-applied readout at 0.995 and the raw one at 0.245, which
settles the convention; the control shares 0.047, 0.231 and 0.152 of the top-10 through the lenses
of three other layers and 1.000 through its own; measured at every layer, the overlap curve is
single-peaked at the hooked layer, rises slowly from below and falls fast from above, and a map
near the identity is far from the readout, which is why controls are chosen from that curve and
not from a matrix norm.

**A2**, the decomposition at one position, computes every term of the identity for a real
activation and its emitted token, reports the residual share $\|L\varepsilon\|/\|Lh\|$, and
**refuses to rank features when that share exceeds a declared dominance**, because the
decomposition is then describing the dictionary's failure and not the model. A
reconstruction-budget gate reports the distribution of that share over a set of positions before
any per-position table is drawn. A2 has not run on a real activation yet: producing one is a
forward pass, which is the device's.

## 4. J-space as a sparse union of cones, and why coefficients are not read

The derivation defines J-space at budget $k$ as the union of cones over supports of size at most
$k$ in the J dictionary,

$$
\mathcal C_k(B)=\{Bc: c\ge 0,\ \|c\|_0\le k\},
$$

with the nonnegative sparse projection $P_k(h)$ and the fit fraction
$\rho_k(h;B)=\|P_k(h)\|_2^2/\|h\|_2^2\in[0,1]$, licensed by the orthogonality of fitted and
residual at an exact fit (Proposition 1). Three things follow that shape every use of it.
The Gram matrix $G=B^\top B$ carries the overlap between vocabulary directions, so the sparse
coefficients are a Gram-corrected pursuit of the bridge scores,
$c_k(h)=\mathcal Q_{k,G}(B^\top b+Tz+B^\top\varepsilon)$, and any number from them must be
reported with three separate errors: the dictionary's reconstruction error, the lens estimation
error, and the solver's own optimisation gap, which a solver cannot certify from its residual.
A residual left by the fit can itself lie inside the cone at that budget, so *unexplained by this
fit* never means *outside J-space*. And energy is not additive over a non-orthogonal dictionary,
so a per-feature share of $\|h\|^2$ is not a partition and can exceed one; a feature's
contribution to a score and its share of the geometry are different quantities in different units
and never appear in one table. Every device-phase artefact therefore reports at the level of
J-space vectors; coefficient tables are diagnostic only.

Stage B, the cone fit with the enrichment test $\Delta_\rho$ against matched controls, is built
only if A2's residual share on real activations justifies it. That number is the device's.

## 5. The intervention

The derivation's decoder intervention replaces a chosen set $F$ of feature values with targets
$z'_F$ while leaving everything else where it was:

$$
\mathcal I_{F,z'_F}(h)=h+D_F\big(z'_F-E(h)_F\big),
\qquad
h'=b+D_{-F}z_{-F}+D_Fz'_F+\varepsilon .
$$

The base reconstruction residual is preserved by construction, and the code tests that identity
exactly in float32 rather than to a tolerance. What is **not** guaranteed, and the derivation says
so, is that re-encoding attains the target: neither $E(h')_F=z'_F$ nor $E(h')_{-F}=E(h)_{-F}$
holds in general, since $E(h')-E(h)=J_E(h)D_F(z'_F-z_F)+o(\|h'-h\|)$. So every application
reports the re-encoding as numbers, the target error and every unselected feature's change, and
never as an assertion of attainment. The fixture that ships with the wrapper is the reason: a
requested value of 3 re-encodes to 5 and moves a neighbouring feature by 2 while the reconstruction
error is preserved exactly.

Two modes are distinguished in every record because the derivation treats them as different
claims: a **one-shot** patch applied once per fresh prefill, and a **clamp** reapplied at every
forward that recomputes the position, keyed to absolute positions or to the positions of emitted
tokens, which the caller must declare on every forward because the capture cannot see sampling.
A clamp on a position already inside the cache refuses before any block runs. A summary form of
the diagnostic exists for corpus-scale records, keeping the selected readings whole and reducing
the off-target change to its count, maximum, L1 norm and eight largest signed entries.

Finite behavioural effects are reached through the integral form of the score change and a bound
that separates three errors, the variation of the local Jacobian around its mean, the estimation
error of the lens, and the curvature term $\tfrac{M_\omega}{2}\|\delta\|^2$; a low averaged lens
reading is therefore never taken as evidence of low causal influence, since averaging can cancel a
sensitivity.

## 6. The state programme

This is the programme the day's work exists to run. The environment is a partially observed
process: a hidden state $X_e$, actions $A_e$, observations $O_{e+1}\sim\Omega(\cdot\mid
X_{e+1},A_e)$, and the Bayes belief $b_e(x)=\Pr(X_e=x\mid\mathsf H_e)$ with its update. The
candidate representation is a feature state $s_e=\phi(R_e)=\psi(\{E_\ell(h_{\ell,t})_F\})$ read
from the model's residuals at a declared set of positions.

### 6.1 What would make it a state

Two properties are defined, and each is a theorem about maps rather than a correlation.
**Predictive sufficiency**: for a declared family of queries $\mathcal Q$, two model conditions
are equivalent when every query's outcome law agrees, and $\phi$ is sufficient when
$\phi(R)=\phi(R')\Rightarrow R\sim_{\mathcal Q}R'$, equivalently $Y_q\perp R\mid s,q$.
**Update closure**: an update rule on the abstraction exists with $\phi(\mathcal
T_u(R))=f_u(\phi(R))$, which requires $\phi$ to be a congruence of the transition (Proposition 3).
A **causal state abstraction** adds an internal intervention $\mathcal I_{s\to s'}$ that commutes
with the abstraction's own transition,

$$
\phi\circ\mathcal T_u\circ\mathcal I_{s\to s'}=f_u\circ i_{s\to s'}\circ\phi ,
$$

and exact commutation implies agreement of whole counterfactual trajectories (Proposition 4). The
stochastic version replaces equality with a per-step total-variation coupling bound $\epsilon$,
and the trajectory bound is $\operatorname{TV}\le\min\{1,\epsilon_0+n\epsilon\}$ over $n$ steps.

### 6.2 What can be measured with finite episodes

Total variation is not observable; a declared family of $M$ bounded diagnostics
$f_1,\dots,f_M$ gives the restricted distance
$d_{\mathcal F}(P,Q)=\max_j|\mathbb E_Pf_j-\mathbb E_Qf_j|\le\operatorname{TV}(P,Q)$, estimated
from $n$ independent episodes with Hoeffding's guarantee

$$
\Pr\Big(\max_{j\le M}|\widehat\Delta_j-\Delta_j|>\eta\Big)\le 2M\exp(-n\eta^2),
\qquad\text{so}\qquad
n\ \ge\ \frac{\ln(2M/\alpha)}{\eta^2}.
$$

The estimands are all of this form. **Substitution**: the outcome law after patching the
representation of state $v$ to $v'$ against the reference law of a model naturally in $v'$,
$\mathcal E_{\mathrm{sub}}=\mathbb E\,\operatorname{TV}(P^{\mathrm{patch}}_{v\to
v'},P^{\mathrm{ref}}_{v'})$; it is informative only when the natural contrast
$\mathcal D_{\mathrm{nat}}$ between the two states' reference laws is large, which the acceptance
rule requires as $\mathcal D_{\mathrm{nat}}\ge d_{\min}$. **Specificity**: an orthogonal
outcome $Y^\perp$ the state should not move, $\mathcal E_\perp$. **Reuse** across a family of
uses of the same state. **Predictive** and **dynamic** errors against the fitted belief update.
**Retention** $\mathcal R_F(\Delta;\kappa)$ after a delay, under carrier controls, because a
retained transcript can carry a fact the model recomputes and mutual information between a feature
at two times does not identify persistent memory. The acceptance rule is a declared set of
tolerances on these, and the derivation is explicit that the tolerances **cannot be derived from
the geometry**: they encode the scientific resolution required, and finite-data claims carry
intervals against them, with failure to reject never read as equivalence.

### 6.3 The worked model and the variable we chose

The derivation's worked example is an uncertain door: a binary hidden state flipping at rate
$\alpha$, an observation with likelihood ratio $\lambda$, and the belief update in logits,
$s_{e+1}=\operatorname{logit}[\alpha+(1-2\alpha)\sigma(s_e)]+\log\lambda_{e+1}$, read out through
a report, a threshold decision and a cost choice, with two doors in swapped states as the test that
the representation carries relations and not a bag of concepts. Our environment already contains
that structure. The Director chose **file existence** as the first state variable: whether the
file the agent must act on exists, observed only through a tool result. The likelihood ratio the
model needs is tool reliability, and the environment supplies it as a controlled instrument: a
false observation is a fault whose message is a well-formed listing that omits a present file or
includes an absent one, applied at a declared rate and seed, which is $\lambda$.

### 6.4 How the run derives its own tolerances

The run script implements the acceptance rule so that no tolerance is typed by hand.
The **existence family** produces matched pairs: the same task, prompt, paths, distractors and
history, differing only in whether the target exists, with the state entering through the
listing and never through the prompt. A **relation family** produces two files with swapped states
across two episodes. **Ten diagnostics** are fixed functions of a transcript, written as code with
tests before any pilot row exists: the first decision after the state-bearing observation (read
the target; search elsewhere; re-list), what the finish answer asserts (exists; absent; names the
path), the same three after a *contradicting* later observation, and the relation test. A
diagnostic that never reaches its position scores nothing rather than zero, so the absence is
dropped rather than diluted.

A **pilot** of matched pairs and a reliability arm measures the device's episode rate first, then
the natural contrast per diagnostic with a seeded bootstrap lower bound. The **drop rule** removes a
diagnostic whose bound on the contrast does not exceed zero, with its reason, and nothing is added
afterwards. The ladder is then $d_{\min}$ the smallest retained bound, $\varepsilon_{\mathrm{sub}}=d_{\min}/4$ so a passing patch has moved behaviour three-quarters of the way to the reference,
$\varepsilon_\perp=2\varepsilon_{\mathrm{sub}}/3$, $\varepsilon_{\mathrm{reuse}}=\varepsilon_{\mathrm{sub}}$,
and $\varepsilon_{\mathrm{pred}},\varepsilon_{\mathrm{dyn}}$ the split-half log-loss gap of a
Bernoulli belief update fitted per arm on falsified episodes, which is the pilot's own noise floor.
The corpus size is $n=\lceil\ln(2M/\alpha)/\varepsilon_{\mathrm{sub}}^2\rceil$ at $\alpha=0.05$,
which the code checks against two closed forms, 2,397 episodes per arm at ten diagnostics and
tolerance 0.05, and 1,066 at 0.075. A **budget rule** converts $4n$ decodes through the measured
rate into hours; if that exceeds the hours bought, the substitution tolerance is loosened and both
numbers are written before the run, never fewer diagnostics and never no control. The derived table
is **sealed** with the digest of the pilot rows it came from; the main run refuses to start
without the seal and refuses to re-derive once a main-run row exists. Each estimand is then
reported in one of three states, measured against its tolerance, untestable when the tolerance is
exactly zero because the pilot's update never varied, or not measured when no scorable row exists,
and only the first can pass. The relation test enters the table as a contrast against its own
negation, $2\cdot\mathrm{mean}(D_{10})-1$, which is the relation sentence in the table's shape.

Two decoding modes give two estimands and two records that are never compared. Greedy decoding
yields one outcome per context, so its estimand is the context-averaged outcome frequency over the
task distribution. Sampled decoding, ruled today for the state programme only at temperature 1.0
with no truncation, seeded through the device's pinned seed and recorded with the sampler, the
seed and the kernel set in every manifest, measures the model's own distribution over outcomes in a
context, which is the derivation's estimand. Sampled draws reproduce only within one backend and
kernel set.

## 7. What makes the device's numbers comparable to the laptop's

Every device gate has a laptop number it must reproduce, and the migration is measured as an
instrument before anything is measured through it. *Instrument calibrations, laptop-made:* the
torch decode reproduces the MLX golden trajectories at 98 of 103 greedy tokens on the real 4B, and
the five flips are all at positions the model held with low confidence, none at the 78 confident
ones, which is not chance at $p=8.4\times10^{-4}$; the multi-device training gate compares
per-parameter gradients between one device and two, because the loss alone agreed to the last
digit on a run whose gradients were wrong by 1.67 relative, and the joined path agrees at
$1.2\times10^{-7}$; the key-value cache strategies are gate arms with an equivalence claim each,
after one strategy armed late was found to corrupt the stream at 2,749 positions; the lens
estimator matches an independent finite-difference reference at a worst layer of $3.6\times10^{-3}$
with the error falling by 3.9 when the step halves, and the transposed convention is 250 times
worse, which is how the convention was fixed. Determinism is pinned before the first CUDA use and
read back into every manifest. The first hour on the device follows a runbook whose rules are that
each result is written when completed, a resume is keyed on the content of the tree that ran, the
checkpoint digest, the device reading and the gate input, only a pass resumes, and every number
carries a basis.

## 8. What none of this is entitled to say

The derivation's own table governs the reading of every artefact. Accurate reconstruction is a
sparse description of sampled geometry and says nothing about meaning or cause. A large linear
score contribution is a contribution to the specified lens score at that layer under that
averaging and that linearisation, not a token probability, a causal derivative, or evidence of a
workspace. A high fit fraction is a good fit by the selected cone family, not causal state content.
Matched finite intervention effects show the targeted directions affect behaviour, not that the
effect implements state substitution. Counterfactual agreement across tasks and updates is an
approximate causal state abstraction on the tested domain, with generality beyond it unresolved. A
delayed effect under carrier controls is state influence carried by the surviving routes. A model
can pass the causal-state tests and fail J enrichment, and that is evidence about the instrument.
And a fixture run of any of this proves plumbing and provenance, never a draw.

## 9. What was built today

| where | lines added | lines removed |
|---|---:|---:|
| source, `src/` | 10,945 | 1,803 |
| tests | 11,787 | 1,152 |
| scripts and acceptance harness | 3,615 | 353 |
| record scripts | 4,121 | 399 |
| records: text, tables, logs | 15,689 | 3,038 |
| orders, plan, runbook | 4,058 | 121 |
| registry | 202 | 10 |
| other | 1,909 | 14 |
| **total** | **52,326** | **6,890** |

That is 219 non-merge commits across five branches touching 276 files, by five seats: a Chief
scientist, a deputy research officer, two software engineers and an external implementer, each
reviewing and testing the others' work against the merge result before anything reached the
integration branch, with every review and ruling written into the orders as a dated heading. The
components, in the order they landed: the device shim and the text-tower loader; the torch view,
capture and graph-once lens estimator with its device gates; the sharded training path with its
gradient gate; the torch lens fit with its golden harness and finite-difference reference; the
decoder intervention with clamps and the summary diagnostic; Stage A of the bridge on real
matrices with its identity check, adapter and budget gate; the sampled decoding path; the state
programme's run script with its family, diagnostics, derivation table, seal and three-state
estimands; the one-command device bootstrap; and a first-hour runbook. Twenty-nine entries in a
method record name what went wrong along the way and the rule each became.

## 10. Status

The rented card is bootstrapping as this is written: the two Gemma checkpoints and the 4B
dictionary at every layer are fetched and verified, the 12B dictionary is downloading, and the
data corpus and preflight follow. The first hour's gates come next, then the lenses at every layer
of both models, then the state programme's pilot, whose first output is the measured episode rate
from which every later time figure is computed. Results follow when they exist, in the same shape
as this document: each number with its basis, and each claim with what it is not.

## Corrections after an external reading, Chief, 2026-09-10

- **The 98-of-103 comparison is teacher-forced**: the acceptance kit feeds the recorded prompt and
  the recorded emitted prefix and compares the produced argmax at each position against the
  recorded token (`research/acceptance/tolerance.py`, `teacher_forced_agreement`). Free-running
  reproduction across backends is struck from the acceptance and only its first divergence is
  reported; on the smoke episode it parts company at the first near-tie. The sentence above should
  have said so.
- **Gemma 3 has no final logit softcap**: `final_logit_softcapping` and `attn_logit_softcapping` are
  both absent from the 4B and 12B configs (checked on the device's snapshots); the readout L is
  linear as written.
- **Independent gates versus self-certifying tests.** The reviewer's count is the right one to keep:
  of this week's gates, the finite-difference-versus-exact comparison, the MLX bfloat16 reference,
  the CPU control, and Codex's seven file-only audits recomputed from raw artefacts are the kind that
  disagree when wrong; the suite is the other kind. The method record will carry the count.
- The carrier question (the transcript as a redundantly available state) is ruled into the
  substitution tests of both state variables in the orders of this date.

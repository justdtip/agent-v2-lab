# SAE–J-lens geometry and causal state abstraction

*A formal derivation for investigating task and environment representations in transformer agents.*  
*8 September 2026*

The mathematical bridge is between an SAE synthesis map and a **linear J-lens readout**. Sparse J-space reconstruction adds a separate, nonlinear inference problem. Neither construction, by itself, identifies a maintained state or a global workspace.

This document derives the bridge, its error terms, and conditions under which a feature representation would constitute a causal state abstraction. Definitions of proposed measurements are distinguished from propositions that follow mathematically. All claims about a particular model remain empirical; no Gemma experiment is assumed to have succeeded.

## 1. Objects, indices, and assumptions

All vectors are columns. Unless explicitly changed, geometry uses the Euclidean inner product in the model's native residual coordinates. Model weights are fixed during measurement. Probabilistic spaces are finite or standard Borel, and the maps and kernels used below are measurable.

| Symbol | Meaning | Space |
| --- | --- | --- |
| $\ell,t$ | Layer and token position | Discrete indices |
| $e$ | Agent event boundary | Discrete index distinct from $t$ |
| $h_{\ell,t}$ | One observed residual activation | $\mathbb R^d$ |
| $R_e$ | Complete low-level agent configuration at a specified cut | $\mathcal R$ |
| $X_e,O_e,A_e,G_e$ | Environment state, observation, action, goal | Task-dependent spaces |
| $\mathsf H_e$ | Observation/action/instruction history | History space |
| $z=E(h)$ | SAE encoding | $\mathbb R_+^m$ |
| $D=[d_1,\ldots,d_m]$ | SAE decoder dictionary | $\mathbb R^{d\times m}$ |
| $b,\varepsilon(h)$ | SAE decoder bias and reconstruction error | $\mathbb R^d$ |
| $K_\ell(\omega)$ | Context-specific residual Jacobian | $\mathbb R^{d\times d}$ |
| $J_\ell$ | Averaged residual Jacobian | $\mathbb R^{d\times d}$ |
| $W$ | Specified effective linear output readout | $\mathbb R^{V\times d}$ |
| $L=WJ_\ell$ | Linear J-lens score map | $\mathbb R^{V\times d}$ |
| $B=[j_1,\ldots,j_V]$ | J dictionary with normalized nonzero columns | $\mathbb R^{d\times V}$ |
| $c_k(h)$ | Sparse nonnegative J coefficients | $\mathbb R_+^V$ |
| $s_e=\phi(R_e)$ | Proposed abstract state | $\mathcal S$ |

Here $V$ denotes the full vocabulary size. Zero J directions are retained as zero score rows and excluded from pursuit. Dimensions can vary by layer; the equal-width notation avoids unnecessary indices. Every SAE and J dictionary must refer to the **same residual hook, layer, coordinate scaling, and model checkpoint** before their vectors can be combined.

At a computational cut, $R_e$ includes whatever can affect continuation: relevant activations, retained keys and values, transcript, external memory, and controller state. Future random draws are exogenous inputs. An observed residual is a function of this configuration, or of its execution up to a chosen hook:

$$
h_{\ell,t}=H_{\ell,t}(R_e).
$$

A collection of residual observations is not automatically a complete machine state. In particular, no transition equation of the form $h_{t+1}=F(h_t,O_{t+1})$ is assumed.

## 2. What the SAE supplies

A representative SAE is

$$
z=E_\theta(h)=\sigma\!\left(W_E(h-b)+a\right),
\qquad
\widehat h=b+Dz,
$$

with nonnegative sparse output $z$. A possible training objective is

$$
\min_{\theta,D,b}
\mathbb E_{h\sim\mu_\ell}
\left[
\frac12\|h-b-DE_\theta(h)\|_2^2
+\lambda\Omega(E_\theta(h))
\right],
\qquad \|d_i\|_2=1,
$$

where $\Omega$ can penalize activation magnitude or enforce a support budget. The derivation below does not require a particular encoder architecture. This setup follows sparse dictionary learning; the relevant SAE construction is described in [Scaling Monosemanticity, §1.1](https://arxiv.org/html/2605.29358v1#S1.SS1).

Define the reconstruction error rather than discarding it:

$$
\boxed{
\varepsilon(h):=h-b-DE(h),
\qquad h=b+Dz+\varepsilon(h).
}
$$

This is an **exact identity**. Its usefulness depends on the size and downstream importance of $\varepsilon$.

Training requires a distribution of activations. Human concept names are not required by the objective. Semantic interpretation is a subsequent hypothesis about the resulting features.

### 2.1 A dictionary is not an identifiable coordinate basis

For a positive diagonal matrix $S$,

$$
Dz=(DS)(S^{-1}z).
$$

Column normalization removes this scale ambiguity, but feature permutations and alternative dictionaries remain possible. For a fixed overcomplete $D$, if

$$
Dn=0,
\qquad z+n\geq0,
$$

then $D(z+n)=Dz$. Sparsity may restrict this ambiguity without eliminating it.

Thus $z$ is an encoder-selected description of $h$, not a uniquely identified collection of semantic variables. An encoder and decoder also need not satisfy

$$
E(b+Dz)=z.
$$

That distinction becomes essential when interpreting interventions.

## 3. Deriving the J-lens object

### 3.1 A local derivative has a specified computational endpoint

Let $\omega$ specify a context, source position $t$, target position $u\geq t$, and target residual layer $r$. Hold the input token sequence and all other upstream inputs fixed. Let

$$
F_\omega:\mathbb R^d\longrightarrow\mathbb R^d
$$

be the downstream computation from an intervention on $h_{\ell,t}$ to the selected target residual. Its Jacobian at the unmodified activation is

$$
K_\ell(\omega)
=D F_\omega(h_{\ell,t})
=\frac{\partial h_{r,u}}{\partial h_{\ell,t}}.
$$

For a perturbation $\delta$,

$$
F_\omega(h+\delta)-F_\omega(h)
=K_\ell(\omega)\delta+\mathcal R_\omega(\delta).
$$

If the derivative is $M_\omega$-Lipschitz along the intervened segment,

$$
\|\mathcal R_\omega(\delta)\|_2
\leq\frac{M_\omega}{2}\|\delta\|_2^2.
$$

For teacher-forced future positions this differentiates the computation on a fixed token sequence. It does not differentiate through discrete sampling or automatically describe a free-running trajectory after its tokens change.

### 3.2 Averaging is part of the definition

Choose a distribution $\nu$ over $\omega$, including a specified weighting of source/target pairs, and define

$$
\boxed{J_\ell^{\nu}:=\mathbb E_{\omega\sim\nu}[K_\ell(\omega)].}
$$

Its sample estimator is

$$
\widehat J_\ell=\sum_{n=1}^N w_n K_\ell(\omega_n),
\qquad w_n\geq0,\quad \sum_nw_n=1.
$$

The paper constructs an averaged residual Jacobian and uses its pullback of vocabulary directions as a lens. Its default Sonnet implementation targets the penultimate residual; the main exposition uses a final-residual schematic. Endpoint, pair weights, and output normalization must therefore be explicit. See [J-lens methods](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens) and [methodological details](https://transformer-circuits.pub/2026/workspace/index.html#app-method-details).

A source-averaged sum over future targets has the convention

$$
J_\ell^{\mathrm{sum}}
=\mathbb E_p\left[
\frac1{|V_p|}\sum_{t\in V_p}
\sum_{\substack{u\in V_p\\u\geq t}}K_{p,u,t}^{(\ell)}
\right],
$$

where $V_p$ is the eligible-position set. For unmasked equal-length sequences of length $T_0$,

$$
J_\ell^{\mathrm{sum}}
=\frac{T_0+1}{2}\,
\mathbb E_{p,\,\mathrm{uniform\ eligible\ pairs}}K_{p,u,t}^{(\ell)}.
$$

The positive global factor leaves normalized directions and cones unchanged. Variable lengths can change the relative weighting of prompts and hence directions. Averaging future targets separately for each source position is another weighting again.

There is no assertion that $J_\ell h$ estimates the actual target activation without an intercept. Taylor expansion is anchored at an observed activation; applying an average Jacobian to an absolute activation is an additional lens construction.

### 3.3 Pullback directions and normalization

For a specified linear readout $W$, write its vocabulary rows as $w_v^\top$. Then

$$
L=WJ_\ell,
\qquad
\widetilde j_v=J_\ell^\top w_v,
\qquad
(Lh)_v=\widetilde j_v^\top h.
$$

Set

$$
\lambda_v=\|\widetilde j_v\|_2,
\qquad
j_v=\begin{cases}\widetilde j_v/\lambda_v,&\lambda_v>0,\\0,&\lambda_v=0,\end{cases}
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_V).
$$

Consequently,

$$
\boxed{L=\Lambda B^\top.}
$$

Normalizing nonzero dictionary columns preserves their positive rays and hence their cone, but changes coefficient magnitudes and score rankings. Keep $\Lambda$ if reconstructing the unnormalized lens scores. Set coefficients of zero columns to zero throughout; softmax still uses the full vocabulary.

The full displayed lens can include a normalization and a softmax. For example, if final normalization has the form

$$
N(y)=\Gamma Cy/a(y)+\beta,
\qquad a(y)>0,
$$

with fixed centering matrix $C$ and gain $\Gamma$, then choosing $W=W_U\Gamma C$ gives

$$
W_U N(Jh)+b_U
=\frac{Lh}{a(Jh)}+W_U\beta+b_U.
$$

This normalized readout is not linear in $h$, and its softmax probabilities are not additive over features. If the model uses another readout, it must be differentiated or specified separately.

For an actual final target residual $y=F_\omega(h)$, the exact local derivative of the output logits $g=W_U N(y)+b_U$ is

$$
D_hg=W_U DN(y)K_\ell(\omega).
$$

For $p=\operatorname{softmax}(g)$,

$$
\nabla_h\log p_v
=K_\ell(\omega)^\top DN(y)^\top W_U^\top(e_v-p).
$$

This is generally different from $J_\ell^\top w_v$. In particular, a linear lens score is neither an exact token probability nor its context-specific causal derivative.

### 3.4 Metric dependence and cancellation

The pullback is naturally a covector. Identifying it with an intervention vector uses a metric. Under $h'=Th$,

$$
\widetilde j_v'=T^{-\top}\widetilde j_v,
\qquad
\delta'=T\delta.
$$

Covectors and perturbation vectors therefore transform differently. With positive definite metric $M$, the corresponding gradient direction is $M^{-1}\widetilde j_v$. All cone geometry below fixes $M=I$; whitening defines a different geometry unless the metric is transported consistently.

Also, averaging can cancel causal sensitivities. With $\Delta_\omega=K_\ell(\omega)-J_\ell$,

$$
\mathbb E[K_\ell^\top K_\ell]
=J_\ell^\top J_\ell+
\mathbb E[\Delta_\omega^\top\Delta_\omega].
$$

Hence

$$
\|J_\ell\delta\|_2^2
\leq \mathbb E\|K_\ell(\omega)\delta\|_2^2.
$$

A small averaged effect does not imply a small context-specific effect. For example, equally likely $K=I$ and $K=-I$ give $J=0$ despite $\|K\delta\|=\|\delta\|$.

## 4. J-space as a sparse union of cones

The paper's operational construction uses sparse nonnegative combinations of J vectors and approximate pursuit. Its formal appendix also uses spans and orthogonal projections, which do not impose nonnegativity. We use the cone definition consistently. See [J-space methods](https://transformer-circuits.pub/2026/workspace/index.html#methods-jspace) and [sparse-frame appendix](https://transformer-circuits.pub/2026/workspace/index.html#app-sparse-frame).

For a fixed budget $k$, define

$$
\boxed{
\mathcal C_k(B)
=\{Bc:c\geq0,\ \|c\|_0\leq k\}
=\bigcup_{|S|\leq k}\operatorname{cone}(B_S).
}
$$

Each constituent cone has dimension at most $\operatorname{rank}(B_S)\leq k$. The union is generally nonconvex and is not a vector space. Its ambient linear span can still be all of $\mathbb R^d$.

Define the ideal reconstruction problem

$$
c_k(h)\in
\underset{c\geq0,\ \|c\|_0\leq k}{\arg\min}
\frac12\|h-Bc\|_2^2,
\qquad
P_k(h):=Bc_k(h).
$$

There are finitely many finitely generated closed cones, so a nearest point exists. It need not be unique. Even when $P_k(h)$ is unique, coefficients can be nonunique. Fix a tie rule when treating either as a function.

An approximate solver produces $\widehat c_k$, $\widehat P_k$. Its optimization gap is

$$
\eta_{\mathrm{opt}}(h)
=\|h-B\widehat c_k(h)\|_2^2
-\min_{c\geq0,\,\|c\|_0\leq k}\|h-Bc\|_2^2
\geq0.
$$

This gap is distinct from SAE reconstruction error and Jacobian estimation error. If the optimum is unavailable, a solver cannot certify this gap merely by reporting its own residual.

### Proposition 1. An exact fit has an orthogonal fitted/residual pair

Let $p=P_k(h)$ be any exact minimizer and $r=h-p$. Because $\alpha p$ is feasible for every $\alpha\geq0$, a nonzero minimizer satisfies

$$
0=\left.\frac{d}{d\alpha}\|h-\alpha p\|_2^2\right|_{\alpha=1}
=-2p^\top(h-p).
$$

The equality is also true for $p=0$. Therefore

$$
\|h\|_2^2=\|P_k(h)\|_2^2+\|h-P_k(h)\|_2^2.
$$

For $h\ne0$, the exact geometric fit fraction is

$$
\boxed{
\rho_k(h;B)
=1-\frac{\operatorname{dist}(h,\mathcal C_k(B))^2}{\|h\|_2^2}
=\frac{\|P_k(h)\|_2^2}{\|h\|_2^2}
\in[0,1].
}
$$

This does **not** make $P_k$ a linear orthogonal projector. Nor does it make $r$ orthogonal to every J direction.

### Counterexample: the residual can itself belong to J-space

Take $B=[e_1,e_2]$, $k=1$, and $h=(2,1)^\top$. Then

$$
P_1(h)=(2,0)^\top,
\qquad h-P_1(h)=(0,1)^\top\in\mathcal C_1(B).
$$

Thus the residual means **unexplained by this fit and budget**. It does not mean nonverbalizable, outside every J cone, or non-workspace computation.

For approximate fits, the two expressions defining $\rho_k$ can disagree. A feasible solver with error no worse than the zero solution gives a reconstruction-improvement score in $[0,1]$; use that expression unless orthogonality has been verified. A small objective gap alone also does not ensure coefficient stability near competing supports.

## 5. The exact SAE-to-J bridge

### 5.1 Linear score decomposition

Apply the linear readout to the exact SAE identity:

$$
Lh=Lb+LDz+L\varepsilon.
$$

Define the unnormalized score transfer matrix

$$
\widetilde T:=LD\in\mathbb R^{V\times m}.
$$

Then

$$
\boxed{Lh=Lb+\widetilde Tz+L\varepsilon.}
$$

For normalized dictionary scores, define

$$
r(h):=B^\top h,
\qquad T:=B^\top D,
\qquad \widetilde T=\Lambda T.
$$

Therefore

$$
\boxed{r(h)=B^\top b+Tz+B^\top\varepsilon.}
$$

For a particular vocabulary direction,

$$
r_v(h)=j_v^\top b+\sum_i z_i\,j_v^\top d_i+j_v^\top\varepsilon.
$$

The signed term $z_iT_{vi}$ is the contribution of SAE decoder term $i$ to this **linear score**, conditional on the chosen decomposition. It is not automatically a probability contribution or a semantic atom.

The error bounds are immediate:

$$
\|r(h)-B^\top b-Tz\|_2
\leq\|B^\top\|_{\mathrm{op}}\,\|\varepsilon\|_2,
$$

$$
\|Lh-Lb-\widetilde Tz\|_2
\leq\|L\|_{\mathrm{op}}\,\|\varepsilon\|_2.
$$

These are exact algebra plus norm inequalities; no semantic assumptions are used.

### 5.2 Sparse coefficients require a Gram correction and support selection

Let

$$
G:=B^\top B.
$$

Expanding the J reconstruction objective gives

$$
\frac12\|h-Bc\|_2^2
=\frac12\|h\|_2^2-r(h)^\top c+\frac12c^\top Gc.
$$

Consequently, with the same tie convention,

$$
\boxed{
c_k(h)=\mathcal Q_{k,G}\!\left(B^\top b+Tz+B^\top\varepsilon\right),
}
$$

where, for attainable scores $r\in\operatorname{range}(B^\top)$,

$$
\mathcal Q_{k,G}(r)\in
\underset{c\geq0,\ \|c\|_0\leq k}{\arg\min}
\left(\frac12c^\top Gc-r^\top c\right).
$$

This is the complete bridge. The score transfer $T$ is linear. The inference map $\mathcal Q_{k,G}$ generally is not. Top scores need not be the best sparse reconstruction because $G$ includes overlap among dictionary directions.

No inversion of the full $V\times V$ Gram matrix is implied. It is singular whenever the retained dictionary has more columns than ambient dimensions. Restricted solves are the relevant objects.

### Proposition 2. A local linear bridge exists on a stable active support

Suppose the exact solution has positive active support $S$, the columns of $B_S$ are linearly independent, and the minimizing face remains the same in a neighborhood of $h$. Then

$$
c_S(h)
=(B_S^\top B_S)^{-1}B_S^\top h,
\qquad c_{S^c}(h)=0.
$$

**Proof.** On this face positivity constraints are inactive. Differentiating the restricted least-squares objective gives

$$
B_S^\top(B_Sc_S-h)=0.
$$

Invertibility of $B_S^\top B_S$ yields the result. The neighborhood assumption is needed to retain the same active face. A coefficient reaching zero or another cone becoming preferable can invalidate it.

Writing $G_{SS}=B_S^\top B_S$,

$$
\boxed{
c_S(h)=G_{SS}^{-1}
\left(B_S^\top b+T_{S,:}z+B_S^\top\varepsilon\right).
}
$$

For a perturbation that preserves this face,

$$
\Delta c_S=B_S^\dagger\delta,
\qquad
\Delta P_k=B_SB_S^\dagger\delta.
$$

For a decoder intervention $\delta=D_F\Delta z_F$,

$$
\Delta c_S
=G_{SS}^{-1}T_{S,F}\Delta z_F.
$$

This is a valid local feature-to-coefficient matrix. It depends on the active support. Near-collinear columns make coordinates unstable because

$$
\|B_S^\dagger\varepsilon\|_2
\leq\frac{\|\varepsilon\|_2}{\sigma_{\min}(B_S)}.
$$

By contrast, $B_SB_S^\dagger$ has operator norm at most one. A stable reconstructed vector need not have stable token coefficients.

### 5.3 Why independently decomposing every decoder does not solve the problem

Suppose

$$
d_i=Ba_i+r_i,
\qquad a_i\geq0,\quad\|a_i\|_0\leq k,
$$

and collect $A=[a_1,\ldots,a_m]$. Then

$$
D=BA+R_D,
\qquad
h=b+BAz+R_Dz+\varepsilon.
$$

This identity is useful, with

$$
\|R_Dz\|_2\leq\sum_i z_i\|r_i\|_2.
$$

But the support of $Az$ can contain as many as $k\|z\|_0$ entries. Even zero reconstruction errors do not make $Az$ the $k$-sparse coefficient vector for $h$.

For $B=D=I_2$, $b=\varepsilon=0$, $k=1$, independently decomposing each decoder gives $A=I_2$. Yet

$$
z=(2,1)^\top,
\qquad Az=(2,1)^\top,
\qquad c_1(Dz)=(2,0)^\top.
$$

Thus

$$
\boxed{c_k(b+Dz)\ne A z\quad\text{in general}.}
$$

Exact decoder fits and a sufficiently small union of active J supports would make $Az$ a feasible exact reconstruction of $Dz$. Even then, equality with a particular coefficient output requires uniqueness or an appropriate tie rule; a nonzero bias must still be handled.

## 6. Feature contributions and localization

### 6.1 Energy is not additive over a nonorthogonal dictionary

Even with $b=\varepsilon=0$,

$$
\|h\|_2^2
=\sum_i z_i^2\|d_i\|_2^2
+2\sum_{i<j}z_i z_j d_i^\top d_j.
$$

Therefore $\|z_id_i\|^2/\|h\|^2$ is not a partition of explained variance and need not lie below one. A feature contribution to a fixed linear score is well-defined as in §5.1; a unique allocation of quadratic energy requires an additional convention for cross terms.

### 6.2 A frozen-support geometric decomposition

For a specified independent support $S$, let

$$
\Pi_S=B_SB_S^\dagger.
$$

Then the exact SAE identity yields

$$
\Pi_Sh=\Pi_Sb+\sum_i z_i\Pi_Sd_i+\Pi_S\varepsilon,
$$

$$
(I-\Pi_S)h=(I-\Pi_S)b+
\sum_i z_i(I-\Pi_S)d_i+(I-\Pi_S)\varepsilon.
$$

This is an exact feature-resolved decomposition into a chosen span and its orthogonal complement. It agrees with the local J fit only under Proposition 2's conditions. Re-selecting $S$ after each change restores nonlinearity.

### 6.3 Finite removal and coalition effects

For decoder removal at the measured activation, define

$$
\Delta_i^P(h):=P_k(h)-P_k(h-z_id_i).
$$

For a coalition $F$,

$$
\Delta_F^P(h):=P_k(h)-P_k(h-D_Fz_F).
$$

The decomposition interaction is

$$
\mathcal I_F^P(h)
=\Delta_F^P(h)-\sum_{i\in F}\Delta_i^P(h).
$$

A nonzero $\mathcal I_F^P$ can arise solely from J support selection. It is not by itself evidence of a neural interaction. Neural interactions require measuring the corresponding finite interventions on the model's downstream computation.

### 6.4 States and state changes require different geometric measurements

The score $\rho_k(d_i;B)$ describes one oriented decoder ray. The score $\rho_k(D_Fz_F;B)$ describes one active coalition contribution. Neither is automatically the fraction of **causal state information** in J-space.

A state replacement has signed displacement

$$
\delta=D_F(z_F'-z_F),
$$

which need not be nonnegative in either dictionary. In general,

$$
\rho_k(\delta;B)\ne\rho_k(-\delta;B).
$$

For a sign-symmetric measurement, choose a support union $U$ from the natural base and donor fits, freeze it, and measure

$$
\eta_U(\delta)
=\frac{\|\Pi_U\delta\|_2^2}{\|\delta\|_2^2}.
$$

This statistic is defined for $\delta\ne0$. It measures **span alignment**, not sparse-cone occupancy. Its null comparison must match $\operatorname{rank}(B_U)$ and the support-selection procedure. Enlarging $U$ can make alignment trivially high.

Adding both $j_v$ and $-j_v$ to the dictionary would define a different sparse set. It must not be reported as the original nonnegative J-space.

### 6.5 Bias and centering are substantive choices

The raw-activation fit uses $h$; a centered analysis might use $h-\mu$. These define different questions:

$$
P_k(h-\mu)\ne P_k(h)-\mu
$$

in general. If centered coordinates are desired, transform the SAE identity to

$$
h-\mu=(b-\mu)+Dz+\varepsilon
$$

and retain the offset. Centering cannot silently remove the SAE bias or translate an origin-anchored cone without changing the estimand.

## 7. From geometric changes to actual model effects

### 7.1 A decoder intervention is a concrete operation

For a feature set $F$, base activation $h$, and desired decoder coefficients $z_F'$, define

$$
\boxed{
\mathcal I_{F,z_F'}(h)
=h+D_F\bigl(z_F'-E(h)_F\bigr).
}
$$

Because $h=b+Dz+\varepsilon$, this preserves the base reconstruction residual in the algebraic replacement:

$$
h'=b+D_{-F}z_{-F}+D_Fz_F'+\varepsilon.
$$

It avoids replacing the entire activation with its SAE reconstruction. But it does not guarantee

$$
E(h')_F=z_F'
\quad\text{or}\quad
E(h')_{-F}=E(h)_{-F}.
$$

To first order, where the encoder is differentiable, write its Jacobian as $J_E(h)$:

$$
E(h')-E(h)
=J_E(h)D_F(z_F'-z_F)+o(\|h'-h\|).
$$

Thus target attainment and changes to other encoded features must be measured. Re-encoding is a diagnostic; it is not a proof that the intervened activation is on the natural activation distribution.

The notation $\operatorname{do}(z_F=z_F')$ is justified only relative to a specified intervention mechanism. The SAE features are usually measurements of the model, not literal variables inserted into its original graph.

### 7.2 Local and finite behavioral effects

For a scalar downstream quantity $q_\omega(h)$,

$$
\left.\frac{d}{d\alpha}q_\omega(h+\alpha d_i)\right|_{\alpha=0}
=\nabla_h q_\omega(h)^\top d_i.
$$

For a finite coalition perturbation $\delta$, the exact path identity is

$$
q_\omega(h+\delta)-q_\omega(h)
=\int_0^1\nabla_h q_\omega(h+\alpha\delta)^\top\delta\,d\alpha,
$$

assuming $q_\omega$ is continuously differentiable near the segment. More generally, absolute continuity and the gradient chain rule almost everywhere along the path suffice. A linear approximation is valid only to the extent that the gradient remains stable.

### 7.3 Separating Jacobian errors

For a target residual and a fixed linear readout $W$,

$$
\begin{aligned}
&\|W[F_\omega(h+\delta)-F_\omega(h)]-W\widehat J\delta\|_2\\
&\quad\leq
\|W\|_{\mathrm{op}}
\left[
\|K_\ell(\omega)-J\|_{\mathrm{op}}\|\delta\|_2
+\|J-\widehat J\|_{\mathrm{op}}\|\delta\|_2
+\frac{M_\omega}{2}\|\delta\|_2^2
\right].
\end{aligned}
$$

These are respectively context mismatch, estimation error, and finite-perturbation curvature. If final normalization or softmax is part of the target, include it in the differentiated function instead of silently using this bound for probabilities.

Comparing general-context and agent-context Jacobians is meaningful only with matched endpoints and weighting conventions. Their difference measures a change of averaged geometry; it does not establish that an agent-specific workspace has been isolated.

## 8. What a task or environment state would mean

### 8.1 External state and available evidence

Let the environment obey

$$
X_{e+1}\sim T(\cdot\mid X_e,A_e),
\qquad
O_{e+1}\sim\Omega(\cdot\mid X_{e+1},A_e).
$$

The experimenter may know $X_e$, but the agent only receives $\mathsf H_e$. For a specified environment model, the Bayesian belief is

$$
b_e(x)=\Pr(X_e=x\mid\mathsf H_e).
$$

For discrete environment states, its update, when the denominator is positive, is

$$
b_{e+1}(x')
=\frac{
\Omega(O_{e+1}\mid x',A_e)
\sum_x T(x'\mid x,A_e)b_e(x)
}{
\sum_{\bar x}\Omega(O_{e+1}\mid\bar x,A_e)
\sum_xT(\bar x\mid x,A_e)b_e(x)
}.
$$

For continuous states, use densities and integrals under the corresponding regularity assumptions. This is a normative reference derived by Bayes' rule, not an assumption that the language model performs Bayesian filtering. A model can represent a mistaken belief. A state it cannot observe need not be represented at all.

Consequently, a proposed semantic variable should specify whether it concerns true state, available evidence, the model's inferred state, a goal, or an intended action. Those labels are not interchangeable.

### 8.2 Candidate feature states

At an event boundary, one possible candidate is

$$
s_e=\phi(R_e)
=\psi\!\left(\{E_\ell(h_{\ell,t})_F:(\ell,t)\in\mathcal A_e\}\right),
$$

where $\mathcal A_e$ specifies aligned sites and $\psi$ is a learned or declared readout. The simplest case is $s_e=z_{F,e}$. The sites must be available from the current configuration or reproducibly measurable from it. Historical activation logs unavailable to the agent would give the analyst extra memory and cannot silently be treated as part of the agent's current state.

The candidate dimension, feature set, event alignment, and admissible $\psi$ are fixed before final evaluation. An unrestricted history-dependent $\psi$ could reconstruct the answer itself, defeating the proposed localization claim.

If separate SAEs are used at different layers, equal feature indices do not identify equal semantics. Layer transport must be learned and validated, for example by testing

$$
\psi_\ell(E_\ell(h_{\ell,t}))
\approx
\psi_{\ell'}(E_{\ell'}(h_{\ell',t}))
$$

on matched semantic variables and interventions.

## 9. Predictive state: sufficiency and update closure

Let $\mathcal Q$ be a declared family of future experiments. A member $q$ specifies future inputs or an adaptive interaction policy, a horizon, and semantic outcome $Y_q$. It also specifies the initial external environment or its law and the input-generating mechanism; the agent configuration $R$ alone does not determine those. A test may ask for a report, route selection, a calculation, or an action. Define

$$
p_q(\cdot\mid R)
=\mathcal L(Y_q\mid\text{start from }R,\ q).
$$

These are controlled continuations, not observational conditioning on whichever task happened to follow a given state.

### Definition 1. Predictive equivalence

$$
R\sim_{\mathcal Q}R'
\iff
p_q(\cdot\mid R)=p_q(\cdot\mid R')
\quad\forall q\in\mathcal Q.
$$

The equivalence class $[R]_{\mathcal Q}$ retains exactly the distinctions visible to this experiment family. It is minimal relative to $\mathcal Q$, not necessarily minimal for every possible use of the model.

A candidate $s=\phi(R)$ is predictively sufficient if

$$
\boxed{
\phi(R)=\phi(R')\ \Longrightarrow\ R\sim_{\mathcal Q}R'.
}
$$

Equivalently, there exist kernels $\overline p_q$ with

$$
p_q(\cdot\mid R)=\overline p_q(\cdot\mid\phi(R)).
$$

For a random mixture over admissible configurations, this factorization implies

$$
Y_q\perp R\mid s,q.
$$

The pointwise factorization is stronger than a conditional-independence fit on a narrow observational distribution.

### Proposition 3. Deterministic update closure requires a congruence

Let a controlled low-level update be $R^+=\mathcal T_u(R)$. There exists a well-defined abstract update $f_u$ satisfying

$$
\boxed{\phi(\mathcal T_u(R))=f_u(\phi(R))}
$$

if and only if

$$
\phi(R)=\phi(R')
\ \Longrightarrow\
\phi(\mathcal T_u(R))=\phi(\mathcal T_u(R'))
\quad\forall u.
$$

**Proof.** Necessity follows by applying $f_u$ to the common value. For sufficiency, define $f_u(s)=\phi(\mathcal T_u(R))$ using any representative $R$ with $\phi(R)=s$. The implication makes that definition independent of the representative.

For the minimal predictive quotient, closure follows when $\mathcal Q$ includes every continuation obtained by prepending any admissible controlled input $u$: distinguishing the successors would then distinguish the original states. This uses the deterministic relation $p_{u;q}(\cdot\mid R)=p_q(\cdot\mid\mathcal T_u(R))$. Stochastic updates require the separate kernel condition in §10.1. A finite set of probes does not guarantee closure.

A candidate can be sufficient for current outputs while containing extra, nonclosed coordinates. Conversely, a recurrent-looking coordinate can be closed while failing to predict the behaviors of interest. Both conditions must be checked.

### 9.1 An empirical information criterion

For finite or discretized outcomes and a declared nuisance variable $V_0$, the best possible log-loss improvement from adding $s$ is

$$
H(Y\mid V_0)-H(Y\mid s,V_0)
=I(Y;s\mid V_0).
$$

The residual predictive information available from the full configuration is

$$
H(Y\mid s,V_0)-H(Y\mid R,V_0)
=I(Y;R\mid s,V_0),
$$

because $s=\phi(R)$. Exact sufficiency makes the second quantity zero. With restricted fitted predictors, observed log-loss differences also reflect approximation and estimation error; they are not exact mutual-information estimates.

There is no requirement that a valid state retain its information after conditioning on the entire transcript. If the state is computed deterministically from that transcript, such conditioning can remove all remaining entropy.

## 10. Causal state abstraction

Predictive sufficiency alone permits a measurement that reports the computation without mediating it. Causal abstraction adds a correspondence between interventions. Interchanging internal representations to test a high-level causal model is established methodology; see [Geiger et al., §3 and Appendix F](https://arxiv.org/html/2106.02997v2#S3). The dynamic conditions and bounds here are stated explicitly for this proposal.

Let $\mathcal I_{s\to s'}$ be a specified low-level intervention, and let $i_{s\to s'}$ be its high-level counterpart. For a replacement of one component of $s$, the other abstract components are retained from the base case.

The first requirement is intervention alignment:

$$
\boxed{
\phi\!\left(\mathcal I_{s\to s'}(R)\right)
=i_{s\to s'}(\phi(R)).
}
$$

The second is update alignment on natural **and intervened** configurations:

$$
\phi(\mathcal T_u(R))=f_u(\phi(R)).
$$

The third is output alignment, for semantic output map $o$ and abstract map $\bar o$:

$$
o(R)=\bar o(\phi(R)).
$$

Together they imply the commutation relation

$$
\boxed{
\phi\circ\mathcal T_u\circ\mathcal I_{s\to s'}
=f_u\circ i_{s\to s'}\circ\phi.
}
$$

### Proposition 4. Exact commutation implies counterfactual trajectory agreement

For an admissible input sequence $u_0,\ldots,u_{n-1}$,

$$
\phi\!\left(
\mathcal T_{u_{n-1}}\circ\cdots\circ\mathcal T_{u_0}
\circ\mathcal I_{s\to s'}(R)
\right)
=
f_{u_{n-1}}\circ\cdots\circ f_{u_0}
\circ i_{s\to s'}(\phi(R)).
$$

**Proof.** Intervention alignment supplies the base case. Apply update alignment at each step and use induction. Output alignment then yields equality of semantic outputs at every evaluated step.

This proposition supplies the mathematical target for state substitution. A change in one final answer is only one consequence of these stronger conditions.

### 10.1 A stochastic version with an explicit error bound

Let the low-level controlled kernel be

$$
\mathsf K(dy,dR'\mid R,u),
$$

and the proposed abstract kernel be

$$
\overline{\mathsf K}(dy,ds'\mid s,u).
$$

The pushforward $(\mathrm{id},\phi)_\#\mathsf K$ is the joint distribution of emitted outcome $y$ and next abstract state $\phi(R')$.

Assume, uniformly over all relevant reachable natural and patched configurations,

$$
\operatorname{TV}\!\left(
(\mathrm{id},\phi)_\#\mathsf K(\cdot\mid R,u),
\overline{\mathsf K}(\cdot\mid\phi(R),u)
\right)\leq\epsilon.
$$

Here $\operatorname{TV}(P,Q)=\sup_A|P(A)-Q(A)|$, equal to $\frac12\sum_y|P(y)-Q(y)|$ on a finite outcome space. Suppose the patched initial abstract law differs from the intended law by at most $\epsilon_0$ in TV. Then for $n$ steps,

$$
\boxed{
\operatorname{TV}\!\left(
\mathcal L(Y_{0:n-1},s_{0:n})_{\mathrm{low}},
\mathcal L(Y_{0:n-1},s_{0:n})_{\mathrm{abstract}}
\right)
\leq\min\{1,\epsilon_0+n\epsilon\}.
}
$$

**Proof.** Couple initial abstract states with disagreement probability at most $\epsilon_0$. While histories agree, choose the same controlled input and couple the next joint outcome/state with disagreement probability at most $\epsilon$. A union bound over steps gives the result. TV cannot increase when internal states are marginalized out.

The same argument permits adaptive inputs when the same continuation policy acts on the coupled observed histories. Here $Y$ must include every observation the continuation policy uses; otherwise augment the coupled state and output. For live environments, their state must be included in the joint system and their transition kernels matched.

A finite test estimates average errors on its sampled domain; it does not establish the uniform premise of this bound. Also, small Euclidean error between deterministic continuous states is not small TV between their point masses. Continuous-state variants require a suitable metric and additional continuity assumptions.

### 10.2 What finite behavioral measurements can estimate

For a prespecified finite family of diagnostic functions $\mathcal F=\{f_1,\ldots,f_M\}$, with $f_j(Y)\in[0,1]$, define

$$
d_{\mathcal F}(P,Q)
=\max_{j\leq M}\left|\mathbb E_Pf_j(Y)-\mathbb E_Qf_j(Y)\right|.
$$

This restricted integral probability metric satisfies

$$
d_{\mathcal F}(P,Q)\leq\operatorname{TV}(P,Q).
$$

Small error on a finite diagnostic family therefore does not establish small TV over all possible text. Full TV is a tractable target for a small declared action space; reports and long trajectories may instead warrant an explicitly restricted $d_{\mathcal F}$. The trajectory bound above requires its TV premise and cannot simply inherit it from a weaker diagnostic bound.

For $n$ independent draws from each of two arms, Hoeffding's inequality applied to the difference of sample means gives

$$
\Pr\!\left(
\max_{j\leq M}|\widehat\Delta_j-\Delta_j|>\eta
\right)
\leq 2M\exp(-n\eta^2),
$$

where $\Delta_j=\mathbb E_Pf_j-\mathbb E_Qf_j$. Thus, at confidence $1-\alpha$,

$$
d_{\mathcal F}(P,Q)
\leq \widehat d_{\mathcal F}
+\sqrt{\frac{\log(2M/\alpha)}{n}}.
$$

All selected contrasts must be included in $M$, or separately adjusted; dependence within an episode does not create additional independent draws. An equivalence claim at tolerance $\epsilon$ requires the upper bound to be below $\epsilon$. This illustrative bound is conservative and concerns fixed diagnostics, not adaptively discovered ones.

## 11. A counterfactual substitution estimand

Let the proposed abstract state be $s=(v,n)$, with target variable $v$ and other task-relevant variables $n$. Choose a base configuration and a donor matched on $n$. A feature patch supplies the donor's target-related values while retaining the base's remaining configuration.

For a continuation experiment $q$, define

$$
P^{\mathrm{patch}}_{v\to v',n,q}
=\mathcal L(Y_q\mid\mathcal I_{F,v'}(R_{v,n})),
$$

and a high-level reference

$$
P^{\mathrm{ref}}_{v',n,q}
=\mathcal L(Y_q\mid\text{abstract state }(v',n),q).
$$

The substitution error is

$$
\boxed{
\mathcal E_{\mathrm{sub}}
=\mathbb E_{v,v',n,q}
\operatorname{TV}\!\left(
P^{\mathrm{patch}}_{v\to v',n,q},
P^{\mathrm{ref}}_{v',n,q}
\right).
}
$$

A natural donor continuation may serve as the reference only when its relevant nuisance variables and future conditions match. The high-level reference should itself describe natural model behavior on held-out cases; agreement with an ideal policy alone is a steering result, not a faithful abstraction of the original model.

### 11.1 The contrast must be behaviorally informative

Define the natural/reference contrast

$$
\mathcal D_{\mathrm{nat}}
=\mathbb E_{v,v',n,q}
\operatorname{TV}\!\left(
P^{\mathrm{ref}}_{v,n,q},P^{\mathrm{ref}}_{v',n,q}
\right).
$$

Require $\mathcal D_{\mathrm{nat}}$ to exceed a prespecified nontrivial threshold. Otherwise a small substitution error can be achieved by an intervention that does nothing.

For any base/donor pair, the triangle inequality gives

$$
\begin{aligned}
&\operatorname{TV}(P^{\mathrm{patch}},P^{\mathrm{ref}}_{v,n,q})\\
&\quad\geq
\operatorname{TV}(P^{\mathrm{ref}}_{v',n,q},P^{\mathrm{ref}}_{v,n,q})
-\operatorname{TV}(P^{\mathrm{patch}},P^{\mathrm{ref}}_{v',n,q}).
\end{aligned}
$$

Thus successful substitution on a large reference contrast must move behavior away from the base **reference**. If the actual natural base law is $P^{\mathrm{base}}$, another triangle inequality gives

$$
\begin{aligned}
\operatorname{TV}(P^{\mathrm{patch}},P^{\mathrm{base}})
\geq{}&\operatorname{TV}(P^{\mathrm{ref}}_{v'},P^{\mathrm{ref}}_v)
-\operatorname{TV}(P^{\mathrm{patch}},P^{\mathrm{ref}}_{v'})\\
&-\operatorname{TV}(P^{\mathrm{base}},P^{\mathrm{ref}}_v),
\end{aligned}
$$

with $n,q$ suppressed. Fidelity of the reference to the original model is therefore part of the conclusion.

### 11.2 Physical state and represented state are separate intervention axes

For a putative belief variable $v$, define

$$
P_{x,v}=
\mathcal L(Y\mid\text{physical state }X=x,
\text{specified internal intervention setting }v).
$$

Before any new state-revealing observation, successful substitution predicts a dependence on $v$. Once physical state changes the observations or action outcomes, later behavior should generally depend on both axes.

For a binary example, the first decision can be organized as follows:

| Physical state | Intervened representation | First decision before new evidence |
| --- | --- | --- |
| Locked | Locked | Consistent with locked representation |
| Locked | Unlocked | Consistent with unlocked representation |
| Unlocked | Locked | Consistent with locked representation |
| Unlocked | Unlocked | Consistent with unlocked representation |

This table is not a prediction that the entire physical trajectory follows the implanted belief. A locked door remains locked. For longer trajectories, the reference must start with the same physical state and the substituted belief, then process whatever observations its actions actually produce.

### 11.3 Specificity and repeated clamps

For declared outcomes $Y^\perp$ that should be unchanged by $v$, measure

$$
\mathcal E_\perp
=\mathbb E\operatorname{TV}\!\left(
\mathcal L(Y^\perp\mid\mathcal I_{F,v'}(R)),
\mathcal L(Y^\perp\mid R)
\right).
$$

These outcomes must be invariant under the target intervention in the proposed abstract model:

$$
\mathcal L_{\mathrm{ref}}(Y^\perp\mid v,n,q)
=\mathcal L_{\mathrm{ref}}(Y^\perp\mid v',n,q).
$$

Here the two laws are generated by setting the abstract variable to the stated value, not by observational conditioning. Paraphrases, irrelevant entities, task identity, and action-label permutations provide distinct nuisance tests; one global performance score cannot substitute for them.

A one-shot patch is applied once and released. A repeated clamp is a sequence of new interventions. If only repeated clamping changes distant behavior, it establishes control under sustained forcing, not autonomous retention of the substituted state.

Failure of a clamp also does not refute all state representations: the targeted coordinates may be incomplete or bypassed. It refutes the stronger claim that this intervention blocks the proposed state update under the tested conditions.

## 12. Temporal persistence and its causal carrier

The earlier proposal

$$
I(z_e;z_{e+\Delta}\mid\text{intervening tokens})>0
$$

does not identify persistent neural memory.

For example, let a historical observation contain bit $B_0$, and suppose each later query recomputes $z_e=B_0$ from the retained transcript. If intervening text $U$ is independent of $B_0$, then

$$
I(z_e;z_{e+\Delta}\mid U)=H(B_0)>0,
$$

even though the earlier $z_e$ is never used to calculate the later $z_{e+\Delta}$. The common cause is the retained record. Conversely, conditioning on that entire deterministic record can yield zero conditional information despite effective state use.

### 12.1 Retention as a controlled intervention effect

Let $\kappa_\Delta$ specify a delay protocol, including distractors, controller behavior, observations, and textual channels. Define

$$
\mathcal R_F(\Delta;\kappa)
=\mathbb E\operatorname{TV}\!\left(
\mathcal L(Y_{e+\Delta}\mid\mathcal I_{F,v'},\kappa_\Delta),
\mathcal L(Y_{e+\Delta}\mid\mathcal I_{F,v},\kappa_\Delta)
\right).
$$

A positive value demonstrates a delayed causal effect under $\kappa$. It does not yet identify the path carrying the effect. State substitution additionally requires agreement with the correct delayed reference, not merely separation of the two conditions.

One possible path is

$$
\text{patch}\longrightarrow\text{generated text}
\longrightarrow\text{future transcript/cache}
\longrightarrow\text{later action}.
$$

Another can pass through retained neural values while emitted tokens are held fixed. These are different mechanisms.

Controlled replay of identical intermediate tokens can block the generated-text path while retaining legitimate downstream cached consequences of the original patch. Resetting or exchanging selected retained components then tests those channels separately. Conditioning observationally on generated tokens is not equivalent to this intervention and can introduce selection bias.

If a new model call reconstructs its full input and retained state from a transcript, and that transcript is identical in both conditions, a vanished transient activation cannot affect the new call through no carrier. Persistent effects must have a path in the actual computation. The instrument should identify that path rather than infer it from a repeated label.

## 13. Flexible access and the scope of “workspace”

J geometry cannot establish broad access by construction alone. Define a family of distinct downstream uses $q\in\mathcal Q_{\mathrm{use}}$, with task-specific semantic output maps $g_q(v,n)$.

Reuse the same state alignment and intervention rule across these uses. Measure

$$
\epsilon_q
=\mathbb E_{v,v',n}
\operatorname{TV}\!\left(
P^{\mathrm{patch}}_{v\to v',n,q},
P^{\mathrm{ref}}_{v',n,q}
\right).
$$

The vector $(\epsilon_q)_q$, or a prespecified worst-case error over a finite family, measures flexible causal reuse. Separate alignments optimized for each task do not demonstrate that the same representation is being reused. A stronger test selects the future task after patching and releasing a common retained checkpoint, then branches it into the different uses.

A local auxiliary diagnostic can use several downstream quantities $g_1,\ldots,g_Q$. At a fixed context, stack

$$
\mathsf B_F(R)=
\begin{bmatrix}
D_hg_1(R)D_F\\
\vdots\\
D_hg_Q(R)D_F
\end{bmatrix}.
$$

Nonzero blocks show local sensitivity through the chosen feature directions. Rank measures independent infinitesimal effects. Neither sensitivity nor rank alone establishes correct semantic use, persistence, or access by every circuit.

Operational “workspace-like state” therefore means a state abstraction that is selectively encoded, causally reusable across a declared family of computations, and associated with the specified J geometry. It remains a functional claim with an explicit experimental scope.

## 14. Testing preferential J localization without circular selection

First discover candidate state features using semantic, dynamic, and causal criteria independently of their J alignment. On held-out episodes, compare them with control features or coalitions matched for layer, activation prevalence, norm, coalition size, and selection opportunity.

For oriented feature geometry, one possible estimand is

$$
\Delta_\rho
=\mathbb E_{i\sim\mathcal F_{\mathrm{state}}}\rho_k(d_i;B)
-\mathbb E_{i\sim\mathcal F_{\mathrm{control}}}\rho_k(d_i;B).
$$

For context-dependent coalition geometry, define a separate estimand using

$$
x_{F,e}=D_Fz_{F,e},
\qquad \rho_k(x_{F,e};B),
$$

with a declared policy for zero vectors. For intervention displacement geometry, use the span statistic $\eta_U$ from §6.4 and match the support-union rank. These quantities answer different questions and should not be averaged into one occupancy number.

A geometric enrichment hypothesis can be written

$$
H_0^{\mathrm{geom}}:\Delta_\rho\leq0,
\qquad
H_1^{\mathrm{geom}}:\Delta_\rho>0.
$$

It does not assert that all state lies in the cone. If candidates are selected because they already have large $\rho_k$, this contrast cannot independently support enrichment.

Nor should $k$ be selected to maximize the reported effect on the evaluation set. For a fixed dictionary,

$$
\mathcal C_k(B)\subseteq\mathcal C_{k+1}(B)
\quad\Longrightarrow\quad
\rho_k(h;B)\leq\rho_{k+1}(h;B).
$$

Report the prespecified sparsity curve and matched controls. Random uniform directions alone need not match the model's anisotropic activation distribution.

For actual attribution of behavior to J-aligned and complementary components, split a signed patch with a frozen projector,

$$
\delta_J=\Pi_U\delta,
\qquad \delta_\perp=(I-\Pi_U)\delta,
$$

then separately evaluate the base, full patch, each component, and their joint patch. For scalar behavior $q$,

$$
\mathcal I_q
=q(h+\delta_J+\delta_\perp)-q(h+\delta_J)
-q(h+\delta_\perp)+q(h).
$$

A nonzero $\mathcal I_q$ makes the behavioral effects nonadditive. Geometric energy fractions cannot be substituted for these causal measurements.

## 15. Worked state model: uncertain door status

This example fixes an abstract model to make the proposed tests precise. It does not assert that a transformer implements it.

Let $X_e\in\{0,1\}$, where $1$ means locked. Suppose the physical state flips independently with probability $\alpha$ between events, and the current belief is

$$
p_e=\Pr(X_e=1\mid\mathsf H_e).
$$

Prediction gives

$$
\bar p_{e+1}
=(1-\alpha)p_e+\alpha(1-p_e)
=\alpha+(1-2\alpha)p_e.
$$

For an observation with likelihood ratio

$$
\lambda_{e+1}
=\frac{\Pr(O_{e+1}\mid X_{e+1}=1)}
{\Pr(O_{e+1}\mid X_{e+1}=0)},
$$

Bayes' rule yields

$$
\boxed{
p_{e+1}
=\frac{\lambda_{e+1}\bar p_{e+1}}
{\lambda_{e+1}\bar p_{e+1}+1-\bar p_{e+1}}.
}
$$

For interior probabilities, write $s_e=\operatorname{logit}(p_e)$. Then

$$
\boxed{
s_{e+1}
=\operatorname{logit}\!\left[
\alpha+(1-2\alpha)\sigma(s_e)
\right]+\log\lambda_{e+1}.
}
$$

If $\alpha=0$ and no new evidence arrives, $\lambda=1$ and $s_{e+1}=s_e$. For $0<\alpha<1$, repeated predictions without fresh evidence converge toward $p=1/2$, since the deviation is multiplied by $1-2\alpha$ each step. At $\alpha=1$, the belief instead alternates. A correct persistent belief need not have constant activation.

Suppose the proposed SAE-based alignment is

$$
\widehat p_e=\sigma\!\left(a^\top z_{F,e}+\beta\right).
$$

Its semantic error against the chosen reference can be measured with a proper scoring rule; its dynamic error is separately measured by

$$
\mathcal E_{\mathrm{dyn}}^{\mathrm{door}}
=\mathbb E\left[
\left(\widehat p_{e+1}-f(\widehat p_e,O_{e+1})\right)^2
\right],
$$

where $f$ is the probability update above. Low observational error in both does not establish mediation.

Now interchange candidate features from a matched donor with belief $p'$. Evaluate three distinct future uses, without fitting a different patch for each:

$$
g_{\mathrm{report}}(p')=p',
$$

$$
g_{\mathrm{decision},\tau}(p')=\mathbf1[p'<\tau],
$$

$$
g_{\mathrm{cost}}(p')=p'C_{\mathrm{locked}}+(1-p')C_{\mathrm{unlocked}}.
$$

After a later observation, all three should use $f(p',O)$, with appropriate output noise or calibration specified in the abstract model. Vary $\tau$, costs, paraphrase, entity identity, and observation reliability independently of the donor belief.

A feature that directly steers an “enter” action may pass one threshold decision while failing probability reports, different thresholds, expected-cost calculations, or later evidence updates. These failures distinguish an action bias from the stronger proposed state abstraction.

For multiple doors, compare assignments such as

$$
(A:\mathrm{locked},B:\mathrm{unlocked})
\quad\text{and}\quad
(A:\mathrm{unlocked},B:\mathrm{locked}).
$$

The words and marginal concepts can be identical while their relations differ. If the candidate representation cannot distinguish the assignments under entity-specific queries, a bag of “door” and “locked” features is insufficient.

## 16. What can be concluded, and what must be estimated

For a candidate $(F,\phi,\mathcal I,\overline{\mathsf K})$, retain a vector of separate quantities:

$$
\mathbf e=
\left(
\mathcal E_{\mathrm{sem}},
\mathcal E_{\mathrm{pred}},
\mathcal E_{\mathrm{dyn}},
\mathcal E_{\mathrm{sub}},
\mathcal E_\perp,
\{\epsilon_q\}_q,
\{\mathcal R_F(\Delta;\kappa)\}_{\Delta,\kappa},
\Delta_\rho
\right).
$$

Here semantic error measures the stated meaning; predictive error measures lost future distinctions; dynamic error measures update closure; substitution error measures counterfactual agreement; specificity measures unintended effects; task-wise errors measure flexible use; retention is indexed by delay and carrier controls; and enrichment measures geometry. No product of these measurements has a privileged mathematical interpretation.

For example, choose an evaluation law $\mu_{\mathrm{eval}}$, a declared distribution metric $d$, and the reference kernels from §§9–10. Explicit general definitions are

$$
\mathcal E_{\mathrm{pred}}
=\mathbb E_{R,q\sim\mu_{\mathrm{eval}}}
d\!\left(p_q(\cdot\mid R),\overline p_q(\cdot\mid\phi(R))\right),
$$

$$
\mathcal E_{\mathrm{dyn}}
=\mathbb E_{R,u\sim\mu_{\mathrm{eval}}}
d\!\left(
\phi_\#\mathsf K_R(\cdot\mid R,u),
\overline{\mathsf K}_S(\cdot\mid\phi(R),u)
\right),
$$

where $\mathsf K_R$ and $\overline{\mathsf K}_S$ are the next-state marginals of their joint kernels. Define $\mathcal E_{\mathrm{sem}}$ as expected prespecified semantic loss against the stated target in §8, which may be a belief rather than physical truth. Natural and patched configurations must both be included when claiming intervened update closure. These average errors do not provide the uniform guarantee of §10.1, and a restricted diagnostic metric must retain the limited interpretation of §10.2.

An acceptance rule is a declared set of tolerances and contrasts, for example

$$
\begin{aligned}
\mathscr A_{\epsilon,\mathcal Q}
=\{&(F,\phi,\mathcal I,\overline{\mathsf K}):\\
&\mathcal E_{\mathrm{pred}}\leq\epsilon_{\mathrm{pred}},\quad
\mathcal E_{\mathrm{dyn}}\leq\epsilon_{\mathrm{dyn}},\\
&\mathcal E_{\mathrm{sub}}\leq\epsilon_{\mathrm{sub}},\quad
\mathcal E_\perp\leq\epsilon_\perp,\\
&\max_{q\in\mathcal Q_{\mathrm{use}}}\epsilon_q\leq\epsilon_{\mathrm{use}},\quad
\mathcal D_{\mathrm{nat}}\geq d_{\min}\}.
\end{aligned}
$$

The numeric tolerances cannot be derived from SAE or J geometry. They encode the scientific resolution required. Finite-data claims should use uncertainty intervals against those tolerances; failure to reject a difference is not evidence of equivalence. Discovery, alignment fitting, and final evaluation must use separate episodes or task families. Token-level resampling is not a substitute for independent episodes.

Semantic validation, carrier-specific retention tests, and J enrichment are additional declared requirements for the corresponding stronger interpretation. A model can pass the causal-state tests and fail J enrichment; that is evidence about the instrument and the localization hypothesis, not a contradiction.

| Evidence | Entitled conclusion | Unresolved question |
| --- | --- | --- |
| Accurate SAE reconstruction | Sparse description of the sampled activation geometry | Semantic and causal validity |
| Large linear J score contribution | Contribution to the specified lens score | Sparse-cone membership or workspace function |
| High $\rho_k$ | Good fit by the selected J cone family | Causal state content |
| Held-out semantic prediction | The candidate carries decodable task information | Whether the model uses it |
| Matched finite intervention effects | The targeted directions affect behavior | Whether the effect implements state substitution |
| Counterfactual agreement across tasks and updates | Approximate causal state abstraction on the tested domain | Generality beyond that domain |
| Delayed effect under carrier controls | State influence carried by the surviving tested routes | Other routes and longer horizons |
| Independent J enrichment of validated candidates | Preferential geometric association | Universal J localization or global access |

### 16.1 The final object is an abstraction, not an assumed manifold

For a validated alignment, the observed state set is

$$
\mathcal S_{\mathrm{obs}}=\{\phi(R):R\in\mathcal R_{\mathrm{tested}}\}.
$$

Nothing above proves that it is a smooth manifold. It can be discrete, branching, stratified, or redundant. Smooth manifold language would require additional regularity and dimension assumptions.

If $\mathcal F_{\mathrm{SAE}}$ denotes SAE coefficient vectors, the expression $\mathcal F_{\mathrm{SAE}}\cap\mathcal J$ mixes different ambient spaces. An intersection using decoder directions in residual space would be well-typed, but would not establish a state interpretation. A geometrically meaningful coefficient subset is

$$
\{z\in\mathbb R_+^m:b+Dz\in\mathcal C_k(B)\},
$$

or an approximate-fit version of it. This set alone has no temporal or causal interpretation.

The proposed research object is therefore the tuple

$$
\boxed{
\left(
\phi,\ \mathcal I,\ \overline{\mathsf K},\ \mathcal Q,
\ \text{SAE/J dictionaries},\ \text{measured errors}
\right),
}
$$

with three logically separate mathematical relations:

$$
\boxed{
\begin{aligned}
\text{Score decomposition:}\quad
&Lh=Lb+LDz+L\varepsilon.\\[2pt]
\text{Sparse reconstruction:}\quad
&c_k(h)=\mathcal Q_{k,B^\top B}
\left(B^\top b+B^\top Dz+B^\top\varepsilon\right).\\[2pt]
\text{Causal state correspondence:}\quad
&\phi\circ\mathcal T_u\circ\mathcal I
\approx f_u\circ i\circ\phi,
\quad\text{with aligned outputs.}
\end{aligned}
}
$$

The first two relations describe the measurement instrument. The third is the empirical hypothesis that would make part of that measurement a state representation.

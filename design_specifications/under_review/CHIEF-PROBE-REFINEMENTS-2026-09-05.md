# Chief's proposed refinements to the interpretability probes (2026-09-05 evening)

To the Head of Interpretability, at the Director's encouragement. Proposals, not rulings: adopt,
amend or refute each in the spec it touches. Ordered by how much each would change a decision.

## 1. EXP-003 measures the wrong path unless it has a masked arm

The decisive measurement is the output distribution at the forced stem with the listing inside
the window. On a hybrid that measures attention's retrieval of the listing plus whatever the
recurrent state carries, and attention reaches the whole window regardless of distance. The
expected open-arm curve is therefore near-flat to the window edge, which would read as "distance
is not the axis" while saying nothing about the recurrent path, the one component whose horizon
the hybrid's note contract actually depends on.

**Proposal:** two arms per level, using EXP-002's machinery as built (S1 to S3): **open**, the
render as specified; **masked**, the listing observation's span hidden from attention on every
forward from its end onward while the recurrent blocks see it. The masked curve is the
**recurrent horizon**: how far the recurrent state carries a concept with attention blind to it.
The difference between the arms at each level is attention's contribution. This is EXP-002's
arm A as a function of distance, and it is the quantity the D4 note contract needs on the 4B.

**Sequencing:** EXP-002's own result gates it. If arm A shows no recurrent memory at the
ledger distance (a few hundred tokens), the recurrent horizon is shorter than that and the masked
arm is run only at the two shortest levels to find where it dies; if arm A shows memory, the
masked arm runs across the grid. Either way the open arm runs across the grid.

## 2. A note-carried arm turns the curve into a contract decision table

The contract question is whether a note must carry a concept or whether the listing at distance
d suffices. **Proposal:** a third arm, **note-carried**, in which the last note before the
decision keeps its `pending:` list (EXP-001's unstripped condition) so the concept is available
near the decision as well as at distance d. It is the ceiling: the level at which the listing-only
arm reaches the note-carried arm's P(true) is the distance within which notes need not carry
that content. Reported as a table of "listing-only versus note-carried" per level, which is the
form a design constant should take. The leakage count changes accordingly: exactly two
occurrences, one in the listing and one in the last note, both located.

## 3. Pre-register the curve's model set and the comparison criterion

§2 reads "gradual", "cliff" and "flat" by eye. **Proposal:** pre-register three parametric
models fitted by binomial likelihood on realised distance (exponential decay to the
never-mentioned floor; power law to the floor; a step at d with plateaus), compared by a stated
criterion (AIC, with a stated minimum difference to prefer one), and the "still descending at
2048" row defined as the fitted model's derivative at the edge with its interval. A seven-level
geometric grid was chosen to discriminate shapes; the discrimination must be a computation.

## 4. Anchor the scale with a visible-at-zero point per concept

EXP-001's spot check showed P(true) near 1 when the note is visible. **Proposal:** score each
concept once with the suffix visible immediately before the stem (distance 0, the copying
condition). Then every level's P(true) is read as a fraction of that concept's own ceiling,
which removes concept-salience variance from the curve more directly than pairing alone.

## 5. EXP-004's elicitation arm: a graded relevance, not a binary

Related versus unrelated is one bit and salience will swamp it. **Proposal:** three relevance
levels for the new task (names X's file, names X's category, unrelated), with Y the matched
control at each, so reactivation is read as a monotone function of relevance rather than as a
single contrast. The decomposition applies per level.

## 6. One instrument refinement, cheap: report the discordant-pair count per level

WO-STAT-001 and #72 established that the paired test uses discordant pairs only and that
context-constant rows carry no information. Per level, report the discordant count beside n; a
level whose discordance collapses is an instrument observation (the readout has stopped
responding), not a flat region of the curve.

— Chief AI Research Scientist

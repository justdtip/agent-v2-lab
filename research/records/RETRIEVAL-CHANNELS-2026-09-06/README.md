# RETRIEVAL-CHANNELS-2026-09-06: the recurrent and attention channels on retrieval-heavy text

**Order.** The Director, 2026-09-06 00:00: "I want the probes rerun on retrieval heavy text." The
target is the limit the Head placed on the ledger-run finding (F8, the recurrent channel as a
write-strength-weighted running summary; query cosine 0.83 measured on ledger tasks at about
1,500 tokens): "it is possible queries move more on text where retrieval genuinely matters, and I
have not shown they do not."

**Design, reviewed by the Head before the run (five changes adopted; R38(g) recorded).** Eight
seeded contexts of four kinds, 775 to 2,000 tokens: sixteen key-value needles; reference-back prose
with a two-hop form; code constants defined early and asserted late; in-context learning of an
arbitrary symbol-to-digit mapping (thirty-two examples of eight symbols). Facts early, questions at
the end with at least 300 tokens between, so every source is a far source (gap >= 257) and each
fact's token span is known. Six retrieval questions and three no-lookup controls per context
(seventy-two items), each its own forward, so read positions differ only in the question; controls
use a placebo span from the context's own facts. Both channels are hooked in the same forward
(gated_delta_update for the recurrent blocks; qwen3_next's scaled_dot_product_attention for the
attention blocks, post-split, post-norm, post-rotary, at the block's own scale) and go through ONE
span statistic: the share of far-source read-weight mass on the answer's span, the uniform share,
hit-at-10, and the same under NDRAW = 50 random unit queries at the real norm. Recurrent heads with
gate constant >= 200 tokens on the context are measured (|alpha| by the backward scan); every
attention head is measured behind a reconstruction gate that rebuilds the block's own output at
the read position from the recomputed probabilities (relative error <= 0.05 of the output scale,
hard fail; proven to fail at 1.21 on a deliberately wrong head pairing; passes at 0.026). Answers
are scored under teacher forcing over the whole answer, because the first token of a numeric answer
is a bare space. Query cosine is reported within a context between two retrieval questions and
between two control questions (the paired contrast: movement beyond wording is dynamic addressing),
and across contexts (the only number set beside the ledger's 0.83). Every record carries the
model's correctness on its item and the summary stratifies by it. Checkpoint: the same 4-bit
model as the ledger run.

**Prediction, written before the run (the Head, adopted).** The needles and the in-context pairs
sit inside the store's capacity of about a hundred-odd distinguishable addresses per head, so those
two conditions are where the recurrent mechanism should work if it ever does; the single-fact
conditions are easier still. If retrieval shows anywhere it should show there, and a null across
all four is the strong result. A positive on the in-context condition is expected, not a surprise.
A recurrent null on its own is uninterpretable; the attention channel's mass on the same span,
under the same statistic, is what makes the comparison the result.

**Files.** `make_corpus.py`, `corpus.json` (seed 20260906), `retrieval_probe.as-run.py.txt` (the
driver as it ran; unimportable by name), `build_page.py`. Results are appended below when the run
lands, with `out-full/` and the page.

## Result, 2026-09-06 00:58 (run 00:36 to 00:50, 840 s, 72 items, all far; the box clear and unlocked afterwards)

**The model retrieved on every item.** 48 of 48 retrieval questions answered correctly under
teacher forcing over the whole answer, on all four kinds; 23 of 24 controls. So retrieval mattered
and succeeded everywhere, and the correct-versus-incorrect stratification has no incorrect stratum:
the corpus was within the model's reach. The attention reconstruction gate held on all 576 blocks
(worst 0.049 of the output scale against the 0.05 gate).

**The recurrent heads read the answer's span no more than a random query would, on every kind,
including the two the prediction singled out.** Over 13,479 head-records (heads with gate constant
>= 200 on the context), the far-source mass on the answer span as a multiple of the random-query
null is median 1.04, 90th percentile 1.64, 99th 2.7; 0.6 percent of records exceed 3 and none
exceed 10; the answer span is in a head's top-10 far sources 12.7 percent of the time against a
null of 10.0; the median span share is 0.0082 against a uniform share of 0.0088. The controls, with
a placebo span and no lookup, give the same distribution (median 0.97). By kind: in-context learning
1.10 (99th 2.4), reference-back 1.03 (3.5), code 1.04 (2.6), needles 0.99 (2.0). The strongest
recurrent head-record reaches 5.0 with a span share of 0.9 percent. The prediction that retrieval,
if it lived in this channel, would show on the needles and the in-context pairs is answered: it did
not show there, and the null is the strong one.

**The attention heads did the reading.** Over 6,144 head-records the same statistic is median 5.5,
90th percentile 36, 99th 102; 64 percent of records exceed 3 and 36 percent exceed 10; the answer
span is in the top-10 far sources 56 percent of the time against a null of 7.7; the median span
share is 4.6 percent against uniform 0.9. On the controls the concentration vanishes (median 0.71,
hit 9.8 percent against 6.8), so attention reads the span when the question asks for it and not
otherwise. By kind: in-context learning 9.5 (hit 73 percent), reference-back 8.2 (64), code 3.9
(43), needles 3.0 (44). The strongest heads put 72 to 88 percent of their far-source mass on the
answer span, 200 to 340 times the null, in attention blocks 4 to 6 (layers 20 to 28) and block 0
(layer 4) on code.

**The query does not move with what is being looked up, in either channel's direction.** Recurrent
heads: median cosine between the read queries of two retrieval questions on the same context 0.983,
between two no-lookup controls 0.878, across contexts 0.74. Attention heads: 0.93, 0.75, 0.53. In
both channels the query moves less between two retrieval questions than between two controls,
because the retrieval questions share a template; the paired contrast the Head specified says query
movement tracks wording, not the lookup. Attention retrieves anyway, because its keys at the answer
span match a query that barely changes direction and the softmax over two thousand keys is sharp;
the recurrent read, one fixed projection of a decaying store, lands at the uniform share. The
across-context 0.74 is set beside the ledger's 0.83 as the Head asked: the recurrent query is
nearly as static across different contexts as it was across positions within one.

**Reading.** F8 stands on text where retrieval genuinely matters and succeeds: the recurrent channel
is a running-summary engine and the lookups are attention-borne, measured on the same items under
one statistic. The division of labour the Head proposed is what the model does. Caveats: eight
contexts, two per kind; heads with gate constants under 200 tokens were not measured, since they
cannot carry a far fact; every item was answered correctly, so whether concentration tracks
correctness is untested here and needs a harder corpus; the reconstruction gate's worst block sat
at 0.049 against 0.05.

**Files added.** `out-full/summary.json`, `out-full/retrieval_probe.json` (every head-record),
`run-full.log` (progress lines with the gate value per item), `retrieval-channels.html` (the page;
also a private artifact).

## Corrections on the Head's review, 01:15 (adopted; the reading above is amended by this section)

**The mechanism is the read operation, not the query.** The paragraph above on query cosine is
wrong as an explanation and is retracted as one: attention's read query turns barely more than the
recurrent one between two retrieval questions (0.93 against 0.983) and attention retrieves anyway
(hit-at-10 56 percent against 7.7). Query mobility was never the mechanism, in either channel.
What separates the channels is the read: attention's read is a softmax over positions, a
competition in which a small margin in the dot product becomes a large margin in mass, so a nearly
static query still selects; the recurrent read is a linear sum in which every stored item
contributes in proportion to its raw alignment with no sharpening. At about 1,500 positions the one
matching key contributes roughly eleven times a non-matching one and there are about 1,499
non-matching ones, so the crosstalk swamps the signal by two orders of magnitude and the span share
lands at uniform: not because the store was not addressed, but because a linear read of a
superposed store cannot separate one entry from the rest at that count. This is the classical
capacity result for linear associative memories, on the order of one item per dimension because
retrieval signal-to-noise falls with every stored pattern, reproduced in a production model; softmax
attention holds far more because the exponential suppresses the non-matches. Same store, different
reader. "The query does not move" is not the finding and should not be repeated as one.

**The three weak key-head candidates from the ledger run close.** Block 1 key-head 7, block 6
key-head 0 and block 15 key-head 3 were each measured on every context here (96 records each). Their
median ratios are 0.92, 1.11 and 1.25, their best single records 1.77, 1.83 and 2.07, ranking 952nd,
789th and 478th of 13,479. On the corpus where they should have declared themselves they did not.
The item, open since the ledger run, is closed.

**Block numbering.** "Blocks 4 to 6 and block 0" above are indices into the eight attention blocks.
In the repository's numbering those are blocks 19, 23, 27 and 3, writing layers 20, 24, 28 and 4.
The per-layer attention profile (median ratio, hit-at-10): layer 4: 1.8, 0.30; layer 8: 2.6, 0.41;
layer 12: 2.6, 0.36; layer 16: 3.5, 0.46; layer 20: 15.2, 0.75; layer 24: 19.5, 0.84; layer 28: 9.1,
0.74; layer 32: 6.3, 0.64. The reading is done in the band, layers 20 to 28, which is where R41e's
pairs sit.

**The within-context query contrast is a corpus limitation, not a result.** The retrieval
questions share a template and the controls do not, so the contrast measures wording variety and
not retrieval intent; it cannot support a claim about queries in either direction. The pairing the
Head asked for did not work as designed and needs matched templates on both arms to run.

**Counts and what remains untested.** Measured recurrent heads per context (gate constant at least
200 tokens): 276, 272, 273, 256, 287, 288, 299, 298 of 768, mean 281; the other 487 cannot carry a
far fact and were not measured. The correctness stratification could not run at 48 of 48 and is
untested, not passed; whether attention's concentration tracks retrieval rather than accompanying it
needs a harder corpus, which is the next step before anyone writes that it does. The Deputy's note
on the instrument: the attention reconstruction gate passed at 0.0486 against 0.05, 97 percent of its
limit, discriminating by a factor of 25 against the wrong-pairing mutant; a longer context can fail
it without announcing that it is about to, so the next run re-measures the gate at the lengths it
uses, before any statistic, and reports n of N.

## Three additions on the Head's second review, 01:30 (adopted)

**The candidates are below chance's best, not merely unremarkable.** Each of the three was measured
96 times, and the best of 96 random draws from the measured recurrent distribution has a median at
its 99.28th percentile, which is 2.92 (2.92 by resampling; the 5th percentile of that maximum is 2.16).
Their best single records, 1.77, 1.83 and 2.07, are all below it, and 97 percent of random heads
would exceed 2.07 somewhere in 96 draws by chance. The strongest thing each candidate managed across
the whole retrieval corpus is weaker than a random head's best would have been; the ranks of 952nd,
789th and 478th are best-of-96 selections and not evidence of anything. The item closes without
residue.

**The layer profile is a finding in its own right (F9).** Attention's concentration on the answer
span rises gently across layers 4 to 16 (medians 1.8, 2.6, 2.6, 3.5), jumps roughly fourfold to 15.2
at layer 20, peaks at 19.5 at layer 24, and falls to 9.1 and 6.3 at 28 and 32: an inverted U centred
on layers 20 to 24. That coincides with the fan-out region of the lens geometry at layers 18 to 25
from the hosted-lens run, and its two peak layers are the attention members of two of R41e's pairs
(19/20 and 23/24). Two instruments with no shared machinery, a persistence-and-dimensionality band
read through the lens and a span-concentration statistic on attention probabilities, land on the same
place. It bears on WP5 directly: the distance curve reads at exactly those pairs and should be most
sensitive there.

**The null, scoped precisely.** 256 to 299 of 768 recurrent heads were measured per context, and the
excluded ones were excluded because their gate constants are too short to carry anything across
three hundred tokens. The claim is therefore not that no recurrent head retrieved; it is that among
the heads that could physically have retrieved at that range, none did, which forecloses the
objection that the interesting heads were filtered out.

The sentence marking "the query does not move" as not to be repeated stays, at the Head's request,
since the wrong version was put in front of the Director and would otherwise return.

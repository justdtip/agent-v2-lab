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

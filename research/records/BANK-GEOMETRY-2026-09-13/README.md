# The concept bank is ten dimensions wide and most of each vector is the same direction

2026-09-13, WS-D. Gemma 4 31B, 240 concepts at layers 20/32/40/48, extracted as the residual for
`Tell me about {word}.` minus the mean over 50 baseline words. No forward pass: this is the file.

## What a concept vector is made of

| layer | mean ‖v‖ | ‖mean of the bank‖ | as a share of ‖v‖ | participation ratio | top-10 eigenvalues | variance along the mean |
|---|---|---|---|---|---|---|
| 20 | 9.35 | 6.59 | 0.705 | 10.4 | 77.3% | 51.1% |
| 32 | 26.14 | 18.52 | 0.708 | 7.5 | 81.4% | 50.8% |
| 40 | 78.21 | 59.76 | 0.764 | 10.8 | 73.9% | 60.3% |
| 48 | 58.39 | 44.51 | 0.762 | 11.2 | 71.5% | 59.6% |

Two facts, and both are large.

**The shared direction is most of the vector.** The bank's mean has 70 to 76 per cent of a typical
concept's norm, and more than half the uncentred variance lies along it. Subtracting fifty baseline
words removed the part that is common to *words*, not the part that is common to *this recipe*.

**The bank spans about ten directions, not 240.** The participation ratio — total variance squared
over the sum of squared eigenvalues, which needs no threshold — sits between 7.5 and 11.2 in a
5,376-dimensional space, and ten eigenvalues carry three quarters of the centred variance.

## What that does to the null

Our null generator centres the bank and draws from the covariance. It therefore has no component
along the mean, and a concept has a large one. Projecting on the mean and thresholding separates
them, on the real banks:

| layer | AUC, mean removed from the null | AUC, mean restored |
|---|---|---|
| 20 | 0.997 | 0.514 |
| 32 | 1.000 | 0.510 |
| 40 | 0.988 | 0.548 |
| 48 | 0.989 | 0.551 |

A readout of one scalar, which never looks at which concept is present, separates the arms almost
perfectly. The estimate from synthetic banks was 0.79; on the real thing it is essentially 1.0. So
the adapter's 100 per cent against 12 per cent needs no concept content to explain, and the
mean-matched arm is a sharp prediction rather than a formality.

## But the vectors do carry meaning

| layer | within-family cosine | between-family | separation |
|---|---|---|---|
| 20 | +0.486 | −0.068 | +2.67 sd |
| 32 | +0.560 | −0.072 | +2.75 sd |
| 40 | +0.493 | −0.069 | +2.65 sd |
| 48 | +0.485 | −0.069 | +2.56 sd |

With the mean removed, concepts from the same hand-assigned family point the same way and concepts
from different families are slightly anti-aligned. The eight families were written down to stratify
negatives, not from the geometry, and the geometry reproduces them at two and a half standard
deviations. The similarity structure also survives depth: the 240×240 cosine matrices correlate at
r = 0.88 between layers 20 and 32 and r = 0.94 between 40 and 48.

## The consequence nobody had costed

Two hundred and forty concepts, about ten effective dimensions, and same-family members at cosine
+0.5. That is a heavily degenerate code. Asking a readout to name **which** of 240 concepts is
present, from one injected token, is close to asking it to invert a rank-10 map.

Asking which **family** is present is an eight-way question that the geometry demonstrably
supports. And the degenerate evaluation run already showed exactly that signature before anyone
looked here: of the nine trained-concept YES rows that named anything, four named the concept
exactly and four named a different word **from the same family** — custard and porridge both drew
yoghurt, fjords drew dunes, pelicans drew salamanders.

So "it cannot name the concept" and "the representation is family-resolved, not instance-resolved"
predict the same data, and only the second one predicts the family-adjacency. The next naming
measurement should be scored at family level as well as instance level, which costs nothing: the
per-candidate log-probabilities are already computed.

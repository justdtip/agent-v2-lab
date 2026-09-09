# WS-B on the rented card: the CUDA seam is faithful, and the gate that failed was measuring the wrong thing

**RTX PRO 6000 Blackwell, 97,887 MiB, torch 2.14.0+cu130, transformers 5.16.1, Python 3.13.15,
2026-09-10.** Determinism pinned before the first CUDA use every run: TF32 off both paths,
`float32_matmul_precision: highest`, cuDNN deterministic, `attn_implementation: eager`, seed 0,
`CUBLAS_WORKSPACE_CONFIG` set before initialisation. Every run held an announced window as
`swe-1`; the card was idle and no other seat's process was touched.

The headline is two facts that have to be held together, because either one alone is misleading.

**The port is device-consistent.** CUDA and CPU, same code and same checkpoint, agree with each
other to within one position in 5,245, and thirteen of fifteen episodes produce the *identical*
set of confident disagreements on both. Two independent CUDA runs of the same episode are
byte-identical. Nothing here is a CUDA defect.

**The acceptance gate failed, and the gate was wrong.** Twenty-four positions where the MLX
recording's own probability was at least 0.99 produced a different argmax, which my rule calls a
defect. It fires on CPU exactly as on CUDA, so it was already true on the laptop and invisible
because **only one of the fifteen episodes had ever been measured**, and that one is clean.
The precision-matched arm (§2a) then resolved it: **twenty-one of the twenty-four are
quantisation** — MLX at bfloat16 agrees with torch and only the 4-bit record dissents — and the
remaining **three are ties at the bfloat16 grid**, gaps of nought to six units of last place.
No port defect is implicated at any of them. The rule's premise, that quantisation cannot move a
confident argmax, is false at 4-bit, and the rule needs a precision-matched reference and a
margin in ULPs before it can call anything a defect.

Nothing here says the port is correct; it says these twenty-four positions were never evidence
against it. What stands as positive evidence is the device-consistency above and WS-A's gates.

---

## 1. What ran, and what it produced

| run | compared | agreed | rate | flips at P ≥ 0.99 | gate |
|---|---:|---:|---:|---:|:--|
| `calculate-0158`, CUDA, run 1 | 103 | 97 | 0.9417 | 0 | PASS |
| `calculate-0158`, CUDA, run 2 | 103 | 97 | 0.9417 | 0 | PASS |
| `calculate-0158`, CPU (this card) | 103 | 98 | 0.9515 | 0 | PASS |
| `calculate-0158`, CPU (laptop, 2026-09-09) | 103 | 98 | 0.9515 | 0 | PASS |
| **all fifteen, CUDA** | **5,245** | **4,999** | **0.9531** | **24** | **FAIL** |
| **all fifteen, CPU (this card)** | **5,245** | **4,998** | **0.9529** | **24** | **FAIL** |

The corpus reads identically to the laptop's: 5,245 emitted tokens over fifteen episodes, 4,801
agentic, forward-to-emission join 5,245 agree and 0 disagree, and 4,199 of 5,245 emissions
recorded at P ≥ 0.99 (0.8006). Every laptop basis therefore still describes this corpus.

**Determinism, the within-backend half of G-2.** Two CUDA runs in separate processes: same
agreement, same flip positions, same peak. Pinning holds and no kernel drifts.

**Peak memory, measured here.** 7.92 GiB for one episode and 8.81 GiB for the full corpus, on
`cuda:0`. The laptop projected 7.56 GiB and measured 7.88 GiB on CPU. The projection was low by
4.8% against the single-episode figure and by 16.5% against the corpus, in the safe direction.

**Speed.** `calculate-0158` takes 4.0 s on the card against 83.5 s on the laptop's CPU, taken
while the card was idle. The recapture's per-episode timings in `cuda-all15-v2.json` were taken
while the D-CRO's fit shared the card and run 2.29x slower than the same episodes in
`cuda-all15.json`; **treat every timing in the v2 file as `shared-card`**. The agreement figures
are unaffected and identical, and the peak is per-process allocated, so it stands.

---

## 2. The failure, characterised rather than asserted

Twenty-four confident flips out of 4,199 confident positions is **0.57%**. The comparison rate at
unconfident positions is **21.2%** (222 of 1,046). The rule is doing what it was built to do:
disagreement concentrates thirty-seven-fold at the near-ties. It is the residue that fails.

The residue is not sitting on the threshold:

| recorded probability of the flipped token | count |
|---|---:|
| 0.99 to 0.999 | 13 |
| 0.999 to 0.9999 | 6 |
| ≥ 0.9999 | 5 |

The most extreme is `batch_update-0166`, turn 1 position 735: the recording gives token 2818 a
probability of **1.000000** and the port produces token 107. A threshold artefact would crowd the
0.990–0.995 band and this does not.

**Where they fall.** Both inside and beyond the 1,024 sliding window, and in short episodes as
well as long. `update-0028` is the only episode that exercises the window at all (a 2,607-position
turn) and it is among the *best*: 0.9781 agreement with 3 confident flips on both devices. **The
signature does not change past the window**, which answers step 4 of the device checklist and
argues against the mask-dispatch hypothesis the checklist named as the thing to fear.

**The two hypotheses this run does not separate.**

1. *A port defect* — a mask, position, entry or norm error that survives on both devices because
   it is in shared code. The 95.3% agreement and the clean window behaviour argue against
   anything structural, but do not exclude something narrow.
2. *The rule's premise is too strong.* The comparison is MLX **4-bit** against torch **bfloat16**.
   The premise was that quantisation cannot move a confident argmax. Four-bit quantisation is not
   a small perturbation, and the recorded probability is the 4-bit model's confidence in its own
   preference, not a shared ground truth. A 4-bit model can be certain about a token the bf16
   model does not choose, because they are not the same function.

Nothing I ran distinguishes these, and I am not going to pick between them from the armchair.

**The test that would, and did.** Compare torch bfloat16 against **MLX bfloat16**: same
precision, different framework, quantisation removed. Authorised by the Chief and run on the
laptop, because only the laptop has MLX. It is §2a below, and it settles the question.

`confident-flips.json` was its input — all twenty-four positions with turn, position, recorded
and produced token, and the recorded probability.

---

## 2a. The precision-matched arm: the premise was wrong, and the port is not implicated

`scripts/mlx_reference.py`, teacher-forced over the turns carrying the twenty-four positions,
MLX bfloat16 against the torch bfloat16 the card produced. At each position: if MLX agrees with
torch, the 4-bit record is the outlier and the premise was wrong; if MLX still produces the
recorded token, torch differs from MLX with quantisation excluded, and that is the port.

| verdict | count |
|---|---:|
| **quantisation** — the two bfloat16 implementations agree, the 4-bit record is the outlier | **21** |
| **port** — MLX at matched precision still produces the recorded token | **3** |

So twenty-one of the twenty-four were never evidence about the port at all. **The hard rule's
premise was wrong**: 4-bit quantisation moves confident argmaxes, and the recorded probability
is the 4-bit model's confidence in its own preference, which does not bound what a bfloat16
model will do. `batch_update-0166` position 735, the P = 1.000000 case that looked most like a
defect, is quantisation: both bfloat16 frameworks produce 107 and only the 4-bit record says
2818.

**The three that remained are ties, not defects** (`scripts/flip_margin.py`). The instrument
that says so is the logit gap, not the probability margin — and measuring the probability
margin first was a mistake worth recording, because it called all three "not a tie" at
margins of 0.12 to 0.64. The logits are **bfloat16**, so their gaps are quantised to the
bfloat16 grid; softmax is monotone, so `ln(p1/p2)` recovers the gap exactly. Every gap came
back an exact multiple of 0.25, which is the bfloat16 step at the magnitude these logits sit
at — the check that it is the right grid:

| position | recorded P (4-bit) | MLX gap | torch gap |
|---|---:|---:|---:|
| `read-0108` turn 0 position 521 | 0.999739 | 6 ULP | **0 ULP** |
| `update-0028` turn 0 position 556 | 0.999873 | 2 ULP | 2 ULP, opposite sign |
| `list-0149` turn 0 position 522 | 0.991508 | 1 ULP | 3 ULP, **same token** |

At `read-0108` the two candidates are **exactly equal in bfloat16** and the argmax is settling a
coin toss. At `update-0028` the same 2-ULP gap points opposite ways in the two frameworks. At
`list-0149`, torch on the laptop CPU picks the *recorded* token, agreeing with MLX: that
position was one of the two episodes where the card's CPU and CUDA already differed, so the
verdict of "port" was a device tie, not a framework disagreement.

**Reading: no port defect is implicated at any of the twenty-four positions.** Twenty-one are
quantisation and three are ties at the resolution of the numbers being compared.

**One honest limit.** The margin arm ran torch on the laptop's CPU, not the card's CUDA, so for
`list-0149` it did not reproduce the exact run that produced the verdict. That is why the row
above says what it says rather than claiming the verdict was wrong.

**What the rule should become.** "A confident flip is a defect" needs a precision-matched
reference and a margin measured in ULPs of the stored dtype. Against a differently-quantised
recording it is not a defect test, and this corpus is the demonstration: it fired twenty-four
times and found nothing.

---

## 2b. Re-based on MLX bfloat16: one flip in 5,245, at three ULPs

The Chief ruled the rule restated and the records re-based. Both are done, and the re-based
number is the one that means something.

`scripts/mlx_reference.py --reference` built the precision-matched reference over all fifteen
episodes on the laptop: 5,245 positions in 399 s, written per episode as each landed. The
reference is decisive nearly everywhere — median top-two gap **63 ULP** — with **126 positions
(2.40%) inside the two-ULP tie band** and 21 of those exact ties, where the reference itself
cannot tell its top two apart.

Then the kit ran once more on the card against it, beside the D-CRO's fit under a separate
box-state directory:

| reference | agreed | rate | outside ties | ties |
|---|---:|---:|---:|---:|
| MLX 4-bit, the stage-two recording | 4,999 / 5,245 | 0.9531 | 24 | not measurable |
| **MLX bfloat16, precision-matched** | **5,213 / 5,245** | **0.9939** | **1** | **30** |

Agreement rises by four points for no change to the port. That gap was the two precisions, and
the old gate was reading it as the port's error.

**The one that remains: `read-0108` turn 0 position 521, reference gap 3.0 ULP.** The reference
prefers token 2234; the port produces 1399. It sits one ULP outside the band, and I am reporting
it rather than widening the band to clear it — a threshold moved to make a run green is not a
threshold. It is also the position where torch's *own* two devices disagree: the laptop's CPU
puts the two candidates at an exact tie there while the card's CUDA prefers 1399, so the port is
at the edge of its own resolution at a position where the reference is three steps clear.
Whether three ULPs is a defect or a band drawn one step too tight is the Chief's to rule; the
evidence for either is in `rebased-cuda.json`.

**So the gate is FAIL, on one position, and the failure is legible.** That is a different object
from the twenty-four it reported yesterday, which were legible only after two more runs.

---

## 3. Gates 5 and 6

**Gate 5 FAILS**, and now for a reportable reason rather than a stale guard. On the smoke episode
it diverges at the very first generated token: position 506, recorded token 818 with probability
**0.5155**, generated 107. That is a three-way near-tie, and free-running comparison across
backends separates completely at the first one. This is the outcome `tolerance.py` predicts in
prose — *"free-running token-for-token agreement across backends is not a gate and is not
attempted"* — so gate 5 as specified asks for the thing the programme already ruled invalid. It
should be restated as within-backend reproduction, which is the question exact agreement can
answer, or struck. **That is a decision for the Chief, not a change I made.**

One inconsistency worth someone's attention: at that same position the teacher-forced pass
produces token 236777 while the free-running loop produces 107, from the same prefix. Two torch
paths, one answer each. The teacher-forced pass is a single forward over the whole sequence; the
decode loop is a chunked prefill plus cache. At a 0.52 near-tie a different accumulation order is
enough to explain it, but it has not been checked and it is exactly the kind of thing that is
cheap now and expensive later.

**Gate 6 is UNAVAILABLE**, correctly: the backend loads, and the lens read path is not ported.
That is WS-A's producing side, unchanged by anything here.

---

## 4. Five defects the card found, all fixed here

**The corpus was not on the card.** No stage-two records existed anywhere on the box; the
archives under `/workspace` are the training corpus, not the captures. They cannot be regenerated
there — the captures are MLX and the card has no MLX. Uploaded, 424 MB, checksums verified on
both sides. Note that `vast-capabilities` reports `workspace_is_volume: false`, so **nothing on
that box survives a recycle** and everything in this directory was pulled back for that reason.

**macOS `tar` shipped AppleDouble sidecars.** Fifteen `._name.jsonl` files, 163 bytes each,
invisible on the Mac and ordinary files on Linux, where they match the reader's `*.jsonl` glob.
The record reader died on `UnicodeDecodeError: byte 0xa3`. Use `COPYFILE_DISABLE=1` when taring
records from a Mac. The reader could also skip `._*`, which would have saved this trip; not done
here because it is a change to a shared reader mid-run.

**The tolerance runner loaded on CPU while every variable said `cuda:0`** (fixed, `4e2cd38`).
`_load` passed `device="cpu"` as a literal, written on a laptop that had no CUDA. The first
measurement of the day was a CPU number that would have been recorded as the port's CUDA result;
the only thing that said otherwise was `device: cpu` inside the load report, and the record's own
`port_precision` field said `cpu bfloat16` because that was a literal too. **A number's label
came from the source and not from the run, which is the failure the provenance fields exist to
prevent, appearing in the field that carries provenance.** Now the target comes from
`device.select()`, the load report is checked against it and the run refuses on a mismatch, and
the record names what actually ran. Peak memory is measured on CUDA in the same commit, because
nothing in the script measured one and the laptop's 7.88 GiB came from outside it.

**The kit could not load a backend at all** (fixed, `16fd7e3`). `_load_backend` raised *"the torch
architecture view does not exist yet (WS-A)"* — a placeholder that outlived its condition by
days, while the tolerance runner had been loading through that very view all evening. Separately,
`--results` passed the registry name into `current_identity`, which hashes a checkpoint path, so
the resume store died on `FileNotFoundError: .../gemma3-4b-cuda-bf16`. Both fixed; `--checkpoint`
now names the snapshot directory, because on the device the weights are under `$HF_HOME` and the
weights-cache resolver looks under the checkout.

**The flip printer hid the flips that gate** (fixed, `8773a1d`). The cap was twelve flips per
episode in position order. An episode with many near-ties pushed its confident flips past it, so
the run reported 24 gating flips and the log carried 12 — and the JSON row kept probabilities but
not positions, so the other twelve existed nowhere. This was found only by trying to build the
input for the follow-up test out of the log and coming up half short. **A cap that can hide the
evidence for the failure it is reporting is the wrong cap.** Every hard flip now prints, only the
soft ones are capped, the withheld count is stated so a short list is not read as a complete one,
and each gating flip's position and tokens go into the record. The corpus was then re-run on the
card and **reproduced the first run exactly** — 4,999/5,245 and the same 24 — which is a third
determinism reading as well as the recapture.

That last point generalises: **`_snapshot` resolves weights from the primary checkout, which is a
laptop convention.** On the device, pass `--checkpoint` explicitly to both scripts. Worth making
the resolver device-aware rather than leaving a flag people must remember.

---

## 5. What is not done

- **Gate 6**, whose producing side is unported (WS-A).
- **Gate 7** (WS-D).
- **The cache strategies' fidelity gate.** `trim` and `snapshot` still refuse; they were never
  going to run before the `none` path passed, and it has not.
- **The precision-matched reference run** of §2, which is the next thing anyone should do with
  this result, and which does not need the card.

## 6. Files

**`rebased-cuda.json` / `.jsonl` / `.log` — the corpus re-based on the precision-matched
reference, which is the number that counts. `mlx-bf16-reference.json` / `.jsonl` — the reference
itself, every deciding position with its top-two gap in ULPs.** `mlx-bf16-arm.json` / `.jsonl` —
the precision-matched arm, one row per position with its verdict. `flip-margins.json` — the logit gaps in ULPs for the three that arm left open.**
`confident-flips.json` — the twenty-four failing positions, the input to both. `cuda-all15-v2.json` / `.jsonl` and `cuda-v2.log` — the recapture that produced it,
identical to the first run in every figure. `cuda-all15.json` / `.jsonl`, `cpu-all15.json` /
`.jsonl` — the two full-corpus runs, per-episode rows written and flushed as each completed. `cuda-0158-run1/run2` — the determinism pair.
`accidental-cpu-0158-run1.*` — the first measurement of the day, kept because it is the evidence
for the device defect above and because a record that quietly drops its own wrong turn is worth
less. `cuda-chain.log`, `cpu-chain.log`, `gate5.log` — every flip printed with its recorded
probability, which is what makes the base rate above readable rather than assertable.

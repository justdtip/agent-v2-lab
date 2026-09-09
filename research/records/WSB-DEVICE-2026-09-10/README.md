# WS-B on the rented card: the CUDA seam is faithful, and the acceptance gate fails anyway

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

**The acceptance gate fails.** Twenty-four positions where the MLX recording's own probability
was at least 0.99 produced a different argmax. My rule says that is a defect and not
quantisation. The rule fires on CPU exactly as it fires on CUDA, so whatever it has found was
already true on the laptop and was invisible because **only one of the fifteen episodes had ever
been measured**, and that one episode is clean.

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

**Speed.** `calculate-0158` takes 4.0 s on the card against 83.5 s on the laptop's CPU.

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

**The test that would.** Compare torch bfloat16 against **MLX bfloat16** on the same prompts:
same precision, different framework, quantisation removed from the comparison. Confident flips
that survive are the port; confident flips that vanish were the 4-bit gap and the hard rule needs
restating in terms of a precision-matched reference. That run needs MLX, so it is a laptop run,
and it is small: the failing positions are 24 and their contexts are known.

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

## 4. Four defects the card found, all fixed here

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

`cuda-all15.json` / `.jsonl`, `cpu-all15.json` / `.jsonl` — the two full-corpus runs, per-episode
rows written and flushed as each completed. `cuda-0158-run1/run2` — the determinism pair.
`accidental-cpu-0158-run1.*` — the first measurement of the day, kept because it is the evidence
for the device defect above and because a record that quietly drops its own wrong turn is worth
less. `cuda-chain.log`, `cpu-chain.log`, `gate5.log` — every flip printed with its recorded
probability, which is what makes the base rate above readable rather than assertable.

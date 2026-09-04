# Execution lift request: P6 causal patching on the 3B (run C adapter)

To the Director, from the Deputy. Format per the Chief's standard (issue #18): artifact paths,
checksums, the recompute command, and one evidence line per condition, so the lift is a
decision on facts. Recompute any hash with `shasum -a 256 <path>` from the repository root.

**Status: FINAL — every artifact cited is committed and on origin.**

## The command (final; the tool refuses without the version binding)

    uv run agent-v2-probe-patch \
      --passing-eval outputs/agent-v2b/evals/runB-test180.json \
      --failing-eval outputs/agent-v2c/evals/best-adapter-test.json \
      --policy C --model qwen25-coder-3b \
      --data-seed 20260902 --generator-version 1 \
      --output outputs/probes/patch-C-2026-09-04

Writes `patch.json` and `patch.md` there and nothing else. Roughly 30 minutes on the 3B.

## Inputs, with checksums

| Artifact | SHA-256 |
| --- | --- |
| `outputs/agent-v2b/evals/runB-test180.json` (the passing run) | `ce0d8f4eade894389688eba250090bba24c8ef68b01c8ca7f0a87694c98ef630` |
| `outputs/agent-v2c/evals/best-adapter-test.json` (the failing run) | `2cabb9a40a644d2dd5a6688b48dd769bd99af684fad4b07ed01507e426668ddc` |
| `outputs/agent-v2c/best-adapter/adapters.safetensors` (policy C) | `f4f22886cfa93d7362d7db29e8d482e071692988f06f7d30efd7ca4e4f80ab74` |
| `outputs/preflight/qwen25-coder-3b.json` (the 3B control, `passed: true`) | `c536353043eb5d0eb53dcb8ecd1eb275616b398ad089bdd7c3b08cdccae36e04` |

Both evaluations are also in the verified 2026-09-04 backup (1,249 files, all hashes matched).

## Conditions, one evidence line each

1. **Preflight on the base this runs on — met.** `outputs/preflight/qwen25-coder-3b.json`,
   schema 2, R18a gate, `passed: true`, native max_abs 0.0, Frobenius 0.0, JVP forward,
   memory 3.09 GiB of 22. Run by the Director at 15:18.
2. **The view sees the adapter — met.** Wave-1 regression fixed at `89dfb56` (issue #27, the
   Chief's direct read): wrapped projections resolve to the same paths as unwrapped, proven on
   real `LoRALinear`-wrapped fakes including a quantized base.
3. **Selection is the spec's universe under the ratified binding — met.** Pass-under-B and
   fail-under-C, recomputed under generator v1 per R23 (the Director's #25 ratification) via
   `replay_task_from_id`, never HEAD; fail-closed without the binding. Five cases, all
   `test-ledger_reconcile-*-clean`: 0031, 0127, 0139, 0163, 0175; decision steps 7/7/7/7/6;
   every counterfactual note from run B's own saved transcript, zero generator fallbacks.
   Pinned by a real-files test and reproduced by the Director's first live attempt.
4. **Scoring stability recorded per case (R24) — met, all five stable.** Decision step and
   dropped value identical under the bound version and HEAD: 0031 step 7 / 85, 0127 step 7 / 89,
   0139 step 7 / 100, 0163 step 7 / 32, 0175 step 6 / 54. Verified three ways (Deputy's own
   checker readout, the tool's own records, the implementer's table); the headline covers all
   five, and there is nothing to list as unstable.
5. **Execution claim — free.** No live agent holds the `model-execution` action; the
   coordination lanes are orphaned since 12:45 and nothing has executed since the 15:20
   preflight.
6. **Cost stated:** about 30 minutes of GPU on the 3B; no training, no eval; writes two files.
7. **Code at the commit — met.** #26 committed as `07c6657`, suite 647 passed, exit 0, pushed;
   every reviewed slice this run depends on (#19, #21, #23, #24, #27, #26) is on origin.

## How to read the result, agreed in advance

- **Five cases is the whole universe**, because `aggregate_report` fails under both B and C,
  so only `ledger_reconcile` supplies pass/fail pairs with an empirically passing note. The heat
  map is indicative, not decisive; intervals over five tasks are wide. Report it as such.
- The headline flip rate covers stable cases only; any unstable case is listed separately.
- Controls (unrelated-task patch, random-position patch) share the patched path's numerics,
  so flip comparisons are internally consistent regardless of the fp32-versus-native gap.
- Bound follow-up, not part of this run: a secondary condition on the `aggregate_report`
  failures using the generator-v4 note as a designed-correct counterfactual, labelled so.

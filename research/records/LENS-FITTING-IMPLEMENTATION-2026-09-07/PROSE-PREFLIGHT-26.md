# Prose producer real-input correction — 7 September 2026

Source055f63b corrects the real-input failure recorded in INTEGRATION-STATUS-25.md. The frozen corpus stores resolved Hugging Face blob paths with opaque names. The producer now resolves one pinned offline snapshot, verifies all five tokenizer/config roles against the frozen descriptor by resolved path and SHA-256, loads locally, then checks snapshot identity again. No corpus mutation or download.

Fresh28 pure tests passed in8.68s. The actual plan CLI ran under a guard rejecting every MLX/MLX-LM import: all51 held windows matched their frozen first824 IDs exactly. Plan SHA256985b3f1660a57390e4d701edc95ecca65dbc77d2bb28d6144b99d5f49750a9c1. Source and real snapshot inventory received a scoped independent GPT-6 Astra rereview with no remaining findings. Detailed commands and evidence paths are preserved in the tracked Task5a implementer report.

This is tokenizer/planning acceptance only. No continuation capture, native replay or profile was produced.

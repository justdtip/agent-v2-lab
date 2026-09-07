# Calibration attempt02 — refused by pre-launch inventory

The retry used fresh output names and an escalated mapped-MLX check in the launching Python process before invoking the preflight script. That check found foreign pytest PID85070 mapping the primary virtualenv MLX library and exited3. resource-preflight-02.log preserves the refusal. The preflight script never ran, so resource-preflight-02.jsonl was not created. No checkpoint, forward, resource measurement, scientific output or lens artifact. No process killed or lock cleared.

The next attempt, after the box handoff recorded in BOX-SCHEDULE-19.md under the implementation record, must use fresh attempt03 paths and both primary lock and mapped-MLX checks.

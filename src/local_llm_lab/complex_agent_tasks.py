from __future__ import annotations

import random
from dataclasses import replace

from local_llm_lab.agent_protocol import Action
from local_llm_lab.agent_tasks import AgentTask, _calculate


def _noise(rng: random.Random, lines: int, label: str) -> str:
    return "".join(
        f"context-{label}-{line}=historical observation {rng.randrange(100000, 999999)} "
        f"for unrelated component {rng.choice(('amber', 'blue', 'green', 'silver'))}\n"
        for line in range(lines)
    )


def _difficulty(split: str) -> int:
    if split == "train":
        return 0
    if split == "valid":
        return 1
    if split == "test":
        return 2
    raise ValueError(f"unsupported split: {split}")


def make_complex_tasks(split: str, count: int, seed: int = 20260902) -> list[AgentTask]:
    """Create longer, split-isolated tasks with test-time length extrapolation."""
    if count < 1:
        raise ValueError("count must be positive")
    difficulty = _difficulty(split)
    categories = (
        "pointer_chain",
        "ledger_reconcile",
        "cross_reference",
        "conditional_update",
        "batch_update",
        "aggregate_report",
    )
    tasks = []
    for index in range(count):
        rng = random.Random(f"complex:{seed}:{split}:{index}")
        category = categories[index % len(categories)]
        maker = _MAKERS[category]
        task = _with_visible_plan(maker(split, index, difficulty, rng))
        tasks.append(replace(task, task_id=f"complex-{task.task_id}"))
    return tasks


_PLAN_STEPS = {
    "pointer_chain": [
        "read the starting node",
        "follow every exact Next path",
        "stop only at a Result field",
        "report the observed Result",
    ],
    "ledger_reconcile": [
        "list and read every invoice",
        "separate approved from held amounts",
        "calculate the approved total",
        "update and re-read the summary",
    ],
    "cross_reference": [
        "search the initial reference",
        "read each matching record",
        "repeat with every Next-Key",
        "report the observed Resolution",
    ],
    "conditional_update": [
        "read the policy and list services",
        "read every service load",
        "select the highest load above threshold",
        "update and re-read the selected service",
    ],
    "batch_update": [
        "read the deployment manifest",
        "inspect every managed worker",
        "apply each exact mode replacement",
        "re-read every changed worker",
    ],
    "aggregate_report": [
        "read every metric value",
        "calculate both subtotals",
        "calculate the grand total",
        "update and re-read the report",
    ],
}


def _with_visible_plan(task: AgentTask) -> AgentTask:
    steps = _PLAN_STEPS[task.category]
    plan = Action(
        "set_plan",
        {"goal": f"complete {task.category} accurately", "steps": steps},
    )
    progress = Action(
        "update_plan",
        {
            "completed": steps,
            "next": "finish with the verified answer",
            "evidence": task.expected_answer,
        },
    )
    actions = (plan, *task.expert_actions[:-1], progress, task.expert_actions[-1])
    return replace(
        task,
        expert_actions=actions,
        required_tools=task.required_tools | {"set_plan", "update_plan"},
    )


def _pointer_chain(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/chain"
    hops = rng.randrange(4 + difficulty, 7 + difficulty)
    paths = [f"{root}/node-{step}-{rng.randrange(100, 999)}.txt" for step in range(hops)]
    result = f"artifact-{rng.randrange(10000, 99999)}"
    files = {}
    for step, path in enumerate(paths):
        context = _noise(rng, 8 + 3 * difficulty, f"node-{step}")
        if step + 1 < len(paths):
            files[path] = f"Node: {step}\n{context}Next: {paths[step + 1]}\n"
        else:
            files[path] = f"Node: final\n{context}Result: {result}\n"
    files[f"{root}/decoy.txt"] = "Result: ignore-this-decoy"
    actions = tuple(Action("read_file", {"path": path}) for path in paths) + (
        Action("finish", {"answer": result}),
    )
    prompt = (
        f"Start by reading {paths[0]}. Follow each exact Next path until a Result field is "
        "reached, then report that Result exactly. Do not use the decoy."
    )
    return AgentTask(
        f"{split}-pointer_chain-{index:04d}",
        "pointer_chain",
        prompt,
        files,
        actions,
        result,
        frozenset({"read_file"}),
    )


def _ledger_reconcile(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/ledger"
    entries = 4 + difficulty
    files = {}
    approved = []
    entry_paths = []
    for item in range(entries):
        amount = rng.randrange(20, 180)
        status = "approved" if item % 3 != 1 else "held"
        path = f"{root}/invoice-{item}-{rng.randrange(100, 999)}.txt"
        files[path] = (
            f"invoice={item}\n"
            + _noise(rng, 12 + 4 * difficulty, f"invoice-{item}")
            + f"amount={amount}\nstatus={status}\n"
        )
        entry_paths.append(path)
        if status == "approved":
            approved.append(amount)
    summary_path = f"{root}/summary.txt"
    old_summary = "approved_total=PENDING"
    total = str(sum(approved))
    new_summary = f"approved_total={total}"
    files[summary_path] = old_summary
    expression = " + ".join(map(str, approved))
    actions = (
        Action("list_files", {"directory": root}),
        *(Action("read_file", {"path": path}) for path in sorted(entry_paths)),
        Action("calculate", {"expression": expression}),
        Action("read_file", {"path": summary_path}),
        Action(
            "replace_text",
            {"path": summary_path, "old": old_summary, "new": new_summary},
        ),
        Action("read_file", {"path": summary_path}),
        Action("finish", {"answer": new_summary}),
    )
    prompt = (
        f"Reconcile {root}. List the files, read every invoice, sum only amounts whose status is "
        f"approved using the calculator, replace PENDING in {summary_path} with the numeric total, "
        "read the summary again to verify it, and report the complete approved_total setting."
    )
    return AgentTask(
        f"{split}-ledger_reconcile-{index:04d}",
        "ledger_reconcile",
        prompt,
        files,
        actions,
        new_summary,
        frozenset({"list_files", "read_file", "calculate", "replace_text"}),
        {summary_path: new_summary},
    )


def _cross_reference(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/records"
    links = 3 + difficulty
    keys = [
        f"REF-{split.upper()}-{index:04d}-{rng.randrange(1000, 9999)}-{n}" for n in range(links)
    ]
    answer = f"resolution-{rng.randrange(10000, 99999)}"
    files = {}
    target_paths = []
    for hop, key in enumerate(keys):
        path = f"{root}/record-{rng.randrange(100, 999)}-{hop}.txt"
        target_paths.append(path)
        if hop + 1 < links:
            files[path] = (
                f"Lookup-Key: {key}\n"
                + _noise(rng, 12 + 4 * difficulty, f"record-{hop}")
                + f"Next-Key: {keys[hop + 1]}\n"
            )
        else:
            files[path] = (
                f"Lookup-Key: {key}\n"
                + _noise(rng, 12 + 4 * difficulty, f"record-{hop}")
                + f"Resolution: {answer}\n"
            )
    for decoy in range(3 + difficulty):
        files[f"{root}/decoy-{decoy}.txt"] = f"Lookup-Key: DECOY-{rng.randrange(99999)}"
    action_list = []
    for key, path in zip(keys, target_paths, strict=True):
        action_list.append(Action("search_files", {"query": key}))
        action_list.append(Action("read_file", {"path": path}))
    actions = tuple(action_list) + (Action("finish", {"answer": answer}),)
    prompt = (
        f"Search for {keys[0]}. Read its matching record, then repeatedly search each Next-Key "
        "and read its record until you reach Resolution. Report the Resolution exactly."
    )
    return AgentTask(
        f"{split}-cross_reference-{index:04d}",
        "cross_reference",
        prompt,
        files,
        actions,
        answer,
        frozenset({"search_files", "read_file"}),
    )


def _conditional_update(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/services"
    service_count = 3 + difficulty
    threshold = rng.randrange(55, 75)
    loads = rng.sample(range(30, 99), service_count)
    target_index = max(range(service_count), key=loads.__getitem__)
    if loads[target_index] <= threshold:
        loads[target_index] = threshold + rng.randrange(5, 20)
    policy_path = f"{root}/policy.txt"
    files = {policy_path: f"threshold={threshold}\naction=change highest load to throttled\n"}
    service_paths = []
    for service, load in enumerate(loads):
        path = f"{root}/service-{service}.ini"
        files[path] = (
            f"name=service-{service}\n"
            + _noise(rng, 12 + 4 * difficulty, f"service-{service}")
            + f"load={load}\nmode=active\n"
        )
        service_paths.append(path)
    target_path = service_paths[target_index]
    expected = files[target_path].replace("mode=active", "mode=throttled")
    answer = f"service-{target_index}:mode=throttled"
    actions = (
        Action("read_file", {"path": policy_path}),
        Action("list_files", {"directory": root}),
        *(Action("read_file", {"path": path}) for path in service_paths),
        Action(
            "replace_text",
            {"path": target_path, "old": "mode=active", "new": "mode=throttled"},
        ),
        Action("read_file", {"path": target_path}),
        Action("finish", {"answer": answer}),
    )
    prompt = (
        f"Read {policy_path}, list and inspect every service file in {root}, identify the service "
        "with the highest load, and if it exceeds the policy threshold change its mode from active "
        "to throttled. Read the changed file to verify it and report service-N:mode=throttled."
    )
    return AgentTask(
        f"{split}-conditional_update-{index:04d}",
        "conditional_update",
        prompt,
        files,
        actions,
        answer,
        frozenset({"read_file", "list_files", "replace_text"}),
        {target_path: expected},
    )


def _batch_update(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/deployment"
    target_count = 2 + difficulty
    manifest_path = f"{root}/manifest.txt"
    files = {}
    targets = []
    expected_files = {}
    manifest_lines = []
    modes = ("safe", "audit", "strict", "fast", "observe")
    for target in range(target_count):
        old, new = rng.sample(modes, 2)
        path = f"{root}/worker-{target}.ini"
        before = (
            f"worker={target}\n"
            + _noise(rng, 12 + 4 * difficulty, f"worker-{target}")
            + f"mode={old}\nversion={rng.randrange(1, 10)}\n"
        )
        files[path] = before
        expected_files[path] = before.replace(f"mode={old}", f"mode={new}")
        targets.append((path, old, new))
        manifest_lines.append(f"{path}|mode={old}->mode={new}")
    files[manifest_path] = "\n".join(manifest_lines) + "\n"
    files[f"{root}/unmanaged.ini"] = "worker=unmanaged\nmode=leave-alone\n"
    action_list = [Action("read_file", {"path": manifest_path})]
    action_list.extend(Action("read_file", {"path": path}) for path, _, _ in targets)
    action_list.extend(
        Action(
            "replace_text",
            {"path": path, "old": f"mode={old}", "new": f"mode={new}"},
        )
        for path, old, new in targets
    )
    action_list.extend(Action("read_file", {"path": path}) for path, _, _ in targets)
    answer = f"updated-and-verified={target_count}"
    actions = tuple(action_list) + (Action("finish", {"answer": answer}),)
    prompt = (
        f"Read {manifest_path}. For every managed worker listed there, inspect its file, apply the "
        "exact mode replacement, then read every changed file again. Do not alter unmanaged.ini. "
        f"When all {target_count} updates are verified, report updated-and-verified={target_count}."
    )
    return AgentTask(
        f"{split}-batch_update-{index:04d}",
        "batch_update",
        prompt,
        files,
        actions,
        answer,
        frozenset({"read_file", "replace_text"}),
        expected_files,
    )


def _aggregate_report(split: str, index: int, difficulty: int, rng: random.Random) -> AgentTask:
    root = f"lab/{split}/{index:04d}/metrics"
    metric_count = 4 + difficulty
    values = [rng.randrange(10, 90) for _ in range(metric_count)]
    files = {}
    metric_paths = []
    for metric, value in enumerate(values):
        path = f"{root}/metric-{metric}.txt"
        files[path] = (
            f"metric={metric}\n"
            + _noise(rng, 12 + 4 * difficulty, f"metric-{metric}")
            + f"value={value}\n"
        )
        metric_paths.append(path)
    split_at = metric_count // 2
    first_expression = " + ".join(map(str, values[:split_at]))
    second_expression = " + ".join(map(str, values[split_at:]))
    first_total = _calculate(first_expression)
    second_total = _calculate(second_expression)
    grand_expression = f"{first_total} + {second_total}"
    grand_total = _calculate(grand_expression)
    report_path = f"{root}/report.txt"
    old_report = "grand_total=PENDING"
    new_report = f"grand_total={grand_total}"
    files[report_path] = old_report
    actions = (
        *(Action("read_file", {"path": path}) for path in metric_paths),
        Action("calculate", {"expression": first_expression}),
        Action("calculate", {"expression": second_expression}),
        Action("calculate", {"expression": grand_expression}),
        Action("read_file", {"path": report_path}),
        Action(
            "replace_text",
            {"path": report_path, "old": old_report, "new": new_report},
        ),
        Action("read_file", {"path": report_path}),
        Action("finish", {"answer": new_report}),
    )
    prompt = (
        f"Read all {metric_count} metric files in {root}. Use the calculator to subtotal the first "
        "half, subtotal the second half, and add the two subtotals. Inspect report.txt, replace "
        "PENDING with the grand total, read it again to verify, and report the complete setting."
    )
    return AgentTask(
        f"{split}-aggregate_report-{index:04d}",
        "aggregate_report",
        prompt,
        files,
        actions,
        new_report,
        frozenset({"read_file", "calculate", "replace_text"}),
        {report_path: new_report},
    )


_MAKERS = {
    "pointer_chain": _pointer_chain,
    "ledger_reconcile": _ledger_reconcile,
    "cross_reference": _cross_reference,
    "conditional_update": _conditional_update,
    "batch_update": _batch_update,
    "aggregate_report": _aggregate_report,
}

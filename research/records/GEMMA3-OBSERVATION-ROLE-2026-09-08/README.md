# The observation role: how the same tasks were not the same tasks

**2026-09-08. A finding about our rendering, not about Gemma, discovered by stage one of the
representation map and demonstrated here rather than argued.**

## What was measured

Stage one ran fifteen episodes across all twelve task families. The instrument passed: 138,240
rank rows, 106,322 reading rows, peak 2.5 to 4.1 GiB. On the eleven episodes where both models
have a result on the same task id:

| | passes of 11 |
|---|---|
| Qwen3.5-4B base | 7 |
| Gemma 3 4B base | 2 |

Two loop detections, four exhaustions. The failures are uniform in kind: `batch_update-0166` reads
one file eight times consecutively; `ledger_reconcile-0163` issues `replace_text` with an empty
`old` eight times and receives the same error each time; `update-0028` cycles read, list, search,
read on a path that never resolves. **Every one is the model failing to notice that its own
previous action produced the text in front of it.**

## The rendering, side by side

The same four-message conversation, rendered through each model's own chat template with
`build_prompt`. No model was loaded; this is the tokenizer and the template only.

**Qwen3.5-4B**, `observation_role: tool`:

```
<|im_start|>assistant
reading the manifest first.
```{"name": "read_file", "arguments": {"path": "lab/worker-0.ini"}}```<|im_end|>
<|im_start|>user
<tool_response>
worker=0
mode=safe
</tool_response><|im_end|>
```

**Gemma 3 4B**, `observation_role: user`:

```
<start_of_turn>model
reading the manifest first.
```{"name": "read_file", "arguments": {"path": "lab/worker-0.ini"}}```<end_of_turn>
<start_of_turn>user
worker=0
mode=safe<end_of_turn>
```

Qwen's observation is wrapped and marked as a tool response. **Gemma's is bare, and formally
indistinguishable from the user's original instruction.** Nothing in what Gemma receives says that
this text is a result, or that it came from the tool the model just called.

## Why it happened

Gemma's chat template has branches for user, assistant and system only, enforces strict
alternation with an explicit exception, and has no tool role. The Chief ruled that observations
render as user turns, which is correct and necessary, and **said nothing about preserving what the
role had been carrying.** The implementation follows the ruling and drops the tool's name as well,
reasoning that a template with no tool role has no use for it.

That reasoning is the error. The `tool` role was not a formatting detail of one template. It
carried two facts the model needs: *this text is a result*, and *it came from this tool*. A
template that cannot express a role has not thereby made the facts unnecessary; it has only
removed the channel they were travelling in, and they must be moved to another one.

**This is the third instance in one day of the same failure in the Chief's rulings, in the same
direction each time** (see R59): reaching for a property of the artefact when the question was
about the thing the artefact represents. A file path for a model, a registry name for a model, a
template role for the fact the role was carrying.

## What this does and does not establish

It **does** establish that the two models were not given the same task, so the seven-against-two
comparison cannot be read as a capability gap.

It **does not** establish that Gemma is fine. The marking may be worth little; Gemma may be
genuinely weaker at this protocol. The transcripts are consistent with either.

**The distinguishing test is three episodes**, the three whose failures are most diagnostic, re-run
with the marking preserved and compared against stage one, which is retained as the control. If the
loops go, the gap was ours. If they stay, Gemma is weaker and the map is still worth running with
that stated plainly in its record.

## The technique, stated for transfer

**When a model's chat template cannot express a role your protocol uses, moving the message to a
role the template does have is half the job. The other half is carrying the discarded role's
information into the content, because a role is not decoration — it is a channel, and a protocol
that used it was relying on what it carried.** Check by rendering the same conversation under both
templates and reading them side by side; the difference is visible in seconds and needs no model.

Declare the substitute marking in the model registry rather than hardcoding it, so the next family
states its own instead of inheriting one model's convention — which is how ChatML's assistant
marker came to be hardcoded for every model in this repository.

---

## Amendment: the Director's correction, and a confound the Chief missed

Two corrections arrived after this record was written, and both make it smaller than it claimed.

**The step ceiling, which is the simpler confound and was missed.** The pilot runs twelve steps.
The evaluation that produced Qwen's comparison numbers runs twenty-four. **Four of Gemma's eleven
failures are exhaustion at a ceiling half the height its comparator was given.** That accounts for
more of the gap than the rendering does, and it was visible in the manifest before any hypothesis
was needed. Looking for a subtle cause without first checking the obvious one is the error, and it
is worse than the subtle one it found. The re-run raises the ceiling to twenty-four.

**The Director's correction to the framing, which is the more important half.** This record framed
the defect as an information-channel problem: the `tool` role carried facts, the template cannot
express the role, move the facts into the content. His framing:

> *Observations should render in a manner that teaches the model to understand what it is doing. A
> model executing a tool should not assume that in doing so it responds to the user.*

Under the current rendering, a Gemma episode is a strict two-party conversation — user, model,
user, model — so every tool call the model makes is followed by what is formally **a user turn**.
The model is taught, turn after turn, that *when it executes a tool, the person answers*, and that
its next output is a reply to that person rather than the next step of its own work. That is a
different task from *you are operating a workspace and observing the results of your own actions*.

**It also predicts the failures better than the missing-label account did.** A model that believes
the user speaks after each of its actions behaves conversationally: it re-acknowledges, re-reads,
restates and confirms. Re-reading one file eight times and re-issuing one call eight times is what
a model does when it thinks it is in a dialogue rather than executing a plan. The label account
explains that the model is confused; this one explains the shape of the confusion.

**So the fix is not a prefix string.** The observation must present as the environment's response
to the model's own action, and **the system prompt must teach the convention**, because the
template forces the user role and the model cannot infer why. Both belong in the model registry
rather than in code, so the next family declares its own instead of inheriting Gemma's.

**And the technique section above is correspondingly wrong where it says the other half of the job
is carrying the discarded role's information into the content.** That is necessary and not
sufficient. The full statement:

> **When a model's chat template cannot express a role your protocol uses, moving the message to a
> role the template does have changes what the model is being taught about the interaction, not
> only what it is being told. Ask what the substitute role implies about who is speaking and what
> the model's next turn is for, and where the implication is false, say so explicitly in the
> content and in the system prompt. A role is not a label on a message; it is a claim about the
> structure of the exchange.**

**A note on what the re-run can and cannot say.** It changes the ceiling and the rendering
together. Both are independently correct, so neither is worth holding back to preserve a clean
comparison, but the consequence is that the re-run tests whether Gemma can do these tasks when
fairly presented — not which of the two errors mattered. Attribution between them is a cheap
follow-up if the result makes it interesting.

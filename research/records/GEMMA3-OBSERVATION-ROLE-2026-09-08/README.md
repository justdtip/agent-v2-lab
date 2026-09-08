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

---

## Amendment, 09:20Z: the marking alone was tried, and the loops stayed

Before the Director's correction arrived, the marking-only version ran: observations re-roled to
`user` and wrapped in a bare `<tool_response>`, no convention paragraph, ceiling still at 12.
One episode completed, `agentic-d2-update-0028`, against stage one's control for the same task.

**It changed nothing that matters.** The trajectory is the same loop and not a milder one: read a
path, list an empty directory, search, read the same path again, four times over twelve turns.
Neither run succeeds, neither detects the loop, and both exhaust.

The mechanism is in both transcripts and is worth stating on its own, because it is sharper than
"the model is confused". `search_files` returns

```
MATCHES: workspace/test/0028/config.ini
```

and the model's next call reads `/workspace/test/0028/config.ini`, with a leading slash it
supplied itself, which does not resolve. It never once tries the string the observation handed it.
The model is not failing to find the file. It is failing to use the result as a source of fact,
and re-deriving the path from its own earlier guess instead — which is what the Director's framing
predicts a model does when it believes it is in a conversation rather than reading its own
instrument.

That run is kept, superseded, at `pilot/superseded/marking-only-12-steps/`, because it is the only
run that isolates the marking from the ceiling, and the run that replaces it changes both.

**Its timings are worthless and the directory says so.** The laptop suspended mid-run. The episode
took 940 s against the control's 381 s for slightly less work, and a second episode was writing at
about 360 bytes per second against the control's 33,000 when it was stopped. Nothing about the
cost of this rendering should be read from those figures.

## What landed

In the registry, per family, as the Director's ruling requires:

- `chat.observation_template` now names the call the observation answers:
  `<tool_response tool="{name}">…</tool_response>`. It was a bare wrapper matching Qwen's, chosen
  so the pilot's two panels rendered observations identically. That choice is retired: a rendering
  does not get to misdescribe the episode in order to keep a comparison tidy. **The cost is real
  and is recorded rather than avoided** — the two panels no longer render observations the same
  way, so a difference between them is no longer attributable to the model alone.
- `chat.observation_convention` is new and required on exactly the same condition as the wrapper:
  whenever the role is not `tool`. It is the paragraph the system prompt carries telling the model
  that a turn the format marks as the user's may be the workspace answering its own last call.
  The wrapper marks each observation; this gives the model the key to the mark. A marker whose
  meaning the model was never told is not a channel.

In code, only the plumbing: `protocol.system_prompt` takes the spec and appends the family's
convention, and `runner.run_task` renders the system message through it. Qwen's prompt is
byte-identical to what every existing Qwen number was produced under, asserted by a test.

**A boundary that is not visible from the code.** The three training-row builders — `build_rows`,
`trajectory_rows` and jlens's probe-context builder — still render the convention-free prompt, and
none of them takes a spec. That is correct today and only today, because Gemma is the sole family
declaring a convention and its training is deferred. The day anyone builds a supervised target on
a re-roling family, those three take the spec or training teaches a prompt inference does not use.

## The ceiling, which was the larger error

The pilot ran `--max-steps 12`. The evaluation that produced the Qwen numbers it is compared
against runs 24, which is `run_task`'s own default. Four of Gemma's eleven failures were
exhaustion. A comparison against numbers made at 24 is not valid at 12, and this is the plainer
confound: it was found by looking at the harness rather than at the model, after a subtler
hypothesis had already been written up.

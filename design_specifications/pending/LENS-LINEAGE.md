# A lens identity is a lineage: base, training, depth — R57 and R60

Patch: `design_specifications/pending/LENS-LINEAGE.patch`, made with **`git add -A && git diff
HEAD`**, so new files are in it. Thirteen files, 244 insertions. **Not landed.**

**1,994 passed, 14 skipped, exit 0**, with the box free — nothing skipped for a window.

## What the Director ruled, and what it replaces

**R60**: *if the lens ontologically fits, and the only thing preventing it from loading is
essentially a metadata mismatch, we should modify to allow loading.*

So the loader decides on what a lens **is** a map of — base checkpoint, training applied, decoder
depth, with width checked against the maps themselves — and **records** what merely travels with
it: the artefact's path, a local registry name, the stored precision, the file's digest.

This replaces two of my own attempts in one day, and the pattern in both is the same: I reached for
a property of the artefact when the question was about the model. First the registry entry name and
the `hf_id`; then `source`, a single string, which was closer and still wrong — because it cannot
distinguish a base checkpoint from that checkpoint after training.

**Why that distinction is not pedantry, in this repository's own numbers.** A base and a trained
checkpoint share family, depth and width, so nothing architectural separates them. A Jacobian lens
is a map of residual geometry, and the adapter-depth ablation measured what training does to it:
every one of thirty-two layers moved by about a fifth of its weight norm, and two arms' updates to
the layers they shared were orthogonal. `source` would have let a base-fitted lens read a trained
model and call it a match.

## The three cases, and what each does

| | outcome |
|---|---|
| same base, same training, different artefact or precision | **permitted**, difference recorded in the manifest |
| different base, or different family | **refused** — issue 99's case, and the one no other check catches |
| same base, different training | **refused by default**, requestable deliberately as a declared cross-condition read |

The third is worth keeping requestable: reading a trained model through its base's lens shows what
training moved, which is one of the more interesting measurements available. What it must never be
is an accident of two entries agreeing on architecture.

## The loader resolves; the artefact is never rewritten

A lens stamped before this rule carries an artefact path or a registry name where the base belongs.
`base_of_artifact` asks the registry which base that artefact descends from, so **Codex's fitted
lens needs no archive rewrite and the digest published in four committed records stays valid.**
An unrecognised name passes through unchanged, deliberately: it should reach the comparison and be
refused there with both names shown, rather than be quietly turned into something else.

## Two smaller decisions a reviewer should push on

**`base` and `training` are keyword-only with defaults, and `base` defaults to `hf_id`.** The
registry always supplies both; the defaults serve direct construction, where a checkpoint nobody
converted is its own base. That is a default expressing an identity relation, not a default
assuming a model — which is the distinction the Chief drew on `generation_prefix`, where the
default was refused.

**`training_lineage: {}` is written explicitly in both Gemma entries**, not omitted. A reader must
never have to decide whether an absent block means *no training* or *not recorded*.

## Three things stage one required, now in the same patch

**An aborted record is no longer skipped as a finished episode.** The record writer already stamps
`status: complete` or `aborted` in its own footer, and the resume check looked only at whether the
file existed. Stage one hit it: a run that died mid-episode left a partial record and the retry
skipped it, so a truncated episode would have gone into the map silently. It now reads the footer,
and **refuses** rather than overwriting — a launcher that deletes records it finds inconvenient is
a launcher with no records.

**`attention_span` per layer in the manifest**, which the pre-registration requires and which
`band` cannot supply because neither Gemma entry declares one.

**Two statements that must not be inferred from absence.** That the secondary comparison is not
attempted, with both reasons — the hosted lens was fitted at 128 tokens against a 1,024 window, and
one episode has any position past 1,024 at all. And the precision mismatch, present **whether or
not the precisions differ**, so a reader never reads agreement out of a missing line.

## And one more instance of the day's recurring fault

`_resolve_checkpoint` resolved `models/...` against the **running** checkout, so a worktree found
nothing: `models/` is git-ignored and exists once, in the primary. That is the box window's own bug
before `box_state_root`, from the same cause, and it is fixed with the same reader. Third instance
today of behaviour that depended on where a process was standing.

## Callers outside this tree that the signature change reaches

`LensIdentity` went from three fields to `(base, num_layers, training=None)`. One caller is known
to still pass the old shape, and it is not in this checkout:

`research/records/GEMMA3-REGRESSION-2026-09-08/compare_maps.py`, in the collaborator's
`gemma-lens-fitting` worktree, constructs it twice with three positional arguments — a registry
entry name, an artefact path, and the depth. Under the new signature that binds the path to
`num_layers` and the depth to `training`, which fails at `int(self.num_layers)` rather than
silently, but fails.

Both calls become one identity, because both artefacts are the same model at two precisions:

```python
LensIdentity("google/gemma-3-4b-it", depth)
```

That is the point of the change rather than a consequence of it. The two entries differ only in
stored precision, which is a property of the file; a lens is a map of the model. And their fitted
lens now loads unchanged under R60 — the stamp keeps the absolute path it was written with, the
loader resolves it to the base through the registry, and the digest published in four committed
records stays valid with no archive rewrite.

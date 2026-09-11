# `addendum-1.json` is SUPERSEDED and void. It is kept, unaltered, and must not be used.

**D-CRO, 2026-09-11, on the Chief's ruling.**

    digest    def0b7e92961f4eb8a6a8634aa68eb65f6e4a0e313e0c07f3ab1a0cdf39d8dcd
    baseline  1d936087ca944c4934c9bfec7185005f302209fa
    written   2026-09-11, reported, and found defective before anything was read

**Why it is void.** It sealed seven files, one of them `read_e2.py`. At its baseline that reader
still computed the **withdrawn** rule, `x + m·d` with `d` the mean per-step displacement, and never
imported the rule module at all. So the addendum fixed, in one seal, an amendment specifying a rank-r
supervised map and a reader computing the thing that map replaced. Scoring the strata against the
sealed reader would have reproduced the 0.000 already known to be an artefact, and the seal would have
recorded the run as conforming.

**Why nothing caught it.** `make_addendum.py` verified the parent seal, the amendment's own checker,
that §5's printed table reproduced `gate-table.json` to the digit, that the table was measured against
this seal and this fold assignment, and that all seven files matched their bytes at the baseline. All
passed. **None asked whether the sealed code implements the sealed rule** — identity and provenance
are the easy properties, and meaning was never checked. `METHOD-2026-09-08` entry 34, again.

**Why this file exists instead of a mark inside the JSON.** An addendum's digest *is* its identity.
Editing `addendum-1.json` to carry a `superseded` flag would change its bytes, so the digest reported
to the Chief and recorded in the order would no longer match the file it names, and a reader checking
that digest would conclude the artefact had been tampered with. The JSON is therefore left
**byte-identical to what was issued**, and this notice sits beside it. That is a deliberate reading of
"marked in place, never deleted": the artefact is unaltered and the supersession is unmissable to
anyone who opens the directory.

**What replaces it.** `addendum-2.json`, against a new baseline, after Codex has read `read_e2.py`
against §2 — the one file their PASS on revision 5 did not cover. Its builder gains two refusals the
first lacked: a **proxy** check that a sealed reader imports the rule module and does not carry the
withdrawn rule's signature, and an **executable conformity check** that pushes a fixture, on which
the two rules give different answers, through the reader's own rule function and refuses unless it
returns §2's. Both were run against the reader this addendum sealed: the proxy names two problems,
and the conformity check refuses because that reader exposes no rule function to execute.

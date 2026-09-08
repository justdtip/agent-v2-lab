> **SUPERSEDED — do not apply.** The Chief landed the stack by another route and this patch is
> for a landing that did not happen. Its work is in the tree as **3c8dac8**, verified there at
> 1,985 passed with the window held. Applying this now would conflict.
>
> Kept rather than deleted because `design_specifications/pending/` is where the next implementer
> looks, and a verified-looking patch that no longer applies is a trap. The reasoning below still
> reads as the record of why each fix was the fix.

# The 28 failures, fixed — and the whole suite green on the box for the first time today

Patch: `design_specifications/pending/STACK-FIXES.patch`. **Apply last, after all six.** Nine
files, 211 insertions.

**The verification is the point of this document.** In one window, on the box, against the six
patches stacked:

```
1976 passed, 14 skipped, 1 warning in 101.70s
```

Nothing skipped for a window. The seven files the skip list had been hiding — `test_patch`,
`test_pipeline`, `test_probes`, `test_jlens`, `test_lens_regression`, `test_live_lens_native`,
`test_arch` — went from **36 failed** on the first honest run to **377 passed**.

The Chief's rule, which I accept and which was mine to have applied: *a patch whose changes are
exercised by files in the skip list is not verified by a suite that skipped them, and the reported
skip count is the coverage statement rather than a footnote.* Every report I wrote today carried
that count and I read past it because the exit code was zero. The terminal summary that says so on
every run is one I wrote.

## Two of the 36 were design signals, not test breakage

**`input_embeddings` was the wrong way to drive the observation, and 26 failures said so.**
`masks` observed the model by handing it a dummy through `input_embeddings`. That keyword is a
footgun on Gemma, whose entry scale sits *after* the branch — and it turned out to be a
requirement several stand-in decoders do not meet. **Now the observation is driven by token ids of
the right length**, which every decoder accepts, because masks need a sequence length and a cache
and never the token values. The footgun is gone rather than documented, and the mechanism can be
observed on a stand-in with no checkpoint.

**Masks now come back in the model's own embedding dtype, which is a precision change nobody
asked for.** They used to be built from the caller's `h`, so a float32 loop got float32 masks. An
additive mask carried at a narrower precision than the stream it is added to is a silent precision
change, so the content stays the model's and the dtype follows the residual. `test_arch.py` proves
the constructors see the model's dtype and the returned mask follows the caller's.

## A cost of the port, measured and flagged rather than hidden

`embed` and `masks` are each one observing forward, and **a forward builds masks whether or not
the caller wanted them**, so `embed`'s pair is built and discarded. On a long sequence that
discarded attention mask is an N-squared allocation.

The Chief's ruling was to pay the forward, never cache it, and hoist at the call site if
measurement says otherwise. `_assert_uncached_hybrid_traversal` now asserts one pair per observing
forward, so this is where such a change would first show up. **This is the item I would put in
front of the Chief before the port lands**, because it is the one number the design argument did
not have.

## Three fakes gained a forward, which is the port's real demand on a view

`_ArchitectureText`, `_JLensInner` and the probe hybrids exposed the attributes a view looks for
and no way to run. That was enough while the view re-implemented the decoder loop; since it
observes one, **a stand-in for a model has to have a forward**. That is the right ripple rather
than a cost: a fake with no forward was a fake of a mask factory.

## The rest, in one line each

- `generation_prefix` added to the fake chat specs in `test_patch` and `test_jlens` — 20 failures.
- `probes.partner_tie_breaks` added to the fake specs the sweep reads — 2.
- `probe_partner_tie_breaks` added to the spec field set in `test_probes` — 1.
- `write_lens(..., identity=)` at the caller in `test_lens_regression`, and the archive's
  `identity` entry in the file-set assertion — 2.
- `counts` carries `residual_source`, asserted — 1.
- `masks` keyed by block index at eight assertion sites in `test_arch` and one in `test_probes`,
  through one `_mask_of` helper so the translation happens in one place.
- My own rotating-cache test handed KV caches to a hybrid's recurrent blocks, which is a different
  error than the one under test. It now replaces only the attention entries.
- My own `(8, 7)` band case asserted a partner for the final layer, which `in_band_layers` reports
  rather than decides on. Removed, with the reason.

## One thing that went wrong in the verifying itself

The first verify script put a **relative** `.venv/bin/python` in its `trap` and then changed
directory, so the trap could not run and the window outlived the job. I found it in `status`,
closed it with its own nonce, and made every path in the script absolute. Same shape as the orphan
that cost the group eight hours, from the same cause: a teardown that depends on where the process
happens to be standing.

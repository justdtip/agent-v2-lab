# Director resolves prose cross-read: requirements section14

7 September2026. The Director's reply and primary commitafb3c35 resolve PROSE-CROSS-READ-QUESTION-20.md. Use greedy continuations of the51 heldwindows; rawfirst824tokenprompt, atmost200generatedtokens, EOSearlier; dropandcountunder32. No fitwindows or furtherdownload. State that these heldwindows selectedridgealpha amongfivevalues; this is not a fully independentprose generalisation sample. Authoredtrailingtokensmayberecordeddescriptivelybutnotscored. Preserve foreknowledgeh1/4/8 overactualemissions and authoredpromptagreement. Base ratescome fromthe prosecapturefinaldistributions, notoldpilotconstants.

The exact tokenizer-only preflight confirms51/51 rawprefixes encode backto their frozen824ids with the currentcached snapshot and corpusBOSpolicy. Evidence PROSE-PREFIX-TOKENIZER-21.json; noMLXimport or weightsload. Its preliminary check stoppedbeforeprocessing due to assuming the reader returned a tuple; the correctedcall used its actualrows-listAPI.

Implementation detail requiring a deliberate choice, already determined by source: generate_turn_with_count adds tool-close stopping, unsuitable for the EOS/cap-onlyproseprotocol. Use installedstream_generate directlyinsidepublicCaptureSession.generation, forwarding every yieldedresponse.token as thepilotdoes andclosing thestream. The installedstream yields anEOSterminalresponse; record it as anemittedtoken as thepilotdoes, retainfinish_reason, and make thecountingconvention explicit. Do not synthesiseemissionsfromauthoredtokens.

The existinglegacyatlas emits span note for anynon-chatkind; newprosesupportmust explicitlyname continuation withoutchangingagentic/chatsummaries. Nativepilotidentity remainsmandatory beforeprofiles. This is anauthorisedsourceextension, notanewsummary or metric. No modeltime isreserved: BOX-SCHEDULE-19.md stillapplies.

## Readable restatement (appended; prior text retained)

The Director's §14, committed in primary at afb3c35, resolves the question in record 20. Use the 51 held Wikipedia windows: first 824 tokens as a raw prompt, then up to 200 greedy emitted tokens; stop at EOS and drop/count continuations shorter than 32 tokens. State that the held windows selected the ridge weight. No further download is authorised. Authored trailing tokens may be stored for comparison but are not scored as foreknowledge.

The two summaries remain emitted-token foreknowledge at horizons 1/4/8 and authored prompt agreement. Every prose foreknowledge row uses the corresponding prose final-distribution base rate. The tokenizer-only check in record 21 confirms exact prefix IDs for all 51 windows and imported no MLX.

The existing agent generator stops on tool-close text, so the prose producer must call the installed stream_generate inside the public CaptureSession context directly. It records every yielded token, including the terminal EOS response, as the pilot does; this counting convention must be explicit. Source code for the legacy atlas currently labels non-chat emissions as note; a narrow prose branch must label them continuation while preserving original pilot values. No metric or model-time reservation is added.

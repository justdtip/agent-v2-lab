# Prose cross-read target origin needs a ruling

7 September 2026. Requirements section2 defines foreknowledge and its final-layer base rate over **emitted tokens**. Section3.5 asks for the agentic lens on prose with the same summaries. The frozen prose corpus is authored Wikipedia text in token windows; it carries ids, spans and source positions but no model-emission events or captured forwards. The held51 windows contain52,224 tokens. The three pilot chat episodes have emissions but are not the specified Wikipedia prose evaluation.

There are two different experiments here. Feeding the known authored tokens would measure prediction of an externally supplied continuation. Recording greedy model continuations would measure prediction of the model's own later outputs, as the pilot does. They need different target-origin labels and cannot silently share the emitted-token definition. Feeding known text through CaptureSession.emitted would create the appearance of native emissions without a declared teacher-forcing protocol.

Asked the Director which experiment is intended. Recommended recording greedy continuations to preserve the pilot metric, with a continuation length fixed before the run. No length, decoding budget or new prose capture protocol has been chosen by the implementer. Exact source token ids remain authoritative; any decode/encode alignment change at a window boundary must be rejected or explicitly handled in an approved protocol.

Dependent prose-profile implementation/execution is held. The existing-record replay work can finish independently and continues with pure tests while issue88 has scheduling priority. No native run or scientific profile accompanies this question.

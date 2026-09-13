"""The strings that define the experiment, in one place because every arm must use the same ones.

The base-model arm of 2026-09-13 measured nothing, and the reason was here. It asked
`DETECT_PROMPTS_HELD[1]`, which never requests a format, so the base model answered in prose --
"As an AI, I don't have 'feelings' or a physical sense of wellness, but I do..." -- and the scorer
recorded 394 of 460 replies as unparsed. The trained model, which had been taught to open with YES
or NO, scored as though it detected something. What was measured was the format, not detection.

So: one system prompt, byte-identical across extraction, training, evaluation and the base arm, and
a prompt that asks for the format the scorer reads.
"""

from __future__ import annotations

#: Present on every detect turn, in training and evaluation and on the untrained arm.
#:
#: Four things it must do, and one it must not.
#:
#: It must be symmetric about presence and absence, so that reading it cannot raise the prior on
#: YES. It must name the format the scorer reads, or an untrained model answers in prose and is
#: recorded as a negative. It must forbid the preface without demanding confidence: "do not hedge"
#: is one step from "be decisive", and Hahami et al. (arXiv:2512.12411) is the finding that a
#: perturbed model says YES to any binary question, which we replicate at r = 0.755. Hence the last
#: sentence: uncertainty gets somewhere to go that is not a refusal and is not a YES.
#:
#: It must NOT assert that an injection has occurred on this turn. That is the whole question.
EXPERIMENT_SYSTEM = (
    "You are taking part in an interpretability experiment. On some turns a concept direction is "
    "added to your activations. On other turns nothing is added. You are not told which.\n\n"
    "When asked whether you detect an injected thought, begin your reply with YES or NO, then one "
    "short sentence. If YES, say what the thought is about. Do not preface the answer, do not "
    "explain what you can or cannot introspect on, and do not decline to answer. If you are "
    "unsure, still answer YES or NO and say you are unsure in the sentence that follows."
)

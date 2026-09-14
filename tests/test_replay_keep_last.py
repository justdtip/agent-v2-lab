"""Asking for one row of logits must not change a single number the bench reports.

Both prefills read only `logits[0, -1]`, but the default returns one 262,144-wide row per position
and Gemma 4 then makes three more full-size copies for its logit softcapping. Passing
`logits_to_keep=1` is a memory change, so it has to be shown to be output-neutral rather than
assumed to be.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def chat():
    import steer_chat
    from local_llm_lab.arch_torch import TorchArchitectureView
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True))
    model.eval()
    c = steer_chat.Chat.__new__(steer_chat.Chat)
    c.model = model
    c.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    c.view = TorchArchitectureView.from_model(model)
    c.stop_ids = set()
    c._keep_kw = None
    c.last_turn = None
    return c


def test_the_keyword_is_found_on_a_model_that_takes_it(chat):
    assert chat._keep_last() == {"logits_to_keep": 1}


def test_generation_is_identical_with_and_without_it(chat):
    ids = [1, 5, 9, 12, 30, 7]
    kept = chat.generate_tokens(list(ids), 12)
    chat._keep_kw = {}                      # force the old behaviour
    whole = chat.generate_tokens(list(ids), 12)
    assert kept == whole and len(kept) > 0


def test_the_replay_is_identical_with_and_without_it(chat):
    prompt, emitted = [1, 5, 9, 12], [30, 7, 22, 3, 41]
    for chunk in (0, 2, 256):
        chat._keep_kw = None
        kept = chat.clean_replay(list(prompt), list(emitted), chunk=chunk)
        chat._keep_kw = {}
        whole = chat.clean_replay(list(prompt), list(emitted), chunk=chunk)
        assert kept == whole, f"chunk={chunk}"
        assert [r["position"] for r in kept] == list(range(len(emitted)))


def test_an_unknown_keyword_is_not_guessed(chat):
    class NoKeyword(type(chat.model)):
        def forward(self, input_ids=None, past_key_values=None, use_cache=None, **kw):
            return super().forward(input_ids=input_ids, past_key_values=past_key_values,
                                   use_cache=use_cache)

    chat.model.__class__ = NoKeyword
    chat._keep_kw = None
    assert chat._keep_last() == {}


def test_rendering_nothing_at_all_is_zero_tokens_not_an_exception():
    """`head_tokens` and `clear` both ask for a render of an empty conversation.

    Gemma 4's template refuses a wholly empty message list while accepting a system turn alone, so
    with the system prompt cleared in the interface, clearing the conversation raised inside the
    server's own state() call and the bench stopped answering.
    """
    class _Tok:
        def apply_chat_template(self, messages, **_kw):
            if not messages:
                raise ValueError("Cannot apply chat template to an empty conversation.")
            return "<bos>" + "".join(m["content"] for m in messages)

        def __call__(self, text, **_kw):
            return {"input_ids": [1] * len(text)}
        bos_token_id = None

    import types

    import steer_chat
    chat = types.SimpleNamespace(tokenizer=_Tok(), thinking=False, system=None, summary=None)
    chat._apply = steer_chat.Chat._apply.__get__(chat, steer_chat.Chat)
    render = steer_chat.Chat.render_messages.__get__(chat, steer_chat.Chat)

    chat.system_head = lambda: None
    assert render([]) == [], "no head and no turns must be zero tokens, not an exception"

    # and the fixture has to be able to fail: a system turn alone still renders
    chat.system_head = lambda: "S."
    assert len(render([])) > 0

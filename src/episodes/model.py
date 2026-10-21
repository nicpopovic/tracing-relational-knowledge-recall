"""Model loading (patched LLaMA / Qwen3) and tokenizer padding."""

from __future__ import annotations


def load_model(model_name: str, device_map_auto: bool = False):
    """Load a patched LLaMA or Qwen3 model (see `src/probingmods`) in eval mode."""
    if "llama" in model_name.lower():
        from src.probingmods.llama import get_model
    elif "qwen3" in model_name.lower():
        from src.probingmods.qwen3 import get_model
    else:
        raise NotImplementedError(f"Model {model_name} not supported")
    model = get_model(model_name, device_map="auto" if device_map_auto else None)
    model.eval()
    return model


def ensure_padding_token(tokenizer) -> None:
    """Set `pad_token` to the EOS (or UNK) token if the tokenizer has none."""
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return
    raise ValueError("Tokenizer must expose eos_token or unk_token so a pad_token can be assigned")

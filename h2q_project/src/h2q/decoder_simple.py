"""Lightweight decoder pairing with SimpleTokenizer.

This module mirrors the tokenizer contract and keeps logic explicit:
- Converts token ids back to text.
- Provides helpers to clip/pad sequences and batch decode.
"""

from typing import List

from .tokenizer_simple import SimpleTokenizer, default_tokenizer


class SimpleDecoder:
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer

    def decode(self, token_ids: List[int]) -> str:
        """Decode a single sequence of token ids to string."""
        return self.tokenizer.decode(token_ids, skip_specials=True)

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Batch decode multiple sequences."""
        return self.tokenizer.decode_batch(batch_ids, skip_specials=True)

    def trim_at_eos(self, token_ids: List[int]) -> List[int]:
        """Trim sequence at first EOS token."""
        if self.tokenizer.eos_id in token_ids:
            cut = token_ids.index(self.tokenizer.eos_id)
            return token_ids[:cut]
        return token_ids

    def trim_padding(self, token_ids: List[int]) -> List[int]:
        """Remove trailing padding tokens."""
        # Work backwards to find last non-pad token
        end = len(token_ids)
        while end > 0 and token_ids[end - 1] == self.tokenizer.pad_id:
            end -= 1
        return token_ids[:end]

    def trim_all(self, token_ids: List[int]) -> List[int]:
        """Trim both EOS and padding from sequence."""
        trimmed = self.trim_at_eos(token_ids)
        return self.trim_padding(trimmed)


default_decoder = SimpleDecoder(default_tokenizer)

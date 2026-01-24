"""Lightweight tokenizer for optional text I/O.

Design goals:
- Zero external deps, portable to edge devices.
- Small default vocab (printable ASCII) with <pad>/<unk>/<bos>/<eos>.
- Deterministic encode/decode to unblock end-to-end tests.
- Batch encoding support for efficient processing.

This is intentionally minimal: it is not a production tokenizer but
provides a stable contract for the FastAPI /generate route and e2e
benchmarks.
"""

from typing import List, Union, Optional


class SimpleTokenizer:
    def __init__(self):
        # Reserve specials first so ids stay stable across runs.
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        ascii_chars = [chr(i) for i in range(32, 127)]  # printable ASCII

        self.id_to_token = specials + ascii_chars
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        """Number of tokens in vocabulary."""
        return len(self.id_to_token)

    def encode(
        self,
        text: str,
        add_specials: bool = True,
        max_length: int = 256,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        """Encode a single string to token ids.
        
        Args:
            text: Input string to encode
            add_specials: Whether to add <bos> and <eos> tokens
            max_length: Maximum sequence length (0 = unlimited)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
        
        Returns:
            List of token ids
        """
        ids: List[int] = []

        if add_specials:
            ids.append(self.bos_id)

        for ch in text:
            if truncation and max_length > 0 and len(ids) >= max_length - (1 if add_specials else 0):
                break
            ids.append(self.token_to_id.get(ch, self.unk_id))

        if add_specials:
            ids.append(self.eos_id)

        # Pad to max_length for stable tensor shapes if needed by callers.
        if padding and max_length > 0 and len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_specials: bool = True,
        max_length: int = 256,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[List[int]]:
        """Batch encode multiple strings.
        
        Args:
            texts: List of input strings
            add_specials: Whether to add <bos> and <eos> tokens
            max_length: Maximum sequence length (0 = unlimited)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
        
        Returns:
            List of token id lists
        """
        return [
            self.encode(t, add_specials=add_specials, max_length=max_length, padding=padding, truncation=truncation)
            for t in texts
        ]

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """Decode token ids to string.
        
        Args:
            ids: List of token ids
            skip_specials: Whether to skip special tokens in output
        
        Returns:
            Decoded string
        """
        tokens: List[str] = []
        specials = {self.pad_id, self.unk_id, self.bos_id, self.eos_id} if skip_specials else set()

        for idx in ids:
            if skip_specials and idx in specials:
                continue
            if idx < 0 or idx >= len(self.id_to_token):
                continue
            tokens.append(self.id_to_token[idx])

        return "".join(tokens)

    def decode_batch(self, batch_ids: List[List[int]], skip_specials: bool = True) -> List[str]:
        """Batch decode multiple sequences.
        
        Args:
            batch_ids: List of token id lists
            skip_specials: Whether to skip special tokens
        
        Returns:
            List of decoded strings
        """
        return [self.decode(ids, skip_specials=skip_specials) for ids in batch_ids]


# Singleton to avoid repeatedly re-initializing in server contexts.
default_tokenizer = SimpleTokenizer()

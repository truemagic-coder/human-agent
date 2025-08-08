import re
from typing import List, Dict, Iterable, Union, Optional

class Tokenizer:
    """A simple whitespace/punctuation tokenizer with special tokens."""

    def __init__(self, vocab_size: int = 16000, special_tokens: Optional[Iterable[str]] = None):
        self.max_vocab_size = vocab_size
        self._frozen = False  # <-- add
        base_specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        if special_tokens:
            base_specials.extend([t for t in special_tokens if t not in base_specials])

        self.special_tokens: List[str] = list(dict.fromkeys(base_specials))
        self.vocab: Dict[str, int] = {}
        for tok in self.special_tokens:
            self._add_to_vocab(tok)
        self.id_to_token: List[str] = [None] * len(self.vocab)
        for tok, idx in self.vocab.items():
            self.id_to_token[idx] = tok
        # Reverse map: id -> token (for legacy callers)
        self.reverse_vocab: Dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}

        # Cache special token IDs
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    def _add_to_vocab(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        idx = len(self.vocab)
        self.vocab[token] = idx
        return idx

    def freeze(self) -> None:
        """Prevent further vocabulary growth."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Allow vocabulary growth."""
        self._frozen = False

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        """Add new special tokens and update maps."""
        updated = False
        for tok in tokens:
            if tok not in self.vocab:
                self.special_tokens.append(tok)
                self._add_to_vocab(tok)
                updated = True
        if updated:
            # Rebuild id_to_token
            self.id_to_token = [None] * len(self.vocab)
            for tok, idx in self.vocab.items():
                self.id_to_token[idx] = tok
            # Rebuild reverse map
            self.reverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def _basic_tokenize(self, text: str) -> List[str]:
        # Split on non-word characters; keep simple tokens
        return [t for t in re.split(r"(\W)", text) if t and not t.isspace()]

    def encode(self, text: Union[str, List[str]], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if isinstance(text, list):
            tokens = text
        else:
            tokens = self._basic_tokenize(text)

        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                # Grow vocab only if not frozen; otherwise use <unk>
                if not self._frozen and len(self.vocab) < self.max_vocab_size:
                    idx = self._add_to_vocab(tok)
                    ids.append(idx)
                    # maintain id_to_token
                    if idx == len(self.id_to_token):
                        self.id_to_token.append(tok)
                    else:
                        # resize if needed
                        if idx >= len(self.id_to_token):
                            self.id_to_token.extend([None] * (idx - len(self.id_to_token) + 1))
                        self.id_to_token[idx] = tok
                    # maintain reverse map
                    self.reverse_vocab[idx] = tok
                else:
                    ids.append(self.unk_token_id)

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens: List[str] = []
        for _id in ids:
            if _id == self.pad_token_id:
                continue
            if _id == self.eos_token_id:
                break
            if 0 <= _id < len(self.id_to_token) and self.id_to_token[_id] is not None:
                tok = self.id_to_token[_id]
                # Skip BOS in output
                if tok == "<bos>":
                    continue
                tokens.append(tok)
            else:
                tokens.append("<unk>")
        # Join with no extra spaces around punctuation
        text = ""
        for tok in tokens:
            if re.fullmatch(r"\W", tok):
                text += tok
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += tok
        return text.strip()

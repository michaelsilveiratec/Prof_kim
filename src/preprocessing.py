from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    sentences: list[list[str]]
    token_count: int


class TextPreprocessor:
    """Realiza a limpeza, normalização e tokenização de textos."""

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        return [token for token in cleaned.split() if token]


def tokenize(text: str) -> list[str]:
    return TextPreprocessor().tokenize(text)


def preprocess_text(text: str) -> CorpusDocument:
    if not text:
        return CorpusDocument(sentences=[], token_count=0)

    parts = re.split(r"[.!?]+", text)
    sentences: list[list[str]] = []
    for part in parts:
        tokens = tokenize(part)
        if tokens:
            sentences.append(tokens)
    return CorpusDocument(sentences=sentences, token_count=sum(len(sentence) for sentence in sentences))


def load_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
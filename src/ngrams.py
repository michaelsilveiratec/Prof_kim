from __future__ import annotations

from collections import Counter

from src.config import END_TOKEN, START_TOKEN, UNK_TOKEN


class NGramModel:
    """Extrai e contabiliza frequências de n-grams do corpus."""

    def __init__(self, order: int = 3, min_word_frequency: int = 1, n: int | None = None):
        if n is not None:
            order = n
        if order <= 0:
            raise ValueError("order must be positive")
        self.order = order
        self.n = order
        self.min_word_frequency = min_word_frequency
        self.unigram_counts: Counter[str] = Counter()
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = set()
        self.candidates: list[str] = []

    @property
    def vocab(self) -> set[str]:
        return self.vocabulary

    @vocab.setter
    def vocab(self, value: set[str]) -> None:
        self.vocabulary = value

    def fit(self, sentences: list[list[str]] | list[str]) -> None:
        if not sentences:
            self.unigram_counts.clear()
            self.ngram_counts.clear()
            self.context_counts.clear()
            self.vocabulary = set()
            self.candidates = []
            return

        if isinstance(sentences, list) and sentences and isinstance(sentences[0], str):
            sentence_list: list[list[str]] = [list(sentences)]
        else:
            sentence_list = [list(sentence) for sentence in sentences]

        flat_tokens = [token for sentence in sentence_list for token in sentence]
        self.unigram_counts = Counter(flat_tokens)
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        for sentence in sentence_list:
            tokens = [START_TOKEN] + sentence + [END_TOKEN]
            for start in range(len(tokens)):
                max_len = min(self.order, len(tokens) - start)
                for length in range(1, max_len + 1):
                    ngram = tuple(tokens[start : start + length])
                    self.ngram_counts[ngram] += 1
                    if length > 1:
                        self.context_counts[ngram[:-1]] += 1

        self.vocabulary = {
            token for token in self.unigram_counts if token not in {START_TOKEN, END_TOKEN}
        }
        self.vocabulary = {
            token for token in self.vocabulary if self.unigram_counts[token] >= self.min_word_frequency
        }
        self.candidates = sorted(self.vocabulary)

    def __contains__(self, token: str) -> bool:
        return token in self.vocabulary

    def unigram_count(self, word: str) -> int:
        return self.unigram_counts.get(word, 0)

    def ngram_count(self, ngram: tuple[str, ...]) -> int:
        return self.ngram_counts.get(ngram, 0)

    def map_tokens(self, tokens: list[str]) -> list[str]:
        return [token if token in self.vocabulary else UNK_TOKEN for token in tokens]
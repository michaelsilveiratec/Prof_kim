from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from src.config import (
    DEFAULT_LAPLACE_ALPHA,
    DEFAULT_NGRAM_ORDER,
    END_TOKEN,
    HIGHER_ORDER_WEIGHT,
    LOWER_ORDER_WEIGHT,
    UNK_TOKEN,
)
from src.ngrams import NGramModel


@dataclass(frozen=True, slots=True)
class Prediction:
    word: str
    probability: float
    confidence_pct: float


class BayesianWordPredictor:
    """Preditor de próxima palavra baseado em n-grams + suavização de Laplace."""

    def __init__(
        self,
        ngram_order: int = DEFAULT_NGRAM_ORDER,
        alpha: float = DEFAULT_LAPLACE_ALPHA,
        higher_order_weight: float = HIGHER_ORDER_WEIGHT,
        lower_order_weight: float = LOWER_ORDER_WEIGHT,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (
            higher_order_weight < 0
            or lower_order_weight < 0
            or not isclose(
                higher_order_weight + lower_order_weight,
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("higher_order_weight + lower_order_weight must add up to 1")
        self.ngram_order = ngram_order
        self.alpha = alpha
        self.higher_order_weight = higher_order_weight
        self.lower_order_weight = lower_order_weight
        self.model = NGramModel(order=ngram_order)
        self.candidates: list[str] = []

    def fit(self, sentences: list[list[str]]) -> None:
        self.model.fit(sentences)
        self.candidates = sorted(self.model.vocabulary)

    def _tokenize_context(self, phrase: str) -> list[str]:
        from src.preprocessing import tokenize

        return tokenize(phrase)

    def next_word_distribution(self, context: list[str] | tuple[str, ...]) -> dict[str, float]:
        if not self.model.vocabulary:
            return {}

        vocab = sorted(self.model.vocabulary)
        total_unigrams = sum(self.model.unigram_counts.values())
        end_count = self.model.ngram_counts.get((END_TOKEN,), 0)
        prior_denominator = total_unigrams + end_count + self.alpha * (len(vocab) + 2)
        priors = {
            word: (self.model.unigram_counts[word] + self.alpha) / prior_denominator
            for word in vocab
        }
        priors[UNK_TOKEN] = self.alpha / prior_denominator
        priors[END_TOKEN] = (end_count + self.alpha) / prior_denominator
        if not context:
            evidence = sum(priors.values())
            return {word: probability / evidence for word, probability in priors.items()}

        context_tokens = list(context)
        max_prefix_len = min(len(context_tokens), max(1, self.ngram_order - 1))
        distributions: list[dict[str, float]] = []

        for prefix_len in range(max_prefix_len, 0, -1):
            context_key = tuple(context_tokens[-prefix_len:])
            count_context = self.model.context_counts.get(context_key, 0)
            if count_context <= 0:
                continue

            continuation_counts = {
                word: self.model.ngram_counts.get(context_key + (word,), 0)
                for word in vocab
            }
            continuation_counts = {
                word: count for word, count in continuation_counts.items() if count > 0
            }
            end_continuation_count = self.model.ngram_counts.get(
                context_key + (END_TOKEN,), 0
            )
            if not continuation_counts and not end_continuation_count:
                continue

            outcomes = {**continuation_counts, UNK_TOKEN: 0}
            if end_continuation_count:
                outcomes[END_TOKEN] = end_continuation_count
            conditional_denominator = count_context + self.alpha * len(outcomes)
            conditional = {
                word: (count + self.alpha) / conditional_denominator
                for word, count in outcomes.items()
            }

            all_contexts = [
                count
                for key, count in self.model.context_counts.items()
                if len(key) == prefix_len
            ]
            total_contexts = sum(all_contexts)
            context_probability = count_context / total_contexts if total_contexts else 0.0

            likelihoods: dict[str, float] = {}
            for word, probability in conditional.items():
                likelihoods[word] = probability * context_probability / priors[word]

            posterior_scores = {
                word: likelihoods[word] * priors[word] for word in outcomes
            }
            evidence = sum(posterior_scores.values())
            if evidence <= 0:
                continue
            distributions.append(
                {word: score / evidence for word, score in posterior_scores.items()}
            )

        if not distributions:
            evidence = sum(priors.values())
            return {word: probability / evidence for word, probability in priors.items()}

        if len(distributions) == 1:
            return distributions[0]

        higher_order, lower_order = distributions[0], distributions[1]
        combined = {
            word: self.higher_order_weight * higher_order.get(word, 0.0)
            + self.lower_order_weight * lower_order.get(word, 0.0)
            for word in set(higher_order) | set(lower_order)
        }
        evidence = sum(combined.values())
        return {word: score / evidence for word, score in combined.items()}

    def predict_word(self, phrase: str) -> Prediction:
        tokens = self._tokenize_context(phrase)
        context = tuple(tokens[-(self.ngram_order - 1):]) if self.ngram_order > 1 else ()
        distribution = self.next_word_distribution(context)
        word_distribution = {
            word: probability
            for word, probability in distribution.items()
            if word not in {UNK_TOKEN, END_TOKEN}
        }
        if not word_distribution:
            return Prediction(word=UNK_TOKEN, probability=0.0, confidence_pct=0.0)

        best_word, best_prob = max(word_distribution.items(), key=lambda item: item[1])
        return Prediction(word=best_word, probability=best_prob, confidence_pct=best_prob * 100)

    def predict(self, phrase: str, top_k: int = 1) -> list[Prediction]:
        tokens = self._tokenize_context(phrase)
        context = tuple(tokens[-(self.ngram_order - 1):]) if self.ngram_order > 1 else ()
        distribution = self.next_word_distribution(context)
        distribution = {
            word: probability
            for word, probability in distribution.items()
            if word not in {UNK_TOKEN, END_TOKEN}
        }
        if not distribution:
            if not self.candidates:
                return [Prediction(word=UNK_TOKEN, probability=0.0, confidence_pct=0.0)]
            options = self.candidates[:top_k]
            return [Prediction(word=word, probability=1.0 / len(options), confidence_pct=100.0 / len(options)) for word in options]

        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        top = ranked[:top_k]
        return [
            Prediction(
                word=word,
                probability=prob,
                confidence_pct=prob * 100,
            )
            for word, prob in top
        ]


class BayesianPredictor(BayesianWordPredictor):
    """Compatibilidade de API antiga do projeto."""

    def __init__(self, ngram_model: NGramModel, alpha: float = 1.0):
        super().__init__(ngram_order=ngram_model.n, alpha=alpha)
        self.model = ngram_model
        self.candidates = sorted(self.model.vocabulary)

    def predict_next_word(self, context_tokens: list[str]) -> tuple[str, float]:
        prediction = self.predict_word(" ".join(context_tokens))
        return prediction.word, prediction.confidence_pct

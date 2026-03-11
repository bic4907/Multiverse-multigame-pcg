from typing import List
import pandas as pd
import numpy as np

class TwoPairComparisonSet:
    def __init__(self, a: np.ndarray, b: np.ndarray, metadata=None):
        self.a = a
        self.b = b
        self.metadata = metadata


class ThreePairComparisonSet:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, metadata=None):
        self.a = a
        self.b = b
        self.c = c
        self.metadata = metadata


class TwoPairComparisonScore:
    def __init__(self, score: float, metadata=None):
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return f"TwoPairComparisonScore(score={self.score})"

class ThreePairComparisonScore:
    def __init__(self, score_ac: float, score_bc: float, metadata=None):
        self.score_ac = score_ac
        self.score_bc = score_bc
        self.metadata = metadata

    @property
    def score(self):
        return self.score_ac * 0.5 + self.score_bc * 0.5

    def __repr__(self):
        return f"ThreePairComparisonScore(score_ab={self.score_ac}, score_ac={self.score_bc}, score={self.score})"

class TwoPairComparisonBatch:
    def __init__(self, batch: List[TwoPairComparisonScore]):
        self.batch = batch

    def __repr__(self):
        return f"TwoPairComparisonBatch(size={len(self.batch)})"

    @property
    def scores(self):
        return np.array([item.score for item in self.batch])

    @property
    def metadatas(self):
        return [item.metadata for item in self.batch]

    def to_dataframe(self):
        rows = list()

        for metadata, score in zip(self.metadatas, self.scores):
            game, text, ratio = metadata
            row = {
                "game": game,
                "text": text,
                "ratio": ratio,
                "score": score,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def __add__(self, other):
        return TwoPairComparisonBatch(self.batch + other.batch)

class ThreePairComparisonBatch:
    def __init__(self, batch: List[ThreePairComparisonScore]):
        self.batch = batch

    def __repr__(self):
        return f"ThreePairComparisonBatch(size={len(self.batch)})"

    @property
    def scores_ac(self):
        return np.array([item.score_ac for item in self.batch])

    @property
    def scores_bc(self):
        return np.array([item.score_bc for item in self.batch])

    @property
    def scores(self):
        return np.array([item.score for item in self.batch])

    @property
    def metadatas(self):
        return [item.metadata for item in self.batch]

    def to_dataframe(self):
        rows = list()

        for metadata, score_ac, score_bc, score in zip(
            self.metadatas, self.scores_ac, self.scores_bc, self.scores
        ):
            level_a_meta, level_b_meta = metadata
            game_a, text_a, ratio_a = level_a_meta
            game_b, text_b, ratio_b = level_b_meta

            row = {
                "game_a": game_a,
                "text_a": text_a,
                "ratio_a": ratio_a,
                "game_b": game_b,
                "text_b": text_b,
                "ratio_b": ratio_b,
                "score_ac": score_ac,
                "score_bc": score_bc,
                "score": score,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def __add__(self, other):
        return ThreePairComparisonBatch(self.batch + other.batch)
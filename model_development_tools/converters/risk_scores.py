import numpy as np


def pd_to_score(
        pd: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return intercept + coefficient * np.log(pd / (1 - pd))


def score_to_pd(
        score: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return 1 / (1 + np.exp(-intercept - coefficient * score))


def pd_to_scaled_score(
        pd: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return 200 + 20 * np.log2((1 + np.exp(-intercept) * ((1 - pd) / pd) ** coefficient) / 50)


def scaled_score_to_pd(
        scaled_score: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return 1 / (1 + np.exp(-intercept) * (50 * 2 ** ((scaled_score - 200) / 20) - 1) ** coefficient)


def score_to_scaled_score(
        score: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return 200 + 20 * np.log2((1 + np.exp(-intercept - coefficient * score)) / 50)


def scaled_score_to_score(
        scaled_score: float,
        intercept: float = 0,
        coefficient: float = 1,
) -> float:
    return intercept + coefficient * np.log(1 / (50 * (2 ** ((scaled_score - 200) / 20)) - 1))

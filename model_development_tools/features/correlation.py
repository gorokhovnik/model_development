from typing import Union, List, Dict, Tuple

import pandas as pd


def get_highly_correlated_features(
        data: pd.DataFrame,
        features: List[str],
        correlation_cutoff: float = 0.95,
        fraction: float = 1,
        corr_method: str = 'pearson',
        return_correlation_matrix: bool = False,
        random_state: int = 16777216,
) -> Union[Dict[str, str], Tuple[Dict[str, str], pd.DataFrame]]:
    correlation_matrix = data.sample(frac=fraction, random_state=random_state)[features].corr(corr_method)

    correlated = {}
    for i in range(len(features)):
        col1 = features[i]
        for j in range(i + 1, len(features)):
            col2 = features[j]
            if abs(correlation_matrix[col1][col2]) > correlation_cutoff and col1 not in correlated:
                correlated.update({col2: col1})

    return (correlated, correlation_matrix) if return_correlation_matrix else correlated

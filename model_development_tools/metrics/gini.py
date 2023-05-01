from typing import Union, Optional, List
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def gini_score(
        y_true: Union[List[int], np.array, pd.Series],
        y_score: Union[List[float], np.array, pd.Series],
        average: str = "macro",
        sample_weight: Optional[Union[List[float], np.array, pd.Series]] = None,
) -> float:
    roc_auc = roc_auc_score(
        y_true=y_true,
        y_score=y_score,
        average=average,
        sample_weight=sample_weight
    )
    return 2 * roc_auc - 1


def gini_avg_score(
        y_true: Union[List[int], np.array, pd.Series],
        y_score: Union[List[float], np.array, pd.Series],
        average: str = "macro",
        sample_weight: Optional[Union[List[float], np.array, pd.Series]] = None,
        y_group: Optional[Union[List[int], np.array, pd.Series]] = None,
) -> float:
    if y_group is None:
        score = gini_score(y_true,
                           y_score,
                           average=average,
                           sample_weight=sample_weight)
    else:
        groups = np.unique(y_group)
        n = len(groups) if sample_weight is None else sum(sample_weight)
        score = 0
        for group in groups:
            grp = (y_group == group)
            y_true_grp = np.array(y_true)[grp]
            if np.max(y_true_grp) != np.min(y_true_grp):
                sample_weight_ = None if sample_weight is None else np.array(sample_weight)[grp]
                divider = n if sample_weight is None else n / sum(sample_weight_)
                if sample_weight is None:
                    score += gini_score(y_true=y_true_grp,
                                        y_score=np.array(y_score)[grp],
                                        average=average,
                                        sample_weight=sample_weight_) / divider

    return score

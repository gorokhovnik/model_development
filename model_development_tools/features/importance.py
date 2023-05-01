from typing import Optional, Union, List
from collections.abc import Iterable

import pandas as pd

import lightgbm as lgb

from model_development_tools.model_selection.cross_validation import StratifiedStratifiedKFold


def get_lgb_features_importance(
        data: pd.DataFrame,
        features: List[str],
        target: str,
        group_by: Optional[Union[str, List[str]]] = None,
        params: Optional[dict] = None,
        importance_type: str = 'gain',
        cv: Union[int, Iterable] = 5,
) -> pd.Series:
    if group_by is not None:
        group_by = data[group_by]
    if params is None:
        params = {}

    use_early_stop = False
    for early_stop in ('early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change'):
        if early_stop in params:
            if params[early_stop] > 0:
                use_early_stop = True

    if isinstance(cv, int):
        cv = [i for i in StratifiedStratifiedKFold(cv, True, 16777216).split(data[features],
                                                                             data[target],
                                                                             group_by)]
    elif not isinstance(cv, Iterable):
        cv = [i for i in cv.split(data[features], data[target], group_by)]

    features_importance = pd.DataFrame()

    for itr, iva in cv:
        xtr, ytr = data.iloc[itr][features], data.iloc[itr][target]
        trn_data = lgb.Dataset(xtr, ytr)
        if use_early_stop:
            xva = data.iloc[iva][features]
            yva = data.iloc[iva][target]
            val_data = lgb.Dataset(xva, yva)
            valid_sets = [trn_data, val_data]
        else:
            valid_sets = None

        model = lgb.train(params,
                          trn_data,
                          valid_sets=valid_sets,
                          verbose_eval=False)

        fold_importance = pd.DataFrame()
        fold_importance['feature'] = features
        fold_importance['importance'] = model.feature_importance(importance_type=importance_type)

        features_importance = pd.concat([features_importance, fold_importance])

    features_importance = features_importance.groupby('feature').mean()['importance']
    return features_importance

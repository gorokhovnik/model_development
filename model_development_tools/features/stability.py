from typing import Callable, Optional, Union, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance

from model_development_tools.utils import pool_map
from model_development_tools.metrics.gini import gini_score


class Stability:
    def __init__(
            self,
            data: pd.DataFrame,
            features: List[str],
    ) -> None:
        self.__data = data
        self.__features = features

    def _feature_fall_and_sign_change(
            self,
            df: pd.DataFrame,
    ) -> str:
        feature = df.columns[0]
        score = pd.DataFrame(df.groupby(self.__stable_by).apply(
            lambda group: pd.Series({'score': self.__metrics(group[self.__target], group[feature])})))
        score.dropna(inplace=True)

        score['base_score'] = self.__metrics(df[self.__target], df[feature])
        score['fell'] = (score['base_score'] / score['score'] > self.__decrease_limit) * 1
        score['changed_sign'] = (score['score'] * score['base_score'] < 0) * 1
        fell = np.sum(score['fell'])
        changed_sign = np.sum(score['changed_sign'])

        if changed_sign > self.__changed_sign_limit:
            return 'changed'
        if fell > self.__fell_limit:
            return 'fell'
        if fell + changed_sign > self.__total_limit:
            return 'total'

        if len(self.__group_by) > 0:
            base_score = pd.DataFrame(df.groupby(self.__group_by).apply(
                lambda group: pd.Series({'base_score': self.__metrics(group[self.__target], group[feature])})))
            score = pd.DataFrame(df.groupby(self.__stable_by + self.__group_by).apply(
                lambda group: pd.Series({'score': self.__metrics(group[self.__target], group[feature])})))
            score.dropna(inplace=True)

            score = score.reset_index().merge(right=base_score.reset_index(), how='left', on=self.__group_by)
            score['fell'] = (score['base_score'] / score['score'] > self.__decrease_limit) * 1
            score['changed_sign'] = (score['score'] * score['base_score'] < 0) * 1
            score = score[self.__group_by + ['fell', 'changed_sign']].groupby(self.__group_by).sum()
            max_fell = np.max(score['fell'])
            max_changed_sign = np.max(score['changed_sign'])
            max_total = np.max(score['fell'] + score['changed_sign'])

            if max_changed_sign > self.__max_changed_sign_limit:
                return 'changed'
            if max_fell > self.__max_fell_limit:
                return 'fell'
            if max_total > self.__max_total_limit:
                return 'total'

        return 'ok'

    def features_fall_and_sign_change(
            self,
            target: str,
            stable_by: Union[str, List[str]],
            group_by: Optional[Union[str, List[str]]] = None,
            metrics: Callable = gini_score,
            decrease_limit: float = 3,
            fell_limit: int = 0,
            changed_sign_limit: int = 0,
            total_limit: int = np.inf,
            max_fell_limit: int = 0,
            max_changed_sign_limit: int = 0,
            max_total_limit: int = np.inf,
            n_threads: int = 1,
    ) -> Tuple[List[str], List[str], List[str]]:
        self.__target = target
        self.__stable_by = stable_by if isinstance(stable_by, list) else [stable_by]
        self.__group_by = [] if group_by is None else group_by if isinstance(group_by, list) else [group_by]
        self.__metrics = metrics
        self.__decrease_limit = decrease_limit
        self.__changed_sign_limit = changed_sign_limit
        self.__fell_limit = fell_limit
        self.__total_limit = total_limit
        self.__max_changed_sign_limit = max_changed_sign_limit
        self.__max_fell_limit = max_fell_limit
        self.__max_total_limit = max_total_limit
        norm_sign_fall_feat = pool_map(
            func=self._feature_fall_and_sign_change,
            iterable=[
                self.__data[[feature, self.__target] + self.__stable_by + self.__group_by]
                for feature in self.__features],
            n_threads=n_threads
        )

        features_fell = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 'fell']
        features_changed_sign = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 'changed']
        features_total = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 'total']

        return features_fell, features_changed_sign, features_total

    def _feature_population_stability_by_time(
            self,
            df: pd.DataFrame,
    ) -> float:
        df.dropna(inplace=True)
        feature = df.columns[-1]
        dfs = []
        mins = []
        maxs = []
        max_ws = -np.inf
        for v in self.__stable_by_val:
            dfs += [df[df[self.__stable_by] == v][feature].to_numpy()]
            if self.__minmax_scale is not None:
                q = np.quantile(dfs[-1], [self.__minmax_scale, 1 - self.__minmax_scale])
                mins += [q[0]]
                maxs += [q[1]]
        for idx in range(len(self.__stable_by_val) - 1):
            if self.__minmax_scale is not None:
                cur_min = np.min(mins[idx:idx + 2])
                cur_max = np.max(maxs[idx:idx + 2])
                cur = self.__metrics((dfs[idx] - cur_min) / (cur_max - cur_min),
                                     (dfs[idx + 1] - cur_min) / (cur_max - cur_min))
            else:
                cur = self.__metrics(dfs[idx], dfs[idx + 1])
            if cur > max_ws:
                max_ws = cur
        return max_ws

    def features_not_stable_population_by_time(
            self,
            stable_by: Union[str, List[str]],
            metrics: Callable = wasserstein_distance,
            cutoff: float = 0.01,
            minmax_scale: float = 0.001,
            return_all_distances: bool = False,
            n_threads: int = 1,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        self.__stable_by = stable_by
        self.__stable_by_val = np.unique(self.__data[stable_by])
        self.__metrics = metrics
        self.__minmax_scale = minmax_scale

        distances = pool_map(
            func=self._feature_population_stability_by_time,
            iterable=[self.__data[[stable_by, feature]]
                      for feature in self.__features],
            n_threads=n_threads
        )
        features_not_stable = [feature for feature, distance in zip(self.__features, distances) if distance > cutoff]

        return (features_not_stable, distances) if return_all_distances else features_not_stable

    def _feature_population_stability(
            self,
            df_dfmodel: pd.DataFrame,
    ) -> float:
        df = df_dfmodel[0].dropna()
        feat = df.columns[-1]
        df_model = df_dfmodel[1].dropna().to_numpy()
        if self.__minmax_scale is not None:
            q_model = np.quantile(df_model, [self.__minmax_scale, 1 - self.__minmax_scale])
        max_ws = 0
        for v in self.__stable_by_val:
            dfs = df[df[self.__stable_by] == v][feat].to_numpy()
            if self.__minmax_scale is not None:
                q = np.quantile(dfs, [self.__minmax_scale, 1 - self.__minmax_scale])
                cur_min = np.min([q[0], q_model[0]])
                cur_max = np.max([q[1], q_model[1]])
                cur = self.__metrics((df_model - cur_min) / (cur_max - cur_min), (dfs - cur_min) / (cur_max - cur_min))
            else:
                cur = self.__metrics(df_model, dfs)
            if cur > max_ws:
                max_ws = cur
        return max_ws

    def _feature_population_stability_(
            self,
            df_dfmodel: pd.DataFrame,
    ) -> float:
        df = df_dfmodel[0].dropna()
        df_model = df_dfmodel[1].dropna()
        if self.__minmax_scale is not None:
            q_model = np.quantile(df_model, [self.__minmax_scale, 1 - self.__minmax_scale])
            q = np.quantile(df, [self.__minmax_scale, 1 - self.__minmax_scale])
            cur_min = np.min([q[0], q_model[0]])
            cur_max = np.max([q[1], q_model[1]])
            max_ws = self.__metrics((df_model - cur_min) / (cur_max - cur_min), (df - cur_min) / (cur_max - cur_min))
        else:
            max_ws = self.__metrics(df_model, df)
        return max_ws

    def features_not_stable_population(
            self,
            data_full: pd.DataFrame,
            stable_by: Union[str, List[str]] = None,
            metrics: Callable = wasserstein_distance,
            cutoff: float = 0.01,
            minmax_scale: float = 0.001,
            return_all_distances: bool = False,
            n_threads: int = 1
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        self.__data_full = data_full
        self.__stable_by = stable_by
        if stable_by is not None:
            self.__stable_by_val = np.unique(self.__data[stable_by])
        self.__metrics = metrics
        self.__minmax_scale = minmax_scale

        if stable_by is not None:
            distances = pool_map(func=self._feature_population_stability,
                                 iterable=[(data_full[[self.__stable_by, feature]], self.__data[feature])
                                           for feature in self.__features],
                                 n_threads=n_threads)
        else:
            distances = pool_map(func=self._feature_population_stability_,
                                 iterable=[(data_full[feature], self.__data[feature])
                                           for feature in self.__features],
                                 n_threads=n_threads)

        features_not_stable = [feature for feature, distance in zip(self.__features, distances) if distance > cutoff]

        return (features_not_stable, distances) if return_all_distances else features_not_stable

from typing import Optional, Union, Iterable, Callable, List, Dict, Tuple

import numpy as np
import pandas as pd

from scipy.stats import mode

from model_development_tools.utils import pool_map
from model_development_tools.metrics.gini import gini_score, gini_avg_score


class Summary:
    def __init__(
            self,
            feature_description_function: Optional[Callable] = None,
            nunique_to_summary: bool = True,
            na_count_to_summary: bool = True,
            stats_to_summary: Optional[List[str]] = None,
            mode_freq_to_summary: bool = True,
            percentiles: Optional[Iterable[float]] = None,
            score_to_summary: bool = True,
            score_fillna_to_summary: bool = True,
            filler_to_summary: bool = True,
            metrics: Optional[Callable] = gini_score,
            score_avg_to_summary: bool = True,
            score_avg_fillna_to_summary: bool = True,
            filler_avg_to_summary: bool = True,
            avg_metrics: Optional[Callable] = gini_avg_score,
    ) -> None:
        self.__feature_description_function = feature_description_function
        self.__nunique_to_summary = nunique_to_summary
        self.__na_count_to_summary = na_count_to_summary
        self.__mode_freq_to_summary = mode_freq_to_summary
        self.__percentiles = [] if percentiles is None else percentiles
        self.__score_to_summary = score_to_summary and metrics is not None
        self.__score_fillna_to_summary = score_fillna_to_summary
        self.__metrics = metrics
        self.__score_avg_to_summary = score_avg_to_summary and avg_metrics is not None
        self.__score_avg_fillna_to_summary = score_avg_fillna_to_summary
        self.__avg_metrics = avg_metrics
        self.__filler_to_summary = filler_to_summary and score_fillna_to_summary
        self.__filler_avg_to_summary = filler_avg_to_summary and score_avg_fillna_to_summary
        self.__stats_list = ['min', 'max', 'mode', 'mean', 'median']
        self.__stats_fill_modifier = {'min': -1, 'max': 1}
        self.__stats_to_summary = set(self.__stats_list if stats_to_summary is None else stats_to_summary)

    def __extract_feat_targ_grby(
            self,
            df_: pd.DataFrame,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        feat = df_.columns[0]

        if len(df_.columns) > 1:
            targ = df_.columns[1]
        else:
            self.__score_to_summary, self.__score_fillna_to_summary = False, False
            self.__score_avg_to_summary, self.__score_avg_fillna_to_summary = False, False

            return feat, None, None

        if len(df_.columns) > 2:
            grby = df_.columns[2]
        else:
            self.__score_avg_to_summary, self.__score_avg_fillna_to_summary = False, False

            return feat, targ, None

        return feat, targ, grby

    def __calc_stats_value(
            self,
            col: pd.Series,
            stats: str,
    ) -> float:
        any_fillna_to_summary = self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary
        if stats in self.__stats_to_summary or any_fillna_to_summary:
            if stats == 'min':
                return col.min()
            elif stats == 'max':
                return col.max()
            elif stats == 'median':
                return col.median()
            elif stats == 'mean':
                return col.astype(np.float64).mean()
            elif stats == 'mode':
                return mode(col, nan_policy='omit')[0][0] if sum(col.isna()) != len(col) else np.nan

    def __score_dict(
            self,
            df_: pd.DataFrame,
            feat: str,
            targ: str,
            grby: str,
            _mode_freq_: float,
            _na_count_: int,
            dfs_filled: Dict[str, pd.DataFrame],
            score_to_summary: bool,
            score_fillna_to_summary: bool,
            is_avg: bool = False,
    ) -> Dict[str, List[Union[str, float]]]:
        suffix = '_avg' if is_avg else ''
        metrics = self.__avg_metrics if is_avg else self.__metrics

        score_summary = {}
        if score_fillna_to_summary:
            if _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                if is_avg:
                    score_wo_na = metrics(df_[targ], df_[feat], y_group=df_[grby])
                else:
                    score_wo_na = metrics(df_[targ], df_[feat])

                score_summary[f'score{suffix}_wo_na'] = [score_wo_na]
                if _na_count_ > 0:
                    best_score = 0
                    for stats in self.__stats_list:
                        if is_avg:
                            score = metrics(dfs_filled[stats][targ], dfs_filled[stats][feat], dfs_filled[stats][grby])
                        else:
                            score = metrics(dfs_filled[stats][targ], dfs_filled[stats][feat])
                        score_summary[f'score{suffix}_na_{stats}'] = [score]
                        if abs(best_score) < abs(score):
                            best_score = score
                            filler = stats
                else:
                    best_score = score_wo_na
                    for stats in self.__stats_list:
                        score_summary[f'score{suffix}_na_{stats}'] = best_score
                    filler = 'na'
            else:
                score_summary[f'score{suffix}_wo_na'] = [0]
                for stats in self.__stats_list:
                    score_summary[f'score{suffix}_na_{stats}'] = [0]
                best_score = 0
                filler = 'na'

        if score_to_summary:
            if score_fillna_to_summary:
                score_summary[f'score{suffix}'] = [best_score]
                if self.__filler_to_summary:
                    score_summary[f'filler{suffix}'] = [filler]
            elif _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                if is_avg:
                    score = metrics(df_[targ], df_[feat], y_group=df_[grby])
                else:
                    score = metrics(df_[targ], df_[feat])
                score_summary[f'score{suffix}'] = [score]
            else:
                score_summary[f'score{suffix}'] = [0]

        return score_summary

    def feature_summary(
            self,
            df_: pd.DataFrame,
    ) -> pd.DataFrame:
        feat, targ, grby = self.__extract_feat_targ_grby(df_)

        n = df_.shape[0]

        feat_summary = {'feature': [feat]}

        if self.__feature_description_function is not None:
            feat_summary['description'] = [self.__feature_description_function(feat)]

        stats_values = {}
        for stats in self.__stats_list:
            stats_values[stats] = self.__calc_stats_value(df_[feat], stats)
            if stats in self.__stats_to_summary:
                feat_summary[stats] = [stats_values[stats]]

        percentile_values = np.nanquantile(df_[feat], self.__percentiles)
        for p, pv in zip(self.__percentiles, percentile_values):
            feat_summary[f'{p * 100}%'] = pv

        if self.__mode_freq_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary or \
                self.__score_to_summary or self.__score_avg_to_summary:
            _mode_freq_ = df_[feat].value_counts(dropna=False).max() / n
        if self.__mode_freq_to_summary:
            feat_summary['mode_freq'] = [_mode_freq_]

        if self.__nunique_to_summary:
            feat_summary['nunique'] = [df_[feat].nunique()]
        if self.__na_count_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary or \
                self.__score_to_summary or self.__score_avg_to_summary:
            _na_count_ = df_[feat].isna().sum()
            if self.__na_count_to_summary:
                feat_summary['na_count'] = [_na_count_]

        dfs_filled = {}
        if self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            if _na_count_ > 0:
                for stats in self.__stats_list:
                    dfs_filled[stats] = df_.fillna(stats_values[stats] + self.__stats_fill_modifier.get(stats, 0))
                df_ = df_.dropna()
        elif self.__score_to_summary or self.__score_avg_to_summary:
            df_ = df_.dropna()

        feat_summary.update(
            self.__score_dict(
                df_, feat, targ, grby, _mode_freq_, _na_count_,
                dfs_filled, self.__score_to_summary, self.__score_fillna_to_summary, is_avg=False
            )
        )

        feat_summary.update(
            self.__score_dict(
                df_, feat, targ, grby, _mode_freq_, _na_count_,
                dfs_filled, self.__score_avg_to_summary, self.__score_avg_fillna_to_summary, is_avg=True
            )
        )

        return pd.DataFrame(feat_summary).set_index('feature')

    def features_summary(
            self,
            df: pd.DataFrame,
            features_list: List[str],
            target_name: Optional[str] = None,
            groupby_name: Optional[str] = None,
            n_threads=1,
    ) -> pd.DataFrame:
        df = df.fillna(np.nan)

        cols = ([target_name] + ([groupby_name] if groupby_name is not None else []) if target_name is not None else [])

        feats_summary = pd.concat(
            pool_map(
                func=self.feature_summary,
                iterable=[df[[feature] + cols] for feature in features_list],
                n_threads=n_threads
            )
        )

        return feats_summary

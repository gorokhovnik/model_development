import numpy as np
import pandas as pd
from scipy.stats import mode, wasserstein_distance
import random
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import datetime
import time
import os
import re
from itertools import product
from functools import wraps, partial
import lightgbm as lgb


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def pool_map(func,
             iterable,
             n_threads=1):
    if n_threads <= 1:
        res = [func(i) for i in iterable]
    else:
        p = Pool(n_threads)
        res = p.map(func=func,
                    iterable=iterable)
        p.close()
    return res


def unpack_dict(d):
    list_ = []
    for l_ in d.values():
        list_ += l_
    return list_


def grepl(pattern, array):
    grep_arr = []
    for string in array:
        if re.search(pattern, string) is not None:
            grep_arr += [string]
    return grep_arr


def gini_score(y_true,
               y_score,
               average="macro",
               sample_weight=None):
    return 2 * roc_auc_score(y_true=y_true,
                             y_score=y_score,
                             average=average,
                             sample_weight=sample_weight) - 1


def gini_avg_score(y_true,
                   y_score,
                   average="macro",
                   sample_weight=None,
                   y_group=None):
    if y_group is None:
        score = gini_score(y_true,
                           y_score,
                           average=average,
                           sample_weight=sample_weight)
    elif sample_weight is None:
        groups = np.unique(y_group)
        n = len(groups)
        score = 0
        for group in groups:
            grp = (y_group == group)
            y_true_grp = np.array(y_true)[grp]
            if np.max(y_true_grp) != np.min(y_true_grp):
                score += gini_score(y_true=y_true_grp,
                                    y_score=np.array(y_score)[grp],
                                    average=average) / n
    else:
        groups = np.unique(y_group)
        n = sum(sample_weight)
        score = 0
        for group in groups:
            grp = (y_group == group)
            sample_weight_ = np.array(sample_weight)[grp]
            y_true_grp = np.array(y_true)[grp]
            if max(y_true_grp) != min(y_true_grp):
                score += gini_score(y_true=y_true_grp,
                                    y_score=np.array(y_score)[grp],
                                    average=average,
                                    sample_weight=sample_weight_) * sum(sample_weight_) / n
    return score


class Log:
    def __init__(self,
                 digits=3,
                 exception_to_log=True):
        self.c = 0
        self.digits = digits
        self.exception_to_log = exception_to_log

        if 'logs' not in os.listdir():
            os.mkdir('logs')

        self.log_dirname = 'logs/log_' + time.strftime('%y_%m_%d_%H_%M_%S')
        os.mkdir(self.log_dirname)

        self.total_start_time = time.time()

    def __log_number(self):
        self.c += 1
        str_c = str(self.c)
        return '0' * (self.digits - len(str_c)) + str_c + '_'

    def log_decorator(self,
                      cell):
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                start_time = time.time()
                log_filename = self.log_dirname + '/' + self.__log_number() + re.sub('\W', '',
                                                                                     re.sub(' ', '_',
                                                                                            cell.lower())) + '.txt'
                log_file = open(log_filename, 'a')

                if self.exception_to_log:
                    try:
                        log_file.write(f(*args, **kwargs))
                    except Exception as e:
                        log_file.write(repr(e) + '\n')
                else:
                    log_file.write(f(*args, **kwargs))

                log_file.write('\nTime: ' + time.strftime('%y/%m/%d %H:%M:%S\n'))
                log_file.write('Spend on ' + cell[0].lower() + cell[1:] + ': ' + str(
                    datetime.timedelta(seconds=time.time() - start_time)) + '\n')
                log_file.write('Spend total: ' + str(datetime.timedelta(seconds=time.time() - self.total_start_time)))
                log_file.close()

            return decorated

        return decorator


class StratifiedStratifiedKFold(StratifiedKFold):
    def split(self,
              X,
              y,
              groups=None):
        if groups is None:
            for spl in super().split(X, y):
                yield spl
        else:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            folds = {i: [] for i in range(self.n_splits)}

            X = pd.Series([i for i in range(X.shape[0])])
            y = pd.Series(y)
            groups = pd.DataFrame(groups)
            X_groups = pd.concat([X, y, groups], axis=1)
            group_cols = ['g' + str(i) for i in range(groups.shape[1] + 1)]
            X_groups.columns = ['X'] + group_cols

            idx = 0
            for group in X_groups.groupby(group_cols):
                group = group[1]
                indices = group['X'].to_numpy()
                if self.shuffle:
                    np.random.shuffle(indices)
                for i in indices:
                    folds[idx] += [i]
                    idx += 1
                    idx %= self.n_splits

            for i in range(self.n_splits):
                yield np.sort(np.concatenate([np.array(folds[j]) for j in range(self.n_splits) if i != j])), \
                      np.sort(np.array(folds[i]))


class MemoryReducer:
    def __init__(self,
                 float_min_type=16,
                 int_min_type=8):
        self.int8_min = np.iinfo(np.int8).min
        self.int8_max = np.iinfo(np.int8).max
        self.int16_min = np.iinfo(np.int16).min
        self.int16_max = np.iinfo(np.int16).max
        self.int32_min = np.iinfo(np.int32).min
        self.int32_max = np.iinfo(np.int32).max
        self.float16_min = np.finfo(np.float16).min
        self.float16_max = np.finfo(np.float16).max
        self.float32_min = np.finfo(np.float32).min
        self.float32_max = np.finfo(np.float32).max
        self.__float_min_type = float_min_type
        self.__int_min_type = int_min_type

    def shrink_column(self,
                      col):
        is_int = col.dtypes.name[:3] == 'int'
        is_float = col.dtypes.name[:3] == 'flo'

        if is_int:
            c_min = col.min()
            c_max = col.max()
            if self.__int_min_type <= 8 and c_min > self.int8_min and c_max < self.int8_max:
                col = col.astype(np.int8)
            elif self.__int_min_type <= 16 and c_min > self.int16_min and c_max < self.int16_max:
                col = col.astype(np.int16)
            elif self.__int_min_type <= 32 and c_min > self.int32_min and c_max < self.int32_max:
                col = col.astype(np.int32)
        elif is_float:
            c_min = col.min()
            c_max = col.max()
            if self.__float_min_type <= 16 and c_min > self.float16_min and c_max < self.float16_max:
                col = col.astype(np.float16)
            elif self.__float_min_type <= 32 and c_min > self.float32_min and c_max < self.float32_max:
                col = col.astype(np.float32)
        return col

    def reduce(self,
               df,
               n_threads=1,
               verbose=0):
        if verbose > 0:
            start_mem = df.memory_usage().sum() / 1024 ** 2
            print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        df = pd.concat(pool_map(func=self.shrink_column,
                                iterable=[df[col] for col in df.columns],
                                n_threads=n_threads),
                       axis=1)

        if verbose > 0:
            end_mem = df.memory_usage().sum() / 1024 ** 2
            print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
            print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df


class Summary:
    def __init__(self,
                 feature_description_function=None,
                 nunique_to_summary=True,
                 na_count_to_summary=True,
                 mode_to_summary=True,
                 mode_freq_to_summary=True,
                 min_to_summary=True,
                 max_to_summary=True,
                 mean_to_summary=True,
                 median_to_summary=True,
                 percentiles=None,
                 score_to_summary=True,
                 score_fillna_to_summary=True,
                 metrics=gini_score,
                 score_avg_to_summary=True,
                 score_avg_fillna_to_summary=True,
                 avg_metrics=gini_avg_score):
        self.__feature_description_function = feature_description_function
        self.__nunique_to_summary = nunique_to_summary
        self.__na_count_to_summary = na_count_to_summary
        self.__mode_to_summary = mode_to_summary
        self.__mode_freq_to_summary = mode_freq_to_summary
        self.__min_to_summary = min_to_summary
        self.__max_to_summary = max_to_summary
        self.__mean_to_summary = mean_to_summary
        self.__median_to_summary = median_to_summary
        self.__percentiles = [] if percentiles is None else percentiles
        self.__score_to_summary = score_to_summary
        self.__score_fillna_to_summary = score_fillna_to_summary
        self.__metrics = metrics
        self.__score_avg_to_summary = score_avg_to_summary
        self.__score_avg_fillna_to_summary = score_avg_fillna_to_summary
        self.__avg_metrics = avg_metrics

    def feature_summary(self,
                        df_):
        feat = df_.columns[0]
        if len(df_.columns) > 1:
            targ = df_.columns[1]
        else:
            self.__score_to_summary = False
            self.__score_fillna_to_summary = False
        if len(df_.columns) > 2:
            grby = df_.columns[2]
        else:
            self.__score_avg_to_summary = False
            self.__score_avg_fillna_to_summary = False
        n = df_.shape[0]

        feat_summary = {'feature': [feat]}
        if self.__feature_description_function is not None:
            feat_summary['description'] = [self.__feature_description_function(feat)]

        if self.__nunique_to_summary:
            feat_summary['nunique'] = [df_[feat].nunique()]
        if self.__na_count_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary or \
                self.__score_to_summary or self.__score_avg_to_summary:
            _na_count_ = df_[feat].isna().sum()
        if self.__na_count_to_summary:
            feat_summary['na_count'] = [_na_count_]

        if self.__mode_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            _mode_ = mode(df_[feat])[0][0] if _na_count_ < n else np.nan
        if self.__mode_to_summary:
            feat_summary['mode'] = [_mode_]
        if self.__mode_freq_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary or \
                self.__score_to_summary or self.__score_avg_to_summary:
            _mode_freq_ = df_[feat].value_counts(dropna=False).max() / n
        if self.__mode_freq_to_summary:
            feat_summary['mode_freq'] = [_mode_freq_]

        if self.__min_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            _min_ = df_[feat].min()
        if self.__min_to_summary:
            feat_summary['min'] = [_min_]
        if self.__max_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            _max_ = df_[feat].max()
        if self.__max_to_summary:
            feat_summary['max'] = [_max_]
        if self.__mean_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            _mean_ = df_[feat].astype(np.float64).mean()
        if self.__mean_to_summary:
            feat_summary['mean'] = [_mean_]
        if self.__median_to_summary or self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            _median_ = df_[feat].median()
        if self.__median_to_summary:
            feat_summary['median'] = [_median_]

        quantiles = np.nanquantile(df_[feat], self.__percentiles)
        for p, q in zip(self.__percentiles, quantiles):
            feat_summary[str(p * 100) + '%'] = q

        if self.__score_fillna_to_summary or self.__score_avg_fillna_to_summary:
            if _na_count_ > 0:
                dfmin = df_.fillna(_min_ - 1)
                dfmax = df_.fillna(_max_ + 1)
                dfmode = df_.fillna(_mode_)
                dfmean = df_.fillna(_mean_)
                dfmedian = df_.fillna(_median_)
                df_ = df_.dropna()
        elif self.__score_to_summary or self.__score_avg_to_summary:
            df_ = df_.dropna()

        if self.__score_fillna_to_summary:
            if _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                feat_summary['score_wo_na'] = [self.__metrics(df_[targ], df_[feat])]
                if _na_count_ > 0:
                    best_score = 0
                    feat_summary['score_na_min'] = [self.__metrics(dfmin[targ], dfmin[feat])]
                    if abs(best_score) < abs(feat_summary['score_na_min'][0]):
                        best_score = feat_summary['score_na_min'][0]
                    feat_summary['score_na_max'] = [self.__metrics(dfmax[targ], dfmax[feat])]
                    if abs(best_score) < abs(feat_summary['score_na_max'][0]):
                        best_score = feat_summary['score_na_max'][0]
                    feat_summary['score_na_mode'] = [self.__metrics(dfmode[targ], dfmode[feat])]
                    if abs(best_score) < abs(feat_summary['score_na_mode'][0]):
                        best_score = feat_summary['score_na_mode'][0]
                    feat_summary['score_na_mean'] = [self.__metrics(dfmean[targ], dfmean[feat])]
                    if abs(best_score) < abs(feat_summary['score_na_mean'][0]):
                        best_score = feat_summary['score_na_mean'][0]
                    feat_summary['score_na_median'] = [self.__metrics(dfmedian[targ], dfmedian[feat])]
                    if abs(best_score) < abs(feat_summary['score_na_median'][0]):
                        best_score = feat_summary['score_na_median'][0]
                else:
                    best_score = feat_summary['score_wo_na'][0]
                    feat_summary['score_na_min'] = best_score
                    feat_summary['score_na_max'] = best_score
                    feat_summary['score_na_mode'] = best_score
                    feat_summary['score_na_mean'] = best_score
                    feat_summary['score_na_median'] = best_score
            else:
                feat_summary['score_wo_na'] = [0]
                feat_summary['score_na_min'] = [0]
                feat_summary['score_na_max'] = [0]
                feat_summary['score_na_mode'] = [0]
                feat_summary['score_na_mean'] = [0]
                feat_summary['score_na_median'] = [0]
                best_score = 0

        if self.__score_to_summary:
            if self.__score_fillna_to_summary:
                feat_summary['score'] = [best_score]
            elif _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                feat_summary['score'] = [self.__metrics(df_[targ], df_[feat])]
            else:
                feat_summary['score'] = [0]

        if self.__score_avg_fillna_to_summary:
            if _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                feat_summary['score_avg_wo_na'] = [self.__avg_metrics(df_[targ], df_[feat], y_group=df_[grby])]
                if _na_count_ > 0:
                    best_score_avg = 0
                    feat_summary['score_avg_na_min'] = [
                        self.__avg_metrics(dfmin[targ], dfmin[feat], y_group=dfmin[grby])]
                    if abs(best_score_avg) < abs(feat_summary['score_avg_na_min'][0]):
                        best_score_avg = feat_summary['score_avg_na_min'][0]
                    feat_summary['score_avg_na_max'] = [
                        self.__avg_metrics(dfmax[targ], dfmax[feat], y_group=dfmax[grby])]
                    if abs(best_score_avg) < abs(feat_summary['score_avg_na_max'][0]):
                        best_score_avg = feat_summary['score_avg_na_max'][0]
                    feat_summary['score_avg_na_mode'] = [
                        self.__avg_metrics(dfmode[targ], dfmode[feat], y_group=dfmode[grby])]
                    if abs(best_score_avg) < abs(feat_summary['score_avg_na_mode'][0]):
                        best_score_avg = feat_summary['score_avg_na_mode'][0]
                    feat_summary['score_avg_na_mean'] = [
                        self.__avg_metrics(dfmean[targ], dfmean[feat], y_group=dfmean[grby])]
                    if abs(best_score_avg) < abs(feat_summary['score_avg_na_mean'][0]):
                        best_score_avg = feat_summary['score_avg_na_mean'][0]
                    feat_summary['score_avg_na_median'] = [
                        self.__avg_metrics(dfmedian[targ], dfmedian[feat], y_group=dfmedian[grby])]
                    if abs(best_score_avg) < abs(feat_summary['score_avg_na_median'][0]):
                        best_score_avg = feat_summary['score_avg_na_median'][0]
                else:
                    best_score_avg = feat_summary['score_avg_wo_na'][0]
                    feat_summary['score_avg_na_min'] = best_score_avg
                    feat_summary['score_avg_na_max'] = best_score_avg
                    feat_summary['score_avg_na_mode'] = best_score_avg
                    feat_summary['score_avg_na_mean'] = best_score_avg
                    feat_summary['score_avg_na_median'] = best_score_avg
            else:
                feat_summary['score_avg_wo_na'] = [0]
                feat_summary['score_avg_na_min'] = [0]
                feat_summary['score_avg_na_max'] = [0]
                feat_summary['score_avg_na_mode'] = [0]
                feat_summary['score_avg_na_mean'] = [0]
                feat_summary['score_avg_na_median'] = [0]
                best_score_avg = 0

        if self.__score_avg_to_summary:
            if self.__score_avg_fillna_to_summary:
                feat_summary['score_avg'] = [best_score_avg]
            elif _mode_freq_ != 1 and df_[targ].max() != df_[targ].min():
                feat_summary['score_avg'] = [self.__avg_metrics(df_[targ], df_[feat], y_group=df_[grby])]
            else:
                feat_summary['score_avg'] = [0]

        return pd.DataFrame(feat_summary).set_index('feature')

    def features_summary(self,
                         df,
                         features_list,
                         target_name=None,
                         groupby_name=None,
                         n_threads=1):
        df = df.fillna(np.nan)

        cols = ([target_name] + ([groupby_name] if groupby_name is not None else []) if target_name is not None else [])

        feats_summary = pd.concat(pool_map(func=self.feature_summary,
                                           iterable=[df[[feature] + cols] for feature in features_list],
                                           n_threads=n_threads))

        return feats_summary


class Stability:
    def __init__(self,
                 data,
                 features):
        self.__data = data
        self.__features = features

    def _feature_fall_and_sign_change(self,
                                      df):
        feature = df.columns[0]
        gini = pd.DataFrame(df.groupby(self.__stable_by).apply(
            lambda group: pd.Series({'gini': self.__metrics(group[self.__target], group[feature])})))
        gini.dropna(inplace=True)

        gini['base_gini'] = self.__metrics(df[self.__target], df[feature])
        gini['fell'] = (gini['base_gini'] / gini['gini'] > self.__decrease_limit) * 1
        gini['changed_sign'] = (gini['gini'] * gini['base_gini'] < 0) * 1
        fell = np.sum(gini['fell'])
        changed_sign = np.sum(gini['changed_sign'])

        if fell > self.__fell_limit:
            return 1
        if changed_sign > self.__changed_sign_limit:
            return 2
        if fell + changed_sign > self.__total_limit:
            return 3

        if len(self.__group_by) > 0:
            base_gini = pd.DataFrame(df.groupby(self.__group_by).apply(
                lambda group: pd.Series({'base_gini': self.__metrics(group[self.__target], group[feature])})))
            gini = pd.DataFrame(df.groupby(self.__stable_by + self.__group_by).apply(
                lambda group: pd.Series({'gini': self.__metrics(group[self.__target], group[feature])})))
            gini.dropna(inplace=True)

            gini = gini.reset_index().merge(right=base_gini.reset_index(), how='left', on=self.__group_by)
            gini['fell'] = (gini['base_gini'] / gini['gini'] > self.__decrease_limit) * 1
            gini['changed_sign'] = (gini['gini'] * gini['base_gini'] < 0) * 1
            gini = gini[self.__group_by + ['fell', 'changed_sign']].groupby(self.__group_by).sum()
            max_fell = np.max(gini['fell'])
            max_changed_sign = np.max(gini['changed_sign'])
            max_total = np.max(gini['fell'] + gini['changed_sign'])

            if max_fell > self.__max_fell_limit:
                return 1
            if max_changed_sign > self.__max_changed_sign_limit:
                return 2
            if max_total > self.__max_total_limit:
                return 3

        return 0

    def features_fall_and_sign_change(self,
                                      target,
                                      stable_by,
                                      group_by=None,
                                      metrics=gini_score,
                                      decrease_limit=3,
                                      fell_limit=0,
                                      changed_sign_limit=0,
                                      total_limit=np.inf,
                                      max_fell_limit=0,
                                      max_changed_sign_limit=0,
                                      max_total_limit=np.inf,
                                      n_threads=1):
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
        norm_sign_fall_feat = pool_map(func=self._feature_fall_and_sign_change,
                                       iterable=[
                                           self.__data[[feature, self.__target] + self.__stable_by + self.__group_by]
                                           for feature in self.__features],
                                       n_threads=n_threads)

        features_fell = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 1]
        features_changed_sign = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 2]
        features_total = [f for i, f in enumerate(self.__features) if norm_sign_fall_feat[i] == 3]

        return (features_fell, features_changed_sign, features_total)

    def _feature_population_stability_by_time(self,
                                              df):
        df.dropna(inplace=True)
        feature = df.columns[-1]
        dfs = []
        mins = []
        maxs = []
        max_ws = -np.inf
        for v in self.__stable_by_val:
            dfs += [df[df[self.__stable_by] == v][feature].to_numpy()]
            if self.__minmax_norm is not None:
                q = np.quantile(dfs[-1], [self.__minmax_norm, 1 - self.__minmax_norm])
                mins += [q[0]]
                maxs += [q[1]]
        for idx in range(len(self.__stable_by_val) - 1):
            if self.__minmax_norm is not None:
                cur_min = np.min(mins[idx:idx + 2])
                cur_max = np.max(maxs[idx:idx + 2])
                cur = self.__metrics((dfs[idx] - cur_min) / (cur_max - cur_min),
                                     (dfs[idx + 1] - cur_min) / (cur_max - cur_min))
            else:
                cur = self.__metrics(dfs[idx], dfs[idx + 1])
            if cur > max_ws:
                max_ws = cur
        return max_ws

    def features_not_stable_population_by_time(self,
                                               stable_by,
                                               metrics=wasserstein_distance,
                                               cutoff=0.01,
                                               minmax_norm=0.001,
                                               return_all_distances=False,
                                               n_threads=1):
        self.__stable_by = stable_by
        self.__stable_by_val = np.unique(stable_by)
        self.__metrics = metrics
        self.__minmax_norm = minmax_norm

        distances = pool_map(func=self._feature_population_stability_by_time,
                             iterable=[self.__data[[stable_by] + [feature]] for feature in self.__features],
                             n_threads=n_threads)
        features_not_stable = [feature for feature, distance in zip(self.__features, distances) if distance > cutoff]

        return (features_not_stable, distances) if return_all_distances else features_not_stable

    def _feature_population_stability(self,
                                      df_dfmodel):
        df = df_dfmodel[0].dropna()
        feat = df.columns[-1]
        df_model = df_dfmodel[1].dropna().to_numpy()
        if self.__minmax_norm is not None:
            q_model = np.quantile(df_model, [self.__minmax_norm, 1 - self.__minmax_norm])
        max_ws = 0
        for v in self.__stable_by_val:
            dfs = df[df[self.__stable_by] == v][feat].to_numpy()
            if self.__minmax_norm is not None:
                q = np.quantile(dfs, [self.__minmax_norm, 1 - self.__minmax_norm])
                cur_min = np.min([q[0], q_model[0]])
                cur_max = np.max([q[1], q_model[1]])
                cur = self.__metrics((df_model - cur_min) / (cur_max - cur_min), (dfs - cur_min) / (cur_max - cur_min))
            else:
                cur = self.__metrics(df_model, dfs)
            if cur > max_ws:
                max_ws = cur
        return np.max(max_ws)

    def features_not_stable_population(self,
                                       data_full,
                                       stable_by=None,
                                       metrics=wasserstein_distance,
                                       cutoff=0.01,
                                       minmax_norm=0.001,
                                       return_all_distances=False,
                                       n_threads=1):
        self.__data_full = data_full
        self.__stable_by = stable_by
        self.__stable_by_val = np.unique(stable_by)
        self.__metrics = metrics
        self.__minmax_norm = minmax_norm

        distances = pool_map(func=self._feature_population_stability,
                             iterable=[(data_full[[self.__stable_by] + [feature]], self.__data[feature])
                                       for feature in self.__features],
                             n_threads=n_threads)

        features_not_stable = [feature for feature, distance in zip(self.__features, distances) if distance > cutoff]

        return (features_not_stable, distances) if return_all_distances else features_not_stable


def get_lgb_features_importance(data,
                                features,
                                target,
                                group_by=None,
                                params=None,
                                importance_type='gain',
                                cv=StratifiedStratifiedKFold(5, True, 16777216)):
    if params is None:
        params = {}

    for early_stop in ('early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change'):
        if early_stop in params:
            if params[early_stop] > 0:
                use_early_stop = True

    features_importance = pd.DataFrame()

    for itr, iva in cv.split(data[features], data[target], data[group_by]):
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


def get_highly_correlated_features(data,
                                   features,
                                   correlation_cutoff=0.95,
                                   fraction=1,
                                   return_correlation_matrix=False,
                                   random_state=16777216):
    correlation_matrix = data.sample(frac=fraction, random_state=random_state)[features].corr()

    correlated = {}
    for i in range(len(features)):
        col1 = features[i]
        for j in range(i + 1, len(features)):
            col2 = features[j]
            if abs(correlation_matrix[col1][col2]) > correlation_cutoff and col1 not in correlated:
                correlated.update({col2: col1})

    return (correlated, correlation_matrix) if return_correlation_matrix else correlated


class FS:
    def __init__(self,
                 data,
                 features,
                 target,
                 group_by=None,
                 monotonic_dict=None,
                 estimator=LogisticRegression(),
                 predict_method='predict_proba',
                 boosting_params=None,
                 metrics=gini_score,
                 increase_metrics=True,
                 total_weight=0,
                 weights=None,
                 total_pos_diff=True,
                 pos_diff=None,
                 cv=StratifiedStratifiedKFold(5, True, 16777216),
                 use_cache=True,
                 initial_cache=None,
                 dropped_from_selection=None,
                 random_state=16777216,
                 verbose=0):
        self.__data = data
        self.__data.index = [i for i in range(data.shape[0])]
        self.__features = features
        self.__selected = []
        self.__target = target
        if group_by is not None:
            if isinstance(group_by, list):
                self.__group_by = group_by
            else:
                self.__group_by = [group_by]
            self.__groups = self.__data[self.__group_by]
        else:
            self.__group_by = []
            self.__groups = None

        if monotonic_dict is not None:
            self.__monotonic = monotonic_dict
            for feature in features:
                if feature not in self.__monotonic:
                    self.__monotonic[feature] = 0

        self.__estimator = estimator
        self.__predict_method = predict_method
        if boosting_params is not None:
            self.__boosting_params = boosting_params
            if estimator in ('lgb', 'lightgbm'):
                self.__boosting_params['verbosity'] = -1
            else:
                self.__boosting_params['verbosity'] = 0
            self.__use_early_stop = False
            for early_stop in ('early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change'):
                if early_stop in boosting_params:
                    if boosting_params[early_stop] > 0:
                        self.__use_early_stop = True

        self.__metrics = metrics
        self.__increase_metric = increase_metrics

        if weights is None or weights == []:
            self.__weights = [total_weight if total_weight != 0 else 1]
            for group in self.__group_by:
                for value in np.unique(self.__data[group]):
                    self.__weights += [1]
        else:
            if isinstance(weights, dict):
                self.__weights = [total_weight]
                for group in self.__group_by:
                    self.__weights += weights[group]
            else:
                self.__weights = [total_weight] + weights
        self.__weights = np.array(self.__weights) / sum(self.__weights)

        self.__is_pos_diff = False
        if pos_diff is None or pos_diff == []:
            self.__pos_diff = [total_pos_diff]
            for group in self.__group_by:
                for value in np.unique(self.__data[group]):
                    self.__pos_diff += [True]
        else:
            if isinstance(pos_diff, dict):
                self.__pos_diff = [total_pos_diff]
                for group in self.__group_by:
                    self.__pos_diff += pos_diff[group]
            else:
                self.__pos_diff = [total_pos_diff] + pos_diff
        for idx in range(len(self.__pos_diff)):
            if self.__pos_diff[idx] == 0 or not self.__pos_diff[idx]:
                self.__pos_diff[idx] = False
            else:
                self.__pos_diff[idx] = True
                self.__is_pos_diff = True
        self.__pos_diff = np.array(self.__pos_diff)

        self.__e_add, self.__e_drop = {}, {}

        self.__cv = cv
        self.__random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        self.__log_columns = ['Selection type', 'Action', 'Features', 'Weighted diff', 'Worst diff', 'TOTAL']
        self.__c = 5
        self.__group_idx = {}
        self.__group_target = {}
        for group in self.__group_by:
            for value in np.unique(self.__data[group]):
                self.__log_columns += [group + '_' + str(value)]
                self.__group_idx[group + '_' + str(value)] = self.__data[group] == value
                self.__group_target[group + '_' + str(value)] = self.__data[self.__data[group] == value][self.__target]
        self.__log = pd.DataFrame(columns=self.__log_columns)
        self.__genetic_log_columns = ['Iteration', 'Features', 'Weighted'] + self.__log_columns[self.__c:]
        self.__genetic_log = pd.DataFrame(columns=self.__genetic_log_columns)
        self.__top_n_log_columns = ['Selection type', 'N', 'Features', 'Weighted'] + self.__log_columns[self.__c:]
        self.__top_n_log = pd.DataFrame(columns=self.__top_n_log_columns)

        self.__use_cache = use_cache
        if use_cache:
            self.__cache = initial_cache if isinstance(initial_cache, dict) else {}
        self.__dropped_from_selection = dropped_from_selection if dropped_from_selection is not None else []

        self.__verbose = verbose

    def get_log(self):
        return self.__log

    def get_top_n_log(self):
        return self.__top_n_log

    def get_genetic_log(self,
                        wide=False):
        if wide:
            return self.__genetic_log
        else:
            return self.__genetic_log[['Iteration', 'Features', 'Weighted'] + self.__log_columns[self.__c:]]

    def get_selected(self):
        return self.__selected

    def get_cache(self):
        return self.__cache

    def _cv_predict(self,
                    features,
                    n_threads=1):
        if len(features) == 0:
            return np.array([np.mean(self.__data[self.__target])] * self.__data.shape[0])
        if self.__estimator in ('lgb', 'lightgbm'):
            self.__boosting_params['monotone_constraints'] = [self.__monotonic[feature] for feature in features]
            self.__boosting_params['nthread'] = n_threads
            pred = np.zeros(self.__data.shape[0])
            for itr, iva in self.__cv.split(self.__data[features],
                                            self.__data[self.__target],
                                            self.__groups):
                xtr, ytr = self.__data.iloc[itr][features], self.__data.iloc[itr][self.__target]
                trn_data = lgb.Dataset(xtr, ytr)
                xva = self.__data.iloc[iva][features]
                if self.__use_early_stop:
                    yva = self.__data.iloc[iva][self.__target]
                    val_data = lgb.Dataset(xva, yva)
                    valid_sets = [trn_data, val_data]
                else:
                    valid_sets = None

                model = lgb.train(self.__boosting_params,
                                  trn_data,
                                  valid_sets=valid_sets,
                                  verbose_eval=False)
                pred[iva] = model.predict(xva)
            return pred
        elif self.__predict_method == 'predict_proba':
            return cross_val_predict(estimator=self.__estimator,
                                     X=self.__data[features],
                                     y=self.__data[self.__target],
                                     groups=self.__groups,
                                     cv=self.__cv,
                                     method='predict_proba')[:, 1]
        elif self.__predict_method == 'predict':
            return cross_val_predict(estimator=self.__estimator,
                                     X=self.__data[features],
                                     y=self.__data[self.__target],
                                     groups=self.__groups,
                                     cv=self.__cv,
                                     method='predict')

    def _scores(self,
                features,
                n_threads=1):
        if self.__use_cache:
            cache_idx = ''.join(['T' if feature in features else 'F' for feature in self.__features])
            if self.__cache.get(cache_idx) is None:
                pred = self._cv_predict(features=features,
                                        n_threads=n_threads)
                scores = [self.__metrics(self.__data[self.__target], pred)]
                scores += [self.__metrics(self.__group_target[group], pred[self.__group_idx[group]])
                           for group in self.__group_idx]
                self.__cache[cache_idx] = scores
            else:
                scores = self.__cache[cache_idx]
        else:
            pred = self._cv_predict(features=features,
                                    n_threads=n_threads)
            scores = [self.__metrics(self.__data[self.__target], pred)]
            scores += [self.__metrics(self.__group_target[group], pred[self.__group_idx[group]])
                       for group in self.__group_idx]
        return np.array(scores)

    def __diff(self,
               scores,
               best_scores):
        weighted_diff = np.dot(scores - best_scores, self.__weights)
        if self.__is_pos_diff:
            worst_diff = np.min(scores[self.__pos_diff] - best_scores[self.__pos_diff])
        else:
            worst_diff = 0
        if not self.__increase_metric:
            weighted_diff *= -1
            worst_diff *= -1
        return weighted_diff, worst_diff

    def __sequential_forward_step(self,
                                  best_scores,
                                  selected,
                                  not_selected,
                                  n_threads=1):
        new_selected = []
        changes = False
        diff = 0
        log = []
        for feature in not_selected:
            scores = self._scores(selected + [feature], n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)

            if weighted_diff >= self.__e_add['f'] and worst_diff >= 0:
                changes = True
                best_scores = scores
                selected += [feature]
                new_selected += [feature]
                diff += weighted_diff
                log += [['Sequential Forward', 'ADDED ' + feature, ','.join(selected),
                         weighted_diff, worst_diff] + list(scores)]
                if self.__verbose > 0:
                    print(Color.GREEN + 'Sequential forward step')
                    print('ADDED ' + feature)
                    print('Features: ' + ', '.join(selected))
                    print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)
            else:
                log += [['Sequential Forward', 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                         weighted_diff, worst_diff] + list(scores)]
                if self.__verbose > 1:
                    print('Sequential forward step')
                    print('TRIED TO ADD ' + feature)
                    print('Features: ' + ', '.join(selected + [feature]))
                    print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        not_selected = [feature for feature in not_selected if feature not in new_selected]

        return changes, best_scores, selected, not_selected, diff, log

    def __sequential_backward_step(self,
                                   best_scores,
                                   selected,
                                   n_threads=1):
        save_selected = selected.copy()
        not_selected = []
        changes = False
        diff = 0
        log = []
        for feature in save_selected:
            tmp_selected = [f for f in selected if f != feature]
            scores = self._scores(tmp_selected, n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)

            if weighted_diff >= self.__e_drop['b'] and worst_diff >= 0:
                changes = True
                best_scores = scores
                selected = tmp_selected
                not_selected += [feature]
                diff += weighted_diff
                log += [['Sequential Backward', 'DROPPED ' + feature, ','.join(selected),
                         weighted_diff, worst_diff] + list(scores)]
                if self.__verbose > 0:
                    print(Color.RED + 'Sequential backward step')
                    print('DROPPED ' + feature)
                    print('Features: ' + ', '.join(selected))
                    print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)
            else:
                log += [['Sequential Backward', 'TRIED TO DROP ' + feature, ','.join(tmp_selected),
                         weighted_diff, worst_diff] + list(scores)]
                if self.__verbose > 1:
                    print('Sequential backward step')
                    print('TRIED TO DROP ' + feature)
                    print('Features: ' + ', '.join(tmp_selected))
                    print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        return changes, best_scores, selected, not_selected, diff, log

    def __all_forward_step(self,
                           best_scores,
                           selected,
                           not_selected,
                           n_threads=1):
        new_selected = []
        changes = False
        diff = 0
        log = []

        scores_ = pool_map(func=self._scores,
                           iterable=[selected + [feature] for feature in not_selected],
                           n_threads=n_threads)

        for idx, feature in enumerate(not_selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= self.__e_add['af'] and worst_diff >= 0:
                changes = True
                new_selected += [feature]
            log += [['All Forward', 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            if self.__verbose > 1:
                print('All forward step')
                print('TRIED TO ADD ' + feature)
                print('Features: ' + ', '.join(selected + [feature]))
                print(pd.DataFrame([scores_[idx]], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        if changes:
            selected += new_selected
            not_selected = [feature for feature in not_selected if feature not in new_selected]
            scores = self._scores(selected, n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)
            best_scores = scores
            diff = weighted_diff
            log += [['All Forward', 'ADDED ' + ','.join(new_selected), ','.join(selected),
                     weighted_diff, worst_diff] + list(scores)]
            if self.__verbose > 0:
                print(Color.GREEN + 'All forward step')
                print('ADDED ' + ', '.join(new_selected))
                print('Features: ' + ', '.join(selected))
                print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)

        return changes, best_scores, selected, not_selected, diff, log

    def __all_backward_step(self,
                            best_scores,
                            selected,
                            n_threads=1):
        not_selected = []
        changes = False
        diff = 0
        log = []

        scores_ = pool_map(func=self._scores,
                           iterable=[[f for f in selected if f != feature] for feature in selected],
                           n_threads=n_threads)

        for idx, feature in enumerate(selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= self.__e_drop['ab'] and worst_diff >= 0:
                changes = True
                not_selected += [feature]
            log += [['All Backward', 'TRIED TO DROP ' + feature, ','.join([f for f in selected if f != feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            if self.__verbose > 1:
                print('All backward step')
                print('TRIED TO DROP ' + feature)
                print('Features: ' + ', '.join([f for f in selected if f != feature]))
                print(pd.DataFrame([scores_[idx]], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        if changes:
            selected = [feature for feature in selected if feature not in not_selected]
            scores = self._scores(selected, n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)
            best_scores = scores
            diff = weighted_diff
            log += [['All Backward', 'DROPPED ' + ','.join(not_selected), ','.join(selected),
                     weighted_diff, worst_diff] + list(scores)]
            if self.__verbose > 0:
                print(Color.RED + 'All backward step')
                print('DROPPED ' + ', '.join(not_selected))
                print('Features: ' + ', '.join(selected))
                print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)

        return changes, best_scores, selected, not_selected, diff, log

    def __stepwise_forward_step(self,
                                best_scores,
                                selected,
                                not_selected,
                                n_threads=1):
        best_selected = ''
        best_diff = self.__e_add['sf']
        best_sc = []
        best_worst_diff = []
        changes = False
        log = []

        scores_ = pool_map(func=self._scores,
                           iterable=[selected + [feature] for feature in not_selected],
                           n_threads=n_threads)

        for idx, feature in enumerate(not_selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= best_diff and worst_diff >= 0:
                changes = True
                best_diff = weighted_diff
                best_selected = feature
                best_sc = scores_[idx]
                best_worst_diff = worst_diff
            log += [['Stepwise Forward', 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            if self.__verbose > 1:
                print('Stepwise forward step')
                print('TRIED TO ADD ' + feature)
                print('Features: ' + ', '.join(selected + [feature]))
                print(pd.DataFrame([scores_[idx]], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        if changes:
            selected += [best_selected]
            not_selected = [feature for feature in not_selected if feature != best_selected]
            best_scores = best_sc
            log += [['Stepwise Forward', 'ADDED ' + best_selected, ','.join(selected),
                     best_diff, best_worst_diff] + list(best_sc)]
            if self.__verbose > 0:
                print(Color.GREEN + 'Stepwise forward step')
                print('ADDED ' + best_selected)
                print('Features: ' + ', '.join(selected))
                print(pd.DataFrame([best_sc], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)
        else:
            best_diff = 0

        return changes, best_scores, selected, not_selected, best_diff, log

    def __stepwise_backward_step(self,
                                 best_scores,
                                 selected,
                                 n_threads=1):
        best_selected = ''
        best_diff = self.__e_drop['sb']
        best_sc = []
        best_worst_diff = []
        changes = False
        log = []

        scores_ = pool_map(func=self._scores,
                           iterable=[[f for f in selected if f != feature] for feature in selected],
                           n_threads=n_threads)

        for idx, feature in enumerate(selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= best_diff and worst_diff >= 0:
                changes = True
                best_diff = weighted_diff
                best_selected = feature
                best_sc = scores_[idx]
                best_worst_diff = worst_diff
            log += [['Stepwise Backward', 'TRIED TO DROP ' + feature, ','.join([f for f in selected if f != feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            if self.__verbose > 1:
                print('Stepwise backward step')
                print('TRIED TO DROP ' + feature)
                print('Features: ' + ', '.join([f for f in selected if f != feature]))
                print(pd.DataFrame([scores_[idx]], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        if changes:
            selected = [feature for feature in selected if feature != best_selected]
            not_selected = [best_selected]
            best_scores = best_sc
            log += [['Stepwise Backward', 'DROPPED ' + best_selected, ','.join(selected),
                     best_diff, best_worst_diff] + list(best_sc)]
            if self.__verbose > 0:
                print(Color.RED + 'Stepwise backward step')
                print('DROPPED ' + best_selected)
                print('Features: ' + ', '.join(selected))
                print(pd.DataFrame([best_sc], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)
        else:
            best_diff = 0
            not_selected = []

        return changes, best_scores, selected, not_selected, best_diff, log

    def __combined_step(self,
                        c,
                        changes,
                        best_scores,
                        selected,
                        not_selected,
                        n_threads=1):
        changes_ = False
        best_diff = 0
        log = []
        if c.lower() in ('f', 'sequential_forward'):
            changes_, best_scores, selected, not_selected, best_diff, log = self.__sequential_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads)
        elif c.lower() in ('b', 'sequential_backward'):
            changes_, best_scores, selected, not_selected_, best_diff, log = self.__sequential_backward_step(
                best_scores,
                selected,
                n_threads)
            not_selected += not_selected_
        elif c.lower() in ('af', 'all_forward'):
            changes_, best_scores, selected, not_selected, best_diff, log = self.__all_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads)
        elif c.lower() in ('ab', 'all_backward'):
            changes_, best_scores, selected, not_selected_, best_diff, log = self.__all_backward_step(
                best_scores,
                selected,
                n_threads)
            not_selected += not_selected_
        elif c.lower() in ('sf', 'stepwise_forward'):
            changes_, best_scores, selected, not_selected, best_diff, log = self.__stepwise_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads)
        elif c.lower() in ('sb', 'stepwise_backward'):
            changes_, best_scores, selected, not_selected_, best_diff, log = self.__stepwise_backward_step(
                best_scores,
                selected,
                n_threads)
            not_selected += not_selected_

        changes = changes or changes_

        return changes, best_scores, selected, not_selected, best_diff, log

    def __genetic_step(self,
                       features,
                       probabilities,
                       iteration,
                       n_threads=1):
        np_features = np.array(features)
        generation = []
        for estimation in range(self.__n_estimation):
            gen = []
            for p in probabilities:
                gen += [np.random.choice([True, False], 1, p=[p, 1 - p])[0]]
            generation += [gen]

        scores = pool_map(func=self._scores,
                          iterable=[np_features[gen] for gen in generation],
                          n_threads=n_threads)

        log = pd.DataFrame(scores, columns=self.__log_columns[self.__c:])
        for idx, feature in enumerate(features):
            log[feature] = [1 if gen[idx] else 0 for gen in generation]
        log['Weighted'] = np.dot(scores, self.__weights)
        log['Features'] = [','.join(np_features[gen]) for gen in generation]
        log['Iteration'] = iteration
        self.__genetic_log = self.__genetic_log.append(log, ignore_index=True)

        log.sort_values('Weighted', ascending=not self.__increase_metric, inplace=True)
        if not self.__increase_metric:
            log['Weighted'] *= -1
        best_score = log['Weighted'].max()
        mean_score = log['Weighted'].mean()
        log['Weighted'] = log['Weighted'].min() - log['Weighted']

        if self.__verbose > 0:
            print('Best features of generation:', log.iloc[0]['Features'])
            print('Best score of generation:', best_score if self.__increase_metric else -best_score)
            print('Mean score of generation:', mean_score if self.__increase_metric else -mean_score)

        if self.__selection_method == 'equal':
            mean_probabilities = list(log[self.__features].iloc[:self.__n_selection].mean())
        elif self.__selection_method[:4] == 'rank':
            if self.__selection_method == 'rank_by_score':
                log['w'] = log['Weighted']
            elif self.__selection_method == 'rank_by_function':
                log['w'] = [self.__rank_function(r=i, n=self.__n_estimation) for i in range(self.__n_estimation)]
            else:
                log['w'] = [i for i in range(self.__n_estimation, 0, -1)]
            log = log[features + ['w']].apply(lambda x: x * log['w'])
            log['w'] = np.sqrt(log['w'])
            mean_probabilities = list(log[self.__features].iloc[:self.__n_selection].sum() / log['w'].sum())
        else:
            mean_probabilities = probabilities
        probabilities = []
        for p in mean_probabilities:
            probabilities += [np.clip(p + random.normalvariate(0, self.__mutation), 0, 1)]

        if self.__verbose > 1:
            print('Probabilities after mutation:')
            print(pd.DataFrame({'Probabilities': probabilities}, features))

        return probabilities, best_score, mean_score

    def top_n(self,
              lower_bound=0,
              upper_bound=None,
              n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'TOP N SELECTION STARTED' + Color.END)
        if upper_bound is None or upper_bound >= len(self.__features):
            upper_bound = len(self.__features) + 1
        else:
            upper_bound += 1
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        scores = pool_map(func=self._scores,
                          iterable=[self.__features[:n] for n in range(lower_bound, upper_bound)],
                          n_threads=n_threads)
        weighted = np.dot(scores, self.__weights)
        n_selected = np.argmax(weighted) + lower_bound

        if self.__verbose > 1:
            for i in range(lower_bound, upper_bound):
                print('TRIED TOP ' + str(i) + ' features: ' + ', '.join(self.__features[:i]))
                print(pd.DataFrame([scores[i - lower_bound]], index=[''], columns=self.__log_columns[self.__c:]), '\n')

        log = pd.DataFrame(scores, columns=self.__log_columns[self.__c:])
        log['Selection type'] = 'Top N'
        log['N'] = [n for n in range(lower_bound, upper_bound)]
        log['Features'] = [','.join(self.__features[:n]) for n in range(lower_bound, upper_bound)]
        log['Weighted'] = weighted
        self.__top_n_log = self.__top_n_log.append(log, ignore_index=True)

        self.__selected = self.__features[:n_selected]
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'TOP N SELECTION FINISHED' + Color.END)
        return self.__selected

    def sequential_forward(self,
                           initial_features=None,
                           max_iter=1,
                           eps_add=0,
                           n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL FORWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        not_selected = [f for f in self.__features if f not in selected + self.__dropped_from_selection]
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_add['f'] = 0 if eps_add < 0 else eps_add
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__sequential_forward_step(best_scores,
                                                                                                          selected,
                                                                                                          not_selected,
                                                                                                          n_threads)
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def sequential_backward(self,
                            initial_features=None,
                            max_iter=1,
                            eps_drop=0,
                            n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL BACKWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_drop['b'] = eps_drop
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__sequential_backward_step(best_scores,
                                                                                                           selected,
                                                                                                           n_threads)
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def all_forward(self,
                    initial_features=None,
                    max_iter=1,
                    eps_add=0,
                    n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL FORWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        not_selected = [f for f in self.__features if f not in selected + self.__dropped_from_selection]
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_add['af'] = 0 if eps_add < 0 else eps_add
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__all_forward_step(best_scores,
                                                                                                   selected,
                                                                                                   not_selected,
                                                                                                   n_threads)
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def all_backward(self,
                     initial_features=None,
                     max_iter=1,
                     eps_drop=0,
                     n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL BACKWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_drop['ab'] = eps_drop
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__all_backward_step(best_scores,
                                                                                                    selected,
                                                                                                    n_threads)
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def stepwise_forward(self,
                         initial_features=None,
                         max_iter=np.inf,
                         eps_add=0,
                         fast=True,
                         n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE FORWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        not_selected = [f for f in self.__features if f not in selected + self.__dropped_from_selection]
        dropped_from_selection = []
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_add['sf'] = 0 if eps_add < 0 else eps_add
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__stepwise_forward_step(best_scores,
                                                                                                        selected,
                                                                                                        not_selected,
                                                                                                        n_threads)
            log = pd.DataFrame(log, columns=self.__log_columns)
            self.__log = self.__log.append(log, ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)
            if fast:
                drop = [f for f in not_selected
                        if log[log['Action'] == 'TRIED TO ADD ' + f]['Weighted diff'].iloc[0] < eps_add]
                if self.__verbose > 0 and len(drop) > 0:
                    print(Color.RED + Color.BOLD + 'Dropped from selection because of low weighted diff: ' + ', '.join(
                        drop) + '\n\n\n' + Color.END)
                dropped_from_selection += drop
                not_selected = [f for f in not_selected if f not in drop]

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def stepwise_backward(self,
                          initial_features=None,
                          max_iter=np.inf,
                          eps_drop=0,
                          n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE BACKWARD SELECTION STARTED' + Color.END)
        selected = initial_features if initial_features is not None else self.__selected
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)
        self.__e_drop['sb'] = eps_drop
        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__stepwise_backward_step(best_scores,
                                                                                                         selected,
                                                                                                         n_threads)
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def combined(self,
                 combination='f c b c sf c sb c af c ab c',
                 initial_features=None,
                 max_iter=100,
                 eps_add=0,
                 eps_drop=0,
                 n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'COMBINED SELECTION STARTED' + Color.END)
        cmd = combination.split(' ')
        if cmd[0] == 'c':
            cmd = cmd[1:]
        if 'c' not in cmd:
            cmd += ['c']

        selected = initial_features if initial_features is not None else self.__selected
        not_selected = [f for f in self.__features if f not in selected]
        best_scores = self._scores(features=selected,
                                   n_threads=n_threads)

        if isinstance(eps_add, dict):
            self.__e_add['f'] = eps_add['sequential_forward'] if 'sequential_forward' in eps_add else eps_add['f']
            self.__e_add['af'] = eps_add['all_forward'] if 'all_forward' in eps_add else eps_add['af']
            self.__e_add['sf'] = eps_add['stepwise_forward'] if 'stepwise_forward' in eps_add else eps_add['sf']
            self.__e_add['f'] = max(self.__e_add['f'], 0)
            self.__e_add['af'] = max(self.__e_add['af'], 0)
            self.__e_add['sf'] = max(self.__e_add['sf'], 0)
        else:
            self.__e_add = {t: max(eps_add, 0) for t in ['f', 'af', 'sf']}
        if isinstance(eps_add, dict):
            self.__e_drop['b'] = eps_drop['sequential_backward'] if 'sequential_backward' in eps_drop else eps_drop['b']
            self.__e_drop['ab'] = eps_drop['all_backward'] if 'all_backward' in eps_drop else eps_drop['ab']
            self.__e_drop['sb'] = eps_drop['stepwise_backward'] if 'stepwise_backward' in eps_drop else eps_drop['sb']
        else:
            self.__e_drop = {t: eps_drop for t in ['b', 'ab', 'sb']}

        changes = False
        stop = False
        i = 0
        while i < max_iter:
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i + 1, 'started' + Color.END)
            for c in cmd:
                if c.lower() in ('c', 'check'):
                    if not changes:
                        if self.__verbose > 0:
                            print(Color.BOLD + 'Iteration', i + 1, 'finished')
                            print('STOPPED ON CHECK\n\n\n' + Color.END)
                        stop = True
                        break
                    changes = False
                else:
                    changes, best_scores, selected, not_selected, best_diff, log = self.__combined_step(c,
                                                                                                        changes,
                                                                                                        best_scores,
                                                                                                        selected,
                                                                                                        not_selected,
                                                                                                        n_threads)
                    self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)

            i += 1
            if stop:
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'COMBINED SELECTION FINISHED' + Color.END)
        return selected

    def genetic(self,
                basic_probabilities=0.1,
                initial_features=None,
                max_iter=10,
                early_stopping_rounds=None,
                n_estimation=50,
                n_selection=None,
                selection_method='rank',
                rank_function=None,
                mutation=0.05,
                feature_selection_method='best',
                n_threads=1):
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'GENETIC SELECTION STARTED' + Color.END)

        initial_features = initial_features if initial_features is not None else self.__selected
        features = [feature for feature in self.__features if feature not in self.__dropped_from_selection]

        self.__genetic_log_columns = ['Iteration', 'Features'] + features + ['Weighted'] + self.__log_columns[self.__c:]
        self.__genetic_log = pd.DataFrame(columns=self.__genetic_log_columns)

        if isinstance(basic_probabilities, dict):
            if basic_probabilities.get('basic prob') is None:
                basic_probabilities.update({'basic prob': 0.1})
            probabilities = [1 if f in initial_features
                             else np.clip(basic_probabilities.get(f, basic_probabilities.get('basic prob')), 0, 1)
                             for f in features]
        else:
            probabilities = [1 if f in initial_features else np.clip(basic_probabilities, 0, 1) for f in features]

        self.__n_estimation = int(n_estimation)
        if n_selection is not None:
            if n_selection > self.__n_estimation:
                self.__n_selection = self.__n_estimation
            elif n_selection >= 1:
                self.__n_selection = int(n_selection)
            elif n_selection > 0:
                self.__n_selection = self.__n_estimation // (1 / n_selection)
            elif selection_method[:4] == 'rank':
                self.__n_selection = self.__n_estimation
            else:
                self.__n_selection = self.__n_estimation // 2
        else:
            if selection_method[:4] == 'rank':
                self.__n_selection = self.__n_estimation
            else:
                self.__n_selection = self.__n_estimation // 2

        self.__selection_method = selection_method
        self.__rank_function = rank_function
        self.__mutation = mutation

        if early_stopping_rounds is None or early_stopping_rounds < 1:
            early_stopping_rounds = max_iter

        no_best_improve_count = 0
        no_mean_improve_count = 0
        scores = self._scores(features=initial_features,
                              n_threads=n_threads)
        best_score = np.dot(scores, self.__weights)
        mean_score = best_score

        i = 0
        while i < max_iter:
            i += 1
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'started' + Color.END)

            probabilities, best_sc, mean_sc = self.__genetic_step(features=features,
                                                                  probabilities=probabilities,
                                                                  iteration=i,
                                                                  n_threads=n_threads)
            if best_sc > best_score:
                best_score = best_sc
                no_best_improve_count = 0
            else:
                no_best_improve_count += 1
            if mean_sc > mean_score:
                mean_score = mean_sc
                no_mean_improve_count = 0
            else:
                no_mean_improve_count += 1
            if self.__verbose > 1:
                print('Rounds without improve of best:', no_best_improve_count)
                print('Rounds without improve of mean:', no_mean_improve_count)

            if no_best_improve_count > early_stopping_rounds and no_mean_improve_count > early_stopping_rounds:
                if self.__verbose > 0:
                    print(Color.BOLD + 'Iteration', i, 'finished. Early stopped\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + 'Iteration', i, 'finished\n\n\n' + Color.END)

        if feature_selection_method == 'best':
            best = self.__genetic_log.sort_values('Weighted', ascending=not self.__increase_metric).iloc[0]
            self.__selected = best['Features'].split(',')
        elif feature_selection_method[:5] == 'above':
            cutoff = float(feature_selection_method[6:])
            self.__selected = list(np.array(features)[np.array(probabilities) > cutoff])

        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'GENETIC SELECTION FINISHED' + Color.END)

        return self.__selected


def join_lgb_boosters(boosters):
    n_boosters = len(boosters)
    params = boosters[0].params

    boosters_txt = []
    for booster in boosters:
        boosters_txt += [booster.model_to_string()]

    feature_infos, tree_sizes, trees, feature_importances = [], [], '', {}

    for booster in boosters_txt:
        cur = booster.split('\n')

        for idx, c in enumerate(cur):
            if c[:5] == 'Tree=':
                meta = cur[:idx]
                trees += re.sub('Tree=[0-9]+',
                                lambda a: 'Tree=' + str(int(a.group()[5:]) + len(tree_sizes)),
                                '\n'.join(cur[idx:cur.index('end of trees')]) + '\n')
                break

        parameters = '\n'.join(cur[cur.index('parameters:'):])

        feat_imp = {i[:i.index('=')]: int(i[i.index('=') + 1:])
                    for i in cur[cur.index('feature importances:') + 1:cur.index('parameters:') - 1]}
        for i in feat_imp:
            if i in feature_importances:
                feature_importances[i] += feat_imp[i]
            else:
                feature_importances[i] = feat_imp[i]

        for c in cur:
            if c[:14] == 'feature_infos=':
                feature_infos += [[[float(j) if j != 'on' else j
                                    for j in i[1:-1].split(':')] for i in c[14:].split(' ')]]
                break

        for c in cur:
            if c[:11] == 'tree_sizes=':
                tree_sizes += c[11:].split(' ')
                break

    feature_infos = [[np.min([feature_infos[j][i][0] if feature_infos[j][i][0] != 'on' else np.inf
                              for j in range(len(feature_infos))]),
                      np.max([feature_infos[j][i][1] if feature_infos[j][i][0] != 'on' else -np.inf
                              for j in range(len(feature_infos))])]
                     for i in range(len(feature_infos[0]))]

    feature_infos = 'feature_infos=' + ' '.join(['[' + re.sub('.0:', ':', str(i[0]) + ':')
                                                 + re.sub('.0]', ']', str(i[1]) + ']')
                                                 if i != [np.inf, -np.inf] else 'none' for i in feature_infos])

    trees = trees.split('\n')
    for idx, c in enumerate(trees):
        if c[:11] == 'leaf_value=':
            trees[idx] = 'leaf_value=' + ' '.join([str(float(cc) / n_boosters) for cc in c[11:].split(' ')])
    trees = '\n'.join(trees)

    tree_sizes = 'tree_sizes=' + ' '.join([str(len(i) + 3) for i in trees.split('\n\n\n')[:-1]])

    for idx, m in enumerate(meta):
        if m[:14] == 'feature_infos=':
            meta[idx] = feature_infos
        if m[:11] == 'tree_sizes=':
            meta[idx] = tree_sizes
    meta = '\n'.join(meta)
    trees += 'end of trees\n'

    feature_importances = 'feature importances:\n' + '\n'.join([i + '=' + str(feature_importances[i])
                                                                for i in sorted(feature_importances,
                                                                                key=feature_importances.get,
                                                                                reverse=True)]) + '\n'

    superbooster_str = '\n'.join([meta, trees, feature_importances, parameters])
    superbooster = lgb.Booster(model_str=superbooster_str, silent=True)
    superbooster_str = superbooster.model_to_string()
    superbooster = lgb.Booster(model_str=superbooster_str, silent=True)
    superbooster.params = params

    return superbooster


def pd_to_score(pd, intercept=0, coef=1):
    return intercept + coef * np.log(pd / (1 - pd)) if 0 < pd < 1 else np.nan


def score_to_pd(score, intercept=0, coef=1):
    return 1 / (1 + np.exp(-intercept - coef * score))


def pd_to_scaled_score(pd, intercept=0, coef=1):
    return 200 + 20 * np.log2((1 + np.exp(-intercept) * ((1 - pd) / pd) ** coef) / 50) if 0 < pd <= 1 else np.nan


def scaled_score_to_pd(scaled_score, intercept=0, coef=1):
    return 1 / (1 + np.exp(-intercept) * (50 * 2 ** ((scaled_score - 200) / 20) - 1) ** coef)


def score_to_scaled_score(score, intercept=0, coef=1):
    return 200 + 20 * np.log2((1 + np.exp(-intercept - coef * score)) / 50)


def scaled_score_to_score(scaled_score, intercept=0, coef=1):
    return intercept + coef * np.log(1 / (50 * (2 ** ((scaled_score - 200) / 20)) - 1))

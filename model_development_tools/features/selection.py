from collections.abc import Iterable
from typing import Optional, Union, Callable, List, Dict, Tuple

import numpy as np
import pandas as pd

import random

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from model_development_tools.utils import pool_map, Color
from model_development_tools.model_selection.cross_validation import StratifiedStratifiedKFold
from model_development_tools.metrics.gini import gini_score


class FS:
    def __init__(
            self,
            data: pd.DataFrame,
            features: List[str],
            target: str,
            group_by: Optional[Union[str, List[str]]] = None,
            monotonic_dict: Optional[Dict[str, int]] = None,
            estimator: Callable = LogisticRegression(),
            predict_method: str = 'predict_proba',
            boosting_params: Optional[dict] = None,
            metrics: Callable = gini_score,
            increase_metrics: bool = True,
            total_weight: Optional[float] = 0,
            weights: List[float] = None,
            total_pos_diff: bool = True,
            pos_diff: Optional[List[bool]] = None,
            cv: Union[int, Iterable] = 5,
            use_cache: bool = True,
            initial_cache: Optional[Dict[str, List[float]]] = None,
            dropped_from_selection: Optional[List[str]] = None,
            random_state: int = 16777216,
            verbose: int = 0,
    ) -> None:
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
        else:
            self.__monotonic = None

        self.__estimator = estimator
        self.__predict_method = predict_method

        self.__use_early_stop = False
        if boosting_params is not None:
            self.__boosting_params = boosting_params
            for early_stop in ('early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change'):
                if early_stop in boosting_params:
                    if boosting_params[early_stop] > 0:
                        self.__use_early_stop = True
        else:
            self.__boosting_params = {}
        if estimator in ('lgb', 'lightgbm'):
            self.__boosting_params['verbosity'] = -1
        else:
            self.__boosting_params['verbosity'] = 0

        self.__metrics = metrics
        self.__increase_metric = increase_metrics

        if weights is None or weights == []:
            self.__weights = [total_weight if total_weight != 0 else 1]
            for group in self.__group_by:
                for _ in np.unique(self.__data[group]):
                    self.__weights += [1]
        elif isinstance(weights, dict):
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
                for _ in np.unique(self.__data[group]):
                    self.__pos_diff += [True]
        elif isinstance(pos_diff, dict):
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

        self.__random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        if isinstance(cv, int):
            self.__cv = [i for i in StratifiedStratifiedKFold(cv, True, random_state).split(self.__data[features],
                                                                                            self.__data[self.__target],
                                                                                            self.__groups)]
        elif not isinstance(cv, Iterable):
            self.__cv = [i for i in cv.split(self.__data[features], self.__data[self.__target], self.__groups)]
        else:
            self.__cv = cv

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

    def get_log(
            self,
    ) -> pd.DataFrame:
        return self.__log

    def get_top_n_log(
            self,
    ) -> pd.DataFrame:
        return self.__top_n_log

    def get_genetic_log(
            self,
            wide: bool = False,
    ) -> pd.DataFrame:
        if wide:
            return self.__genetic_log
        else:
            return self.__genetic_log[['Iteration', 'Features', 'Weighted'] + self.__log_columns[self.__c:]]

    def get_selected(
            self,
    ) -> List[str]:
        return self.__selected

    def get_cache(
            self,
    ) -> Dict[str, List[float]]:
        return self.__cache

    def _cv_predict(
            self,
            features: List[str],
            n_threads: int = 1,
    ) -> np.array:
        if len(features) == 0:
            return np.array([np.mean(self.__data[self.__target])] * self.__data.shape[0])
        if self.__estimator in ('lgb', 'lightgbm'):
            if self.__monotonic is not None:
                self.__boosting_params['monotone_constraints'] = [self.__monotonic[feature] for feature in features]
            self.__boosting_params['num_threads'] = n_threads
            pred = np.zeros(self.__data.shape[0])
            for itr, iva in self.__cv:
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

    def _scores(
            self,
            features: List[str],
            n_threads: int = 1,
    ) -> np.array:
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

    def __diff(
            self,
            scores: np.array,
            best_scores: np.array,
    ) -> Tuple[np.array, np.array]:
        weighted_diff = np.dot(scores - best_scores, self.__weights)
        if self.__is_pos_diff:
            worst_diff = np.min(scores[self.__pos_diff] - best_scores[self.__pos_diff])
        else:
            worst_diff = 0
        if not self.__increase_metric:
            weighted_diff *= -1
            worst_diff *= -1
        return weighted_diff, worst_diff

    def __verbose_step(
            self,
            step_type: str,
            is_add: bool,
            is_trial: bool,
            actioned: str,
            features: List[str],
            scores: np.array,
            verbose_thr: int = 0,
            color: str = Color.END,
    ) -> None:
        if is_trial:
            action = 'TRIED TO ' + ('ADD' if is_add else 'DROP')
        else:
            action = 'ADDED' if is_add else 'DROPPED'

        if self.__verbose > verbose_thr:
            print(color + f'{step_type} step')
            print(f'{action} {actioned}')
            print(f'Features: {", ".join(features)}')
            print(pd.DataFrame([scores], index=[''], columns=self.__log_columns[self.__c:]), '\n' + Color.END)

    def __sequential_forward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            not_selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'Sequential forward'
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
                log += [[step_type, 'ADDED ' + feature, ','.join(selected),
                         weighted_diff, worst_diff] + list(scores)]

                self.__verbose_step(step_type, True, False, feature, selected, scores, 0, Color.GREEN)
            else:
                log += [[step_type, 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                         weighted_diff, worst_diff] + list(scores)]
                self.__verbose_step(step_type, True, True, feature, selected + [feature], scores, 1)

        not_selected = [feature for feature in not_selected if feature not in new_selected]

        return changes, best_scores, selected, not_selected, diff, log

    def __sequential_backward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'Sequential backward'
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
                log += [[step_type, 'DROPPED ' + feature, ','.join(selected),
                         weighted_diff, worst_diff] + list(scores)]
                self.__verbose_step(step_type, False, False, feature, selected, scores, 0, Color.RED)
            else:
                log += [[step_type, 'TRIED TO DROP ' + feature, ','.join(tmp_selected),
                         weighted_diff, worst_diff] + list(scores)]
                self.__verbose_step(step_type, False, True, feature, tmp_selected, scores, 1)

        return changes, best_scores, selected, not_selected, diff, log

    def __all_forward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            not_selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'All forward'
        new_selected = []
        changes = False
        diff = 0
        log = []

        scores_ = pool_map(
            func=self._scores,
            iterable=[selected + [feature] for feature in not_selected],
            n_threads=n_threads
        )

        for idx, feature in enumerate(not_selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= self.__e_add['af'] and worst_diff >= 0:
                changes = True
                new_selected += [feature]
            log += [[step_type, 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            self.__verbose_step(step_type, True, True, feature, selected + [feature], scores_[idx], 1)

        if changes:
            selected += new_selected
            not_selected = [feature for feature in not_selected if feature not in new_selected]
            scores = self._scores(selected, n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)
            best_scores = scores
            diff = weighted_diff
            log += [[step_type, 'ADDED ' + ','.join(new_selected), ','.join(selected),
                     weighted_diff, worst_diff] + list(scores)]
            self.__verbose_step(step_type, True, False, ', '.join(new_selected), selected, scores, 0, Color.GREEN)

        return changes, best_scores, selected, not_selected, diff, log

    def __all_backward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'All backward'
        not_selected = []
        changes = False
        diff = 0
        log = []

        scores_ = pool_map(
            func=self._scores,
            iterable=[[f for f in selected if f != feature] for feature in selected],
            n_threads=n_threads
        )

        for idx, feature in enumerate(selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= self.__e_drop['ab'] and worst_diff >= 0:
                changes = True
                not_selected += [feature]
            log += [[step_type, 'TRIED TO DROP ' + feature, ','.join([f for f in selected if f != feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            self.__verbose_step(step_type, False, True, feature, [f for f in selected if f != feature], scores_[idx], 1)

        if changes:
            selected = [feature for feature in selected if feature not in not_selected]
            scores = self._scores(selected, n_threads)
            weighted_diff, worst_diff = self.__diff(scores, best_scores)
            best_scores = scores
            diff = weighted_diff
            log += [[step_type, 'DROPPED ' + ','.join(not_selected), ','.join(selected),
                     weighted_diff, worst_diff] + list(scores)]
            self.__verbose_step(step_type, False, False, ', '.join(not_selected), selected, scores, 0, Color.RED)

        return changes, best_scores, selected, not_selected, diff, log

    def __stepwise_forward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            not_selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'Stepwise forward'
        best_selected = ''
        best_diff = self.__e_add['sf']
        best_sc = []
        best_worst_diff = []
        changes = False
        log = []

        scores_ = pool_map(
            func=self._scores,
            iterable=[selected + [feature] for feature in not_selected],
            n_threads=n_threads
        )

        for idx, feature in enumerate(not_selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= best_diff and worst_diff >= 0:
                changes = True
                best_diff = weighted_diff
                best_selected = feature
                best_sc = scores_[idx]
                best_worst_diff = worst_diff
            log += [[step_type, 'TRIED TO ADD ' + feature, ','.join(selected + [feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            self.__verbose_step(step_type, True, True, feature, selected + [feature], scores_[idx], 1)

        if changes:
            selected += [best_selected]
            not_selected = [feature for feature in not_selected if feature != best_selected]
            best_scores = best_sc
            log += [[step_type, 'ADDED ' + best_selected, ','.join(selected),
                     best_diff, best_worst_diff] + list(best_sc)]
            self.__verbose_step(step_type, True, False, best_selected, selected, best_sc, 0, Color.GREEN)
        else:
            best_diff = 0

        return changes, best_scores, selected, not_selected, best_diff, log

    def __stepwise_backward_step(
            self,
            best_scores: np.array,
            selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
        step_type = 'Stepwise backward'
        best_selected = ''
        best_diff = self.__e_drop['sb']
        best_sc = []
        best_worst_diff = []
        changes = False
        log = []

        scores_ = pool_map(
            func=self._scores,
            iterable=[[f for f in selected if f != feature] for feature in selected],
            n_threads=n_threads
        )

        for idx, feature in enumerate(selected):
            weighted_diff, worst_diff = self.__diff(scores_[idx], best_scores)
            if weighted_diff >= best_diff and worst_diff >= 0:
                changes = True
                best_diff = weighted_diff
                best_selected = feature
                best_sc = scores_[idx]
                best_worst_diff = worst_diff
            log += [[step_type, 'TRIED TO DROP ' + feature, ','.join([f for f in selected if f != feature]),
                     weighted_diff, worst_diff] + list(scores_[idx])]
            self.__verbose_step(step_type, False, True, feature, [f for f in selected if f != feature], scores_[idx], 1)

        if changes:
            selected = [feature for feature in selected if feature != best_selected]
            not_selected = [best_selected]
            best_scores = best_sc
            log += [[step_type, 'DROPPED ' + best_selected, ','.join(selected),
                     best_diff, best_worst_diff] + list(best_sc)]
            self.__verbose_step(step_type, False, False, best_selected, selected, best_sc, 0, Color.RED)
        else:
            best_diff = 0
            not_selected = []

        return changes, best_scores, selected, not_selected, best_diff, log

    def __combined_step(
            self,
            c: str,
            changes: bool,
            best_scores: np.array,
            selected: List[str],
            not_selected: List[str],
            n_threads: int = 1,
    ) -> Tuple[bool, np.array, List[str], List[str], np.array, List[list]]:
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

    def __genetic_step(
            self,
            features: List[str],
            probabilities: List[float],
            iteration: int,
            n_threads: int = 1,
    ) -> Tuple[List[float], float, float]:
        np_features = np.array(features)
        generation = []
        for estimation in range(self.__n_estimation):
            gen = []
            for p in probabilities:
                gen += [np.random.choice([True, False], 1, p=[p, 1 - p])[0]]
            generation += [gen]

        scores = pool_map(
            func=self._scores,
            iterable=[np_features[gen] for gen in generation],
            n_threads=n_threads
        )

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
            print(f'Best features of generation: {log.iloc[0]["Features"]}')
            print(f'Best score of generation: {best_score if self.__increase_metric else -best_score}')
            print(f'Mean score of generation: {mean_score if self.__increase_metric else -mean_score}')

        if self.__selection_method == 'equal':
            mean_probabilities = list(log[self.__features].iloc[:self.__n_selection].mean())
        elif self.__selection_method[:4] == 'rank':
            if self.__selection_method == 'rank_by_score':
                log['w'] = log['Weighted']
            elif self.__selection_method == 'rank_by_function':
                log['w'] = [self.__rank_function(r=i, n=self.__n_estimation, s=log['Weighted'].iloc[i])
                            for i in range(self.__n_estimation)]
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

    def top_n(
            self,
            lower_bound: int = 0,
            upper_bound: Optional[int] = None,
            n_threads: int = 1,
    ) -> List[str]:
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'TOP N SELECTION STARTED' + Color.END)
        if upper_bound is None or upper_bound >= len(self.__features):
            upper_bound = len(self.__features) + 1
        else:
            upper_bound += 1
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        scores = pool_map(
            func=self._scores,
            iterable=[self.__features[:n] for n in range(lower_bound, upper_bound)],
            n_threads=n_threads
        )
        weighted = np.dot(scores, self.__weights)
        n_selected = (np.argmax(weighted) if self.__increase_metric else np.argmin(weighted)) + lower_bound

        if self.__verbose > 1:
            for i in range(lower_bound, upper_bound):
                print(f'TRIED TOP {i} features: {", ".join(self.__features[:i])}')
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

    def sequential_forward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = 1,
            eps_add: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__sequential_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads
            )
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def sequential_backward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = 1,
            eps_drop: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__sequential_backward_step(
                best_scores,
                selected,
                n_threads
            )
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'SEQUENTIAL BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def all_forward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = 1,
            eps_add: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__all_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads
            )
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def all_backward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = 1,
            eps_drop: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__all_backward_step(
                best_scores,
                selected,
                n_threads
            )
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'ALL BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def stepwise_forward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = np.inf,
            eps_add: float = 0,
            fast: bool = True,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__stepwise_forward_step(
                best_scores,
                selected,
                not_selected,
                n_threads
            )
            log = pd.DataFrame(log, columns=self.__log_columns)
            self.__log = self.__log.append(log, ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)
            if fast:
                drop = [f for f in not_selected
                        if log[log['Action'] == 'TRIED TO ADD ' + f]['Weighted diff'].iloc[0] < eps_add]
                if self.__verbose > 0 and len(drop) > 0:
                    print(Color.RED + Color.BOLD + 'Dropped from selection because of low weighted diff: '
                                                   f'{", ".join(drop)}\n\n\n' + Color.END)
                dropped_from_selection += drop
                not_selected = [f for f in not_selected if f not in drop]

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE FORWARD SELECTION FINISHED' + Color.END)
        return selected

    def stepwise_backward(
            self,
            initial_features: Optional[List[str]] = None,
            max_iter: int = np.inf,
            eps_drop: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)
            changes, best_scores, selected, not_selected, best_diff, log = self.__stepwise_backward_step(
                best_scores,
                selected,
                n_threads
            )
            self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)
            if not changes:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished without changes\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'STEPWISE BACKWARD SELECTION FINISHED' + Color.END)
        return selected

    def combined(
            self,
            combination: str = 'f c b c sf c sb c af c ab c',
            initial_features: Optional[List[str]] = None,
            max_iter: int = 100,
            eps_add: float = 0,
            eps_drop: float = 0,
            n_threads: int = 1,
    ) -> List[str]:
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'COMBINED SELECTION STARTED' + Color.END)
        cmd = combination.split(' ')
        if cmd[0] == 'c':
            cmd = cmd[1:]
        if 'c' not in cmd:
            cmd += ['c']

        selected = initial_features if initial_features is not None else self.__selected
        not_selected = [f for f in self.__features if f not in selected + self.__dropped_from_selection]
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
        if isinstance(eps_drop, dict):
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
                print(Color.BOLD + f'Iteration {i + 1} started' + Color.END)
            for c in cmd:
                if c.lower() in ('c', 'check'):
                    if not changes:
                        if self.__verbose > 0:
                            print(Color.BOLD + f'Iteration {i + 1} finished')
                            print('STOPPED ON CHECK\n\n\n' + Color.END)
                        stop = True
                        break
                    changes = False
                else:
                    changes, best_scores, selected, not_selected, best_diff, log = self.__combined_step(
                        c,
                        changes,
                        best_scores,
                        selected,
                        not_selected,
                        n_threads
                    )
                    self.__log = self.__log.append(pd.DataFrame(log, columns=self.__log_columns), ignore_index=True)

            i += 1
            if stop:
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        self.__selected = selected
        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'COMBINED SELECTION FINISHED' + Color.END)
        return selected

    def genetic(
            self,
            basic_probabilities: Union[float, Dict[str, float]] = 0.1,
            initial_features: List[str] = None,
            max_iter: int = 10,
            early_stopping_rounds: Optional[int] = None,
            n_estimation: int = 50,
            n_selection: Optional[int] = None,
            selection_method: str = 'rank',
            rank_function: Optional[Callable] = None,
            mutation: float = 0.05,
            feature_selection_method: str = 'best',
            n_threads: int = 1,
    ) -> List[str]:
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
                self.__n_selection = int(self.__n_estimation // (1 / n_selection))
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
                print(Color.BOLD + f'Iteration {i} started' + Color.END)

            probabilities, best_sc, mean_sc = self.__genetic_step(
                features=features,
                probabilities=probabilities,
                iteration=i,
                n_threads=n_threads
            )
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
                print(f'Rounds without improve of best: {no_best_improve_count}')
                print(f'Rounds without improve of mean: {no_mean_improve_count}')

            if no_best_improve_count > early_stopping_rounds and no_mean_improve_count > early_stopping_rounds:
                if self.__verbose > 0:
                    print(Color.BOLD + f'Iteration {i} finished. Early stopped\n\n\n' + Color.END)
                break
            if self.__verbose > 0:
                print(Color.BOLD + f'Iteration {i} finished\n\n\n' + Color.END)

        if feature_selection_method == 'best':
            best = self.__genetic_log.sort_values('Weighted', ascending=not self.__increase_metric).iloc[0]
            self.__selected = best['Features'].split(',')
        elif feature_selection_method[:5] == 'above':
            cutoff = float(feature_selection_method[6:])
            self.__selected = list(np.array(features)[np.array(probabilities) > cutoff])

        if self.__verbose > 0:
            print(Color.BOLD + Color.UNDERLINE + 'GENETIC SELECTION FINISHED' + Color.END)

        return self.__selected

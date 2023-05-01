from typing import Optional
from collections.abc import Iterator

import numpy as np
import pandas as pd

import random

from sklearn.model_selection import StratifiedKFold


class StratifiedStratifiedKFold(StratifiedKFold):
    def split(
            self,
            X: np.ndarray,
            y: np.ndarray,
            groups: Optional[np.ndarray] = None,
    ) -> Iterator[np.ndarray, np.ndarray]:
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
            group_cols = [f'g{i}' for i in range(groups.shape[1] + 1)]
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

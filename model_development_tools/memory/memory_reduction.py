import numpy as np
import pandas as pd

from model_development_tools.utils import pool_map


class MemoryReducer:
    def __init__(
            self,
            float_min_type: int = 16,
            int_min_type: int = 8,
    ) -> None:
        self.int8_min = np.iinfo(np.int8).min
        self.int8_max = np.iinfo(np.int8).max
        self.int16_min = np.iinfo(np.int16).min
        self.int16_max = np.iinfo(np.int16).max
        self.int32_min = np.iinfo(np.int32).min
        self.int32_max = np.iinfo(np.int32).max

        self.uint8_max = np.iinfo(np.uint8).max
        self.uint16_max = np.iinfo(np.uint16).max
        self.uint32_max = np.iinfo(np.uint32).max

        self.float16_min = np.finfo(np.float16).min
        self.float16_max = np.finfo(np.float16).max
        self.float32_min = np.finfo(np.float32).min
        self.float32_max = np.finfo(np.float32).max
        self.__float_min_type = float_min_type
        self.__int_min_type = int_min_type

    def shrink_column(
            self,
            col: pd.Series,
    ) -> pd.Series:
        is_int = col.dtypes.name[:3] == 'int'
        is_uint = col.dtypes.name[:3] == 'uin'
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
        elif is_uint:
            c_max = col.max()
            if self.__int_min_type <= 8 and c_max < self.uint8_max:
                col = col.astype(np.int8)
            elif self.__int_min_type <= 16 and c_max < self.uint16_max:
                col = col.astype(np.int16)
            elif self.__int_min_type <= 32 and c_max < self.uint32_max:
                col = col.astype(np.int32)
        elif is_float:
            c_min = col.min()
            c_max = col.max()
            if self.__float_min_type <= 16 and c_min > self.float16_min and c_max < self.float16_max:
                col = col.astype(np.float16)
            elif self.__float_min_type <= 32 and c_min > self.float32_min and c_max < self.float32_max:
                col = col.astype(np.float32)
        return col

    def reduce(
            self,
            df: pd.DataFrame,
            n_threads: int = 1,
            verbose: int = 0,
    ) -> pd.DataFrame:
        if verbose > 0:
            start_mem = df.memory_usage().sum() / 1024 ** 2
            print(f'Memory usage of dataframe is {round(start_mem, 2)} MB')

        df = pd.concat(
            pool_map(
                func=self.shrink_column,
                iterable=[df[col] for col in df.columns],
                n_threads=n_threads
            ),
            axis=1
        )

        if verbose > 0:
            end_mem = df.memory_usage().sum() / 1024 ** 2
            print(f'Memory usage after optimization is: {round(end_mem, 2)} MB')
            print(f'Decreased by {round(100 * (start_mem - end_mem) / start_mem, 1)}%')

        return df

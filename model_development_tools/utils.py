from typing import Callable, Iterable, List

import re
from multiprocessing import Pool


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


def pool_map(
        func: Callable,
        iterable: Iterable,
        n_threads: int = 1,
) -> Iterable:
    if n_threads <= 1:
        res = [func(i) for i in iterable]
    else:
        p = Pool(n_threads)
        res = p.map(func=func,
                    iterable=iterable)
        p.close()
    return res


def unpack_dict(
        d: dict,
) -> list:
    unpacked_list = []
    for el in d.values():
        unpacked_list += el
    return unpacked_list


def grepl(
        pattern: str,
        array: List[str],
) -> List[str]:
    grep_arr = []
    for string in array:
        if re.search(pattern, string) is not None:
            grep_arr += [string]
    return grep_arr

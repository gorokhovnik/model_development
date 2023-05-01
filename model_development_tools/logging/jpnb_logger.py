import os
import re
from functools import wraps
import time
import datetime


class Log:
    def __init__(
            self,
            logs_folder: str = 'logs',
            digits: int = 3,
            exception_to_log: bool = True
    ) -> None:
        self.c = 0
        self.digits = digits
        self.exception_to_log = exception_to_log

        if logs_folder not in os.listdir():
            os.mkdir(logs_folder)

        self.log_dirname = f'{logs_folder}/log_{time.strftime("%y_%m_%d_%H_%M_%S")}'
        os.mkdir(self.log_dirname)

        self.total_start_time = time.time()

    def __log_number(
            self,
    ) -> str:
        self.c += 1
        str_c = str(self.c)
        return f'{str_c.zfill(self.digits)}'

    def log_decorator(
            self,
            cell: str,
    ):
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                start_time = time.time()
                cell_normalized = re.sub('\W', '', re.sub(' ', '_', cell.lower()))
                log_filename = f'{self.log_dirname}/{self.__log_number()}_{cell_normalized}.txt'
                log_file = open(log_filename, 'a')

                if self.exception_to_log:
                    try:
                        log_file.write(f(*args, **kwargs))
                    except Exception as e:
                        log_file.write(repr(e) + '\n')
                else:
                    log_file.write(f(*args, **kwargs))

                log_file.write(f'\nTime: {time.strftime("%y/%m/%d %H:%M:%S")}\n')
                spent_on_cell = datetime.timedelta(seconds=time.time() - start_time)
                spent_total = datetime.timedelta(seconds=time.time() - self.total_start_time)
                log_file.write(f'Spend on {cell[0].lower()}{cell[1:]}: {spent_on_cell}\n')
                log_file.write(f'Spend total: {spent_total}')
                log_file.close()

            return decorated

        return decorator

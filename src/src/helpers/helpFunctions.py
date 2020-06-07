import os
import re
import yaml
import time
import psutil
import argparse

from .logger import setup_logger

from threading import Thread

number = re.compile(r'\d+')


def init(name):
    """
    Init script for init program, log base info and load configuration files from command line. Init the logger.
    """

    args = load_arguments(name=name, l='logfile', ll='loglevel', ec='experimentConfig')
    experiment_config = None

    with open(args.experimentConfig, 'r') as stream:
        try:
            experiment_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-5)

    logger = setup_logger(args.logfile, True, args.loglevel if args.loglevel is not None else "INFO")
    logger.info(name)
    logger.info("Run config file: {}".format(args.experimentConfig))
    return experiment_config['experiment'], logger


def load_arguments(name, **kwargs):
    """
    Parse arguments from command line.
    """

    parser = argparse.ArgumentParser(description=name)
    for kwarg in kwargs:
        parser.add_argument('-' + str(kwarg), '--' + str(kwargs[kwarg]), required=False, default=None)
    return parser.parse_args()


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Compare double with some precision.
    """

    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def parse_str_to_3value_range(text):
    """
    Parse 3 values write in string "1, 2, 3" to values of generator.
    Number of values can be 1, 2 or 3 for one value, min-max and min-max-step
    """

    val = text.split()
    if len(val) == 1:
        return float(val[0]), float(val[0]) + 1, 1
    if len(val) == 2:
        return float(val[0]), float(val[1]), 1
    if len(val) == 3:
        return float(val[0]), float(val[1]), float(val[2])
    raise ValueError(
        "For parse need string with one test value, two values for min and max or three for min, max and step.")


def d_range(b, m, s, round_num=0):
    """
    Range with possibility of multiplication or addition. Range is only incrementing.
    For negative step is use multiplication for next number.
    Range is in integers, next step is always min 1 bigger then previous.
    """

    r = b
    while r < m:
        r = round(r, round_num)
        yield r
        prev = r
        if s > 0:
            r += s
        else:
            r = r * ((-1) * s)
            if is_close(prev, round(r, round_num)):
                r += 1


def count_sum_product(vector):
    """
    Count sum product of one vector.
    """

    value = 0.
    for item in vector:
        value += (vector[item] ** 2)
    return value


class Timer:
    """
    Class for mensuration time for some block of code.
    """

    timers = {}
    counter = 0

    @classmethod
    def start(cls):
        """
        Start new timer
        """

        while cls.counter in cls.timers:
            cls.counter += 1
        cls.timers[cls.counter] = time.time()
        cls.counter += 1
        return cls.counter - 1

    @classmethod
    def restart(cls, counter_id):
        """
        Restart timer under id. If not exist create new one.
        """

        cls.timers[counter_id] = time.time()
        return counter_id

    @classmethod
    def get_time_millisecond(cls, time_id):
        """Return time from timer in milliseconds."""
        return (time.time() - cls.timers[time_id]) * 1000

    @classmethod
    def get_time_second(cls, time_id):
        """
        Return time from timer in milliseconds.
        """

        return time.time() - cls.timers[time_id]

    @classmethod
    def get_time_minute(cls, time_id):
        """
        Return time from timer in milliseconds.
        """

        return (time.time() - cls.timers[time_id]) / 60


def get_number_of_items(raw_data):
    """
    Count items in raw data triplets.
    """

    items = set()
    for i in raw_data:
        if i[1] is not None:
            items.add(i[1])
    return len(items)


class WriteCSVData:
    """
    Create file in csv format for writing stats.
    """

    def __init__(self, file, sep=","):
        self.file = None
        self.file = open(file, 'a')
        self.head = None
        self.sep = sep

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def append_line(self, **kwargs):
        """
        Append one line to file, if it is first line -> create header
        """

        if self.head is not None:
            self.__write_line(kwargs=kwargs)
        else:
            self.__write_head(kwargs)
            self.__write_line(kwargs)
        self.file.flush()

    def __write_head(self, kwargs):
        self.head = []
        for key in kwargs:
            self.head.append(key)
        self.file.write(self.sep.join([str(key) for key in self.head]) + "\n")

    def __write_line(self, kwargs):
        self.file.write(self.sep.join([str(kwargs[key]) for key in self.head]) + "\n")
        self.file.flush()


class PercentCounter:
    """
    Percent counter logging
    """

    def __init__(self, maximum, step, logger):
        self.per = step
        self.step = step
        self.max = maximum
        self.logger = logger
        self.done = 0
        self.timer = Timer.start()

    def increment(self, message):
        """
        Increment value of counter. Check if need print.
        """

        self.done += 1
        if self.done / float(self.max) > self.per:
            while self.done / float(self.max) > self.per:
                self.per += self.step
            self.logger.info(message + " Percent: {} % Count: {} Time: {} s".format(
                round(self.done / float(self.max) * 100., 2), self.done, Timer.get_time_second(self.timer)))
            Timer.restart(self.timer)


class BlockTimer:
    """
    Tuned timer which is possible use as block in Python.
    """

    def __init__(self, name, message, logger):
        self.name = name
        self.message = message
        self.logger = logger
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info('%s; %s; take %s seconds' % (self.name, self.message, time.time()-self.t0))


class MemMon:
    """
    Class for memory monitoring. It is not possible specific memory like Java JVM. To not kill HW run on it. :D
    """

    def __init__(self, logger, max_memory, log_every_secs):
        self.logger = logger
        self.max_memory = max_memory
        self.log_every_secs = log_every_secs
        self.process = psutil.Process(os.getpid())
        self.stopped = False
        self.report_thread = None

    def __get_used_memory_mb(self):
        return self.process.memory_info().rss >> 20

    def get_used_memory_mb(self):
        return self.process.memory_info().rss >> 20

    def get_used_memory(self):
        return self.process.memory_info().rss

    def get_used_memory_ratio(self):
        return float(self.__get_used_memory_mb())/self.max_memory

    def __run(self):

        while not self.stopped:
            mem_info = {'metric_type': 'memory_metric',
                        'used_memory': self.__get_used_memory_mb(),
                        'max_memory': self.max_memory,
                        'ratio': self.get_used_memory_ratio()}
            if self.get_used_memory_ratio() > 1.0:
                self.logger.error("To much memory kill myself. Use {} mb".format(self.get_used_memory_mb()))
                self.logger.error("Cant use more memory")
                os._exit(-1)
            self.logger.info(mem_info)
            time.sleep(self.log_every_secs)

    def stop(self):
        self.stopped = True
        self.report_thread.join()

    def run(self):
        self.report_thread = Thread(target=self.__run, args=(), daemon=True)
        self.report_thread.start()


def get_mf_parameters_from_file_name(file):
    """
    Parse file name to MF parameters.
    """

    table = file.split('_')
    it = iter(table)
    mapping = {'f': 'factors', 'l': 'regularization', 'p': 'pruning', 'b': 'bm25b', 'm': 'bm25m'}
    v = {}
    last = None
    while True:
        try:
            if last is not None:
                i = last
                last = None
            else:
                i = next(it)
            if i[0].lower() not in {'f', 'l', 'p', 'b', 'm'}:
                continue
            num = next(it)
            if number.match(num):
                num2 = next(it)
                if number.match(num2):
                    v[mapping[i[0].lower()]] = int(num) + float(float(num2)/float('1' + '0'*len(num2)))
                else:
                    v[mapping[i[0].lower()]] = int(num)
                    last = num2
            else:
                last = num
        except StopIteration:
            break
    if 'pruning' not in v:
        v['pruning'] = -1
    if 'bm25b' not in v:
        v['bm25b'] = -1
    if 'bm25m' not in v:
        v['bm25m'] = -1
    if 'factors' not in v:
        v['factors'] = -1
    if 'regularization' not in v:
        v['regularization'] = -1
    return v


import csv
import time

from pyspark.sql import SparkSession


def timeit(method):
    """Decorator for measuring running time of functions/methods."""
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        diff = '%2.2f s' % (end - start)
        args_str = str(args)
        kwargs_str = str(args)

        print('*' * 72)
        print('TIME LOG for %r: %s' % (method.__name__, diff))
        print('Args: %s' % args_str)
        print('Keyword args: %s' % kwargs_str)
        print('*' * 72)

        return result
    return timed


def log(message, type='info'):
    """Poor man's logger. Works great with YARN though.
    Args:
      message (string)
      type (string)
    """
    print('-' * 72)
    print('%s: %s' % (type.upper(), message))
    print('-' * 72)


@timeit
def get_spark_session(app_name):
    return SparkSession\
        .builder\
        .appName(app_name)\
        .enableHiveSupport()\
        .getOrCreate()


@timeit
def csv_to_list(spark, filename, separator='\t', headerp=True):
    """Read CSV and return list.
    Can handle local files and files saved in HDFS. In order to read
    local files run yarn in client mode.
    Args:
        filename (string)
        separator (string)
        headerp (boolean)
    Returns:
        list
    """
    if filename.startswith('hdfs://'):
        data = spark.read.load(
            filename,
            format='csv',
            sep=separator,
            inferSchema='true',
            header='true' if headerp else 'false').collect()
    else:
        with open(filename, 'r') as inf:
            tsv_reader = csv.reader(inf, delimiter=separator)
            if headerp:
                next(tsv_reader)
            data = list(tsv_reader)
    return data

import csv
import time

from pyspark.sql import SparkSession

LOGGER = None


def setup_logger(spark):
    global LOGGER
    log4jLogger = spark._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)


def timeit(method):
    """Decorator for measuring running time of functions/methods."""
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        diff = '%2.2f s' % (end - start)
        args_str = str(args)
        kwargs_str = str(args)

        LOGGER.info('-' * 72)
        LOGGER.info('TIME LOG for %r: %s' % (method.__name__, diff))
        LOGGER.info('Args: %s' % args_str)
        LOGGER.info('Keyword args: %s' % kwargs_str)
        LOGGER.info('-' * 72)

        return result
    return timed


def log(message, type='info'):
    """Poor man's logger. Works great with YARN though.
    Args:
      message (string)
      type (string)
    """
    LOGGER.info('-' * 72)
    if type == 'debug':
        LOGGER.debug(message)
    elif type == 'warn':
        LOGGER.warn(message)
    elif type == 'error':
        LOGGER.error(message)
    else:
        LOGGER.info(message)
    LOGGER.info('-' * 72)


@timeit
def get_spark_session(app_name):
    spark = SparkSession\
        .builder\
        .appName(app_name)\
        .enableHiveSupport()\
        .getOrCreate()
    setup_logger(spark)
    return spark


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

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Generate article normalized scores.

This script allows generating article normalized scores that can be used
for sorting missing articles.

Example:
$ python recommend.py 2019/01/01 en uz

Todo:
    * break up the big functions into multiple small functions.
    * add debug statements
    * add unit tests
"""

import argparse
import csv
from datetime import datetime, timedelta
import logging

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F, SparkSession


TOPSITES_FILE = '../data/topsites.tsv'
"""string: Location of the TSV file that contains the top 50 Wikipedias
by article count.
"""

DBLIST_FILE = '../data/wikipedia.dblist'
"""string: Location of the TSV file that contains the list of
Wikipedias.
"""

TRAIN_RANGE_DAYS = 180
"""int: Number of days used for training models.

This variable comes from the research paper. Pageviews data is gathered
using this date range and the end_date argument passed to the script.
"""

WIKIDATA_DUMPS = '/user/joal/wmf/data/wmf/mediawiki/wikidata_parquet/20181001'
"""string: Location of Wikidata dumps in parquet format.
"""

OUTPUT_DIR = '/user/bmansurov/article-recommender'
"""string: Location of the directory in HDFS for storing intermediate
and output data.
"""

LOG_FILE = 'article-recommender.log'
"""string: Location of the log file.
"""


def get_cmd_options():
    """Return command line options passed to the script.

    Returns:
        object: Arguments passed to the script.

    """
    parser = argparse.ArgumentParser(
        description='Generates article normalized scores.')
    parser.add_argument('end_date',
                        help='End date in the yyyy/mm/dd format.',
                        type=lambda x: datetime.strptime(x, "%Y/%m/%d").date())
    parser.add_argument('source_language',
                        help='Source language code, e.g. en or uz.')
    parser.add_argument('target_language',
                        help='Target language code, e.g. en or uz.')
    return parser.parse_args()


def get_wikipedia_dblist():
    """Return a set of Wikipedias from a file.

    Returns:
        set: Wikipedias, e.g. {'enwiki', 'uzwiki', ... }

    """
    with open(DBLIST_FILE, 'r') as inf:
        return set(inf.readlines())


def validate_cmd_options(options):
    """Validate command line options passed by the user.

    Returns:
        bool: In case of error, False is returned. Otherwise, True.

    """
    wikipedias = get_wikipedia_dblist()
    if '%swiki' % options.source_language not in wikipedias:
        logging.error('Unrecognized source language: %s' %
                      options.source_language)
        return False
    if '%swiki' % options.target_language not in wikipedias:
        logging.error('Unrecognized target language: %s' %
                      options.target_language)
        return False
    if options.source_language == options.target_language:
        logging.error('Source and target languages cannot be the same.')
        return False
    if options.end_date > datetime.today():
        logging.error('End date cannot be later than today: %s.' %
                      options.end_date)
        return False
    return True


def get_spark_session():
    """Create or return existing spark session with Hive support.

    Returns:
        an instance of SparkSession with Hive support
    """
    return SparkSession\
        .builder\
        .master('yarn')\
        .appName('article-recommender')\
        .enableHiveSupport()\
        .getOrCreate()


def get_wikidata(spark_session):
    """Return Wikidata data frame from parquet.

    Args:
        spark_session: Instance of SparkSession.

    Returns:
        dataframe <id, site, title>

    """
    return spark_session\
        .read\
        .parquet(WIKIDATA_DUMPS)\
        .where(F.col('typ') == 'item')\
        .select('id', F.explode('siteLinks').alias('sl'))\
        .select('id', 'sl.site', 'sl.title')


def get_pageviews_filename(language, start_date, end_date):
    """Return pageviews filename.

    Filename is identified by language, start and end dates.

    Args:
        language (string)
        start_date (datetime)
        end_date (datetime)

    Returns:
        string: filename in HDFS for given language wiki pageviews data

    """
    return '%s/pageviews-%s-%s-%s' %\
        (OUTPUT_DIR, start_date, end_date, language)


def calculate_pageviews_and_save(spark_session, language, start_date, end_date,
                                 articles, filename):
    """Get pageviews from Hive and save calculated data in parquet.

    Args:
        spark_session: Instance of SparkSession.
        start_date (datetime): Pageviews that happened on this date and
            after will be considered
        end_date (datetime): Pageviews that happened on this date and
            before will be consided
        language (string): Pageviews for the language wiki only will be
            considered
        articles (dataframe): TODO document
        filename (string): Location used to save the data as parquet

    Returns:
        dataframe: Pageviews with normalized and log ranks

    """
    # Cannot create date alias as the where clause is evaluated first
    sql = """
        SELECT page_title AS %s_title, SUM(view_count) AS %s_pageviews,
          RANK() OVER (ORDER BY SUM(view_count) ASC) AS rank
        FROM wmf.pageview_hourly
        WHERE
          TO_DATE(CONCAT(year, "-", month, "-", day)) >= "%s" AND
          TO_DATE(CONCAT(year, "-", month, "-", day)) <= "%s" AND
          project="%s.wikipedia" AND
          agent_type="user" AND
          instr(page_title, ':')=0
        GROUP BY page_title
    """
    pageviews = spark_session.sql(
        sql % (language, language, start_date.strftime('%Y-%m-%d'),
               end_date.strftime('%Y-%m-%d'), language))

    target_article_count = pageviews.count()
    # Calculate normalized and log ranks
    pageviews = pageviews\
        .withColumn('%s_normalized_rank' % language,
                    F.col('rank') / target_article_count)\
        .withColumn('%s_log_rank' % language,
                    F.log(F.col('rank')))

    pageviews = pageviews\
        .alias('p')\
        .join(articles.alias('a'),
              F.col('a.title') == F.col('p.%s_title' % language))

    pageviews = pageviews.withColumnRenamed('id', '%s_id' % language)
    pageviews = pageviews.drop('title').drop('site').drop('rank')
    pageviews.write.parquet(filename)

    return pageviews


def get_pageviews(spark_session, language, start_date, end_date,
                  articles):
    """Return pageviews with normalized ranks from file if exits, otherwise
    from Hive and save to file for later use.

    Args:
        spark_session: Instance of SparkSession.
        start_date (datetime): Pageviews that happened on this date and
            after will be considered
        end_date (datetime): Pageviews that happened on this date and
            before will be consided
        language (string): Pageviews for the language wiki only will be
            considered
        articles (dataframe): TODO document

    Returns:
        dataframe: Pageviews with normalized and log ranks

    """
    filename = get_pageviews_filename(language, start_date, end_date)
    try:
        pageviews = spark_session\
            .read\
            .parquet(filename)
    except Exception:
        pageviews = calculate_pageviews_and_save(
            spark_session, language, start_date, end_date, articles,
            filename)

    return pageviews


def get_top_languages():
    """Return to 50 wikipedia languages by article count.

    Returns:
        list<string>: e.g. ['en', 'de', 'sv', ...]

    """
    with open(TOPSITES_FILE, 'r') as inf:
        tsv_reader = csv.reader(inf, delimiter='\t')
        next(tsv_reader)
        return [x[0] for x in tsv_reader]


def get_combined_pageviews_filename(start_date, end_date):
    """Return filename for combined pageviews.

    Args:
        start_date (datetime): Pageviews that happened on this date and
            after will be considered
        end_date (datetime): Pageviews that happened on this date and
            before will be consided

    Returns:
        string

    """

    return '%s/combined-pageviews-%s-%s' %\
        (OUTPUT_DIR, start_date, end_date)


def calculate_combined_pageviews_and_save(spark_session, wikidata,
                                          start_date, end_date, filename):
    """Calculate combined pageviews and save for later use in HDFS.

    Args:
        spark_session: Instance of SparkSession.
        start_date (datetime): Pageviews that happened on this date and
            after will be considered
        end_date (datetime): Pageviews that happened on this date and
            before will be consided
        filename (string): Where to save data

    Returns:
        dataframe: Combined pageviews with normalized and log ranks for
            top 50 Wikipedias

    """
    top_languages = get_top_languages()
    wikidata_ids = wikidata.select('id').distinct()
    for language in top_languages:
        articles = wikidata\
            .where(F.col('site') == '%swiki' % language)\
            .filter(~F.col('title').contains(':'))
        pageviews = get_pageviews(spark_session, language, start_date,
                                  end_date, articles)
        pageviews = wikidata_ids\
            .alias('w')\
            .join(pageviews.alias('p'),
                  F.col('w.id') == F.col('p.%s_id' % language),
                  'left_outer')\
            .na.fill({
                '%s_pageviews' % language: 0,
                '%s_normalized_rank' % language: 0,
                '%s_log_rank' % language: 0
            })
        pageviews = pageviews\
            .drop('%s_id' % language)\
            .drop('%s_title' % language)

    pageviews.write.parquet(filename)

    return pageviews


def get_combined_pageviews(spark_session, wikidata, start_date, end_date):
    """Return combined pageviews for the top 50 Wikipedias.

    If the parquet file exists, return it, otherwise generate the file,
    save and return it.

    Args:
        spark_session: Instance of SparkSession.
        start_date (datetime): Pageviews that happened on this date and
            after will be considered
        end_date (datetime): Pageviews that happened on this date and
            before will be consided

    Returns:
        dataframe: Combined pageviews with normalized and log ranks for
            top 50 Wikipedias

    """
    filename = get_combined_pageviews_filename(start_date, end_date)
    try:
        pageviews = spark_session\
            .read\
            .parquet(filename)
    except Exception:
        pageviews = calculate_combined_pageviews_and_save(
            spark_session, wikidata, start_date, end_date, filename)

    return pageviews


def get_predictions_filename(start_date, end_date, source_language,
                             target_language):
    """Get filename for saving normalized rank predictions.

    Args:
        start_date (datetime)
        end_date (datetime)
        source_language (string)
        target_language (string)

    Returns:
        string
    """
    return '%s/predictions-%s-%s-%s-%s.tsv' %\
        (OUTPUT_DIR, start_date, end_date,
         source_language, target_language)


def train(spark_session, options):
    """Train models and create article normalized scores.

    Args:
        spark_session: instance of SparkSession
        options (object): command line options passed by the user

    """
    source_language = options.source_language
    target_language = options.target_language
    end_date = options.end_date

    source_wiki = '%swiki' % source_language
    target_wiki = '%swiki' % target_language
    start_date = end_date - timedelta(days=TRAIN_RANGE_DAYS)

    wikidata = get_wikidata(spark_session)
    sitelinks = wikidata.groupBy('id').count()
    articles = wikidata\
        .where((F.col('site') == source_wiki) |
               (F.col('site') == target_wiki))\
        .filter(~F.col('title').contains(':'))
    source_wikidata_ids = articles\
        .filter(articles.site == '%swiki' % source_language)\
        .select('id')
    target_wikidata_ids = articles\
        .filter(articles.site == '%swiki' % target_language)\
        .select('id')
    common_wikidata_ids = source_wikidata_ids.intersect(target_wikidata_ids)
    target_pageviews = get_pageviews(
        spark_session, target_language, start_date, end_date,
        articles)
    target_articles = articles\
        .alias('a')\
        .where(F.col('a.site') == target_wiki)\
        .join(sitelinks.alias('s'), F.col('a.id') == F.col('s.id'))\
        .join(target_pageviews.alias('t'),
              F.col('a.title') == F.col('t.%s_title' % target_language),
              'left_outer')\
        .select([F.col('a.id').alias('wikidata_id'),
                 F.col('a.title'),
                 F.col('s.count').alias('sitelinks_count'),
                 F.col('t.%s_normalized_rank' %
                       target_language).alias('output')])\
        .na.fill(0)
    common_articles = common_wikidata_ids\
        .alias('c')\
        .join(target_articles.alias('a'),
              F.col('c.id') == F.col('a.wikidata_id'))\
        .select('a.*')
    combined_pageviews = get_combined_pageviews(
        spark_session, wikidata, start_date, end_date)
    input_df = common_articles\
        .alias('c')\
        .join(combined_pageviews.alias('cp'),
              F.col('c.wikidata_id') == F.col('cp.id'))
    input_cols = [x for x in combined_pageviews.columns
                  if x.endswith('_pageviews') or x.endswith('_rank')]
    input_cols.append('sitelinks_count')
    vector_assembler = VectorAssembler(inputCols=input_cols,
                                       outputCol='features')
    train_data = vector_assembler\
        .transform(input_df)\
        .select(['features', 'output'])
    rf = RandomForestRegressor(
        featuresCol="features", labelCol="output")
    pipeline = Pipeline(stages=[rf])
    model = pipeline.fit(train_data)

    source_only_wikidata_ids = source_wikidata_ids.subtract(
        common_wikidata_ids)
    source_articles = articles\
        .alias('a')\
        .where(F.col('a.site') == source_wiki)\
        .join(sitelinks.alias('s'), F.col('a.id') == F.col('s.id'))\
        .select([F.col('a.id').alias('wikidata_id'),
                 F.col('a.title'),
                 F.col('s.count').alias('sitelinks_count')])
    source_only_articles = source_only_wikidata_ids\
        .alias('c')\
        .join(source_articles.alias('s'),
              F.col('c.id') == F.col('s.wikidata_id'))\
        .select('s.*')
    output_df = source_only_articles\
        .alias('s')\
        .join(combined_pageviews.alias('c'),
              F.col('s.wikidata_id') == F.col('c.id'))
    vector_assembler = VectorAssembler(
        inputCols=input_cols, outputCol='features')
    prediction_data = vector_assembler\
        .transform(output_df)\
        .select(['features'])
    predictions = model.transform(prediction_data)

    source_only_articles = source_only_articles\
        .withColumn("row_number", F.monotonically_increasing_id())
    predictions = predictions\
        .withColumn("row_number", F.monotonically_increasing_id())
    predictions = source_only_articles\
        .alias('s')\
        .join(predictions.alias('p'),
              F.col('s.row_number') == F.col('p.row_number'))\
        .select([F.col('s.wikidata_id'),
                 F.col('p.prediction').alias('normalized_rank')])
    predictions\
        .repartition(1)\
        .write\
        .csv(get_predictions_filename(start_date, end_date,
                                      source_language, target_language),
             mode='overwrite', sep='\t', header=True,
             compression='bzip2')


def main():
    """Main entry point of the script.

    Parses command line options, trains models, and makes predictions.

    """
    options = get_cmd_options()
    if validate_cmd_options(options):
        train(get_spark_session(), options)


if __name__ == '__main__':
    main()

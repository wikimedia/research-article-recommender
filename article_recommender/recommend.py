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
import os.path

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F, SparkSession


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOPSITES_FILE = BASE_DIR + '/data/topsites.tsv'
"""string: Fallback location of the TSV file that contains the top 50
Wikipedias by article count.
"""

DBLIST_FILE = BASE_DIR + '/data/wikipedia.dblist'
"""string: Fallback location of the TSV file that contains the list of
Wikipedias.
"""

WIKIDATA_DIR = '/user/joal/wmf/data/wmf/mediawiki/wikidata_parquet/20181001'
"""string: Fallback location of Wikidata dumps in parquet format.
"""

OUTPUT_DIR = '/user/bmansurov/article-recommender'
"""string: Fallback location of the directory in HDFS for storing
intermediate and final output.
"""


class NormalizedScores:
    """Class that calculates article normalized scores.
    Instantiate it and call train()."""

    TRAIN_RANGE_DAYS = 180
    """int: Number of days used for training models.
    This variable comes from the research paper. Pageviews data is gathered
    using this date range and the end_date argument passed to the script.
    """

    TOP_LANGUAGES_COUNT = 50
    """int: Number of top languages used in calculating pageviews.
    This variable comes from the research paper.
    """

    def __init__(self, spark, source_language, target_language,
                 end_date, wikidata_dir, topsites_file, output_dir):
        self.spark = spark
        self.source_language = source_language
        self.target_language = target_language
        self.end_date = end_date
        self.wikidata_dir = wikidata_dir
        self.topsites_file = topsites_file
        self.output_dir = output_dir

        self.source_wiki = '%swiki' % source_language
        self.target_wiki = '%swiki' % target_language
        self.start_date = end_date - timedelta(days=self.TRAIN_RANGE_DAYS)

    def get_wikidata(self):
        """Return Wikidata dataframe from parquet.
        Returns:
            dataframe <id, site, title>
        """
        return self.spark\
            .read\
            .parquet(self.wikidata_dir)\
            .where(F.col('typ') == 'item')\
            .select('id', F.explode('siteLinks').alias('sl'))\
            .select('id', 'sl.site', 'sl.title')

    def calculate_pageviews_and_save(self, language, articles, filename):
        """Get pageviews from Hive and save calculated data in parquet.
        Calculated includes normalized and log ranks.
        Args:
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
        pageviews = self.spark.sql(
            sql % (language, language, self.start_date.strftime('%Y-%m-%d'),
                   self.end_date.strftime('%Y-%m-%d'), language))
        target_article_count = pageviews.count()
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
        logging.debug('Calculated pageviews and saved as %s.' % filename)
        return pageviews

    def get_pageviews(self, language, articles):
        """Return pageviews with normalized ranks from file if exits, otherwise
        from Hive and save to file for later use.
        Args:
            language (string): Pageviews for the language wiki only will be
                considered
            articles (dataframe): TODO document
        Returns:
            dataframe: Pageviews with normalized and log ranks
        """
        filename = '%s/pageviews-%s-%s-%s' %\
            (self.output_dir, self.start_date, self.end_date, language)
        try:
            pageviews = self.spark.read.parquet(filename)
            logging.debug('Returning existing pageviews from %s.' % filename)
        except Exception:
            logging.debug('Starting to calculate pageviews and save as %s.'
                          % filename)
            pageviews = self.calculate_pageviews_and_save(
                language, articles, filename)
        return pageviews

    def get_top_languages(self):
        """Return to 50 wikipedia languages by article count.
        Returns:
            list<string>: e.g. ['en', 'de', 'sv', ...]
        """
        with open(self.topsites_file, 'r') as inf:
            tsv_reader = csv.reader(inf, delimiter='\t')
            next(tsv_reader)
            languages = [x[0] for x in tsv_reader]
            ll = len(languages)
            if ll != 50:
                logging.warning('We got %d top languages, and not %d.'
                                % (ll, self.TOP_LANGUAGES_COUNT))

    def calculate_combined_pageviews_and_save(self, wikidata, filename):
        """Calculate combined pageviews and save for later use in HDFS.
        Args:
            wikidata (dataframe)
            filename (string): Where to save data
        Returns:
            dataframe: Combined pageviews with normalized and log ranks for
                top 50 Wikipedias
        """
        top_languages = self.get_top_languages()
        wikidata_ids = wikidata.select('id').distinct()
        for language in top_languages:
            articles = wikidata\
                .where(F.col('site') == '%swiki' % language)\
                .filter(~F.col('title').contains(':'))
            pageviews = self.get_pageviews(language, articles)
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
        logging.debug('Calculated combined pageviews and saved as %s.'
                      % filename)
        return pageviews

    def get_combined_pageviews(self, wikidata):
        """Return combined pageviews for the top 50 Wikipedias.
        If the parquet file exists, return it, otherwise generate the file,
        save and return it.
        Args:
            wikidata (dataframe)
        Returns:
            dataframe: Combined pageviews with normalized and log ranks for
                top 50 Wikipedias
        """
        filename = '%s/combined-pageviews-%s-%s' %\
            (self.output_dir, self.start_date, self.end_date)
        try:
            pageviews = self.spark.read.parquet(filename)
            logging.debug('Returning existing combined pageviews from %s.'
                          % filename)
        except Exception:
            logging.debug(
                'Starting to calculate combined pageviews and save as %s.'
                % filename
            )
            pageviews = self.calculate_combined_pageviews_and_save(
                wikidata, filename
            )
        return pageviews

    def train(self):
        """Train models and create article normalized scores."""
        logging.debug('Starting to train.')
        wikidata = self.get_wikidata()
        sitelinks = wikidata.groupBy('id').count()
        articles = wikidata\
            .where((F.col('site') == self.source_wiki) |
                   (F.col('site') == self.target_wiki))\
            .filter(~F.col('title').contains(':'))
        source_wikidata_ids = articles\
            .filter(articles.site == self.source_wiki)\
            .select('id')
        target_wikidata_ids = articles\
            .filter(articles.site == self.target_wiki)\
            .select('id')
        common_wikidata_ids = source_wikidata_ids.intersect(
            target_wikidata_ids)
        target_pageviews = self.get_pageviews(self.target_language, articles)
        target_articles = articles\
            .alias('a')\
            .where(F.col('a.site') == self.target_wiki)\
            .join(sitelinks.alias('s'), F.col('a.id') == F.col('s.id'))\
            .join(target_pageviews.alias('t'),
                  F.col('a.title') == F.col(
                      't.%s_title' % self.target_language),
                  'left_outer')\
            .select([F.col('a.id').alias('wikidata_id'),
                     F.col('a.title'),
                     F.col('s.count').alias('sitelinks_count'),
                     F.col('t.%s_normalized_rank' %
                           self.target_language).alias('output')])\
            .na.fill(0)
        common_articles = common_wikidata_ids\
            .alias('c')\
            .join(target_articles.alias('a'),
                  F.col('c.id') == F.col('a.wikidata_id'))\
            .select('a.*')
        logging.debug('Got common articles.')
        combined_pageviews = self.get_combined_pageviews(wikidata)
        input_df = common_articles\
            .alias('c')\
            .join(combined_pageviews.alias('cp'),
                  F.col('c.wikidata_id') == F.col('cp.id'))
        input_cols = [x for x in combined_pageviews.columns
                      if x.endswith('_pageviews') or x.endswith('_rank')]
        input_cols.append('sitelinks_count')
        vector_assembler = VectorAssembler(inputCols=input_cols,
                                           outputCol='features')
        logging.debug('Starting to train a model.')
        train_data = vector_assembler\
            .transform(input_df)\
            .select(['features', 'output'])
        rf = RandomForestRegressor(
            featuresCol="features", labelCol="output")
        pipeline = Pipeline(stages=[rf])
        model = pipeline.fit(train_data)
        logging.debug('Finished training the model.')

        source_only_wikidata_ids = source_wikidata_ids.subtract(
            common_wikidata_ids)
        source_articles = articles\
            .alias('a')\
            .where(F.col('a.site') == self.source_wiki)\
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
        logging.debug('Starting to fit the model.')
        vector_assembler = VectorAssembler(
            inputCols=input_cols, outputCol='features')
        prediction_data = vector_assembler\
            .transform(output_df)\
            .select(['features'])
        predictions = model.transform(prediction_data)
        logging.debug('Finished fitting the model.')

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
        filename = '%s/predictions-%s-%s-%s-%s.tsv' %\
            (self.output_dir, self.start_date, self.end_date,
             self.source_language, self.target_language)
        predictions\
            .repartition(1)\
            .write\
            .csv(filename, mode='overwrite', sep='\t', header=True,
                 compression='bzip2')
        logging.debug('Saved predictions file as %.' % filename)
        logging.debug('Finished training.')


def get_cmd_options():
    """Return command line options passed to the script.
    Returns:
        object: Arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        description='Generates article normalized scores.')
    parser.add_argument('end_date',
                        help='End date in the yyyymmdd format.',
                        type=lambda x: datetime.strptime(x, "%Y%m%d").date())
    parser.add_argument('source_language',
                        help='Source language code, e.g. en or uz.')
    parser.add_argument('target_language',
                        help='Target language code, e.g. en or uz.')
    parser.add_argument('--topsites_file',
                        help='Location of top Wikipedias by edit count.',
                        default=TOPSITES_FILE)
    parser.add_argument('--dblist-file',
                        help='Location of list of Wikipedias.',
                        default=DBLIST_FILE)
    parser.add_argument('--wikidata_dir',
                        help='Location of Wikidata dumps in HDFS.',
                        default=WIKIDATA_DIR)
    parser.add_argument('--output_dir',
                        help='Output location in HDFS.',
                        default=OUTPUT_DIR)
    return parser.parse_args()


def get_wikipedia_dblist(dblist_file):
    """Return a set of Wikipedias from a file.
    Args:
        dblist_file (string): Name of the file that contains the list of
          Wikipedias
    Returns:
        set: Wikipedias, e.g. {'enwiki', 'uzwiki', ... }
    """
    with open(dblist_file, 'r') as inf:
        return set([x.strip() for x in inf.readlines()])


def validate_cmd_options(options):
    """Validate command line options passed by the user.
    Args:
        options (object)
    Returns:
        bool: In case of error, False is returned. Otherwise, True.
    """
    wikipedias = get_wikipedia_dblist(options.dblist)
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
    if options.end_date > datetime.today().date():
        logging.error('End date cannot be later than today: %s.' %
                      options.end_date)
        return False
    return True


def main():
    """Main entry point of the script.
    Parses command line options, trains models, and makes predictions.
    """
    options = get_cmd_options()
    if validate_cmd_options(options):
        spark = SparkSession\
            .builder\
            .appName('article-recommender')\
            .enableHiveSupport()\
            .getOrCreate()
        normalized_scores = NormalizedScores(
            spark,
            options.source_language,
            options.target_language,
            options.end_date,
            options.wikidata_dir,
            options.topsites_file,
            options.output_dir
        )
        normalized_scores.train()


if __name__ == '__main__':
    main()

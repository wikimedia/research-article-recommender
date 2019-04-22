#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Generate article normalized scores.

This script allows generating article normalized scores that can be used
for sorting missing articles.

$ python recommend.py en-ru,ru-uz 20190131

Todo:
    * add unit tests
"""

import argparse
from datetime import datetime
import os.path

from article_recommender.util\
    import csv_to_list, get_spark_session, log, timeit
from article_recommender.normalizedscores import NormalizedScores


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

TMP_DIR = '/tmp/article-recommender'
"""string: Fallback location for storing temporary files"""

OUTPUT_DIR = '/user/bmansurov/article-recommender'
"""string: Fallback location of the directory in HDFS for storing
intermediate and final output.
"""


@timeit
def get_cmd_options():
    """Return command line options passed to the script.
    Also apply defaults and transormations.
    Returns:
        object
    """
    parser = argparse.ArgumentParser(
        description="""Generate article normalized scores.

        It's faster to supply all language pairs at once rather than
        supplying them in parts. For example, if you want to train
        en-ru, ru-uz, and ko-uz, do so at once:

        `python recommend.py en-ru,ru-uz,ko-uz 20190420`.

        Don't do this (it will take much longer):

        `python recommend.py en-ru 20190420`,
        `python recommend.py ru-uz 20190420`,
        `python recommend.py ko-uz 20190420`.
        """)
    parser.add_argument('language_pairs',
                        help='Comma separated list of source and target '
                        'language codes, e.g. ru-uz,en-ko,ko-uz.',
                        type=lambda pairs: [
                            x.split('-') for x in pairs.split(',')
                        ])
    parser.add_argument('end_date',
                        help='End date in the yyyymmdd format.',
                        type=lambda x: datetime.strptime(x, "%Y%m%d").date())
    parser.add_argument('--spark_app_name',
                        help='Name of the Spark application.',
                        default='article-recommender')
    parser.add_argument('--topsites_file',
                        help='List of top Wikipedias by edit count.',
                        default=TOPSITES_FILE)
    parser.add_argument('--dblist_file',
                        help='Location of list of Wikipedias.',
                        default=DBLIST_FILE)
    parser.add_argument('--wikidata_dir',
                        help='Location of Wikidata dumps in HDFS.',
                        default=WIKIDATA_DIR)
    parser.add_argument('--output_dir',
                        help='Output location in HDFS.',
                        default=OUTPUT_DIR)
    parser.add_argument('--tmp_dir',
                        help='Location for saving temporary files in HDFS.',
                        default=TMP_DIR)
    return parser.parse_args()


@timeit
def get_wikipedia_dblist(spark, dblist_file):
    """Return a set of Wikipedias from a file.
    Args:
        spark (SparkSession)
        dblist_file (string): Name of the file that contains the list of
          Wikipedias
    Returns:
        set: Wikipedias, e.g. {'enwiki', 'uzwiki', ... }
    """
    dblist = csv_to_list(spark, dblist_file, headerp=False)
    return set([x[0].strip() for x in dblist])


@timeit
def validate_cmd_options(spark, options):
    """Validate command line options passed by the user.
    Args:
        spark (SparkSession)
        options (object)
    Returns:
        bool: In case of error, False is returned. Otherwise, True.
    """
    wikipedias = get_wikipedia_dblist(spark, options.dblist_file)
    for source, target in options.language_pairs:
        if '%swiki' % source not in wikipedias:
            log('Unrecognized source language: %s' % source, 'error')
            return False
        if '%swiki' % target not in wikipedias:
            log('Unrecognized target language: %s' % target, 'error')
            return False
        if source == target:
            log('Source and target languages cannot be the same.', 'error')
            return False
    if options.end_date > datetime.today().date():
        log('End date cannot be later than today: %s.' %
            options.end_date, 'error')
        return False
    return True


@timeit
def main():
    """Main entry point of the script.
    Parses command line options, trains models, and makes predictions.
    """
    options = get_cmd_options()
    spark = get_spark_session(options.spark_app_name)
    if validate_cmd_options(spark, options):
        log('Options: %s' % str(options), 'debug')
        normalized_scores = NormalizedScores(
            spark,
            options.language_pairs,
            options.end_date,
            options.wikidata_dir,
            options.topsites_file,
            options.output_dir,
            options.tmp_dir
        )
        normalized_scores.train()


if __name__ == '__main__':
    main()

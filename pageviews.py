from __future__ import division
from datetime import datetime, timedelta
import sys

from pyspark.sql import functions as F, SparkSession


if len(sys.argv) != 3:
    print("Pass in language code and end date, e.g. pageviews.py uz 05/31/2018")
    exit(1)

lang = sys.argv[1]
end_date = datetime.strptime(sys.argv[2], "%m/%d/%Y").date()

wiki = '%swiki' % lang

spark = SparkSession.builder\
    .master('yarn')\
    .appName('translation-recommendation')\
    .config('spark.executor.memory', '8G')\
    .config('spark.cores.max', '4')\
    .config('spark.driver.memory', '8G')\
    .config("spark.driver.maxResultSize", "8G")\
    .enableHiveSupport()\
    .getOrCreate()
print('---> Started a Spark session')

# Get Wikidata items.
wikidata = spark\
    .read\
    .parquet('/user/joal/wikidata/parquet')\
    .select('id', F.explode('siteLinks').alias('sl'))\
    .select(F.col('id').alias('%s_id' % lang), 'sl.site', 'sl.title')
print('---> Read Wikidata parquet')

# Get articles in the main namespace for the language pair.
articles = wikidata\
    .where(F.col('site') == wiki)\
    .filter(~F.col('title').contains(':'))
print('---> Got articles titles for the wiki')

start_date = end_date - timedelta(days=180)
sql = """
    SELECT page_title AS %s_title, sum(view_count) as %s_pageviews
    FROM wmf.pageview_hourly
    WHERE
        ((year = %d AND month < %d) OR (year = %d AND month >= %d))
        AND project="%s.wikipedia"
        AND agent_type="user"
        AND instr(page_title, ':')=0
    GROUP BY page_title
    ORDER BY %s_pageviews
"""
pageviews = spark.sql(
    sql % (lang, lang, end_date.year, end_date.month, start_date.year,
           start_date.month, lang, lang))
print('---> Queried target pageviews')

target_article_count = pageviews.count()
# Normalize pageviews
pageviews = pageviews\
    .withColumn('%s_normalized_rank' % lang,
                F.col('%s_pageviews' % lang) / target_article_count)\
    .withColumn('%s_log_rank' % lang,
                F.log(F.col('%s_pageviews' % lang)))

pageviews = pageviews\
    .alias('p')\
    .join(articles.alias('a'), F.col('a.title') == F.col('p.%s_title' % lang))

pageviews = pageviews.drop('title').drop('site')

pageviews.write.parquet(
    "/user/bmansurov/%s-pageviews-%s-%s.parquet" %
    (wiki, start_date.strftime('%m%d%Y'), end_date.strftime('%m%d%Y')))
print('---> Saved %s pageviews to as a parquet.' % wiki)

from datetime import datetime, timedelta
import sys

import pandas as pd
from pyspark.sql import functions as F, SparkSession

BASE_DIR = '/user/bmansurov/translation-recommendation/'

if len(sys.argv) != 2:
    print("Pass in the end date, e.g. combined_pageviews.py 05/31/2018")
    exit(1)

end_date = datetime.strptime(sys.argv[1], "%m/%d/%Y").date()
start_date = end_date - timedelta(days=180)

spark = SparkSession.builder\
    .master('yarn')\
    .appName('article-recommender')\
    .enableHiveSupport()\
    .getOrCreate()
print('---> Started a Spark session')

wikidata = spark\
    .read\
    .parquet('/user/joal/wmf/data/wmf/mediawiki/wikidata_parquet/20181001')\
    .select('id')\
    .distinct()

top_wikipedias = list(
    pd.read_csv(
        'topsites.tsv', sep='\t', names=['wiki', 'article_count']
    )['wiki'][1:]
)
for lang in top_wikipedias:
    try:
        pageviews = spark\
            .read\
            .parquet('%s%swiki-pageviews-%s-%s.parquet' %
                     (BASE_DIR,
                      lang,
                      start_date.strftime('%m%d%Y'),
                      end_date.strftime('%m%d%Y')))
    except Exception:
        print('Skipping %s as no pageviews file found.' % lang)
        continue

    wikidata = wikidata\
        .alias('w')\
        .join(pageviews.alias('p'),
              F.col('w.id') == F.col('p.%s_id' % lang),
              'left_outer')\
        .na.fill({
            '%s_pageviews' % lang: 0,
            '%s_normalized_rank' % lang: 0,
            '%s_log_rank' % lang: 0
        })
    wikidata = wikidata.drop('%s_id' % lang).drop('%s_title' % lang)

wikidata.write.parquet(
    "%scombined-pageviews-%s-%s.parquet" %
    (BASE_DIR,
     start_date.strftime('%m%d%Y'),
     end_date.strftime('%m%d%Y')))

from __future__ import division
from datetime import date, timedelta
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import HiveContext
from pyspark.sql import functions as F

# TODO don't hardcode paths, dates, etc.

# Check if we got source and target languages.
if len(sys.argv) != 3:
    print("Pass in source and target languages, e.g. train.py ru uz")
    exit(1)
source_lang, target_lang = sys.argv[1:]
source_wiki = '%swiki' % source_lang
target_wiki = '%swiki' % target_lang

# Check if Wikipedias in given source and target languages exist.
wikipedias = set(pd.read_csv('wikipedia.dblist', names=['wiki'])['wiki'])
if source_wiki not in wikipedias or target_wiki not in wikipedias:
    print("Either source or target language Wikipedia doesn't exist.")
    exit(1)

spark = SparkSession\
    .builder\
    .appName("TranslationRecommendation")\
    .getOrCreate()
sql_context = SQLContext(spark.sparkContext)
hive_context = HiveContext(spark.sparkContext)


# Get Wikidata items.
wikidata = sql_context\
    .read\
    .parquet('/user/joal/wikidata/parquet')\
    .select('id', F.explode('siteLinks').alias('sl'))\
    .select('id', 'sl.site', 'sl.title')

# Wikidata sitelinks
sitelinks = wikidata.groupBy('id').count()

# Get articles in the main namespace for the language pair.
articles = wikidata\
    .where((F.col('site') == source_wiki) | (F.col('site') == target_wiki))\
    .filter(~F.col('title').contains(':'))

# Get pageviews for the target language over the last 6 months.
end = date.today()
start = end - timedelta(days=180)
sql = """
    SELECT page_title, sum(view_count) as pageviews
    FROM wmf.pageview_hourly
    WHERE
        ((year = %d AND month < %d) OR (year = %d AND month >= %d))
        AND project="%s.wikipedia"
        AND agent_type="user"
        AND instr(page_title, ':')=0
    GROUP BY page_title
    ORDER BY pageviews
"""
target_pageviews = hive_context.sql(
    sql % (end.year, end.month, start.year, start.month,
           source_lang))
# Normalize pageviews
target_article_count = target_pageviews.count()
target_pageviews = target_pageviews.withColumn(
    'normalized_rank',
    (F.col('pageviews') / target_article_count)
)

# Get common and source only articles
source_wikidata_ids = articles\
    .filter(articles.site == '%swiki' % source_lang)\
    .select('id')
target_wikidata_ids = articles\
    .filter(articles.site == '%swiki' % target_lang)\
    .select('id')
common_wikidata_ids = source_wikidata_ids.intersect(target_wikidata_ids)
target_articles = articles\
    .alias('a')\
    .where(F.col('a.site') == target_wiki)\
    .join(sitelinks.alias('s'), F.col('a.id') == F.col('s.id'))\
    .join(target_pageviews.alias('t'),
          F.col('a.title') == F.col('t.page_title'))\
    .select([F.col('a.id').alias('wikidata_id'),
            F.col('a.title'),
            F.col('s.count').alias('sitelinks_count'),
             F.col('t.normalized_rank')])
common_articles = common_wikidata_ids\
    .alias('c')\
    .join(target_articles.alias('a'),
          F.col('c.id') == F.col('a.wikidata_id'))\
    .select('a.*')

# Train a model
vector_assembler = VectorAssembler(
    inputCols=['sitelinks_count'], outputCol='features')
train_data = vector_assembler\
    .transform(common_articles)\
    .select(['features', 'normalized_rank'])

rf = RandomForestRegressor(
    featuresCol="features", labelCol="normalized_rank")
pipeline = Pipeline(stages=[rf])
model = pipeline.fit(train_data)

# Predict missing data
source_only_wikidata_ids = source_wikidata_ids.subtract(common_wikidata_ids)
source_articles = articles\
    .alias('a')\
    .where(F.col('a.site') == source_wiki)\
    .join(sitelinks.alias('s'), F.col('a.id') == F.col('s.id'))\
    .select([F.col('a.id').alias('wikidata_id'),
             F.col('a.title'),
             F.col('s.count').alias('sitelinks_count')])
vector_assembler = VectorAssembler(
    inputCols=['sitelinks_count'], outputCol='features')
prediction_data = vector_assembler\
    .transform(source_articles)\
    .select(['features'])
predictions = model.transform(prediction_data)

source_articles = source_articles\
    .withColumn("row_number", F.monotonically_increasing_id())
predictions = predictions\
    .withColumn("row_number", F.monotonically_increasing_id())

predictions = source_articles\
    .alias('s')\
    .join(predictions.alias('p'),
          F.col('s.row_number') == F.col('p.row_number'))\
    .select([F.col('s.wikidata_id'),
             F.col('p.prediction').alias('normalized_rank')])

predictions_filename = '%s-%s_%d_%d-%d_%d.tsv' % (
    source_wiki, target_wiki, start.year, start.month,
    end.year, end.month
)
predictions\
    .toPandas()\
    .to_csv(predictions_filename, sep='\t', index=False)
print('Saved predictions to %s' % predictions_filename)

spark.stop()

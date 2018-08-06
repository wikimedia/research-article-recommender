from __future__ import division
from datetime import datetime, timedelta
import sys

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F, SparkSession

# TODO don't hardcode paths, dates, etc.
BASE_DIR = '/user/bmansurov/translation-recommendation/'

# Check if we got source and target languages.
if len(sys.argv) != 4:
    print('Pass in source and target languages, along with the '
          'end date, e.g. train.py ru uz 05/31/2018')
    exit(1)
source_lang, target_lang = sys.argv[1:3]
source_wiki = '%swiki' % source_lang
target_wiki = '%swiki' % target_lang

end_date = datetime.strptime(sys.argv[3], "%m/%d/%Y").date()
start_date = end_date - timedelta(days=180)

# Check if Wikipedias in given source and target languages exist.
wikipedias = set(pd.read_csv('wikipedia.dblist', names=['wiki'])['wiki'])
if source_wiki not in wikipedias or target_wiki not in wikipedias:
    print("Either source or target language Wikipedia doesn't exist.")
    exit(1)

spark = SparkSession.builder\
    .master('yarn')\
    .appName('translation-recommendation')\
    .enableHiveSupport()\
    .getOrCreate()
print('---> Started a Spark session')

# Get Wikidata items.
wikidata = spark\
    .read\
    .parquet('/user/joal/wmf/data/wmf/mediawiki/wikidata_parquet/20180108')\
    .select('id', F.explode('siteLinks').alias('sl'))\
    .select('id', 'sl.site', 'sl.title')
print('---> Read Wikidata parquet')

# Wikidata sitelinks
sitelinks = wikidata.groupBy('id').count()
print('---> Got wikidata sitelinks')

# Get articles in the main namespace for the language pair.
articles = wikidata\
    .where((F.col('site') == source_wiki) | (F.col('site') == target_wiki))\
    .filter(F.col('id').startswith('Q'))\
    .filter(~F.col('title').contains(':'))
print('---> Got articles titles for the source and target languages')

# Get pageviews for the target language over the last 6 months.
target_pageviews = spark\
    .read\
    .parquet('%s%swiki-pageviews-%s-%s.parquet' %
             (BASE_DIR,
              target_lang,
              start_date.strftime('%m%d%Y'),
              end_date.strftime('%m%d%Y')))
print('---> Read target pageviews parquet')

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
          F.col('a.title') == F.col('t.%s_title' % target_lang), 'left_outer')\
    .select([F.col('a.id').alias('wikidata_id'),
             F.col('a.title'),
             F.col('s.count').alias('sitelinks_count'),
             F.col('t.%s_normalized_rank' % target_lang).alias('output')])\
    .na.fill(0)
common_articles = common_wikidata_ids\
    .alias('c')\
    .join(target_articles.alias('a'),
          F.col('c.id') == F.col('a.wikidata_id'))\
    .select('a.*')
print('---> Computed common articles between the target and source languages')

# Read combined pagviews
combined_pageviews = spark\
    .read\
    .parquet('%scombined-pageviews-%s-%s.parquet' %
             (BASE_DIR,
              start_date.strftime('%m%d%Y'),
              end_date.strftime('%m%d%Y')))
print('---> Read combined pageviews parquet')

# Prepare training data
input_df = common_articles\
    .alias('c')\
    .join(combined_pageviews.alias('cp'),
          F.col('c.wikidata_id') == F.col('cp.id'))

# Train a model
input_cols = [x for x in combined_pageviews.columns
              if x.endswith('_pageviews') or x.endswith('_rank')]
input_cols.append('sitelinks_count')
vector_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
train_data = vector_assembler\
    .transform(input_df)\
    .select(['features', 'output'])
rf = RandomForestRegressor(
    featuresCol="features", labelCol="output")
pipeline = Pipeline(stages=[rf])
model = pipeline.fit(train_data)
print('---> Trained a model')

# Predict missing data
source_only_wikidata_ids = source_wikidata_ids.subtract(common_wikidata_ids)
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
print('---> Made predictions')

# Save predictions to a file
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
predictions_filename = 'predictions-%s-%s_%s-%s.tsv' % (
    start_date.strftime('%m%d%Y'), end_date.strftime('%m%d%Y'),
    source_wiki, target_wiki)
predictions\
    .toPandas()\
    .to_csv(predictions_filename, sep='\t', index=False)
print('---> Saved predictions to %s' % predictions_filename)

spark.stop()

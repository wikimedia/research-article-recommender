"""Outputs Wikidata items that are similar. For example, Q4645365 is
similar to Q4357225 according to this script. The results can be used to
filter out similar items from the recommendation results."""

import string

import numpy as np
from pyspark.ml.feature import StopWordsRemover, Word2Vec
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType


config = {
    'spark': {
        'app_name': 'redirect-resolver',
        'executor_memory': '16G',
        'cores_max': '4',
        'driver_memory': '16G',
        'driver_max_result_size': '16G'
    },
    'lang': 'english',
    'lang_code': 'en',
    'wiki': 'enwiki',
    'wikidata_dump': '/user/joal/wmf/data/wmf/mediawiki/wikidata_parquet/'
                     '20181001',
    # TODO: this is arbitrary, test various values
    'min_similarity_score': 0.75,
    'output_filename': 'similar_items-en-20181001.tsv'
}

spark = SparkSession.builder\
    .master('yarn')\
    .appName(config['spark']['app_name'])\
    .config('spark.executor.memory', config['spark']['executor_memory'])\
    .config('spark.cores.max', config['spark']['cores_max'])\
    .config('spark.driver.memory', config['spark']['driver_memory'])\
    .config('spark.driver.maxResultSize',
            config['spark']['driver_max_result_size'])\
    .enableHiveSupport()\
    .getOrCreate()
print('---> Started a Spark session')


# Get Items from the Wikidata dump and explode sitelinks and take only
# Wikipedia articles that are in the Main namespace
wikidata = spark\
    .read\
    .parquet(config['wikidata_dump'])\
    .where(F.col('typ') == 'item')\
    .select('id', F.explode('siteLinks').alias('sl'),
            'labels', 'descriptions')\
    .select('id', 'sl.site', 'sl.title', 'labels', 'descriptions')\
    .where(~F.col('title').contains(':'))
print('---> Read Wikidata parquet')
# wikidata.take(1)

# # TODO: filter out disambiguation pages, e.g. https://www.wikidata.org/wiki/Q12777830
# # Filter out Wikidata items that point to other Wikipedias
# wikidata_item_and_title = wikidata\
#     .where(F.col('site') == config['wiki'])\
#     .select('id', 'title')
# print('---> Got articles titles for the wiki')

# wikidata_item_and_title.take(1)


# Get wikidata items that have a label and description for the current language
wikidata_label_and_description = wikidata\
    .select('id', 'title', 'site', F.explode('labels'), 'descriptions')\
    .select('id', 'title', 'site', F.col('key').alias('lang'),
            F.col('value').alias('label'), 'descriptions')\
    .where(F.col('lang') == config['lang_code'])\
    .select('id', 'title', 'site', 'label', F.explode('descriptions'))\
    .where(F.col('key') == config['lang_code'])\
    .select('id', 'title', 'site', 'label', F.col('value').alias('description'))
print('---> Got wikidata labels and descriptions')
# wikidata_label_and_description.take(1)

# Get Wikipedia redirects in the Main namespace for the language and
# remove disambiguation pages from the results
sql = """
    SELECT
      replace(r.rd_from_page_title, '_', ' ') as source,
      replace(r.rd_to_page_title, '_', ' ') AS target
    FROM prod.redirect r
    LEFT JOIN (SELECT * FROM prod.page_props
               WHERE lang='%s'
                 AND propname='disambiguation') p
      ON r.rd_to_page_id=p.page_id
    WHERE
        r.rd_to_page_is_redirect=false
        AND r.rd_to_page_namespace=0
        AND r.rd_from_page_is_redirect=true
        AND r.rd_from_page_namespace=0
        AND r.lang='%s'
        AND p.page_id is NULL
"""
redirects = spark.sql(sql % (config['lang_code'], config['lang_code']))
print('---> Got redirects')
# print(redirects.take(1))
# redirects.count()

# Filter out redirects whose source doesn't have a corresponding Wikidata label
# Also filter out Wikidata items who already have sitelinks to the current wiki
redirects_wikidata = redirects.alias('r')\
    .join(wikidata_label_and_description.alias('w'),
          F.lower(F.col('r.source')) == F.lower(F.col('w.label')))\
    .where(~(F.col('w.site') == config['wiki']))\
    .select(F.col('r.source').alias('source_title'),
            F.col('r.target').alias('target_title'),
            F.col('w.id').alias('source_id'),
            F.col('w.label').alias('source_label'),
            F.col('w.description').alias('source_description'))
# print(redirects_wikidata.take(1))
# redirects_wikidata.count()


# Prepare redirects input by splitting and cleaning source descriptions
def description_splitter(description):
    "Split Wikidata description into words and remove punctuation."
    table = str.maketrans({key: None for key in string.punctuation})
    return [x for x in [s.translate(table) for s in
                        description.split(' ')] if x]


description_splitter_udf = F.udf(description_splitter, ArrayType(StringType()))
redirects_wikidata = redirects_wikidata\
    .withColumn('description_words',
                description_splitter_udf('source_description'))
# redirects_wikidata.take(1)

# Remove stopwords from input
stopwords = StopWordsRemover.loadDefaultStopWords(config['lang'])
remover = StopWordsRemover(inputCol="description_words",
                           outputCol="description_words_clean",
                           stopWords=stopwords)
redirects_wikidata = remover.transform(redirects_wikidata)
# redirects_wikidata.take(1)

# Train the model
# TODO: tune parameters of Word2Vec
word2Vec = Word2Vec(inputCol="description_words_clean",
                    outputCol="description_vector")
model = word2Vec.fit(redirects_wikidata)
redirects_wikidata = model.transform(redirects_wikidata)
# redirects_wikidata.take(1)
# redirects_wikidata.count()

# Find unique targets in redirects, add Wikidata ID and description
redirects_unique_targets = redirects.alias('r')\
    .select('r.target')\
    .dropDuplicates()\
    .join(wikidata_label_and_description.alias('w'),
          F.col('r.target') == F.col('w.title'))\
    .select(F.col('r.target').alias('target_title'),
            F.col('w.id').alias('target_id'),
            F.col('w.description').alias('target_description'))
# redirects_unique_targets.take(1)

redirects_unique_targets = redirects_unique_targets\
    .withColumn('description_words',
                description_splitter_udf('target_description'))
# redirects_target_input.take(1)

# Remove stopwords from input
remover = StopWordsRemover(inputCol="description_words",
                           outputCol="description_words_clean",
                           stopWords=stopwords)
redirects_unique_targets = remover.transform(redirects_unique_targets)
# redirects_unique_targets.take(1)

# Generate target vectors
redirects_unique_targets = model.transform(redirects_unique_targets)
# redirects_unique_targets.take(1)
# redirects_unique_targets.count()

# Combine source redirects with target redirects vectors
# Drop items when the source and target point to the same Wikidata item
redirects_combined = redirects_wikidata.alias('r')\
    .join(redirects_unique_targets.alias('t'),
          F.col('r.target_title') == F.col('t.target_title'))\
    .select('r.source_title', 'r.source_id', 'r.source_label',
            'r.source_description',
            F.col('r.description_vector').alias('source_description_vector'),
            't.target_title', 't.target_id', 't.target_description',
            F.col('t.description_vector').alias('target_description_vector'))\
    .where(~(F.col('r.source_id') == F.col('t.target_id')))\
    .dropDuplicates()
# redirects_combined.take(1)
# redirects_combined.count()


def cosine_similarity(v, u):
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)))


cosine_similarity_udf = F.udf(cosine_similarity, FloatType())
redirects_combined = redirects_combined\
    .withColumn('cosine_similarity',
                cosine_similarity_udf(
                    F.col('source_description_vector'),
                    F.col('target_description_vector')))
# redirects_combined.count()
# redirects_combined.take(3)

similar_items = redirects_combined\
    .where(F.col('cosine_similarity') >= config['min_similarity_score'])
# similar_items.count()
# similar_items.take(3)

similar_items\
    .select('source_title', 'source_id', 'source_label', 'source_description',
            'target_title', 'target_id', 'target_description',
            'cosine_similarity')\
    .toPandas()\
    .to_csv(config['output_filename'], sep='\t', index=False)
print('---> Saved similar_items to %s' % config['output_filename'])

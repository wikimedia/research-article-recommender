# research-translation-recommendation-models
scripts used in building models that suggest article recommendations

## Research
Suggest Wikipedia articles for translation: https://arxiv.org/abs/1604.03235


## How to run
0. `ssh stat1007`
1. Clone this repo and `cd` into it.
2. Generate top 50 Wikipedias by article count:
   `python topsites.py 05/31/2018 > topsites.tsv`
3. For each top site, calculate pageviews:
   `PYSPARK_DRIVER_PYTHON=python2 spark2-submit --master yarn --executor-memory 32G --executor-cores 4 --driver-memory 32G --conf spark.driver.maxResultSize=32G topsites_pageviews.py 05/31/2018`
4. Combine pageviews into a single data frame:
   `PYSPARK_DRIVER_PYTHON=python2 spark2-submit --master yarn --executor-memory 32G --executor-cores 4 --driver-memory 32G --conf spark.driver.maxResultSize=32G combined_pageviews.py 05/31/2018`
5. Get the top 10 language pairs from ContentTranslation:
   `echo "USE wikishared; SELECT translation_source_language, translation_target_language, count(translation_id) as count FROM cx_translations WHERE translation_status='published' GROUP BY translation_source_language, translation_target_language ORDER BY count DESC LIMIT 10;" | mysql -h analytics-store.eqiad.wmnet -A > language_pairs.txt`
6. Make predictions for the language pairs from ContentTranslation:
   `PYSPARK_DRIVER_PYTHON=python2 spark2-submit --master yarn --executor-memory 32G --executor-cores 4 --driver-memory 32G --conf spark.driver.maxResultSize=32G toplanguages_train.py 05/31/2018`

   Here `ru` is the source language and `uz` is the target language. The
   script will create a prediction file (tsv) in the current directory.
   The file will looks something like this:

   wikidata_id	normalized_rank\
   Q1031683	0.00114945421299\
   Q7240	0.00641891887845\
   Q7693627	0.00114945421299\
   Q80811	0.00437726904659

   The higher the normalized rank, the more page views an article is
   predicted to receive if created in the target language.

## Misc
1. Generate language list from dblist:
   `cat wikipedia.dblist | rev | cut -c 5- | rev > wikipedia.langlist`
2. Outputting similar Wikidata items:
   `PYSPARK_DRIVER_PYTHON=python2 spark2-submit --master yarn --executor-memory 32G --executor-cores 4 --driver-memory 32G --conf spark.driver.maxResultSize=32G wikidata_similar_items.py`

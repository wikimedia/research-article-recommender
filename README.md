# research-translation-recommendation-models
scripts used in building models that suggest translation recommendations

## How to run
0. ssh stat1005
1. clone this repo and cd into it
2. Run: `PYSPARK_DRIVER_PYTHON=python2 spark2-submit train.py ru uz`

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

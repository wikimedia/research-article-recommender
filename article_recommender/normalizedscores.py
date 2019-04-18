from datetime import timedelta

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import functions as F, Window

from util import csv_to_list, log, timeit


TRAIN_RANGE_DAYS = 180
"""int: Number of days used for training models.
This variable comes from the research paper. Pageviews data is gathered
using this date range and the end_date argument passed to the script.
"""

TOP_LANGUAGES_COUNT = 50
"""int: Number of top languages used in calculating pageviews.
This variable comes from the research paper.
"""


class NormalizedScores:
    """Class that calculates article normalized scores.
    Instantiate it and call train().
    """

    def __init__(self, spark, language_pairs,
                 end_date, wikidata_dir, topsites_file, output_dir,
                 tmp_dir):
        """Args:
          spark (SparkSession)
          language_pairs (str): List of lists of language pairs, where
            the first language is the source language and the second one
            is the target language, e.g.: (('en', 'ru'), ('ru', 'uz')).
          end_date (date): The days until when pageviews are considered.
          wikidata_dir (str): Location of Wikidata dumps in HDFS.
          topsites_file (str): Location of the file that contains top
            Wikipedias by edit count.
          output_dir (str): Where to save normalized scores.
          tmp_dir (str): Location for storing temporary data.
        """
        self.spark = spark
        self.language_pairs = language_pairs
        self.end_date = end_date
        self.wikidata_dir = wikidata_dir
        self.topsites_file = topsites_file
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir

        self.start_date = end_date - timedelta(days=TRAIN_RANGE_DAYS)
        self.language_pairs_set = set(
            [language for pair in self.language_pairs
             for language in pair]
        )
        self.top_languages = self.get_top_languages()

    @timeit
    def get_wikidata(self):
        """Return Wikidata dataframe from parquet.
        Only articles in the main name space are returned.
        Returns:
            dataframe <id (string), site (string), title (string)>
        """
        return self.spark\
            .read\
            .parquet(self.wikidata_dir)\
            .where(F.col('typ') == 'item')\
            .select('id', F.explode('siteLinks').alias('sl'))\
            .select('id', 'sl.site', 'sl.title')\
            .filter(~F.col('title').contains(':'))

    @timeit
    def get_top_languages(self):
        """Return top Wikipedia languages by article count.
        Returns:
            set<string>: e.g. {'en', 'de', 'sv', ...}
        """
        topsites = csv_to_list(self.spark, self.topsites_file)
        languages = {x[0] for x in topsites}
        ll = len(languages)
        if ll != TOP_LANGUAGES_COUNT:
            log('We got %d top languages, and not %d.'
                % (ll, TOP_LANGUAGES_COUNT), 'warning')
        return languages

    @timeit
    def get_pageviews_from_hive(self, languages):
        """Return page view counts for given languages.
        Returns:
            dataframe
        """
        sql = """
            SELECT TRIM(REGEXP_REPLACE(page_title, "_", " ")) as page_title,
                   CONCAT(SPLIT(project, "[\.]")[0], "wiki") as wiki,
                   SUM(view_count) as view_count
            FROM wmf.pageview_hourly
            WHERE
              TO_DATE(CONCAT(year, "-", month, "-", day)) >= "%s" AND
              TO_DATE(CONCAT(year, "-", month, "-", day)) <= "%s" AND
              project in (%s) AND
              agent_type = "user" AND
              -- Special value when Page ID could not be extracted
              page_title != '-' AND
              instr(page_title, ':') = 0
            GROUP BY page_title, project
            """
        projects = ','.join(['"%s.wikipedia"' % x for x in languages])
        pageviews = self.spark.sql(
            sql % (self.start_date.strftime('%Y-%m-%d'),
                   self.end_date.strftime('%Y-%m-%d'),
                   projects)
        )
        return pageviews

    @timeit
    def calculate_pageviews(self, wikidata, languages):
        """Calculate pageviews, normalized ranks, and log ranks.
        Args:
            wikidata (dataframe)
            languages (list of strings)
        Returns:
            dataframe
        """
        pageviews = self.get_pageviews_from_hive(languages)
        pageviews = pageviews\
            .alias('p')\
            .join(
                wikidata.alias('w'),
                (F.col('p.page_title') == F.col('w.title')) &
                (F.col('p.wiki') == F.col('w.site')),
            )\
            .select(F.col('w.id').alias('id'),
                    F.col('p.wiki').alias('wiki'),
                    F.col('p.view_count').alias('view_count'))
        window = Window.partitionBy('wiki').orderBy('view_count')
        pageviews = pageviews.select(
            F.col('*'),
            F.rank().over(window).alias('rank')
        )
        wikis = ['%swiki' % x for x in languages]
        pageviews = pageviews\
            .groupBy('id')\
            .pivot('wiki', wikis)\
            .agg(F.first('view_count'), F.first('rank'))\
            .fillna(0)
        for language in languages:
            pageviews = pageviews\
                .withColumnRenamed(
                    '%swiki_first(view_count, false)' % language,
                    '%s_view_count' % language
                )\
                .withColumnRenamed('%swiki_first(rank, false)' % language,
                                   '%s_rank' % language)
            article_count = pageviews\
                .where(F.col('%s_view_count' % language) != 0)\
                .count()
            pageviews = pageviews\
                .withColumn(
                    '%s_normalized_rank' % language,
                    F.col('%s_rank' % language) / article_count
                )\
                .withColumn('%s_log_rank' % language,
                            F.log('%s_rank' % language))\
                .fillna(0)
        log('Calculated pageviews.', 'debug')
        return pageviews

    @timeit
    def get_missing_pageviews(self, wikidata, pageviews):
        """Return pageviews for languages that don't have them computed.
        Returns:
            dataframe
        """
        existing_languages = set([
            x.split('_')[0] for x in pageviews.columns
            if x.endswith('_view_count')
        ])
        missing_languages = self.language_pairs_set\
                                .difference(existing_languages)
        log('Pageviews for these languages are missing: %s' %
            missing_languages, 'debug')
        missing_pageviews = None
        if len(missing_languages):
            missing_pageviews = self.calculate_pageviews(
                wikidata, missing_languages)
        return missing_pageviews

    @timeit
    def get_pageviews(self, wikidata):
        """Return page view counts, normalized and log ranks.
        Return from file if exists, otherwise generate the file, save
        it, and return it.
        Args:
            wikidata (dataframe)
        Returns:
            dataframe: pageviews, normalized ranks, and log ranks.
        """
        filename = '%s/article-pageviews-%s-%s' %\
            (self.tmp_dir, self.start_date, self.end_date)
        try:
            pageviews = self.spark.read.parquet(filename)
            # If we're training language pairs that were not trained
            # before, we'll need to generate missing pageviews and save
            # them.
            missing_pageviews = self.get_missing_pageviews(wikidata, pageviews)
            if missing_pageviews:
                pageviews = pageviews\
                    .alias('p')\
                    .join(missing_pageviews.alias('m'),
                          F.col('p.id') == F.col('m.id'))\
                    .drop(F.col('m.id'))
                # TODO: think about saving additions only.
                pageviews.write.parquet(filename)
            log('Returning existing pageviews from %s.'
                % filename, 'debug')
        except Exception:
            log('Starting to calculate pageviews and save as %s.'
                % filename, 'debug')
            pageviews = self.calculate_pageviews(
                wikidata,
                self.top_languages.union(self.language_pairs_set)
            )
            pageviews.write.parquet(filename)
        return pageviews

    @timeit
    def get_sitelinks(self, wikidata):
        return wikidata.groupBy('id').count()

    @timeit
    def get_features(self):
        features = []
        for language in self.top_languages:
            features.append('%s_view_count' % language)
            features.append('%s_normalized_rank' % language)
            features.append('%s_log_rank' % language)
        features.append('sitelinks')
        return features

    @timeit
    def get_model(self, common_wikidata_ids, pageviews, sitelinks,
                  features, source_language, target_language):
        """Train a model and return."""
        input_data = common_wikidata_ids\
            .alias('w')\
            .join(pageviews.alias('p'), F.col('w.id') == F.col('p.id'))\
            .join(sitelinks.alias('s'), F.col('w.id') == F.col('s.id'))\
            .select(F.col('p.*'),
                    F.col('s.count').alias('sitelinks'),
                    F.col('%s_normalized_rank' % target_language)
                    .alias('label'))
        vector = VectorAssembler(inputCols=features, outputCol="features")
        rfr = RandomForestRegressor(
            labelCol="label", featuresCol="features")
        pipeline = Pipeline(stages=[vector, rfr])
        model = pipeline.fit(input_data)
        return model

    @timeit
    def get_predictions(self, source_only_wikidata_ids, model,
                        pageviews, sitelinks):
        prediction_input_data = source_only_wikidata_ids\
            .alias('w')\
            .join(pageviews.alias('p'), F.col('w.id') == F.col('p.id'))\
            .join(sitelinks.alias('s'), F.col('w.id') == F.col('s.id'))\
            .select(F.col('p.*'), F.col('s.count').alias('sitelinks'))
        return model.transform(prediction_input_data)

    @timeit
    def save_predictions(self, predictions, source_language, target_language):
        filename = '%s/normalized-scores-%s-%s-%s-%s.tsv' %\
            (self.output_dir, self.start_date, self.end_date,
             source_language, target_language)
        predictions\
            .select('id', 'prediction')\
            .repartition(1)\
            .write\
            .csv(filename, mode='overwrite', sep='\t', header=True,
                 compression='bzip2')

    @timeit
    def train(self):
        """Train models and create article normalized scores."""
        wikidata = self.get_wikidata()
        pageviews = self.get_pageviews(wikidata)
        sitelinks = self.get_sitelinks(wikidata)
        features = self.get_features()

        for source_language, target_language in self.language_pairs:
            source_wiki = '%swiki' % source_language
            target_wiki = '%swiki' % target_language
            source_wikidata_ids = wikidata\
                .filter(wikidata.site == source_wiki)\
                .select('id')
            target_wikidata_ids = wikidata\
                .filter(wikidata.site == target_wiki)\
                .select('id')
            common_wikidata_ids = source_wikidata_ids\
                .intersect(target_wikidata_ids)
            source_only_wikidata_ids = source_wikidata_ids\
                .subtract(common_wikidata_ids)

            model = self.get_model(
                common_wikidata_ids, pageviews, sitelinks, features,
                source_language, target_language)
            predictions = self.get_predictions(
                source_only_wikidata_ids, model, pageviews, sitelinks)
            self.save_predictions(
                predictions, source_language, target_language)

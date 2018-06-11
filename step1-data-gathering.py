from __future__ import division

from pyspark.sql import SparkSession

from pyspark import SparkContext, SQLContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F

# TODO don't hardcode paths, dates, etc.


def get_dblist(spark):
    """Return the list of Wikipedias"""
    return spark.read.format('csv').load('/user/bmansurov/wikipedia.dblist')\
                                   .toDF('wiki')


def get_wikidata(sqlCtx, dblist):
    # Wikidata dump
    wd = sqlCtx.read.parquet('/user/joal/wikidata/parquet')
    # Wikidata dump with sitelinks flattened
    wd = wd.select(wd.id, F.explode(wd.siteLinks).alias('siteLinks'))
    # Wikidata items with flat sitelinks that point to Wikipedia only,
    # rather than Wikiquote, or other similar sites
    wd = wd.join(dblist, wd.siteLinks.site == dblist.wiki)


def get_sitelinks(wikidata):
    # Wikidata items and count of their site links to Wikipedia articles
    return wikidata.groupBy(wikidata.id).count()


def save_sitelinks_to_file(sitelinks):
    # Save Wikipedia sitelinks as parquet
    sitelinks.write.parquet(
        '/user/bmansurov/wikidata/wikipedia_site_links.parquet')


def get_articles(wikidata):
    # List of (qid, title, wiki)
    articles = wikidata.select('id', 'siteLinks.title', 'wiki')
    # remove non-main namespace articles
    return articles.filter(~articles.title.contains(':'))


def save_articles_to_file(articles):
    # save Wikipedia Articles with wiki name and Wikidata ID
    articles.write.parquet(
        '/user/bmansurov/wikidata/wikipedia_articles.parquet')


def get_pageviews(wiki, hc):
    # Article (in Main namespace) pagviews over the last 6 months
    return hc.sql("""
        SELECT page_title, sum(view_count) as pageviews
        FROM wmf.pageview_hourly
        WHERE
            ((year = 2018 AND month < 5) OR (year = 2017 AND month > 10))
            AND project="%s.wikipedia"
            AND agent_type="user"
            AND instr(page_title, ':')=0
        GROUP BY page_title
        ORDER BY pageviews
        """ % wiki)


def save_pageviews_to_file(pageviews):
    pageviews.write.parquet(
        '/user/bmansurov/wikipedia/uzwiki_pageviews_2017_11-2018_4.parquet')


def get_normalized_pageviews(pageviews):
    article_count = pageviews.count()
    # get normalized ranks of page views
    return pageviews.select(
        '*', (pageviews.pageviews/article_count).alias('normalized_rank'))


def save_normalized_pageviews_to_file(pageviews):
    pageviews.write.parquet(
        '/user/bmansurov/wikipedia/uzwiki_pageviews_'
        '2017_11-2018_4_normalized.parquet')


if __name__ == '__main__':
    spark = SparkSession.builder.appName("TranslationRecommendation")\
                                .getOrCreate()
    sc = SparkContext("local", "Simple App")
    sqlCtx = SQLContext(sc)
    hc = HiveContext(sc)

    dblist = get_dblist(spark)
    wikidata = get_wikidata(sqlCtx, dblist)

    sitelinks = get_sitelinks(wikidata)
    save_sitelinks_to_file(sitelinks)

    articles = get_articles(wikidata)
    save_articles_to_file(articles)

    pageviews = get_pageviews('uz', hc)
    save_pageviews_to_file(pageviews)

    normalized_pageviews = get_normalized_pageviews(pageviews)
    save_normalized_pageviews_to_file(normalized_pageviews)

    spark.stop()

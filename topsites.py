# Get top Wikipedias by edit count
# stats1005:/home/ezachte/wikistats_data/dumps/csv/csv_wp/StatisticsMonthly.csv
# The header of the CSV matches the header of the first table from
# https://stats.wikimedia.org/EN/TablesWikipediaEN.htm
# First column is the wiki, second is the date, and seventh column is the
# article count.

import csv
import sys

STATS_FILE = '/home/ezachte/wikistats_data/dumps/csv/'\
    'csv_wp/StatisticsMonthly.csv'

# Check if we got source and target languages.
if len(sys.argv) != 2:
    print("Pass in the date, e.g. topsites.py 05/31/2018")
    exit(1)
day = sys.argv[1]

article_counts = {}
with open(STATS_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if row[1] == day:
            article_counts[row[0]] = int(row[6])

# get top 50
# ignore zz, which is the combined value
top_50 = sorted(
    list(article_counts.items()),
    key=lambda x: x[1],
    reverse=True)[1:51]

print('wiki\tarticle_count')
for w in top_50:
    print('%s\t%d' % (w[0], w[1]))

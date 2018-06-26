import csv
from subprocess import call
import sys


if len(sys.argv) != 2:
    print('Pass in the end date, e.g. topsites_pageviews.py 05/31/2018')
    exit(1)
end_date = sys.argv[1]

with open('topsites.tsv', 'r') as infile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    next(tsv_reader)
    for row in tsv_reader:
        print('Generating top sites for %swiki.' % row[0])
        cmd = 'PYSPARK_DRIVER_PYTHON=python2 spark2-submit '\
            'pageviews.py %s %s' % (row[0], end_date)
        call(cmd, shell=True)

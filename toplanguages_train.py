import csv
from subprocess import call
import sys


if len(sys.argv) != 2:
    print('Pass in the end date, e.g. toplanguages_train.py 05/31/2018')
    exit(1)
end_date = sys.argv[1]

with open('language_pairs.txt', 'r') as infile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    next(tsv_reader)
    for row in tsv_reader:
        print('Training models for %s-%s.' % (row[0], row[1]))
        cmd = 'PYSPARK_DRIVER_PYTHON=python2 spark2-submit --master yarn '\
            '--executor-memory 32G --executor-cores 4 --driver-memory 32G '\
            '--conf spark.driver.maxResultSize=32G '\
            'train.py %s %s %s' % (row[0], row[1], end_date)
        call(cmd, shell=True)


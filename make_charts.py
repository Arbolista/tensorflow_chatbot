import sys
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from camel_snake_kebab import *
import os
dirname = os.path.dirname(os.path.realpath(__file__))

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
def plot_bucket_histogram(source_path, target_path, title, max_size=None):
  data_set = [[] for _ in _buckets]
  source_file = open(source_path, 'r')
  target_file = open(target_path, 'r')
  source, target = source_file.readline(), target_file.readline()
  counter = 0
  while source and target and (not max_size or counter < max_size):
    counter += 1
    if counter % 100000 == 0:
      print("  reading data line %d" % counter)
      sys.stdout.flush()
    source_ids = [int(x) for x in source.split()]
    target_ids = [int(x) for x in target.split()]
    target_ids.append(b"_EOS")
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
      if len(source_ids) < source_size and len(target_ids) < target_size:
        data_set[bucket_id].append([source_ids, target_ids])
        break
    source, target = source_file.readline(), target_file.readline()
  lengths = [len(data) for data in data_set]
  plot_bar(lengths, title)

def plot_bar(data, title):
  index = np.arange(len(data))
  bar_width = 0.35
  plt.bar(index, data, bar_width)
  plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D'))
  plt.title(title)
  plt.xlabel("Value")
  plt.ylabel("Frequency")
  fig = plt.gcf()
  fig.savefig(('data/%s.png' % kebab_case(title)))   # save the figure to file

def make_charts(source, targets, title):
  plot_bucket_histogram(source, targets, ('%s Bucket Distribution' % title))

if __name__ == "__main__":
  enc20000 = ("%s/data/train.enc.ids20000" % dirname)
  dec20000 = ("%s/data/train.dec.ids20000" % dirname)
  enc25000 = ("%s/data/train.enc.ids25000" % dirname)
  dec25000 = ("%s/data/train.dec.ids25000" % dirname)
  make_charts(enc20000, dec20000, "Vocabulary 20000")
  make_charts(enc25000, dec25000, "Vocabulary 25000")


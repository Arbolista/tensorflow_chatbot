# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import csv
import string
import requests
import random
import json
from itertools import izip

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")



def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def _append_augmented_vocab(vocab_list, augmented_data_path, tokenizer=None):
  ivocab_list = iter(vocab_list)
  existing_vocab = dict(izip(ivocab_list, ivocab_list))

  augemented_vocab = {}
  with gfile.GFile(augmented_data_path, mode="rb") as f:
    for line in f:
      tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
      # we don't normalize digits here.
      for w in tokens:
        if w not in existing_vocab:
          augemented_vocab[w] = True
    # replace the least frequent words with the augmented data.
    for _ in augemented_vocab.keys():
      vocab_list.pop()
    for w in augemented_vocab.keys():
      vocab_list.append(w)

    return vocab_list

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, augmented_data_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      vocab_list = _append_augmented_vocab(vocab_list, augmented_data_path)

      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def _get_paraphrase(sentence):
  url = "http://api.microsofttranslator.com/v3/json/paraphrase"
  params = {
    "appId": "b7be71d9b2134ef390d51b43fdb24206",
    "language": "en",
    "category": "general",
    "sentence": sentence,
    "maxTranslation": 20
  }
  res = requests.get(url, params=params)
  if res.ok:
    if "paraphrases" not in res.content:
      print(res.content)
      raise RuntimeError("Microsoft paraphrase API did not work for phrase: " + sentence)
    print("--- Retrieved paraphrase for sentence. ---")
    print(sentence)
    print(res.content["paraphrases"])
    return res.content["paraphrases"]
  else:
    print(res.content)
    raise RuntimeError("Microsoft paraphrase API did not work for phrase: " + sentence)

def _preprocess_climate_change_data(working_directory, augmented_enc, augmented_dec, train_enc, train_dec):
  if not gfile.Exists(augmented_enc) and not gfile.Exists(augmented_dec):
    with gfile.GFile(augmented_enc, mode="w") as augmented_input:
      with gfile.GFile(train_enc, mode="a") as train_input:
        with gfile.GFile(augmented_dec, mode="w") as augmented_output:
          with gfile.GFile(train_dec, mode="a") as train_output:
            with open(working_directory + "/climate_augmented_data.csv", "rb") as climate_augmented_data:
              augmented_rows = csv.reader(climate_augmented_data, delimiter=',')
              augmented_metatokens = {}
              augmented_paraphrases = []
              for row in augmented_rows:

                safe_input = row[0].replace('\n', ' ').replace('\r', '')
                safe_output = row[1].replace('\n', ' ').replace('\r', '')

                # write raw input sentence
                augmented_input.write(safe_input + "\n")
                train_input.write(safe_input + "\n")

                # tokenize output as a random string.
                output_length = random.randrange(15, 20, 1)
                output_metatoken = ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(output_length))
                augmented_metatokens[output_metatoken] = safe_output

                # write output token
                augmented_output.write(output_metatoken + "\n")
                train_output.write(output_metatoken + "\n")
                #paraphrases = _get_paraphrase()
                #augmented_paraphrases += [(paraphrase, output_metatoken) for paraphrase in augmented_paraphrases]

              # write input and output for paraphrases.
              random.shuffle(augmented_paraphrases)
              for (paraphrase, output_metatoken) in augmented_paraphrases:
                augmented_input.write(paraphrase + "\n")
                train_input.write(paraphrase + "\n")
                augmented_output.write(output_metatoken + "\n")
                train_output.write(output_metatoken + "\n")

              # save output metatokens
              with open(working_directory + "climate_augmented_metatokens.json", "w") as climate_augmented_metatokens:
                json.dump(augmented_metatokens, climate_augmented_metatokens, indent=2, sort_keys=True)

def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

    augmented_enc_path = os.path.join(working_directory, "augmented.enc")
    augmented_dec_path = os.path.join(working_directory, "augmented.dec")
    _preprocess_climate_change_data(working_directory, augmented_enc_path, augmented_dec_path, train_enc, train_dec)

    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(enc_vocab_path, train_enc, enc_vocabulary_size, augmented_enc_path, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec, dec_vocabulary_size, augmented_dec_path, tokenizer)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)

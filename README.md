#Tensorflow Chatbot

Data
============

Please download the data/ and working_dir/ directories from Dropbox and unzip them here (they should be unzipped so their contents are accessible within ./data/ and ./working_dir/ paths).

Those directories contain all of the necessary data to run the test in test mode, create new vocabulary sets, and re-train.

Dependencies
============
* numpy
* scipy
* six
* tensorflow r0.12

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies


Usage
===========

To train the bot, edit the `seq2seq.ini` file so that mode is set to train like so

`mode = train`

then run the code like so

``python execute.py``

To test the bot during or after training, edit the `seq2seq.ini` file so that mode is set to test like so

`mode = test`

then run the code like so

``python execute.py``

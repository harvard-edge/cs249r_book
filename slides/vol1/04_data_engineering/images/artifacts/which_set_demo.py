#!/usr/bin/env python3
"""Run TensorFlow's Speech Commands split function on real file names.

The function body below is copied verbatim from
tensorflow/examples/speech_commands/input_data.py, lines 44 and 65-110,
TensorFlow v2.16.1 (github.com/tensorflow/tensorflow, Apache 2.0).
Only `compat.as_bytes` is replaced by its pure-Python equivalent, so the
script runs without TensorFlow. File names are from the Hugging Face card
for google/speech_commands (7846fd85, 8b775397); the _nohash_1 and yes/
variants show that one speaker's other takes follow the same key.

Run: python3 which_set_demo.py   (slide: "One speaker's clips must stay in one split")
"""
import hashlib
import os
import re


class compat:  # stand-in for tensorflow.python.util.compat
    @staticmethod
    def as_bytes(s):
        return s.encode("utf-8")


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


if __name__ == "__main__":
    for f in ["no/7846fd85_nohash_0.wav", "no/7846fd85_nohash_1.wav",
              "yes/7846fd85_nohash_0.wav", "tree/8b775397_nohash_0.wav",
              "tree/8b775397_nohash_1.wav"]:
        # train.py defaults: --validation_percentage=10 --testing_percentage=10
        print(f"{f:28s} -> {which_set(f, 10, 10)}")

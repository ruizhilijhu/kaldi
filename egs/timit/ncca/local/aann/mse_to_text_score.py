#!/usr/bin/python

import sys, os, logging, numpy as np
import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io_funcs import kaldi_io, utils

import numexpr
numexpr.set_num_threads(1)

from optparse import OptionParser
usage = "%prog [options] <mse_dir/feats.scp> <scores.pkl>"
parser = OptionParser(usage)

parser.add_option('--uttLevel', dest='uttLevel',
                  help='if true, same weights for all frames in an utterance [default: %default]',
                  default="true", action='store_true');

(o, args) = parser.parse_args()
if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(mse_feats_scp, scores_pkl) = (args[0], args[1])

scores={}
loss=0
cnt=0
with kaldi_io.KaldiScpReader(mse_feats_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    #0th dim is per-frame mse
    H = np.asarray(X[:,0])
    
    loss+= sum(H)
    cnt+= H.size

    
  mse=loss/cnt
  print mse
  f = open(scores_pkl, 'w')
f.write("%s\n" %mse)
f.close()




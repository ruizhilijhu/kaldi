#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN with monophone target on top of Fbank features. 
# The training is done in 3 stages,
#
# 1) make fbank features
# 2) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 3) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/mono
data_fbank=data-fbank
stage=1 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fbank features, and split the data (9:1) so we can train on them easily,
  # make fbank features
  fbankdir=fbank # ark,scp files
  mkdir -p $data_fbank # kaldi format inputs
  for x in test dev train; do
    cp -r data/$x $data_fbank/$x
    steps/make_fbank.sh --nj 10 \
      $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
    steps/compute_cmvn_stats.sh $data_fbank/$x exp/make_fbank/$x $fbankdir || exit 1    
  done
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $data_fbank/train ${data_fbank}/train_tr90 ${data_fbank}/train_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
	    steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 \
	    --cmvn_opts "--norm-vars=true --norm-means=true" \
	    $data_fbank/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
  dbn=exp/dnn4_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    --cmvn_opts "--norm-vars=true --norm-means=true" \
    ${data_fbank}/train_tr90 ${data_fbank}/train_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $data_fbank/test $dir/decode_test || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $data_fbank/dev $dir/decode_dev || exit 1;
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

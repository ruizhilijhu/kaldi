#!/bin/bash

# This script
# 1 ) extract pre-softmax layer activations
# 2 ) train an autoencoder with pretaining step
# 3 ) compute mse on train and tests on multicondition.

. ./cmd.sh
. ./path.sh 

# Config:
nj=60
stage=1
nnet_dir=exp/dnn4_pretrain-dbn_dnn
remove_last_components=1
fea_dir=data-fbank23-allnoise-sub1
cmd=run.pl
# End of config

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. utils/parse_options.sh || exit 1

# Extract hidden activations from fea_dir 
if [ $stage -le 0 ]; then
  dir=data-bn-${remove_last_components}
  for x in dev test; do  
    steps/nnet/make_bn_feats.sh --nj 10 \
      --cmd "$train_cmd" \
      --remove-last-components ${remove_last_components} \
      $dir/${x}_${fea_dir} $fea_dir/$x $nnet_dir $dir/${x}_${fea_dir}/log $dir/${x}_${fea_dir}/data || exit 1
    steps/compute_cmvn_stats.sh $dir/${x}_${fea_dir} $dir/${x}_${fea_dir}/log $dir/${x}_${fea_dir}/data || exit 1
  done
fi


# Compute per-frame mse; per-utt mse (avg over frame)
if [ $stage -le 1 ]; then
  dir=exp/aann_${remove_last_components} # put mse in here
  for x in dev test; do
    steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj 10 $dir/mse_${x}_${fea_dir} data-bn-${remove_last_components}/${x}_${fea_dir} $dir $dir/mse_${x}_${fea_dir}/log $dir/mse_${x}_${fea_dir}/data
    compute-cmvn-stats scp:$dir/mse_${x}_${fea_dir}/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_${x}_${fea_dir}/mse_${x}_${fea_dir}_utt.txt.ark
  done

   
  
fi

echo Success
exit 0;

#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example


# Begin configuration section.
cmd=run.pl
nnet_forward_opts=
use_gpu=no
mlp=   # truncated mlp
minibatch_size=603
randomizer_size=32768
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 4 ]; then
   echo "usage: $0 [options] <data> <ref_cm> <nnet-dir> <log-dir> ";
   echo "Compute minibatch-based fn loss for a given data set on a dnn " 
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

data=$1
ref_cm=$2
nndir=$3
logdir=$4 # output

######## CONFIGURATION

required="$data/feats.scp $mlp $nndir/final.feature_transform ${ref_cm}"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

mkdir -p $logdir

# Copy feature transform with MLP:
cp $mlp $logdir/
cp $nndir/final.feature_transform $logdir/
nnet-info $mlp >$logdir/mlp-info
nnet-info $nndir/final.feature_transform >$logdir/feature_transform-info


# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nndir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"


# PREPARE LABEL PIPELINE
o_feats=`echo $feats | sed 's@ark:copy-feats@ark,o:copy-feats@g'`
labels="$o_feats feat-to-post ark:- ark:- |"


# Run the forward pass,
$cmd $logdir/compute_fn_loss.log \
     nnet-train-frmshuff-ncca $nnet_forward_opts \
     --objective-function=ncca \
     --ncca-ref-mat=$ref_cm \
     --use-gpu=$use_gpu \
     --cross-validate=true \
     --verbose=1 \
     --randomize=false \
     --minibatch-size=$minibatch_size \
     --randomizer-seed=777 \
     --randomizer-size=$randomizer_size \
     --feature-transform=$D/final.feature_transform \
     "$feats" "$labels" $mlp || exit 1;


info=$(grep AvgLoss $logdir/compute_fn_loss.log | awk '{print $4}')
echo $info > $logdir/fn
echo "Succeeded compute fn loss for $data --> $info (Ncca) minibatch-${minibatch_size}"

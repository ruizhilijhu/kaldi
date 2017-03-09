#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
remove_last_components=4 # remove N last components from the nnet
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

# transform nnet out opts
# transf_nnet_out_opts= # 

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <tgt-data-dir> <src-data-dir> <multi-stream-opts> <pca-dir> <log-dir> <abs-path-to-feat-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
multi_stream_opts=$3
pcadir=$4
logdir=$5
bnfeadir=$6

######## CONFIGURATION

# copy the dataset metadata from srcdata.
mkdir -p $data $logdir $bnfeadir || exit 1;
utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null

# make $bnfeadir an absolute pathname.
[ '/' != ${bnfeadir:0:1} ] && bnfeadir=$PWD/$bnfeadir

required="$srcdata/feats.scp $pcadir/feature_extractor.nnet $pcadir/final.feature_transform $pcadir/pca.mat"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Create nnet in $bnfeadir:
feature_transform=$pcadir/final.feature_transform
nnet=$pcadir/feature_extractor.nnet
pcamat=$pcadir/pca.mat

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$pcadir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  ivector_dim=$(cat $D/ivector_dim)
  [ -z $ivector ] && echo "Missing --ivector, they were used in training! (dim $ivector_dim)" && exit 1
  ivector_dim2=$(copy-vector --print-args=false "$ivector" ark,t:- | head -n1 | awk '{ print NF-3 }') || true
  [ $ivector_dim != $ivector_dim2 ] && "Error, i-vector dimensionality mismatch! (expected $ivector_dim, got $ivector_dim2 in $ivector)" && exit 1
  # Append to feats
  feats="$feats append-vector-to-feats ark:- '$ivector' ark:- |"
fi

# Add Multi-stream options
#feats="$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask $multi_stream_opts ark:- ark:- |"
feats="$feats nnet-forward $feature_transform ark:- ark:- |"

# transf_nnet_out_opts
transf_nnet_out_opts=$(cat $pcadir/transf_nnet_out_opts)
echo "Loading transf_nnet_out_opts=$transf_nnet_out_opts"

# Run the forward pass,
$cmd JOB=1:$nj $logdir/make_pca_transf.JOB.log \
  nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
  transform-nnet-posteriors $transf_nnet_out_opts ark:- ark:- \| \
  transform-feats ${pcamat} ark:- ark,scp:${bnfeadir}/pca_transf_$name.JOB.ark,${bnfeadir}/pca_transf_$name.JOB.scp || exit 1;
# concatenate the .scp files
(
for ((n=1; n<=nj; n++)); do
  cat $bnfeadir/pca_transf_$name.$n.scp
done
) > $data/feats.scp

# check sentence counts,
N0=$(cat $srcdata/feats.scp | wc -l) 
N1=$(cat $data/feats.scp | wc -l)
[[ "$N0" != "$N1" ]] && echo "$0: sentence-count mismatch, $srcdata $N0, $data $N1" && exit 1

echo "Succeeded creating pca_transf_mlpfeats features '$data'"



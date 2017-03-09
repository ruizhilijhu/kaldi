#!/bin/bash

# this script extracts coactivtion matrix from dnn-aann

# CONFIG
feats_nj=20
dnnid="mono_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components=2
aannid=
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging


aannDir=exp/${dnnid}_aann/$aannid/pretrain-aann_aann
dnnDir=exp/$dnnid
outdir=exp/${dnnid}_aann_cm/$aannid
[ ! -d $outdir ] && mkdir -p $outdir
echo "OUTPUT DIR: "${outdir}

# concatenate final.nnet.1.aann.ft + aann.nnet
mlp=$dnnDir/final.nnet
numComponents_dnn=`nnet-info ${mlp} | grep num-components | awk '{print $2}'`
last_components=$(($numComponents_dnn-$first_components))
echo "number of Components in DNN: $numComponents_dnn"
echo "number of first components: $first_components"
echo "number of last components: $last_components"
nnet-copy --remove-last-components=$last_components $mlp $outdir/final.nnet.1 || exit 1
nnet-concat $outdir/final.nnet.1 $aannDir/final.feature_transform $aannDir/final.nnet $outdir/final.nnet.1.aann.ft.aann.nnet || exit 1
nnet-concat $outdir/final.nnet.1 $aannDir/final.feature_transform $outdir/final.nnet.1.aann.ft|| exit 1


# extract output 
# pre-aann
local/ncca/make_bn_feats_ncca.sh --nj ${feats_nj} \
				 --cmd "${train_cmd}" \
				 --remove-last-components 0 \
				 --dnnnet $outdir/final.nnet.1.aann.ft \
				 $outdir/pre-aann/bn data-fbank13-clean/train $dnnDir $outdir/pre-aann/bn/log $outdir/pre-aann/bn/data || exit 1
# post-aann
local/ncca/make_bn_feats_ncca.sh --nj ${feats_nj} \
				 --cmd "${train_cmd}" \
				 --remove-last-components 0 \
				 --dnnnet $outdir/final.nnet.1.aann.ft.aann.nnet \
				 $outdir/post-aann/bn data-fbank13-clean/train $dnnDir $outdir/post-aann/bn/log $outdir/post-aann/bn/data || exit 1


# compute coactivation matrix
est-coact-mat scp:$outdir/pre-aann/bn/feats.scp $outdir/pre-aann/${first_components}.mat || exit 1
est-coact-mat scp:$outdir/post-aann/bn/feats.scp $outdir/post-aann/${first_components}.mat || exit 1

copy-matrix --binary=false $outdir/pre-aann/${first_components}.mat $outdir/pre-aann/${first_components}.txt.mat || exit 1
copy-matrix --binary=false $outdir/post-aann/${first_components}.mat $outdir/post-aann/${first_components}.txt.mat || exit 1

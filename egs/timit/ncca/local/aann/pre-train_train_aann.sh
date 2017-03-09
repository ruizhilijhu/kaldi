#!/bin/bash

# This script does the following by steps given a data set 
# 1) pretrain
# 2) train aann

# CONFIG
feats_nj=10
train_nj=30
decode_nj=20
stage=0

lr=0.0001
hiddenDim="512:24:512"
hid_dim=0
splice=0
splice_step=1

cmvn_opts=
pretrain_opts=
train_opts=
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging


if [ $# != 2 ]; then
    echo "Usage: $0 <bn-data-dir> <out-dir>"
    echo " e.g.: $0 exp/mono_dnn-256-bn/2 exp/mono_dnn-256-bn_aann-2"
    echo "main options (for others, see top of script file)"
    echo "  --config <config-file> # config containing options"
    echo "  --hiddenDim <array> # hidden dim info, by default: 512:24:512"
    exit 1;
fi

data=$1
expid=$2

echo ${expid}

# STEPS

echo ============================================================================
echo "                  Pretrain AANN                                           "
echo ============================================================================
outdir=${expid}/pretrain-aann # the only output
if [ $stage -le 0 ]; then
    mkdir -p $outdir
    (tail --pid=$$ -F $outdir/log/pretrain_dbn.log 2>/dev/null)&
    $cuda_cmd $outdir/log/pretrain_dbn.log \
	      local/aann/pretrain_dbn.sh \
	      ${pretrain_opts} \
	      --splice $splice \
	      --splice-step $splice_step \
	      --hid-dim $hiddenDim \
	      ${cmvn_opts:+ --cmvn_opts "${cmvn_opts}"} \
	      $data $outdir || exit 1;
fi
outdir0=$outdir

echo ============================================================================
echo "                  Train AANN                                           "
echo ============================================================================
outdir=${expid}/pretrain-aann_aann # the only output
if [ $stage -le 1 ]; then

    
    # split train into 90% trn and 10% cv
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $data ${data}_tr90 ${data}_cv10 || exit 1
    
    mkdir -p $outdir
    numHiddenLayers=`echo $hiddenDim | awk -F ':' '{print NF}'`
    dbn=$outdir0/${numHiddenLayers}.dbn
    (tail --pid=$$ -F $outdir/log/train_aann.log 2>/dev/null)&
    $cuda_cmd $outdir/log/train_aann.log \
	      local/aann/train_aann.sh \
	      ${train_opts} \
	      --splice $splice \
	      --splice-step $splice_step \
	      --train-opts "--max-iters 50" \
	      --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
	      --hid-layers 0 \
	      --dbn $dbn \
	      --learn-rate $lr \
	      --copy-feats "false" --skip-cuda-check "true" \
	      --hid-dim $hid_dim \
	      ${cmvn_opts:+ --cmvn_opts "${cmvn_opts}"} \
	      ${data}_tr90 ${data}_cv10 $outdir || exit 1;
fi
outdir1=$outdir

echo ============================================================================
echo "Finished successfully pre-train and train aann on" `date`
echo ============================================================================

exit 0

#!/bin/bash

# This script does cm-based dnn adaptation based on one coactivation matrix
# 0) preadapt DNN preparation -- Split DNN into two parts final.nnet.1 final.nnet.2 and feature transform
# 1) adaptation
# 2) postadapt DNN preparation 

### dnn decoding before and after adaptation
# 3) preadapt DNN decoding on clean and one adapt data (paralell)
# 4) postadapt DNN decoding on clean and one adapt data (paralell)

### CM decoding before and after adaptation
# 5) extract hidden activations for postadapt DNN 
# 6) compute frobenious norm (paralell) pre-adapted DNN
# 7) compute frobenious norm (paralell) post-adapted DNN


# CONFIG
feats_nj=10
train_nj=30
decode_nj=20
stage=0

# dnn
dnnDir=exp/dnn_pretrain-dbn_dnn
first_components=13

# data 
data_fbank_clean=data-fbank23-clean
data_fbank_allnoise=data-fbank23-allnoise
noise=car-20

# aann
aannids=13
aannid=13
aannDir=exp/dnn-bn-13_aann/pretrain-aann_aann
splice=0
splice_step=1

# adapt
mb=18000
adaptid=adapt-frm
lr=0.0008
numLayersFreeze=  # from begining 1 to i
nnetIdx= # two digits 01 02 or 15


. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

echo $aannDir

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging
numComponents_dnn=`nnet-info ${dnnDir}/final.nnet | grep num-components | awk '{print $2}'`
last_components=$((${numComponents_dnn}-${first_components}))
echo "number of Components in DNN: $numComponents_dnn"
echo "number of first components: $first_components"
echo "number of last components: $last_components"

#output dir 
expid=exp/$noise-${adaptid}_${aannids}/${aannid}  
echo ${expid}
[ ! -d ${expid} ] && mkdir -p ${expid}

# STEPS
echo ============================================================================
echo "              Preparation: AANN(non-updatable) and DNN(Trimmed)           "
echo ============================================================================
outdir=${expid}/preAdapt-dnn # the only output
if [ $stage -le 0 ]; then

    # copy dnn dir
    [ ! -d $outdir ] && rsync -ar --exclude 'decode*' $dnnDir/ $outdir
    # select intermediate iteration nnet to adapt
    
    # freeze adapted layers
    if [ ! -z $numLayersFreeze ]; then
    	if [ $numLayersFreeze != 0 ]; then
    	    cp $outdir/final.nnet $outdir/final.nnet.tmp
    	    unlink $outdir/final.nnet
    	    nnet-set-learnrate --components=`seq -s ":" 1 $numLayersFreeze` --coef=0 $outdir/final.nnet.tmp $outdir/final.nnet || exit 1
    	fi
    fi

    # split final.nnet into final.nnet.1 final.nnet.2
    [ -z $outdir/final.nnet.1 ] && rm $outdir/final.nnet.1
    [ -z $outdir/final.nnet.2 ] && rm $outdir/final.nnet.2
    [ -z $outdir/aann.nnet ] && rm $outdir/aann.nnet
    [ -z $outdir/aann.feature_transform ] && rm $outdir/aann.feature_transform
    [ -z $outdir/final.nnet.1.aann.feature_transform ] && rm $outdir/final.nnet.1.aann.feature_transform
    [ -z $outdir/final.nnet.adapted ] && rm $outdir/final.nnet.adapted
    [ -z $outdir/final.nnet.tmp ] && rm $outdir/final.nnet.tmp

    nnet-copy --remove-last-components=$last_components $outdir/final.nnet $outdir/final.nnet.1 2>$outdir/log/trim_dnn.log || exit 1
    nnet-copy --remove-first-components=$first_components $outdir/final.nnet $outdir/final.nnet.2 2>$outdir/log/trim_dnn.log || exit 1
    
    # copy aann.final and aann.feature_transform and make aann.final non-updatable
    numComponents_aann=`nnet-info ${aannDir}/final.nnet | grep num-components | awk '{print $2}'`
    nnet-set-learnrate --components=`seq -s ":" 1 $numComponents_aann` --coef=0 $aannDir/final.nnet $outdir/aann.nnet 2>$outdir/log/create_aann.log
    cp $aannDir/final.feature_transform $outdir/aann.feature_transform
    nnet-concat $outdir/final.nnet.1 $outdir/aann.feature_transform $outdir/final.nnet.1.aann.feature_transform
    i=1
fi
outdir0=$outdir



echo ============================================================================
echo "                 Adaptation based on $noise                               "
echo ============================================================================
outdir=${expid}/postAdapt-dnn-RAW # the only output
if [ $stage -le 1 ]; then
    i=1
    # split data into 90% trn and 10% cv
    utils/subset_data_dir_tr_cv.sh ${data_fbank_allnoise}/dev/$noise ${data_fbank_allnoise}/dev/${noise}_tr90 ${data_fbank_allnoise}/dev/${noise}_cv10 || exit 1
    
    feature_transform=$outdir0/final.feature_transform
    mlp_init=$outdir0/final.nnet.1.aann.feature_transform
    
    (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
    $cuda_cmd $outdir/log/train_aann.log \
    	      steps/multi-stream-nnet/train_aann.sh \
    	      --train-opts "--max-iters 30 " \
    	      --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
    	      --hid-layers 0 --mlp_init $mlp_init --learn-rate $lr \
    	      --copy-feats "false" --skip-cuda-check "true" \
    	      --feature-transform "$feature_transform" \
    	      --cmvn-opts "--norm-means=true --norm-vars=true" \
    	      --train_tool "nnet-train-frmshuff-aann --objective-function=adapt_mse --ae-adaptation-model=$outdir0/aann.nnet" \
    	      ${data_fbank_allnoise}/dev/${noise}_tr90 ${data_fbank_allnoise}/dev/${noise}_cv10 $outdir || exit 1
fi
outdir1=$outdir



echo ============================================================================
echo "                 Preparation : DNN dir (Post Adaptation)                  "
echo ============================================================================
outdir=${expid}/postAdapt-dnn # the only output
if [ $stage -le 2 ]; then
    # copy dnn dir
    [ ! -d $outdir ] && rsync -ar $outdir0/ $outdir
    rm -r $outdir/nnet/*
    cp $outdir1/nnet/* $outdir/nnet/
    
    # create the adapted dnn
    numComponents_aann_feature_transform=`nnet-info $outdir0/aann.feature_transform | grep num-components | awk '{print $2}'`
    unlink $outdir/final.nnet
    nnet-copy --remove-last-components=${numComponents_aann_feature_transform} $outdir1/nnet/*iter${nnetIdx}* - | nnet-concat - $outdir0/final.nnet.2 $outdir/final.nnet.adapted
    ln -s final.nnet.adapted $outdir/final.nnet

    
fi
outdir2=$outdir




echo ============================================================================
echo "Adaptation Finished successfully on $noise" 
echo ============================================================================

exit 0

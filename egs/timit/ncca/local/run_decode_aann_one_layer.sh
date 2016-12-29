#!/bin/bash

# This script does the following by steps for a giving layer of a dnn, giving training data, decoding data train an aann
# 0) extract hidden activations 
# 1) pretrain
# 2) train aann
# 4) compute mse

# CONFIG
feats_nj=10
train_nj=30
decode_nj=20

dnnDir=exp/dnn_pretrain-dbn_dnn
first_components=
stage=0
data_fbank_clean=data-fbank23-clean
data_fbank_allnoise=data-fbank23-allnoise
noise="car-20"

# aann
aannid=

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

numComponents_dnn=`nnet-info ${dnnDir}/final.nnet | grep num-components | awk '{print $2}'`
last_components=$((${numComponents_dnn}-${first_components}))
echo "number of Components in DNN: $numComponents_dnn"
echo "number of first components: $first_components"
echo "number of last components: $last_components"

expid=dnn-bn-${first_components}_aann${aannid}
echo ${expid}

# STEPS
echo ============================================================================
echo "      Extract Hidden Activations from Component $first_components         "
echo ============================================================================
outdir=exp/${expid}/data-bn-${first_components} # the only output
if [ $stage -le 0 ]; then
    # one noise condition
    for x in dev test; do
	if [ ! -d $outdir/$x/$noise ]; then    
	    steps/nnet/make_bn_feats.sh --nj $feats_nj \
					--cmd "$train_cmd" \
					--remove-last-components ${last_components} \
					$outdir/$x/$noise $data_fbank_allnoise/$x/$noise $dnnDir $outdir/$x/$noise/log $outdir/$x/$noise/data || exit 1
	    steps/compute_cmvn_stats.sh $outdir/$x/$noise $outdir/$x/$noise/log $outdir/$x/$noise/data || exit 1
	fi
    done	     
fi
outdir0=$outdir


outdir=exp/${expid}/pretrain-aann # the only output
outdir1=$outdir

outdir=exp/${expid}/pretrain-aann_aann # the only output
outdir2=$outdir

echo ============================================================================
echo "                  Compute MSE                                             "
echo ============================================================================
outdir=exp/${expid}/mse # the only output
if [ $stage -le 1 ]; then
    # one noise condition
    for x in dev test; do
	if [ ! -d $outdir/$x/$noise ]; then

	    steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj $decode_nj $outdir/$x/$noise \
							$outdir0/$x/$noise $outdir2 $outdir/$x/$noise/log $outdir/$x/$noise/data || exit 1
	    compute-cmvn-stats scp:$outdir/$x/$noise/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$outdir/$x/$noise/mse_utt.txt.ark || exit 1
	    ivector-mean ark,t:$outdir/$x/$noise/mse_utt.txt.ark $outdir/$x/$noise/mse_global.vec || exit 1
	fi
    done  
fi
outdir3=$outdir



echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0

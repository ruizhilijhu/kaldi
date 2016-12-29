#!/bin/bash

# This script does decoding on adapted DNN
# giving dnn and aann
# 0) DNN decoding on one set 
# 1) extract hidden activations on one set
# 2) compute MSE one one set

# CONFIG
feats_nj=10
train_nj=30
decode_nj=20
stage=0

# dnn
dnnDir=exp/car-20-adapt-frm-lr0.000005_11/11/postAdapt-dnn
data=data-fbank23-allnoise/dev/car-20
aannDir=exp/dnn-bn-11_aann/pretrain-aann_aann
first_components=11
outDir=exp/car-20-adapt-frm-lr0.000005_11/11/postAdapt-dnn
monoDir=exp/mono

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

echo ${outDir}
[ ! -d ${outDir} ] && mkdir -p ${outDir}
dataID=`basename \`dirname $data\``_`basename $data`
numComponents_dnn=`nnet-info ${dnnDir}/final.nnet | grep num-components | awk '{print $2}'`
last_components=$((${numComponents_dnn}-${first_components}))
echo "number of Components in DNN: $numComponents_dnn"
echo "number of first components: $first_components"
echo "number of last components: $last_components"


# STEPS
echo ============================================================================
echo "              DNN decoding on $dataID                            "
echo ============================================================================
outdir=${outDir}/decode_dnn_$dataID # the only output
if [ $stage -le 0 ]; then
    	(
    	    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    				 $monoDir/graph $data $outdir || exit 1;
    	) & sleep 2
fi
outdir0=$outdir


echo ============================================================================
echo "              Extract Hidden Activations on $dataID                       "
echo ============================================================================
outdir=${outDir}/decode_mse_$dataID/bn # the only output
if [ $stage -le 1 ]; then
    steps/nnet/make_bn_feats.sh --nj $feats_nj \
				--cmd "$train_cmd" \
				--remove-last-components ${last_components} \
				$outdir $data $dnnDir $outdir/log $outdir/data || exit 1
    steps/compute_cmvn_stats.sh $outdir $outdir/log $outdir/data || exit 1
       
fi
outdir1=$outdir

echo ============================================================================
echo "              Compute MSE on $dataID                       "
echo ============================================================================
outdir=${outDir}/decode_mse_$dataID/mse # the only output
if [ $stage -le 2 ]; then
    
    steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj $decode_nj $outdir \
						$outdir1 $aannDir $outdir/log $outdir/data || exit 1
    compute-cmvn-stats scp:$outdir/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$outdir/mse_utt.txt.ark || exit 1
    ivector-mean ark,t:$outdir/mse_utt.txt.ark $outdir/mse_global.vec || exit 1
fi

outdir2=$outdir




echo ============================================================================
echo "     Decoding Finished successfully on $dataID" 
echo ============================================================================

exit 0

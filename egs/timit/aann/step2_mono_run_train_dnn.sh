#!/bin/bash

# This script does the following by steps
# 0) monophone training and decoding 
# 1) monophone aligment
# 2) train dnn
# 3) dnn decoding on clean data set
# 4) extract bn features for all layers (post activations) (dnn-bn)


# CONFIG
feats_nj=10
train_nj=10
decode_nj=20
stage=4
data_fbank_clean=data-fbank13-clean
data_fbank_allnoise=data-fbank13-allnoise
dropout_schedule=0.2,0.2,0.2,0.2,0.2,0.0
dnn_hidden_unit=256
exp=

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

dnnid=mono_dnn-${dnn_hidden_unit}
set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# STEPS
echo ============================================================================
echo "                Monophone training & decoding                             "
echo ============================================================================
outdir=exp/mono # the only output
if [ $stage -le 0 ]; then
    
    steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang $outdir
    utils/mkgraph.sh --mono data/lang_test_bg $outdir $outdir/graph
    
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
		    $outdir/graph data/dev $outdir/decode_dev
    
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
		    $outdir/graph data/test $outdir/decode_test
fi
outdir0=$outdir


echo ============================================================================
echo "                Monophone aligment                                        "
echo ============================================================================
outdir=exp/mono_ali # the only output
if [ $stage -le 1 ]; then
    
    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    		      data/train data/lang $outdir0 $outdir/train
    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    		      data/dev  data/lang $outdir0 $outdir/dev
    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    		      data/test  data/lang $outdir0 $outdir/test
    

    noiseList=`ls $data_fbank_allnoise/dev`
    for x in ${noiseList[@]}; do
	for d in dev test; do
	    (
		steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
				  data-allnoise/$d/$x  data/lang $outdir0 $outdir/data-allnoise/$d/$x || exit 1
	    ) & sleep 10
	done
    done
    
fi
outdir1=$outdir


echo ============================================================================
echo "                Training DNN                                              "
echo ============================================================================
outdir=exp/${dnnid} # the only output
if [ $stage -le 2 ]; then
    
    # input config
    ali_trn=$outdir1/train
    ali_cv=$outdir1/dev
    (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
    # Train
    $cuda_cmd $outdir/log/train_nnet.log \
	      steps/nnet/train.sh \
	      --hid-layers 5 \
	      --splice 5 \
	      --learn-rate 0.008 \
	      --hid-dim ${dnn_hidden_unit} \
	      --cmvn_opts "--norm-vars=true --norm-means=true" \
	      --delta_opts "--delta-order=2" \
	      --proto-opts "--with-dropout" \
	      --scheduler-opts "--keep-lr-iters 5 --dropout-schedule $dropout_schedule" \
	      ${data_fbank_clean}/train ${data_fbank_clean}/dev data/lang $ali_trn $ali_cv $outdir || exit 1
    
fi
outdir2=$outdir



echo ============================================================================
echo "                Decoding on all data set (CLEAN and Allnoise)             "
echo ============================================================================
outdir=exp/${dnnid} # the only output
if [ $stage -le 3 ]; then
    
    #    decode on clean
    for x in train dev test; do
    	(
    	    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    				 $outdir0/graph $data_fbank_clean/$x $outdir/decode_clean_$x || exit 1;
    	) & sleep 10
    done

    # # decode on allnoise
    # noiseList=`ls $data_fbank_allnoise/dev`
    # echo 'noise conditions: '$noiseList
    # for x in ${noiseList[@]}; do
    # 	for d in dev test; do
    # 	    (
    # 		steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    # 				     $outdir0/graph $data_fbank_allnoise/$d/$x $outdir/decode_allnoise_${x}_${d} || exit 1;
    # 	    ) & sleep 10
    # 	done
    # done
fi
outdir3=$outdir



echo ============================================================================
echo "                Extract BN features for each layer               "
echo ============================================================================
outdir=exp/${dnnid}-bn # the only output
if [ $stage -le 4 ]; then
    mkdir -p $outdir
    idxComponents=$(nnet-info $outdir2/final.nnet | grep 'Sigmoid\|Softmax' | awk '{print $2}')
    numComponents=$(nnet-info $outdir2/final.nnet | grep num-components | awk '{print $2}')
    for idx in ${idxComponents[@]}; do
	echo "extract bn features for component $idx by removing last $((numComponents-idx)) components"
	steps/nnet/make_bn_feats.sh --nj ${feats_nj} \
				    --cmd "$train_cmd" \
				    --remove-last-components $((numComponents-idx)) \
				    $outdir/$idx ${data_fbank_clean}/train $outdir2 $outdir/$idx/log $outdir/$idx/data || exit 1
	steps/compute_cmvn_stats.sh $outdir/$idx $outdir/$idx/log $outdir/$idx/data || exit 1
	
	echo "extract pre-nonlinearity bn features for component $idx by removing last $((numComponents-idx+1)) components"
	steps/nnet/make_bn_feats.sh --nj ${feats_nj} \
				    --cmd "$train_cmd" \
				    --remove-last-components $((numComponents-idx+1)) \
				    $outdir/$((idx-1)) ${data_fbank_clean}/train $outdir2 $outdir/$((idx-1))/log $outdir/$((idx-1))/data || exit 1
	steps/compute_cmvn_stats.sh $outdir/$((idx-1)) $outdir/$((idx-1))/log $outdir/$((idx-1))/data || exit 1

	
    done    
fi
outdir4=$outdir






# echo ============================================================================
# echo "                Compute co-activation matrix for each layer               "
# echo ============================================================================
# outdir=exp/${dnnid}-bn_cm # the only output
# if [ $stage -le 5 ]; then
#     mkdir -p $outdir
#     idxComponents=$(nnet-info $outdir2/final.nnet | grep 'Sigmoid\|Softmax' | awk '{print $2}')
#     for idx in ${idxComponents[@]}; do
# 	echo "compute co-activation matrix for component $idx"
# 	( est-coact-mat scp:$outdir4/$idx/feats.scp $outdir/${idx}.mat || exit 1 ) & sleep 2
# 	( est-coact-mat --binary=false scp:$outdir4/$idx/feats.scp $outdir/${idx}.txt.mat || exit 1 ) & sleep 2
	
	
#     done
    
# fi
# outdir5=$outdir




echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0

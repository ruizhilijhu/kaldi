#!/bin/bash

# This script does AE-adaptation for dnn based on new data set.   
# Input: one dataset for adaptation
#        Well-trained AE trained on pre-softmax actications
#        Well-trained DNN 
# 0) duplicate AE which is non-updatable; make dnn only have updatable layers
# 1) adaptation 
# 2) decode based on pre-adapted dnn and post-adapted dnn

# CONFIG
remove_last_components=1
devDataDir=data-fbank23-allnoise-sub1/dev_benz
testDataDir=data-fbank23-allnoise-sub1/dev_benz
dnnDir=exp/dnn4_pretrain-dbn_dnn
aannDir=exp/aann_${remove_last_components}_ctx1_splice2
gmmdir=exp/mono
lr=0.0008


feats_nj=10
train_nj=30
decode_nj=20
stage=2
splice=1
splice_step=5
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

exp=exp/adapt_dnn_${remove_last_components}_lr${lr}_splice${splice}_step${splice_step}


set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

numComponents_dnn=`nnet-info ${exp}/dnn.nnet | grep num-components | awk '{print $2}'`
remove_first_components=$((${numComponents_dnn}-${remove_last_components}))



# STEPS
echo ============================================================================
echo "              Preparation: AANN(non-updatable) and DNN(Trimmed)           "
echo ============================================================================
dir=${exp}
if [ $stage -le 0 ]; then
  dir=${exp}
  logDir=$dir/log  
  mkdir -p $dir $logDir
  
  # aann 
  numComponents_aann=`nnet-info ${aannDir}/final.nnet | grep num-components | awk '{print $2}'` 
  nnet-set-learnrate --components=`seq -s ":" 1 $numComponents_aann` --coef=0 $aannDir/final.nnet $dir/aann.nnet 2>$logDir/create_aann.log || exit 1
  cp $aannDir/final.feature_transform $dir/aann.feature_transform
  nnet-concat $dir/aann.feature_transform $dir/aann.nnet $dir/aann_feature_transform.nnet 2>$logDir/create_aann.log || exit 1
  
  # dnn; (the parts needed for training AE from dnndir)
  cp $dnnDir/final.nnet $dir/dnn.nnet
  cp $dnnDir/final.feature_transform $dir/dnn.feature_transform
  nnet-copy --remove-last-components=$remove_last_components $dir/dnn.nnet $dir/dnn_${remove_last_components}.nnet 2>$logDir/trim_dnn.log || exit 1
  nnet-copy --remove-first-components=$remove_first_components $dir/dnn.nnet $dir/dnn_first_${remove_first_components}.nnet 2>$logDir/trim_dnn.log || exit 1
  nnet-concat $dir/dnn_${remove_last_components}.nnet $dir/aann.feature_transform $dir/dnn_${remove_last_components}_aann_transform.nnet 

  # split dev data into tr_90 and cv_10
  utils/subset_data_dir_tr_cv.sh $devDataDir ${devDataDir}_tr90 ${devDataDir}_cv10 || exit 1
fi
dir0=$dir




echo ============================================================================
echo "              Adaptation based on `basename $devDataDir`                  "
echo ============================================================================
dir=$exp/`basename $devDataDir`
if [ $stage -le 1 ]; then
  logDir=$dir/log
  mkdir -p $dir $logDir
  feature_transform=$dir0/dnn.feature_transform
  mlp_init=$dir0/dnn_${remove_last_components}_aann_transform.nnet
  

  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/train_aann.log \
     steps/multi-stream-nnet/train_aann.sh \
      --train-opts "--max-iters 30" \
      --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
      --hid-layers 0 --mlp_init $mlp_init --learn-rate $lr \
      --copy-feats "false" --skip-cuda-check "true" \
      --feature-transform "$feature_transform" \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --train_tool "nnet-train-frmshuff-aann --objective-function=adapt_mse --ae-adaptation-model=$dir0/aann.nnet" \
      ${devDataDir}_tr90 ${devDataDir}_cv10 $dir || exit 1
fi
dir1=$dir


echo ============================================================================
echo "     Decode on Pre- & Post-Adapted DNN on `basename $devDataDir`          "
echo ============================================================================
dir=$exp/`basename $devDataDir`

if [ $stage -le 2 ]; then
  logDir=$dir/log
  mkdir -p $dir $logDir

  # create adatped DNN dir in $dir
  cp $dnnDir/final.mdl $dir/final.mdl
  [ -e $dnnDir/cmvn_opts ] && cp $dnnDir/cmvn_opts $dir/cmvn_opts
  [ -e $dnnDir/norm_vars ] && cp $dnnDir/norm_vars $dir/norm_vars
  [ -e $dnnDir/delta_opts ] && cp $dnnDir/delta_opts $dir/delta_opts
  [ -e $dnnDir/delta_order ] && cp $dnnDir/delta_order $dir/delta_order
  [ -e $dnnDir/ali_train_pdf.counts ] && cp $dnnDir/ali_train_pdf.counts $dir/ali_train_pdf.counts
  
  # create adapted DNN 
  [ ! -e $dir1/final_pre.nnet ] && cp $dir1/final.nnet $dir1/final_pre.nnet && unlink $dir1/final.nnet
  nnet-copy --remove-last-components=3 $dir1/final_pre.nnet $dir1/final_dnn_first_part.nnet 2>$logDir/trim_dnn.log || exit 1
  
  nnet-concat $dir1/final_dnn_first_part.nnet $dir0/dnn_first_${remove_first_components}.nnet $dir/final.nnet 2>$logDir/create_aann.log || exit 1

  # decode  
  # decode on pre-adapted dnn
#  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
#    --acwt 0.2 --srcdir $dnnDir \
#    $gmmdir/graph $testDataDir $dir/decode_pre-dnn_`basename $testDataDir` || exit 1
  
  # decode on post-adapted dnn
  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --acwt 0.2 \
    $gmmdir/graph $testDataDir $dir/decode_post-dnn_`basename $testDataDir` || exit 1     
fi
dir2=$dir



echo ============================================================================
echo "Extract Presoftmax Feats on adapted DNN on `basename $devDataDir`"
echo ============================================================================
if [ $stage -le 3 ]; then
  dir=$exp/`basename $devDataDir`/bn-${remove_last_components}_`basename $testDataDir`
  logDir=$dir/log
  mkdir -p $dir $logDir
  
  steps/nnet/make_bn_feats.sh --nj $feats_nj \
      --cmd "$train_cmd" \
      --remove-last-components ${remove_last_components} \
      $dir $testDataDir $dir1 $logDir $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1
fi
dir3=$dir


echo ============================================================================
echo "                Compute adapted MSE on for `basename $devDataDir`         "
echo ============================================================================
if [ $stage -le 4 ]; then
  dir=$exp/`basename $devDataDir`/aann_decode_post-dnn_`basename $testDataDir`
  logDir=$dir/log
  mkdir -p $dir $logDir
  steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj $decode_nj $dir $dir3 $aannDir $dir/log $dir/data || exit 1
  compute-cmvn-stats scp:$dir/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_utt.txt.ark || exit 1 
  ivector-mean ark,t:$dir/mse_utt.txt.ark $dir/mse_global.vec || exit 1
  
fi
dir4=$dir





echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0

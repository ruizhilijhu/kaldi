#!/bin/bash

# This script
# 0 ) Extract fbank features on multicondition test data
# 1 ) Extract hidden activations over all data.
# 2 ) Pre-train an autoencoder.
# 3 ) Train an autoencoder.
# 4 ) Compute Mean-Square Error (MSE) loss on all datasets.
# 5 ) Compute Co-Activation matrix on Train data.
# 6 ) Compute Frobenius Norm (FN) loss on all datasets.
# 7 ) Decode on DNN on all data set.

# Goal: Have three numbers for one utterance. 
 
. ./cmd.sh
. ./path.sh 

# Config:
splice=1
splice_step=5
nj=60
stage=4
nnet_dir=exp/dnn4_pretrain-dbn_dnn
remove_last_components=1
fea_dir=data-fbank
cmd=run.pl
gmmdir=exp/mono
exp=ctx1_splice2

# End of config

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. utils/parse_options.sh || exit 1

# Extract fbank features on multicondition test data
if [ $stage -le 0 ]; then
  for data_fbank in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
      steps/make_fbank.sh --nj 10 \
	$data_fbank/$x $data_fbank/$x/log $data_fbank/$x/data || exit 1;
      steps/compute_cmvn_stats.sh $data_fbank/$x $data_fbank/$x/log $data_fbank/$x/data || exit 1    
    done
  done
fi


# Extract hidden activations from fea_dir 
if [ $stage -le 1 ]; then
  dir=data-bn-${remove_last_components}
  # clean   
  for x in train dev test; do  
#!/bin/bash

# This script
# 0 ) Extract fbank features on multicondition test data
# 1 ) Extract hidden activations over all data.
# 2 ) Pre-train an autoencoder.
# 3 ) Train an autoencoder.
# 4 ) Compute Mean-Square Error (MSE) loss on all datasets.
# 5 ) Compute Co-Activation matrix on Train data.
# 6 ) Compute Frobenius Norm (FN) loss on all datasets.
# 7 ) Decode on DNN on all data set.

# Goal: Have three numbers for one utterance. 
 
. ./cmd.sh
. ./path.sh 

# Config:
splice=1
splice_step=5
nj=60
stage=4
nnet_dir=exp/dnn4_pretrain-dbn_dnn
remove_last_components=1
fea_dir=data-fbank
cmd=run.pl
gmmdir=exp/mono
exp=ctx1_splice2

# End of config

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. utils/parse_options.sh || exit 1

# Extract fbank features on multicondition test data
if [ $stage -le 0 ]; then
  for data_fbank in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
      steps/make_fbank.sh --nj 10 \
	$data_fbank/$x $data_fbank/$x/log $data_fbank/$x/data || exit 1;
      steps/compute_cmvn_stats.sh $data_fbank/$x $data_fbank/$x/log $data_fbank/$x/data || exit 1    
    done
  done
fi


# Extract hidden activations from fea_dir 
if [ $stage -le 1 ]; then
  dir=data-bn-${remove_last_components}
  # clean   
  for x in train dev test; do  
    steps/nnet/make_bn_feats.sh --nj 10 \
      --cmd "$train_cmd" \
      --remove-last-components ${remove_last_components} \
      $dir/$fea_dir/$x $fea_dir/$x $nnet_dir $dir/$fea_dir/$x/log $dir/$fea_dir/$x/data || exit 1
    steps/compute_cmvn_stats.sh $dir/$fea_dir/$x $dir/$fea_dir/$x/log $dir/$fea_dir/$x/data || exit 1
  done
  utils/subset_data_dir_tr_cv.sh $dir/$fea_dir/train $dir/$fea_dir/train_tr90 $dir/$fea_dir/train_cv10 || exit 1
  # noise
  for d in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
	steps/nnet/make_bn_feats.sh --nj 10 \
	  --cmd "$train_cmd" \
	  --remove-last-components ${remove_last_components} \
	  $dir/$d/$x $d/$x $nnet_dir $dir/$d/$x/log $dir/$d/$x/data || exit 1;
	steps/compute_cmvn_stats.sh $dir/$d/$x $dir/$d/$x/log $dir/$d/$x/data || exit 1    
    done
  done
fi


#Pretrain AE
if [ $stage -le 2 ]; then
  dir=exp/aann_dbn_${remove_last_components}_${exp}
  mkdir -p $dir
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)&
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" \
    data-bn-${remove_last_components}/$fea_dir/train $dir || exit 1;
  dbn=$dir/5.dbn
fi
#    --cmvn-opts "--norm-means=true --norm-vars=true" \



# Train AE
if [ $stage -le 3 ]; then

  dbn=exp/aann_dbn_${remove_last_components}_${exp}/5.dbn

  dir=exp/aann_${remove_last_components}_${exp}
  mkdir -p $dir
  (tail --pid=$$ -F $dir/log/train_aann.log 2>/dev/null)&
  $cuda_cmd $dir/log/train_aann.log \
    steps/multi-stream-nnet/train_aann.sh \
      --splice $splice --splice-step $splice_step --train-opts "--max-iters 30" \
      --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
      --hid-layers 0 --dbn $dbn --learn-rate 0.0008 \
      --copy-feats "false" --skip-cuda-check "true" \
      data-bn-${remove_last_components}/$fea_dir/train_tr90 data-bn-${remove_last_components}/$fea_dir/train_cv10 $dir || exit 1;
fi

# Compute per-frame mse; per-utt mse (avg over frame), global 
if [ $stage -le 4 ]; then
  dir=exp/aann_${remove_last_components}_${exp} # put mse in here  
  # clean
  for x in train dev test; do  
    steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj 10 $dir/mse_$fea_dir/$x data-bn-${remove_last_components}/$fea_dir/$x $dir $dir/mse_$fea_dir/$x/log $dir/mse_$fea_dir/$x/data || exit 1
    compute-cmvn-stats scp:$dir/mse_$fea_dir/${x}/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_$fea_dir/${x}/mse_utt.txt.ark || exit 1
    ivector-mean ark,t:$dir/mse_$fea_dir/$x/mse_utt.txt.ark $dir/mse_$fea_dir/$x/mse_global.vec || exit 1
  done
  # noise
  for d in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
      steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj 10 $dir/mse_$d/$x data-bn-${remove_last_components}/$d/$x $dir $dir/mse_$d/$x/log $dir/mse_$d/$x/data || exit 1 
      compute-cmvn-stats scp:$dir/mse_$d/${x}/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_$d/${x}/mse_utt.txt.ark || exit 1
      ivector-mean ark,t:$dir/mse_$d/$x/mse_utt.txt.ark $dir/mse_$d/$x/mse_global.vec || exit 1
    done
  done
fi



# Compute Co-Activtion Matrix on Train data.
if [ $stage -le 5 ]; then
  dir=exp/co-Act_${remove_last_components}_${exp}
  mkdir -p $dir  
  est-coact-mat scp:data-bn-${remove_last_components}/$fea_dir/train/feats.scp $dir/ref_coact_mat.ark || exit 1
fi


# Compute Frobenius Norm (FN) loss on all datasets. per-utt and global
if [ $stage -le 6 ]; then
  dir=exp/co-Act_${remove_last_components}_${exp}
  for d in data-fbank data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do  
    for x in dev test; do
      mkdir -p $dir/$d/$x
      compute-fn-loss $dir/ref_coact_mat.ark scp:data-bn-${remove_last_components}/$d/$x/feats.scp ark,t:$dir/$d/$x/fn_utt.txt.ark || exit 1
      ivector-mean ark,t:$dir/$d/$x/fn_utt.txt.ark $dir/$d/$x/fn_global.vec || exit 1
    done      
  done
  mkdir -p $dir/data-fbank/train
  compute-fn-loss $dir/ref_coact_mat.ark scp:data-bn-${remove_last_components}/data-fbank/train/feats.scp ark,t:$dir/data-fbank/train/fn_utt.txt.ark || exit 1
  ivector-mean ark,t:$dir/data-fbank/train/fn_utt.txt.ark $dir/data-fbank/train/fn_global.vec || exit 1
fi


exit 0

# Decode on DNN with all dataset 
if [ $stage -le 7 ]; then
  dir=exp/dnn4_pretrain-dbn_dnn # put mse in here    
  for d in data-fbank data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do  
    for x in dev test; do
      (steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 $gmmdir/graph $d/$x $dir/decode_${d}_${x}_${remove_last_components} || exit 1) & sleep 5
    done      
  done
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 $gmmdir/graph data-fbank/train $dir/decode_data-fbank_train_${remove_last_components} || exit 1
fi


echo Success
exit 0;


# TODO:
# 1. get sentence-level score (stage 7)
# 2. create a spk-utt condition-utt mapping
# 3. Create a score file which has uttid err-cnt total-cnt (for dnn decoding) uttid score for PM in kaldi vector form
# 4. modify i-vector mean to get mean scores.
# 5. one binary to compute corelation for two inputs

    steps/nnet/make_bn_feats.sh --nj 10 \
      --cmd "$train_cmd" \
      --remove-last-components ${remove_last_components} \
      $dir/$fea_dir/$x $fea_dir/$x $nnet_dir $dir/$fea_dir/$x/log $dir/$fea_dir/$x/data || exit 1
    steps/compute_cmvn_stats.sh $dir/$fea_dir/$x $dir/$fea_dir/$x/log $dir/$fea_dir/$x/data || exit 1
  done
  utils/subset_data_dir_tr_cv.sh $dir/$fea_dir/train $dir/$fea_dir/train_tr90 $dir/$fea_dir/train_cv10 || exit 1
  # noise
  for d in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
	steps/nnet/make_bn_feats.sh --nj 10 \
	  --cmd "$train_cmd" \
	  --remove-last-components ${remove_last_components} \
	  $dir/$d/$x $d/$x $nnet_dir $dir/$d/$x/log $dir/$d/$x/data || exit 1;
	steps/compute_cmvn_stats.sh $dir/$d/$x $dir/$d/$x/log $dir/$d/$x/data || exit 1    
    done
  done
fi


#Pretrain AE
if [ $stage -le 2 ]; then
  dir=exp/aann_dbn_${remove_last_components}_${exp}
  mkdir -p $dir
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)&
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/multi-stream-nnet/pretrain_dbn.sh \
    --splice $splice --splice-step $splice_step --hid-dim "512:512:24:512:512" \
    data-bn-${remove_last_components}/$fea_dir/train $dir || exit 1;
  dbn=$dir/5.dbn
fi
#    --cmvn-opts "--norm-means=true --norm-vars=true" \



# Train AE
if [ $stage -le 3 ]; then

  dbn=exp/aann_dbn_${remove_last_components}_${exp}/5.dbn

  dir=exp/aann_${remove_last_components}_${exp}
  mkdir -p $dir
  (tail --pid=$$ -F $dir/log/train_aann.log 2>/dev/null)&
  $cuda_cmd $dir/log/train_aann.log \
    steps/multi-stream-nnet/train_aann.sh \
      --splice $splice --splice-step $splice_step --train-opts "--max-iters 30" \
      --proto-opts "--no-softmax --activation-type=<Sigmoid>" \
      --hid-layers 0 --dbn $dbn --learn-rate 0.0008 \
      --copy-feats "false" --skip-cuda-check "true" \
      data-bn-${remove_last_components}/$fea_dir/train_tr90 data-bn-${remove_last_components}/$fea_dir/train_cv10 $dir || exit 1;
fi

# Compute per-frame mse; per-utt mse (avg over frame), global 
if [ $stage -le 4 ]; then
  dir=exp/aann_${remove_last_components}_${exp} # put mse in here  
  # clean
  for x in train dev test; do  
    steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj 10 $dir/mse_$fea_dir/$x data-bn-${remove_last_components}/$fea_dir/$x $dir $dir/mse_$fea_dir/$x/log $dir/mse_$fea_dir/$x/data || exit 1
    compute-cmvn-stats scp:$dir/mse_$fea_dir/${x}/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_$fea_dir/${x}/mse_utt.txt.ark || exit 1
    ivector-mean ark,t:$dir/mse_$fea_dir/$x/mse_utt.txt.ark $dir/mse_$fea_dir/$x/mse_global.vec || exit 1
  done
  # noise
  for d in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
    for x in dev test; do
      steps/multi-stream-nnet/compute_mse_aann.sh --cmd "$decode_cmd" --nj 10 $dir/mse_$d/$x data-bn-${remove_last_components}/$d/$x $dir $dir/mse_$d/$x/log $dir/mse_$d/$x/data || exit 1 
      compute-cmvn-stats scp:$dir/mse_$d/${x}/feats.scp ark:- | compute-cmvn-stats-avg ark:- ark,t:$dir/mse_$d/${x}/mse_utt.txt.ark || exit 1
      ivector-mean ark,t:$dir/mse_$d/$x/mse_utt.txt.ark $dir/mse_$d/$x/mse_global.vec || exit 1
    done
  done
fi



# Compute Co-Activtion Matrix on Train data.
if [ $stage -le 5 ]; then
  dir=exp/co-Act_${remove_last_components}_${exp}
  mkdir -p $dir  
  est-coact-mat scp:data-bn-${remove_last_components}/$fea_dir/train/feats.scp $dir/ref_coact_mat.ark || exit 1
fi


# Compute Frobenius Norm (FN) loss on all datasets. per-utt and global
if [ $stage -le 6 ]; then
  dir=exp/co-Act_${remove_last_components}_${exp}
  for d in data-fbank data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do  
    for x in dev test; do
      mkdir -p $dir/$d/$x
      compute-fn-loss $dir/ref_coact_mat.ark scp:data-bn-${remove_last_components}/$d/$x/feats.scp ark,t:$dir/$d/$x/fn_utt.txt.ark || exit 1
      ivector-mean ark,t:$dir/$d/$x/fn_utt.txt.ark $dir/$d/$x/fn_global.vec || exit 1
    done      
  done
  mkdir -p $dir/data-fbank/train
  compute-fn-loss $dir/ref_coact_mat.ark scp:data-bn-${remove_last_components}/data-fbank/train/feats.scp ark,t:$dir/data-fbank/train/fn_utt.txt.ark || exit 1
  ivector-mean ark,t:$dir/data-fbank/train/fn_utt.txt.ark $dir/data-fbank/train/fn_global.vec || exit 1
fi


exit 0

# Decode on DNN with all dataset 
if [ $stage -le 7 ]; then
  dir=exp/dnn4_pretrain-dbn_dnn # put mse in here    
  for d in data-fbank data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do  
    for x in dev test; do
      (steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 $gmmdir/graph $d/$x $dir/decode_${d}_${x}_${remove_last_components} || exit 1) & sleep 5
    done      
  done
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 $gmmdir/graph data-fbank/train $dir/decode_data-fbank_train_${remove_last_components} || exit 1
fi


echo Success
exit 0;


# TODO:
# 1. get sentence-level score (stage 7)
# 2. create a spk-utt condition-utt mapping
# 3. Create a score file which has uttid err-cnt total-cnt (for dnn decoding) uttid score for PM in kaldi vector form
# 4. modify i-vector mean to get mean scores.
# 5. one binary to compute corelation for two inputs


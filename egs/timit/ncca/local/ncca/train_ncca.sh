#!/bin/bash

# Copyright 2012/2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
#

# FEATURE PROCESSING
copy_feats=true # resave the train/cv features into /tmp (disabled by default)
copy_feats_tmproot= # tmproot for copy-feats (optional)
# feature config (applies always)
cmvn_opts=
delta_opts=
# feature_transform:

# LABELS
labels=            # use these labels to train (override deafault pdf alignments, has to be in 'Posterior' format, see ali-to-post) 

# TRAINING SCHEDULER
adapt_scp_opt=     # options, for another train_schedule script, "_no_rdn" for utterance level adaptation
learn_rate=0.00001 # initial learning rate
train_opts=        # options, passed to the training script
train_tool_opts=
train_tool="nnet-train-frmshuff-ncca --objective-function=ncca"
                   # optionally change the training tool
frame_weights=     # per-frame weights for gradient weighting
scheduler_opts=    # options, passed to the training scheduler,

# OTHER
seed=777    # seed value used for training data shuffling and initialization
skip_cuda_check=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
   echo "Usage: $0 <data-train> <data-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv exp/ncca"
   echo ""
   echo " Training data : <data-train>  (for optimizing ncca or mse)"
   echo " Held-out data : <data-dev> (for learn-rate/model selection based on ncca or mse)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --mlp-init <file>       # select initialized MLP"
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo ""
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo "  --copy-feats <bool>      # copy input features to /tmp (it's faster)"
   echo ""
   echo "  --train-tool \"net-train-frmshuff-ncca --object-function=ncca --ncca-ref-mat=ref.mat\""
   exit 1;
fi

data=$1
data_cv=$2
dir=$3

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data \n"
printf "\t CV-set    : $data_cv \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

# check if CUDA is compiled in,
if ! $skip_cuda_check; then
  cuda-compiled || { echo 'CUDA was not compiled in, skipping! Check src/kaldi.mk and src/configure' && exit 1; }
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

# re-save the train/cv features to /tmp, reduces LAN traffic, avoids disk-seeks due to shuffled features
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d $copy_feats_tmproot); mv $dir/train.scp{,_non_local}; mv $dir/cv.scp{,_non_local}
  copy-feats scp:$dir/train.scp_non_local ark,scp:$tmpdir/train.ark,$dir/train.scp || exit 1
  copy-feats scp:$dir/cv.scp_non_local ark,scp:$tmpdir/cv.ark,$dir/cv.scp || exit 1
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
fi

#create a 10k utt subset for global cmvn estimates
head -n 10000 $dir/train.scp > $dir/train.scp.10k


###### PREPARE FEATURE PIPELINE ######

# optionally import feature setup from pre-training,
if [ ! -z $feature_transform ]; then
  D=$(dirname $feature_transform)
  [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  echo "Imported config : cmvn_opts='$cmvn_opts' delta_opts='$delta_opts'"
else
  echo "Missing feature_transform file" && exit 1;
fi

# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"
# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
else
  echo "apply-cmvn is not used"
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "add-deltas with $delta_opts"
fi

# keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts 
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# get feature dim
echo "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"

# Now we will start building complex feature_transform which will 
# be forwarded in CUDA to have fast run-time.
#
# We will use 1GPU for both feature_transform and MLP training in one binary tool. 
# This is against the kaldi spirit to have many independent small processing units, 
# but it is necessary because of compute exclusive mode, where GPU cannot be shared
# by multiple processes.

if [ ! -z "$feature_transform" ]; then
  echo "Using pre-computed feature-transform : '$feature_transform'"
  tmp=$dir/$(basename $feature_transform) 
  cp $feature_transform $tmp; feature_transform=$tmp
else
  echo "Missing feature_transform file" && exit 1;
fi


###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
(cd $dir; [ ! -f final.feature_transform ] && ln -s $(basename $feature_transform) final.feature_transform )

echo "PREPARING LABELS"

if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else 
  o_feats_tr=`echo $feats_tr | sed 's@ark:copy-feats@ark,o:copy-feats@g'`
  o_feats_cv=`echo $feats_cv | sed 's@ark:copy-feats@ark,o:copy-feats@g'`
  labels_tr="$o_feats_tr nnet-forward ${dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |"
  labels_cv="$o_feats_cv nnet-forward ${dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |"
fi

###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
[ ! -z "$mlp_init" ] && echo "Using pre-initialized network '$mlp_init'";

###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
local/ncca/train_scheduler${adapt_scp_opt}.sh \
  ${scheduler_opts} \
  --feature-transform $feature_transform \
  --learn-rate $learn_rate \
  ${train_opts} \
  ${train_tool_opts:+ --train-tool-opts "$train_tool_opts"} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${config:+ --config $config} \
  $mlp_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir || exit 1

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0

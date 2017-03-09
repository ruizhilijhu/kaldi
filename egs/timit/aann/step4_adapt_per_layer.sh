#!/bin/bash

# this script do per-layer training for one layer

# CONFIG
stage=0
begin=

# dnn & data
dnnid="mono_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components=
iter= # option
noise=babble-10

# adapt 
adapt_scp_opt="_no_rdn" # options for other training scp
minibatch_size=19306 # num of frames in one update and back propagation
randomizer_size=32768 # num of frames for one epoch training
lr=0.001
lr_adapt=no
keep_lr_iters=100
max_iters=400
objective_function="aann" # "aann-ncca"
numLayersFreeze=
numLayersFreezeStart=
adapt_options=
l1_penalty=0
l2_penalty=0
dropout_rate=

# decode 
decode_subsets="test"
decode_noises="$noise CLEAN"

# monophone model
gmmdir="exp/mono"

# aann # assumption: bn layer is 24
aannid= 

# coactivation matrix
ref_cm_opt= # post or pre

# out
outdir_suffix=  #"_ncca" for parameter analysis 

. utils/parse_options.sh || exit 1
set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# aann
aannDir=exp/${dnnid}_aann/${aannid}/pretrain-aann_aann

# outdir 
adaptid=${adapt_scp_opt}_aann-${aannid}${outdir_suffix}
outdir=exp/${dnnid}_${noise}${adaptid}
expid=$outdir/$first_components
[ ! -d $expid ] && mkdir -p $expid

# coactivtion matrix
if [ $objective_function == 'aann-cm' ];then
    [ ! -z $ref_cm_opt ] && ref_cm=exp/${dnnid}_aann_cm/${aannid}/${ref_cm_opt}-aann/${first_components}.mat
elif [ $objective_function == 'ncca' ];then
    ref_cm=exp/${dnnid}-bn_cm/${first_components}.mat
fi
    
if [ $begin == 'yes' ];then
    dnnDir=exp/${dnnid}
elif [ $begin == 'no' ];then
    layer=$((first_components-3))
    dnnDir=$outdir/${layer}/postAdapt-dnn
    mlp=$dnnDir/nnets_decode/${iter}/final.nnet    
    [ ! -f $mlp ] && echo "missing mlp! " && exit 1
    [ -z $iter ] && echo "misiing iter number! " && exit 1
fi

if [ ! -z $dropout_rate ]; then
    numDpIter=$((${keep_lr_iters}-5))
    dropout_schedule=
    for ((c=1;c<=$numDpIter;c++));do
	dropout_schedule="$dropout_rate,$dropout_schedule"
    done
    dropout_schedule="${dropout_schedule}0.0"
    echo $dropout_schedule
fi



bash local/ncca/adapt_scheduler.sh $adapt_options \
     ${stage:+ --stage "$stage"} \
     ${dnnid:+ --dnnid "$dnnid"} \
     ${first_components:+ --first_components "$first_components"} \
     ${mlp:+ --mlp "$mlp"} \
     ${dnnDir:+ --dnnDir "$dnnDir"} \
     ${noise:+ --noise "$noise"} \
     ${adapt_scp_opt:+ --adapt_scp_opt "$adapt_scp_opt"} \
     ${minibatch_size:+ --minibatch_size "$minibatch_size"} \
     ${randomizer_size:+ --randomizer_size "$randomizer_size"} \
     ${lr:+ --lr "$lr"} \
     ${lr_adapt:+ --lr_adapt "$lr_adapt"} \
     ${keep_lr_iters:+ --keep_lr_iters "$keep_lr_iters"} \
     ${max_iters:+ --max_iters "$max_iters"} \
     ${objective_function:+ --objective_function "${objective_function}"} \
     ${decode_subsets:+ --decode_subsets "$decode_subsets"} \
     ${decode_noises:+ --decode_noises "$decode_noises"} \
     ${gmmdir:+ --gmmdir "$gmmdir"} \
     ${ref_cm:+ --ref_cm "$ref_cm"} \
     ${numLayersFreeze:+ --numLayersFreeze "$numLayersFreeze"} \
     ${numLayersFreezeStart:+ --numLayersFreezeStart "$numLayersFreezeStart"} \
     ${dropout_rate:+ --dropout_schedule "$dropout_schedule"} \
     --l1-penalty ${l1_penalty} \
     --l2-penalty ${l2_penalty} \
     --aannDir $aannDir \
     --expid $expid \
     2>&1 | tee ${expid}/step4_adapt_scheduler.log    

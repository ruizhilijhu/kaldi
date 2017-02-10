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
lr_adapt=yes
keep_lr_iters=100
max_iters=400
objective_function="aann" # "aann-ncca"
adapt_options=

# decode 
iter_step=2
decode_subsets="test"
decode_noises="$noise CLEAN"
decode_epochs=
frm_decode="yes"
mse_decode="yes"
wer_decode="no"
fn_decode="no"

# monophone model
gmmdir="exp/mono"

# aann # assumption: bn layer is 24
aann_hl=3
aann_hu=512

# out
outdir_suffix=  #"_ncca" for parameter analysis 

. utils/parse_options.sh || exit 1
set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# aann
aannid=${aann_hl}hl_${aann_hu}hu${aann_suffix}
aannDir=exp/${dnnid}_aann/${first_components}_${aann_hl}hl_${aann_hu}hu/pretrain-aann_aann

# outdir 
adaptid=${adapt_scp_opt}_aann-${aann_hl}hl-${aann_hu}hu${outdir_suffix}
outdir=exp/${dnnid}_${noise}${adaptid}
expid=$outdir/$first_components

if [ $begin == 'yes' ];then
    dnnDir=exp/${dnnid}
elif [ $begin == 'no' ];then
    layer=$((first_components-3))
    dnnDir=$outdir/${layer}/postAdapt-dnn
    mlp=$dnnDir/nnets_decode/${iter}/final.nnet    
    [ ! -f $mlp ] && echo "missing mlp! " && exit 1
    [ -z $iter ] && echo "misiing iter number! " && exit 1
fi

bash local/aann/adapt_scheduler_aann.sh $adapt_options \
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
     ${iter_step:+ --iter_step "$iter_step"} \
     ${decode_subsets:+ --decode_subsets "$decode_subsets"} \
     ${decode_noises:+ --decode_noises "$decode_noises"} \
     ${decode_epochs:+ --decode_epochs "$decode_epochs"} \
     ${frm_decode:+ --frm_decode "$frm_decode"} \
     ${mse_decode:+ --mse_decode "$mse_decode"} \
     ${wer_decode:+ --wer_decode "$wer_decode"} \
     ${fn_decode:+ --fn_decode "$fn_decode"} \
     ${gmmdir:+ --gmmdir "$gmmdir"} \
     --aannDir $aannDir \
     --expid $expid \ 
2>&1 | tee ${outdir}/step4_adapt_scheduler.log    

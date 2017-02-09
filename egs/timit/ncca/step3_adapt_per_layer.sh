#!/bin/bash

# this script do per-layer training for one layer

# begin first_compoennt noise adapt_scp_opt fn_mb_opt id
# mlp

# CONFIG
stage=0
begin=yes

# dnn
dnnid="mono_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components=2
iter= # option
noise=babble-10
adapt_scp_opt='_no_rdn' # options for other training scp
minibatch_size=19306 # num of frames in one update and back propagation
randomizer_size=32768 # num of frames for one epoch training
lr=0.001
lr_adapt=no
keep_lr_iters=250
max_iters=400
frm_acc_decode="yes"
fn_decode="yes"
wer_decode="no"
id="_no_rdn"
iter_step=10
decode_subsets="test"
decode_noises="$noise CLEAN"
fn_mb_opt="yes" # err used for lr adaptation 
gmmdir="exp/mono"

. utils/parse_options.sh || exit 1

#adaptid="_minibatch${minibatch_size}${id}"
adaptid="${id}"
outdir=exp/${dnnid}_${noise}${adaptid}/${first_components}
mkdir -p $outdir

if [ $begin == 'yes' ];then
    dnnDir=exp/${dnnid}
    bash local/ncca/adapt_scheduler.sh \
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
	 ${adaptid:+ --adaptid "$adaptid"} \
	 ${frm_acc_decode:+ --frm_acc_decode "$frm_acc_decode"} \
	 ${fn_decode:+ --fn_decode "$fn_decode"} \
	 ${wer_decode:+ --wer_decode "$wer_decode"} \
	 ${iter_step:+ --iter_step "$iter_step"} \
	 ${decode_subsets:+ --decode_subsets "$decode_subsets"} \
	 ${decode_noises:+ --decode_noises "$decode_noises"} \
	 ${fn_mb_opt:+ --fn_mb_opt "$fn_mb_opt"} \
	 ${gmmdir:+ --gmmdir "$gmmdir"} \
	 2>&1 | tee ${outdir}/step3_adapt_scheduler.log
elif [ $begin == 'no' ];then
    # for others
    layer=$((first_components-3))
    dnnDir=exp/${dnnid}_${noise}${adaptid}/${layer}/postAdapt-dnn
    mlp=$dnnDir/frm_decode/${iter}/test/${noise}/final.nnet 
    [ ! -f $mlp ] && echo "missing mlp! " && exit 1
    [ -z $iter ] && echo "misiing iter number! " && exit 1
    bash local/ncca/adapt_scheduler.sh \
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
	 ${adaptid:+ --adaptid "$adaptid"} \
	 ${frm_acc_decode:+ --frm_acc_decode "$frm_acc_decode"} \
	 ${fn_decode:+ --fn_decode "$fn_decode"} \
	 ${wer_decode:+ --wer_decode "$wer_decode"} \
	 ${iter_step:+ --iter_step "$iter_step"} \
	 ${decode_subsets:+ --decode_subsets "$decode_subsets"} \
	 ${decode_noises:+ --decode_noises "$decode_noises"} \
	 ${fn_mb_opt:+ --fn_mb_opt "$fn_mb_opt"} \
	 ${gmmdir:+ --gmmdir "$gmmdir"} \
	 2>&1 | tee ${outdir}/step3_adapt_scheduler.log  
fi

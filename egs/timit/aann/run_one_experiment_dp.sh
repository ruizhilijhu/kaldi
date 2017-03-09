#!/bin/bash
first_components=17
hiddenDim="512:24:512"
splice=0
lr="1.0e-06"
keep_lr_iters=50
iter_step=3
layer_s=1
layer_e=14
opt=only17
dp=0.2
. ./cmd.sh
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

num_hl=`echo $hiddenDim | awk -F ':' '{print NF}'`
aannid=${first_components}_${num_hl}hl_512hu_s${splice}s1
echo $aannid

if [ ! -d exp/mono_dnn-256_aann/$aannid ];then
    bash step3_train_aann_cm.sh --first_components $first_components --hiddenDim "$hiddenDim" --splice $splice || exit 1
fi

outdir_suffix="_lr${lr}${opt}_dp${dp}_new"
bash step4_adapt_per_layer.sh --begin yes --first_components $first_components --lr $lr --keep_lr_iters $keep_lr_iters --numLayersFreezeStart $layer_s --numLayersFreeze $layer_e --aannid $aannid --outdir_suffix $outdir_suffix --dropout_rate $dp|| exit 1


bash step5_compute_all_stats.sh --expid exp/mono_dnn-256_babble-10_no_rdn_aann-${aannid}${outdir_suffix} --first_components $first_components --iter_step $iter_step --keep_lr_iters $keep_lr_iters --mlp_opt 'dnn1' --statsList "frm mse fn mm" || exit 1

#"frm mse fn mm" || exit 1

bash step5_compute_all_stats.sh --expid exp/mono_dnn-256_babble-10_no_rdn_aann-${aannid}${outdir_suffix} --first_components $first_components --iter_step $iter_step --keep_lr_iters $keep_lr_iters --mlp_opt 'dnn' --statsList "fn mm" || exit 1


bash local/aann/get_results_aann_new.sh --expDir exp/mono_dnn-256_babble-10_no_rdn_aann-${aannid}${outdir_suffix}/${first_components} --outdir RESULTS/mono_dnn-256_babble-10_no_rdn_aann-${aannid}${outdir_suffix}/${first_components} || exit 1

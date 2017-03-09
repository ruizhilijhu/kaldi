#!/bin/bash

# this script do aann pretrain and train # also the cm extraction 

# CONFIG
stage=0
dnnid="mono_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components=2
lr=0.0001
hiddenDim="512:24:512"
train_aann_opts=
splice=0
splice_step=1
aann_suffix=

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

data=exp/${dnnid}-bn/${first_components}
hid_dim=$(echo $hiddenDim | awk -F':' '{print $1}')
num_hl=`echo $hiddenDim | awk -F ':' '{print NF}'`
aannid="${first_components}_${num_hl}hl_${hid_dim}hu_s${splice}s${splice_step}"${aann_suffix}
expid=exp/${dnnid}_aann
outdir=${expid}/${aannid}
[ ! -d $outdir ] && mkdir -p $outdir
echo "OUTPUT DIR: "${outdir}

if [ $stage -le 0 ]; then
    bash local/aann/pre-train_train_aann.sh $train_aann_opts\
	 ${stage:+ --stage "$stage"} \
	 ${lr:+ --lr "$lr"} \
	 ${hiddenDim:+ --hiddenDim "$hiddenDim"} \
	 ${hid_dim:+ --hid_dim "$hid_dim"} \
	 ${splice:+ --splice "$splice"} \
	 ${splice_step:+ --splice_step "${splice_step}"} \
	 $data $outdir \
	 2>&1 | tee ${outdir}/step3_train_aann.log  
fi


if [ $stage -le 1 ]; then
    out=${expid}_cm/${aannid}
    [ ! -d $out ] && mkdir -p $out
    bash local/aann/compute_aann_cm.sh --dnnid $dnnid \
	 --first_components ${first_components} \
	 --aannid "${aannid}" \
	 2>&1 | tee ${out}/step3_get_aann_cm.log
fi


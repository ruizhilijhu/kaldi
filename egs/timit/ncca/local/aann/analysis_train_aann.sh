#!/bin/bash

# this script do aann pretrain and train aann and decode 

# CONFIG
lr=0.0001
nj=10
stage=0
first_components=1
dnnid="mono_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
hiddenDim="512:512:24:512:512"
splice=5
splice_step=1
data_fbank_allnoise=data-fbank13-allnoise
data_fbank_clean=data-fbank13-clean
nnet_forward_opts=
train_opts=
pretrain_opts=
cmvn_opts="--norm-means=true --norm-vars=true"
decodeExpId=exp/mono_dnn-256_babble-10_no_rdn_aann-3hl-512hu_lr1.0e-05/2
iter_step=5
keep_lr_iters=200
decodeNoise='babble-10'
decodeSubset='test'
aannSuffix=
utt2utt=
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1
set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging
dnnDir=exp/${dnnid}
numComponents=`nnet-info $dnnDir/final.nnet | grep num-components | awk '{print $2}'`
last_components=$((${numComponents}-${first_components}))
num_hl=`echo $hiddenDim | awk -F ':' '{print NF}'`
hid_dim=$(echo $hiddenDim | awk -F':' '{print $1}')

# set up aann id
aannId=${first_components}
[[ $cmvn_opts != "${cmvn_opts/--norm-means=true --norm-vars=true/}" || $cmvn_opts != "${cmvn_opts/--norm-vars=true --norm-means=true/}" ]] && aannId="${aannId}_spkCMVN"
[ $utt2utt == "utt2utt" ] && aannId="${aannId}_utt2utt"
aannId=${aannId}_${num_hl}hl_${hid_dim}hu_s${splice}s${splice_step}${aannSuffix}
aannDir=exp/analysis_${dnnid}_aann/${aannId}

decodeExpDir=${decodeExpId}/postAdapt-dnn/nnets_decode
decodeEpochs=`seq -f"%03g" 0 ${iter_step} ${keep_lr_iters}`

echo "AANN-ID: $aannId"    

dataBN=exp/${dnnid}-bn/${first_components}



# train aann 
out=$aannDir
if [ $stage -le 2 ]; then
    [ ! -d $out ] && mkdir -p $out
    # if uttCMVN
    if [ $utt2utt == "utt2utt" ]; then
	# copy data directoty and change utt2spk and spk2utt to utt2utt
	utils/copy_data_dir.sh $dataBN $out/data_utt2utt || exit 1
	dataBN=$out/data_utt2utt
	rm $dataBN/spk2utt || exit 1
	paste <(awk '{print $1}' $dataBN/utt2spk) <(awk '{print $1}' $dataBN/utt2spk) > $dataBN/spk2utt || exit 1
	cp $dataBN/spk2utt $dataBN/utt2spk || exit 1
	steps/compute_cmvn_stats.sh $dataBN $dataBN/log $dataBN/data || exit 1

    fi
    
    local/aann/pre-train_train_aann.sh --lr $lr \
				       --hiddenDim "${hiddenDim}" \
				       --hid_dim "${hid_dim}" \
				       --splice "${splice}" \
				       --splice_step "${splice_step}" \
				       ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} \
				       ${train_opts:+ --train_opts "${train_opts}"} \
				       ${pretrain_opts:+ --pretrain_opts "${pretrain_opts}"} \
				       $dataBN $aannDir \
				       2>&1 | tee $out/pre-train_train_aann.log || exit 1

fi
out2=$out

# decode aann over experiments   # assumtopn : aann is train with spk cmvn
out=$aannDir/$decodeExpId/$decodeSubset/$decodeNoise
if [ $stage -le 3 ]; then
    [ ! -d $out ] && mkdir -p $out
    for iter in ${decodeEpochs}; do
	O=$out/$iter; mkdir -p $O
	echo "$O"
	nnet-copy --remove_last_components=${last_components} $decodeExpDir/$iter/final.nnet $O/feats_ex.nnet
	(
	    D=$data_fbank_allnoise/$decodeSubset/$decodeNoise
	    
	    local/aann/combine_make_mlpfeats_compute_mse_aann.sh \
		--cmd "${decode_cmd}" \
		--nj $nj \
		--mlp $O/feats_ex.nnet \
		--aann_cmvn_opts "${cmvn_opts}" \
		${utt2utt:+ --utt2utt "${utt2utt}"} \
		$O $D 'haha' $decodeExpDir/../ $aannDir/pretrain-aann_aann $O/log $O/data || exit 1
	    python local/aann/mse_to_text_score.py $O/feats.scp $O/score.text
	) && sleep 5
    done
fi
out3=$out

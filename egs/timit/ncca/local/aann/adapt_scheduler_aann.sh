#!/bin/bash

# This script does cm-based dnn adaptation based on one coactivation matrix (one layer for one data)
# this script does everything but postAdapting decoding 

# 0) preadapt DNN preparation -- Split DNN into two parts final.nnet.1 final.nnet.2 and feature transform
# + preadapt aann preparation
# 1) extract bn feats: clean and adapted data
# 2) compute mse,wer,fn,frm: clean and adapted data 
# 3) adaptation 
# 4) Preparation : DNN dir (Post Adaptation)
# 5) Post Adaptation: MSE,WER,FN,FRM,BN

# note, it can do decoding for any epoch on any options by setting --stage 5 --decode_epochs "011 012" 

# CONFIG
feats_nj=10
train_nj=30
decode_nj=20
stage=0

# dnn
dnnid= # "mon_dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components= # 2
mlp= # option for non begining layer
dnnDir= # dnnDir to be copied from 

# data 
data_fbank_clean=data-fbank13-clean
data_fbank_allnoise=data-fbank13-allnoise
noise= # babble-10

# adapt
adapt_scp_opt= # options for other training scp
minibatch_size=19306 # num of frames in one update and back propagation
randomizer_size=32768 # num of frames for one epoch training
randomizer_seed=777
lr=0.001
lr_adapt=no
keep_lr_iters=250
max_iters=400
objective_function="aann"  # aann-ncca 
numLayersFreeze=

# decode
iter_step=20
decode_subsets= # "dev test"
decode_noises= # "babble-10 CLEAN"
decode_epochs= # specify which ones to decode
frm_decode="yes"
mse_decode="yes"
wer_decode="no"
fn_decode="no"

# monophone model
gmmdir=exp/mono

# aann dir
aannDir=

# output
expid=

# coactivation mat reference
ref_cm=

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# aann
[ -z $aannDir ] && echo "Missing AANN dir!" && exit 1

# dnn
if [ ! -z $mlp ]; then
    numComponents_dnn=`nnet-info ${mlp} | grep num-components | awk '{print $2}'`
else
    numComponents_dnn=`nnet-info ${dnnDir}/final.nnet | grep num-components | awk '{print $2}'`
fi
[ -z $numLayersFreeze ] && numLayersFreeze=$((${first_components}-3)) 
last_components=$((${numComponents_dnn}-${first_components}))
echo "number of Components in DNN: $numComponents_dnn"
echo "number of first components: $first_components"
echo "number of last components: $last_components"

# aann-cm
[ $objective_function == 'aann-cm' -a -z $ref_cm ] && echo "Missing referece coactivation mat!" && exit 1

# decode
[ -z $decode_epochs ] && decode_epochs=`seq -f"%03g" 1 ${iter_step} ${keep_lr_iters}`
    
# output dir 
echo "OUTPUT DIR: "${expid}
[ ! -d ${expid} ] && mkdir -p ${expid}


###### STEPS #####
echo ============================================================================
echo "              Preparation: DNN(Trimmed) & AANN(Non-updatable)             "
echo ============================================================================
outdir=${expid}/preAdapt-dnn # the only output
if [ $stage -le 0 ]; then

    # copy dnn dir
    [ ! -d $outdir ] && rsync -avr --exclude '*decode*' $dnnDir/ $outdir || exit 1

    # if specify initial nnet other than the defalut final.net
    if [ ! -z $mlp ]; then
	[ -L $outdir/final.nnet ] && unlink $outdir/final.nnet
	[ -f $outdir/final.nnet ] && rm $outdir/final.nnet
	[ -L $outdir/mlp_init ] && unlink $outdir/mlp_init
	
	mlp_name=$(basename $mlp)
	
	ln -s $PWD/$mlp $outdir/mlp_init
	ln -s $PWD/$mlp $outdir/final.nnet
    fi
    
    # freeze adapted layers
    if [ $numLayersFreeze -ge 1 ]; then
    	if [ -L $outdir/final.nnet ]; then

	    orig_mlp=$(readlink -f $outdir/final.nnet)
	    unlink $outdir/final.nnet
	    cp $orig_mlp $outdir/final.nnet.tmp || exit 1
	else
	    cp $outdir/final.nnet $outdir/final.nnet.tmp || exit 1
	fi
	
	[ -f $outdir/final.nnet ] && rm $outdir/final.nnet
    	nnet-set-learnrate --components=`seq -s ":" 1 $numLayersFreeze` --coef=0 ${outdir}/final.nnet.tmp $outdir/final.nnet 
    fi

    
    # split final.nnet into final.nnet.1 final.nnet.2
    [ -f $outdir/final.nnet.1 ] && rm $outdir/final.nnet.1
    [ -f $outdir/final.nnet.2 ] && rm $outdir/final.nnet.2
    [ -f $outdir/final.nnet.tmp ] && rm $outdir/final.nnet.tmp
    [ -f $outdir/final.nnet.1.aann.ft ] && rm $outdir/final.nnet.1.aann.ft

    nnet-copy --remove-last-components=$last_components $outdir/final.nnet $outdir/final.nnet.1 >$outdir/log/trim_dnn.log 2>&1 || exit 1
    nnet-copy --remove-first-components=$first_components $outdir/final.nnet $outdir/final.nnet.2 >>$outdir/log/trim_dnn.log 2>&1|| exit 1

    # AANN: copy aann.nnet and aann.ft and make aann.nnet non-updatable
    [ -d $outdir/aann ] && rm -r $outdir/aann
    mkdir -p $outdir/aann
    
    aann_mlp=$outdir/aann/aann.nnet
    aann_ft=$outdir/aann/aann.ft
    
    numComponents_aann=`nnet-info $aannDir/final.nnet | grep num-components | awk '{print $2}'`
    nnet-set-learnrate --components=`seq -s ":" 1 $numComponents_aann` --coef=0 $aannDir/final.nnet $aann_mlp 
    cp $aannDir/final.feature_transform $aann_ft
    nnet-concat $aann_ft $aann_mlp $outdir/aann/aann.ft.nnet # used for adaptation    
    nnet-concat $outdir/final.nnet.1 $outdir/aann/aann.ft $outdir/final.nnet.1.aann.ft
fi
outdir0=$outdir


echo ============================================================================
echo "       Pre-Adaptation: Extract BN features for clean and adapted data     "
echo ============================================================================
outdir=${expid}/preAdapt-dnn/mse_decode_bn # the only output
if [ $stage -le 1 ]; then
    mkdir -p $outdir
    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do	    
	    steps/nnet/make_bn_feats.sh --nj ${feats_nj} \
					--cmd "$train_cmd" \
					--remove-last-components $last_components \
					$outdir/$d/$n ${data_fbank_allnoise}/$d/$n $outdir0 $outdir/$d/$n/log $outdir/$d/$n/data || exit 1
	    steps/compute_cmvn_stats.sh $outdir/$d/$n $outdir/$d/$n/log $outdir/$d/$n/data || exit 1
	done
    done
fi
outdir1=$outdir


echo ============================================================================
echo "      Pre-Adaptation: Compute MSE, FRM, FN, WER for clean and adapted data"
echo ============================================================================
outdir=${expid}/preAdapt-dnn # the only output
if [ $stage -le 2 ]; then
    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do
	    # echo "Pre-Adaptation Decoding: WER"
	    # out_wer=$outdir/wer_decode/$d/$n
	    # mkdir -p $out_wer
	    # ( 
	    # 	local/ncca/decode_ncca.sh --nj $decode_nj \
	    # 				  --cmd "$decode_cmd" \
	    # 				  --acwt 0.2 \
	    # 				  --srcdir $outdir \
	    # 				  --nnet $outdir/final.nnet \
	    # 				  $gmmdir/graph $data_fbank_allnoise/$d/$n $out_wer || exit 1	    
	    # ) & sleep 2

	    echo "Pre-Adaptation Decoding: FRM"
	    out_frm=$outdir/frm_decode/$d/$n
	    mkdir -p $out_frm
	    ( 
	    	local/ncca/compute_frm_acc.sh --mlp $outdir/final.nnet \
	    				      --cmd "$decode_cmd" \
	    				      $data_fbank_allnoise/$d/$n ${gmmdir}_ali/data-allnoise/$d/$n $outdir $out_frm || exit 1
	    ) & sleep 2
	    
	    echo "Pre-Adaptation Decoding: MSE"
	    out_mse=$outdir/mse_decode/$d/$n
	    mkdir -p $out_mse
	    local/aann/compute_mse_aann.sh --mlp $outdir/final.nnet.1.aann.ft \
	    				   --cmd "$decode_cmd" \
	    				   --aann_mlp $outdir/aann/aann.nnet \
	    				   $data_fbank_allnoise/$d/$n $outdir $out_mse || exit 1
	    
	    if [ ${objective_function} == "aann-ncca" -o ${objective_function} == "aann-cm" ]; then
		echo "Pre-Adaptation Decoding: FN"
		out_fn=$outdir/fn_decode/$d/$n		
		mkdir -p $out_fn			    
		local/aann/compute_fn_loss_aann-ncca.sh --cmd "$decode_cmd" \
							--mlp $outdir/final.nnet.1.aann.ft \
							--minibatch_size ${minibatch_size} \
							--aann_mlp $outdir/aann/aann.nnet \
							--objective-function $objective_function \
							${ref_cm:+ --ref_cm ${ref_cm}} \
							$data_fbank_allnoise/$d/$n $outdir $out_fn || exit 1
	    fi	    
	done
    done
fi
outdir2=$outdir


echo ============================================================================
echo "                 Adaptation based on $noise                               "
echo ============================================================================
outdir=${expid}/postAdapt-dnn-RAW # the only output
if [ $stage -le 3 ]; then

    # # learning rate modification
    if [ "${lr_adapt}" == "yes" ]; then
	
	if [ $objective_function == "aann" ];then
	    echo "Objective Function: aann"
	    echo "ERR is calculated in frame base (mse)"
	    err=$(cat $outdir2/mse_decode/test/$noise/mse_frm)	    
	elif [ $objective_function == "aann-ncca" -o $objective_function == "aann-cm" ]; then
	    echo "Objective Function: aann-ncca"
	    echo "ERR is calculated in minibatch base (fn)"
	    err=$(cat $outdir2/fn_decode/test/$noise/fn_mb)
	else
	    echo "wrong objective function $objective_function" && exit 1
	fi
	lr=$(awk "BEGIN{print($lr/$err)}")
	echo "Learning rate is adjusted by eer ($err) --> $lr"
    fi
    
    feature_transform=$outdir0/final.feature_transform
    mlp_init=$outdir0/final.nnet.1.aann.ft

    # train options passed as options for schedule
    train_tool_opts="--minibatch-size=${minibatch_size} --randomizer-seed=${randomizer_seed} --randomizer-size=${randomizer_size}"
    train_options="--keep-lr-iters ${keep_lr_iters}"
    train_options="${train_options} --max-iters $max_iters"
    
    train_tool="nnet-train-frmshuff-ncca --objective-function=${objective_function} --aann-mlp-filename=$outdir0/aann/aann.nnet"
    [ $objective_function == "aann-cm" ] && train_tool="${train_tool} --ncca-ref-mat=${ref_cm}"
    
    
    (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
    $cuda_cmd $outdir/log/train_ncca.log \
    	      local/ncca/train_ncca.sh \
     	      --mlp-init $mlp_init \
	      --train-opts "$train_options" \
	      --learn-rate $lr \
    	      --copy-feats "false" --skip-cuda-check "true" \
    	      --feature-transform "$feature_transform" \
    	      --train-tool "${train_tool}" \
	      --train-tool-opts "${train_tool_opts}" \
	      ${adapt_scp_opt:+ --adapt_scp_opt "$adapt_scp_opt"} \
    	      ${data_fbank_allnoise}/test/${noise} ${data_fbank_allnoise}/dev/${noise} $outdir || exit 1
fi
outdir3=$outdir


echo ============================================================================
echo "   Preparation : DNN dir (Post Adaptation)                                "
echo ============================================================================
outdir=${expid}/postAdapt-dnn # the only output
if [ $stage -le 4 ]; then
    
    # copy dnn dir
    [ ! -d $outdir ] && rsync -ar --exclude '*decode*' $outdir0/ $outdir
    [ -L $outdir/final.nnet ] && unlink $outdir/final.nnet
    [ -d $outdir/nnet ] && rm -r $outdir/nnet
    [ -d $outdir/log ] && rm -r $outdir/log
    cp -r $outdir3/nnet $outdir/nnet || exit 1
    
    # initial decode results (copy from preAdapt)
    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do
	    for s in wer frm mse fn; do
		out=$outdir/${s}_decode/000/$d/$n
		if [ ! -d $out -a -d $outdir2/${s}_decode ]; then
		    mkdir -p $out 
		    cp -r $outdir2/${s}_decode/$d/$n/* $out || exit 1		    
		fi
	    done
	done
    done
    
    # organize nnets over epoch in nnets_decode
    [ -d $outdir/nnets_decode ] && rm -r $outdir/nnets_decode
    # # initial epoch 000
    [ ! -d ${outdir}/nnets_decode/000 ] && mkdir -p ${outdir}/nnets_decode/000
    for f in final.nnet.1.aann.ft final.nnet.1 final.nnet; do
	ln -s $PWD/$outdir0/${f} $outdir/nnets_decode/000/${f} || exit 1
    done
    # # iterate over epochs
    for iter in `seq -f"%03g" 1 1 ${keep_lr_iters}` ;do
	
	# create nnets dir
	[ ! -d ${outdir}/nnets_decode/$iter ] && mkdir -p ${outdir}/nnets_decode/$iter
	out=$outdir/nnets_decode/${iter}
	
	# create the adapted dnn
	nnet_id=$(basename ${outdir}/nnet/*iter${iter}*)
	echo "Organizing nnets in epoch ${iter}: ${nnet_id}"
	
	ln -s $PWD/${outdir}/nnet/${nnet_id} $out/final.nnet.1.aann.ft # final.nnet.1.aann.ft
	numComponents_aann_ft=$(nnet-info $outdir0/aann/aann.ft | grep 'num-components' | awk '{print $2}')
	nnet-copy --remove-last-components=${numComponents_aann_ft} ${outdir}/nnet/${nnet_id} $out/final.nnet.1 # final.nnet.1
	nnet-concat $out/final.nnet.1 $outdir0/final.nnet.2 $out/final.nnet # final.nnet	
    done
fi
outdir4=$outdir


echo ============================================================================
echo "   Post-Adaptation: Compute MSE, FRM, FN, WER for clean and adapted data  "
echo ============================================================================
outdir=${expid}/postAdapt-dnn # the only output
if [ $stage -le 5 ]; then
    # iterate over selected epochs
    for iter in ${decode_epochs} ;do
	for d in ${decode_subsets}; do
	    for n in ${decode_noises}; do

		# create the adapted dnn
		nnet_id=$(basename ${outdir}/nnet/*iter${iter}*)
		echo "Decoding in epoch ${iter}: ${nnet_id}"
		
		if [ $wer_decode == "yes" ]; then	    
		    echo "Post-Adaptation Decoding: WER -- iter${iter}"
		    out_wer=$outdir/wer_decode/$iter/$d/$n
		    mkdir -p $out_wer
		    ( 
			local/ncca/decode_ncca.sh --nj $decode_nj \
						  --cmd "$decode_cmd" \
						  --acwt 0.2 \
						  --srcdir $outdir \
						  --nnet $outdir/nnets_decode/$iter/final.nnet \
						  $gmmdir/graph $data_fbank_allnoise/$d/$n $out_wer || exit 1	    
		    ) & sleep 2
		fi
		
		if [ $frm_decode == "yes" ]; then	    
		    echo "Post-Adaptation Decoding: FRM -- iter${iter}"
		    out_frm=$outdir/frm_decode/$iter/$d/$n
		    mkdir -p $out_frm
		    (		
			local/ncca/compute_frm_acc.sh --mlp $outdir/nnets_decode/$iter/final.nnet \
						      --cmd "$decode_cmd" \
						      $data_fbank_allnoise/$d/$n ${gmmdir}_ali/data-allnoise/$d/$n $outdir $out_frm || exit 1					
		    ) & sleep 2
		fi

		
		if [ $fn_decode == "yes" ]; then
		    if [ $objective_function == 'aann-cm' -o $objective_function == 'aann-ncca' ];then
			echo "Post-Adaptation Decoding: FN -- iter${iter}"
			out_fn=$outdir/fn_decode/$iter/$d/$n
			mkdir -p $out_fn
			(
			    local/aann/compute_fn_loss_aann-ncca.sh --cmd "$decode_cmd" \
								    --mlp $outdir/nnets_decode/$iter/final.nnet.1.aann.ft \
								    --minibatch_size ${minibatch_size} \
								    --aann_mlp $outdir/aann/aann.nnet \
								    --objective-function $objective_function \
								    ${ref_cm:+ --ref_cm ${ref_cm}} \
								    $data_fbank_allnoise/$d/$n $outdir $out_fn || exit 1		       
			) & sleep 2
		    fi
		fi

		
		if [ $mse_decode == "yes" ]; then  # bn are produced here as well	    
		    echo "Post-Adaptation Decoding: MSE -- iter${iter}"
		    out_mse=$outdir/mse_decode/$iter/$d/$n
		    mkdir -p $out_mse
		    (
			local/ncca/make_bn_feats_ncca.sh --nj ${feats_nj} \
							 --cmd "$train_cmd" \
							 --remove-last-components $last_components \
							 --dnnnet $outdir/nnets_decode/$iter/final.nnet \
							 $out_mse/bn ${data_fbank_allnoise}/$d/$n $outdir $out_mse/bn/log $out_mse/bn/data || exit 1
			local/aann/compute_mse_aann.sh --mlp $outdir/nnets_decode/$iter/final.nnet.1.aann.ft \
						       --cmd "$decode_cmd" \
						       --aann_mlp $outdir/aann/aann.nnet \
						       $data_fbank_allnoise/$d/$n $outdir $out_mse || exit 1			
		    ) & sleep 2
		fi
		
	    done
	done
    done
fi
outdir5=$outdir


echo ============================================================================
echo "Adaptation Finished successfully on $noise" 
echo ============================================================================

exit 0

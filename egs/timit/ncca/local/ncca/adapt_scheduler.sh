#!/bin/bash

# This script does cm-based dnn adaptation based on one coactivation matrix (one layer for one data)
# this script does everything but postAdapting decoding 
# 0) preadapt DNN preparation -- Split DNN into two parts final.nnet.1 final.nnet.2 and feature transform
# 1) extract bn feats: clean and adapted data
# 2) compute fn loss: clean and adapted data
# 3) dnn decode: clean and adapted data (paralell): (WER, FRM_ACC)
# 4) adaptation 
# 5) Preparation : DNN dir (Post Adaptation) & FN decode & dnn decode
# 6) Post Adaptation: FN decode 
# 7) Post Adaptation: DNN decode: WER FRM_ACC 


# CONFIG
feats_nj=10
train_nj=30
decode_nj=20
stage=0

# dnn
dnnid="dnn-256"  # mono_dnn-256 as a dnnid (used only for output dir)
first_components=2
mlp= # option
dnnDir= # dnnDir to be copied from 

# data 
data_fbank_clean=data-fbank13-clean
data_fbank_allnoise=data-fbank13-allnoise
noise=babble-10

# adapt
adapt_scp_opt= # options for other training scp
minibatch_size=19306 # num of frames in one update and back propagation
randomizer_size=32768 # num of frames for one epoch training
randomizer_seed=777
lr=0.001
lr_adapt=no
keep_lr_iters=250
max_iters=400
adaptid= # options for parameter analysis (used only for output dir)
frm_acc_decode="yes"
fn_decode="yes"
wer_decode="no"

# post decode
iter_step=20
decode_subsets="dev test"
decode_noises="babble-10 CLEAN"
fn_mb_opt="no"

# monophone model
gmmdir=exp/mono

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# reference coactivation matrix
ref_cm=exp/${dnnid}-bn_cm/${first_components}.mat

# dnn
if [ ! -z $mlp ]; then
    numComponents_dnn=`nnet-info ${mlp} | grep num-components | awk '{print $2}'`
else
    numComponents_dnn=`nnet-info ${dnnDir}/final.nnet | grep num-components | awk '{print $2}'`
fi
    numLayersFreeze=$((${first_components}-3)) 
    last_components=$((${numComponents_dnn}-${first_components}))
    echo "number of Components in DNN: $numComponents_dnn"
    echo "number of first components: $first_components"
    echo "number of last components: $last_components"

# output dir 
expid=exp/${dnnid}_${noise}${adaptid}/${first_components}
echo "OUTPUT DIR: "${expid}
[ ! -d ${expid} ] && mkdir -p ${expid}



###### STEPS #####
echo ============================================================================
echo "              Preparation: DNN(Trimmed)                                   "
echo ============================================================================
outdir=${expid}/preAdapt-dnn # the only output
if [ $stage -le 0 ]; then

    # copy dnn dir
    [ ! -d $outdir ] && rsync -ar --exclude '*decode*' $dnnDir/ $outdir || exit 1


    # if specify initial nnet other than the defalut final.net
    if [ ! -z $mlp ]; then
	[ -L $outdir/final.nnet ] && unlink $outdir/final.nnet
	[ -f $outdir/final.nnet ] && rm $outdir/final.nnet
	[ -L $outdir/mlp_init ] && unlink $outdir/mlp_init
	
	mlp_name=$(basename $mlp)
	
	ln -s $mlp $outdir/mlp_init
	ln -s $mlp $outdir/final.nnet
    fi
    
    # freeze adapted layers
    if [ $numLayersFreeze -ge 0 ]; then
    	if [ -L $outdir/final.nnet ]; then

	    orig_mlp=$(readlink $outdir/final.nnet)
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

    nnet-copy --remove-last-components=$last_components $outdir/final.nnet $outdir/final.nnet.1 >$outdir/log/trim_dnn.log 2>&1 || exit 1
    nnet-copy --remove-first-components=$first_components $outdir/final.nnet $outdir/final.nnet.2 >>$outdir/log/trim_dnn.log 2>&1|| exit 1
    
fi
outdir0=$outdir



echo ============================================================================
echo "       Pre-Adaptation: Extract BN features for clean and adapted data     "
echo ============================================================================
outdir=${expid}/preAdapt-dnn_bn # the only output
if [ $stage -le 1 ]; then
    if [ ${fn_decode} == "yes" ];then
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
fi
outdir1=$outdir


echo ============================================================================
echo "      Pre-Adaptation: Compute fn loss for clean and adapted data          "
echo ============================================================================
outdir=${expid}/preAdapt-dnn_bn_fn # the only output
if [ $stage -le 2 ]; then
    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do
	    mkdir -p $outdir/$d/$n
	    compute-fn-loss --global=true ${ref_cm} scp:${outdir1}/$d/$n/feats.scp ark,t:$outdir/$d/$n/fn-loss || exit 1
	    # mb level fn loss
	    local/ncca/compute_fn_loss.sh \
		--cmd "$decode_cmd" \
		--mlp $outdir0/final.nnet.1 \
		${data_fbank_allnoise}/$d/$n ${ref_cm} $outdir0 $outdir/$d/$n/mb${minibatch_size} || exit 1
	done
    done
fi
outdir2=$outdir


echo ============================================================================
echo "     Pre-Adaptation: Decoding on all data set (CLEAN and Allnoise)        "
echo ============================================================================
outdir=${expid}/preAdapt-dnn # the only output
if [ $stage -le 3 ]; then

    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do
	    (
    		    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    		 			 $gmmdir/graph $data_fbank_allnoise/$d/$n $outdir/decode_${d}_${n} || exit 1;
		    local/ncca/compute_frm_acc.sh --mlp $outdir/final.nnet --cmd "$decode_cmd" $data_fbank_allnoise/$d/$n ${gmmdir}_ali/data-allnoise/$d/$n $outdir $outdir/frm_decode/$d/$n || exit 1
    	    ) & sleep 10	    
	done
    done    
fi
outdir3=$outdir


echo ============================================================================
echo "                 Adaptation based on $noise                               "
echo ============================================================================
outdir=${expid}/postAdapt-dnn-RAW # the only output
if [ $stage -le 4 ]; then

    # # learning rate modification
    if [ "${lr_adapt}" == "yes" ]; then
	if [ "${fn_mb_opt}" == "yes" ]; then
 	    err=$(grep AvgLoss $outdir2/test/$noise/mb${minibatch_size}/compute_fn_loss.log | awk '{print $4}') 	    
	else
	    err=$(awk '{print $3}' $outdir2/test/$noise/fn-loss) 
	fi
	lr=$(awk "BEGIN{print($lr/$err)}")
	echo "Learning rate is adjusted by eer ($eer) --> $lr, [minibatch-level: ${fn_mb_opt}]"
    fi

    
    feature_transform=$outdir0/final.feature_transform
    mlp_init=$outdir0/final.nnet.1

    # train options passed as options for schedule
    train_tool_opts="--minibatch-size=${minibatch_size} --randomizer-seed=${randomizer_seed} --randomizer-size=${randomizer_size}"
    train_options="--keep-lr-iters ${keep_lr_iters}"
    train_options="${train_options} --max-iters $max_iters"
    
    (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
    $cuda_cmd $outdir/log/train_ncca.log \
    	      local/ncca/train_ncca.sh \
     	      --mlp-init $mlp_init \
	      --train-opts "$train_options" \
	      --learn-rate $lr \
    	      --copy-feats "false" --skip-cuda-check "true" \
    	      --feature-transform "$feature_transform" \
    	      --train-tool "nnet-train-frmshuff-ncca --objective-function=ncca --ncca-ref-mat=${ref_cm}" \
	      --train-tool-opts "${train_tool_opts}" \
	      ${adapt_scp_opt:+ --adapt_scp_opt "$adapt_scp_opt"} \
    	      ${data_fbank_allnoise}/test/${noise} ${data_fbank_allnoise}/dev/${noise} $outdir || exit 1
fi
outdir4=$outdir


echo ============================================================================
echo "   Preparation : DNN dir (Post Adaptation)                                "
echo ============================================================================
outdir=${expid}/postAdapt-dnn # the only output
if [ $stage -le 5 ]; then
    # copy dnn dir
    [ ! -d $outdir ] && rsync -ar --exclude '*decode*' $outdir0/ $outdir
    [ -L $outdir/final.nnet ] && unlink $outdir/final.nnet
    rm -r $outdir/nnet
    cp -r $outdir4/nnet $outdir/nnet
fi
outdir5=$outdir



echo ============================================================================
echo "       Post-Adaptation: Extract BN and Compute FN-loss                    "
echo ============================================================================
outdir=${expid}/postAdapt-dnn/fn_decode # the only output
if [ $stage -le 6 ]; then
    if [ ${fn_decode} == "yes" ];then

	# initial results (copy from preAdapt)
	for d in ${decode_subsets}; do
	    for n in ${decode_noises}; do
		if [ ! -d $outdir/000/$d/$n ]; then
		    mkdir -p $outdir/000/$d/$n
		    cp -r $outdir2/$d/$n/* $outdir/000/$d/$n
		fi
	    done
	done
				
	
	# iterate over selected epochs
	for iter in `seq -f"%03g" 1 ${iter_step} ${keep_lr_iters}` ;do
	    
	    # create fn decode dir
	    [ ! -d ${outdir}/$iter ] && mkdir -p ${outdir}/$iter
	    
	    # create the adapted dnn
	    nnet_id=$(basename ${outdir5}/nnet/*iter${iter}*)
	    echo "FN decoding: ${nnet_id}"
	    nnet-concat ${outdir5}/nnet/${nnet_id} $outdir0/final.nnet.2 $outdir/${iter}/final.nnet
	    ln -s ${outdir5}/nnet/${nnet_id} $outdir/${iter}/final.nnet.1
	    for d in ${decode_subsets}; do
		for n in ${decode_noises}; do
		    out_fn=$outdir/${iter}/$d/$n
		    (			
			# extract bn 
			local/ncca/make_bn_feats_ncca.sh --nj ${feats_nj} \
							 --cmd "$train_cmd" \
							 --remove-last-components $last_components \
							 --dnnnet $outdir/${iter}/final.nnet \
							 $out_fn/bn ${data_fbank_allnoise}/$d/$n $outdir5 $out_fn/bn/log $out_fn/bn/data || exit 1
			# compute fn loss
			compute-fn-loss --global=true ${ref_cm} scp:${out_fn}/bn/feats.scp ark,t:$out_fn/fn-loss || exit 1
		     	# mb level fn loss
			local/ncca/compute_fn_loss.sh \
			    --cmd "$decode_cmd" \
			    --mlp ${outdir5}/nnet/${nnet_id} \
			    ${data_fbank_allnoise}/$d/$n ${ref_cm} $outdir5 $out_fn/mb${minibatch_size}	 || exit 1
		    ) & sleep 2
		done
	    done
	done
    fi
fi
outdir6=$outdir



echo ============================================================================
echo "       Post-Adaptation: DNN deocde (WER, FRM_ACC)                                       "
echo ============================================================================
outdir=${expid}/postAdapt-dnn/dnn_decode # the only output
if [ $stage -le 7 ]; then

    # initial results (copy from preAdapt)
    for d in ${decode_subsets}; do
	for n in ${decode_noises}; do
	    if [ ! -d $outdir/000/$d/$n ]; then
		mkdir -p $outdir/000/$d/$n
		mkdir -p ${outdir5}/frm_decode/000/$d/$n
		cp -r $outdir3/decode_${d}_${n}/* $outdir/000/$d/$n
		cp -r $outdir3/frm_decode/$d/$n/* ${outdir5}/frm_decode/000/$d/$n
	    fi
	done
    done
	
    # iterate over selected epochs
    for iter in `seq -f"%03g" 1 ${iter_step} ${keep_lr_iters}` ;do
	
	# create dnn decode dir 
	[ ! -d ${outdir}/$iter ] && mkdir -p ${outdir}/$iter
	
	# create the adapted dnn
	nnet_id=$(basename ${outdir5}/nnet/*iter${iter}*)
	echo "dnn decoding: ${nnet_id}"
	nnet-concat ${outdir5}/nnet/${nnet_id} $outdir0/final.nnet.2 $outdir/${iter}/final.nnet
	
	# decode in parallel
	for d in ${decode_subsets}; do
	    for n in ${decode_noises}; do
		(
		    if [ ${wer_decode} == "yes" ];then
    			local/ncca/decode_ncca.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
		    				  --srcdir ${outdir5} \
		    				  --nnet $outdir/${iter}/final.nnet \
    		    				  $gmmdir/graph $data_fbank_allnoise/$d/$n $outdir/${iter}/$d/$n || exit 1; 
		    fi
		    if [ ${frm_acc_decode} == "yes" ];then
			local/ncca/compute_frm_acc.sh --mlp $outdir/${iter}/final.nnet --cmd "$decode_cmd" $data_fbank_allnoise/$d/$n ${gmmdir}_ali/data-allnoise/$d/$n ${outdir5} ${outdir5}/frm_decode/${iter}/$d/$n || exit 1;
		    fi
    		) & sleep 10	    
	    done
	done
    done
fi
outdir7=$outdir


echo ============================================================================
echo "Adaptation Finished successfully on $noise" 
echo ============================================================================

exit 0

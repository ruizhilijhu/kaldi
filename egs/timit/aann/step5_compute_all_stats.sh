#!/bin/bash

# option: speficy nnet --> final.nnet final.nnet.1 final.nnet.1.aann.ft and aann.nnet 
# ENT: final.nnet or final.nnet.1
# MM: last layer or adapted layer
# MD:
# bn: extract bottle neck
#" frm wer aannfn mm mse ent fn md "


# CONFIG
decode_nj=20
# in and out files and dirs
dnnid=mono_dnn-256
expid=exp/mono_dnn-256_babble-10_no_rdn_aann-3hl-512hu
first_components=2
outDir= # option, $expid/$first_components/postAdapt-dnn by default
gmmdir=exp/mono
ref_cm= # option 
data_fbank_allnoise=data-fbank13-allnoise
# decode 
iter_step=5
keep_lr_iters=10
minibatch_size=19306 # for fn computation
decode_subsets="test" # "dev test"
decode_noises="babble-10 CLEAN" # "babble-10 CLEAN"
decode_epochs= # option, 0 ${iter_step} ${keep_lr_iters} by default
# decode options
mlp_opt="dnn1" # dnn1|dnn|dnn1_aannft ; only for the following stats
statsList="frm mse fn mm"   #"frm wer aannfn mm mse ent fn"

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# decode
[ -z $decode_epochs ] && decode_epochs=`seq -f"%03g" 0 ${iter_step} ${keep_lr_iters}`
echo $decode_epochs
# output dir
expDir=$expid/$first_components/postAdapt-dnn
[ -z $outDir ] && outDir=$expid/$first_components/postAdapt-dnn
echo "OUTPUT DIR: "${outDir}
[ ! -d ${outDir} ] && mkdir -p ${outDir}
# reference matrix
if [ -z $ref_cm ]; then
    if [ $mlp_opt == "dnn" ]; then
	ref_cm=exp/${dnnid}-bn_cm/17.mat
    elif [ $mlp_opt == "dnn1" ]; then
	ref_cm=exp/${dnnid}-bn_cm/${first_components}.mat
    fi
fi			 
# check nnets_decode iter000
if [ ! -d ${expDir}/nnets_decode/000 ]; then
    mkdir -p ${expDir}/nnets_decode/000
    for f in final.nnet.1.aann.ft final.nnet.1 final.nnet; do
	ln -s $PWD/${expid}/preAdapt-dnn/${f} $expDir/nnets_decode/000/${f} || exit 1
    done
fi
###### STEPS #####
# iterate over selected epochs

for iter in ${decode_epochs} ;do

    # dnns
    nnetDir=$expDir/nnets_decode/$iter
    dnn=$nnetDir/final.nnet
    dnn1=$nnetDir/final.nnet.1
    dnn1_aannft=$nnetDir/final.nnet.1.aann.ft
    aann=$expDir/aann/aann.nnet
    # mlp pick
    if [ $mlp_opt == "dnn" ];then
	mlp=$dnn
    elif [ $mlp_opt == "dnn1" ]; then
	mlp=$dnn1
    fi

    echo $mlp
    
    for f in $dnn $dnn1 $dnn1_aannft; do
	[ ! -f $f -a ! -L $f ] && echo "Missing $f!" 
    done

    # compute
    for s in ${statsList}; do
	for d in ${decode_subsets}; do
	    for n in ${decode_noises}; do
	     
		out=$outDir/${s}_decode/$iter/$d/$n
		echo $out
		
		#[ -d $out ] && continue

		case $s in
		    frm)
			echo "COMPUTING: FRAME ACCURACY -- ITER${iter}"
			(
			    local/ncca/compute_frm_acc.sh --mlp $dnn \
							  --cmd "$decode_cmd" \
							  $data_fbank_allnoise/$d/$n \
							  ${gmmdir}_ali/data-allnoise/$d/$n $expDir $out || exit 1
			) & sleep 5
		    	;;
		    aannfn)
			echo "COMPUTING: AANN FROBENIUS NORM -- ITER${iter}"
			(
			    local/aann/compute_fn_loss_aann-ncca.sh --cmd "$decode_cmd" \
								    --mlp $dnn1_aannft \
								    --minibatch_size ${minibatch_size} \
								    --aann_mlp $aann \
								    --objective-function 'aann-ncca' \
								    $data_fbank_allnoise/$d/$n $expDir $out || exit 1		       			) & sleep 5
			;;
		    mse)
			echo "COMPUTING: MSE -- ITER${iter}"
			(
			    local/aann/compute_mse_aann.sh --mlp $dnn1_aannft \
							   --cmd "$decode_cmd" \
							   --aann_mlp $aann \
							   $data_fbank_allnoise/$d/$n $expDir $out || exit 1			
			) & sleep 5
			;;		    
		    fn)
			echo "COMPUTING: FROBENIUS NORM -- ITER${iter}"
			# TODO: BN global
			# TODO: compute-fn-loss --global=true ${ref_cm} scp:${out_fn}/bn/feats.scp ark,t:- | awk '{print $3}' > $out_fn/fn-loss || exit 1
			if [ $mlp_opt == "dnn" ];then
			    out=$outDir/${s}_sm_decode/$iter/$d/$n
			elif [ $mlp_opt == "dnn1" ]; then
			    out=$outDir/${s}_decode/$iter/$d/$n
			fi
			echo $out
			[ -z $ref_cm -o ! -f $ref_cm ] && echo "Missing reference coactivation matrix!" && exit 1			
			(
			    local/ncca/compute_fn_loss.sh \
				--minibatch_size $minibatch_size \
 				--cmd "$decode_cmd" \
				--mlp $mlp \
				${data_fbank_allnoise}/$d/$n ${ref_cm} $expDir $out	 || exit 1
			) & sleep 5		       
			;;
		    wer)
			echo "COMPUTING: WER -- ITER${iter}"
			( 			
			    local/ncca/decode_ncca.sh --nj $decode_nj \
						      --cmd "$decode_cmd" \
						      --acwt 0.2 \
						      --srcdir $expDir \
						      --nnet $dnn \
						      $gmmdir/graph $data_fbank_allnoise/$d/$n $out || exit 1	    
			) & sleep 5
			;;
		    mm)
			echo "COMPUTING: M Measure -- ITER${iter}"
			if [ $mlp_opt == "dnn" ];then
			    out=$outDir/${s}_sm_decode/$iter/$d/$n
			elif [ $mlp_opt == "dnn1" ]; then
			    out=$outDir/${s}_decode/$iter/$d/$n
			fi
			(
			    local/aann/compute_m_measure.sh --nj $decode_nj \
							    --cmd "$decode_cmd" \
							    --remove_last_components 0 \
							    --mlp $mlp \
							    $out $data_fbank_allnoise/$d/$n $expDir $out/log $out/data 'haha'
			) & sleep 5
 			;;
		    ent)
			echo "COMPUTING: AANN FROBENIUS NORM -- ITER${iter}"
			;;
		    md)
			echo "COMPUTING: AANN FROBENIUS NORM -- ITER${iter}"
			;;
		    *)
			echo "Not a valid option $s! " && exit 1
			;;
		esac
	    done
	done
    done
done
exit 0

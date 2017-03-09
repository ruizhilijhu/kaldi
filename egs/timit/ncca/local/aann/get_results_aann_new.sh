#!/bin/bash

# This script get results and stats (fn(aaann_ncca) frm mse)
# note: fn and fn come together as fn_decode option
# assumption: every layer should have same iters for each results


# in 
expDir=exp/mono_dnn-256_babble-10_mb_lr_adapt/2
# out
outdir=RESULTS/mono_dnn-256_babble-10_mb_lr_adapt/2

# options
frmDecode="yes"
fnDecode="yes"
mseDecode="yes"
fnSmDecode="yes"
mmDecode="yes"
mmSmDecode="yes"

noise="babble-10 CLEAN"
subset="dev test"

# minibatch size (for fn computation)
minibatch_size=19306

[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1


set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# in 
fnDecodeDir=${expDir}/postAdapt-dnn/fn_decode
fnSmDecodeDir=${expDir}/postAdapt-dnn/fn_sm_decode
frmDecodeDir=${expDir}/postAdapt-dnn/frm_decode
mseDecodeDir=${expDir}/postAdapt-dnn/mse_decode
mmSmDecodeDir=${expDir}/postAdapt-dnn/mm_sm_decode
mmDecodeDir=${expDir}/postAdapt-dnn/mm_decode

# out
fnDecodeOutDir=$outdir/fn_decode
fnSmDecodeOutDir=$outdir/fn_sm_decode
frmDecodeOutDir=$outdir/frm_decode
mseDecodeOutDir=$outdir/mse_decode
mmSmDecodeOutDir=$outdir/mm_sm_decode
mmDecodeOutDir=$outdir/mm_decode



if [ $frmDecode == "yes" ]; then

    # output dir
    mkdir -p $frmDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $frmDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $frmDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $frmDecodeDir/000/$d/$n ]; then
		cat $frmDecodeDir/*/$d/$n/frm_acc > $frmDecodeOutDir/frm_${d}_${n} || exit 1
		numLines=$(wc $frmDecodeOutDir/frm_${d}_${n} | awk '{print $1}')
		numIters=$(wc $frmDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "FRM: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi



if [ $fnDecode == "yes" ]; then

    # output dir
    mkdir -p $fnDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $fnDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $fnDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $fnDecodeDir/000/$d/$n ]; then
	        echo "$fnDecodeDir/*/$d/$n/fn"
		cat $fnDecodeDir/*/$d/$n/fn > $fnDecodeOutDir/fn_${d}_${n} || exit 1
		numLines=$(wc $fnDecodeOutDir/fn_${d}_${n} | awk '{print $1}')
		numIters=$(wc $fnDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "FN: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi


if [ $fnSmDecode == "yes" ]; then

    # output dir
    mkdir -p $fnSmDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $fnSmDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $fnSmDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $fnSmDecodeDir/000/$d/$n ]; then
		cat $fnSmDecodeDir/*/$d/$n/fn > $fnSmDecodeOutDir/fn_${d}_${n} || exit 1
		numLines=$(wc $fnSmDecodeOutDir/fn_${d}_${n} | awk '{print $1}')
		numIters=$(wc $fnSmDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "FN-SM: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi


if [ $mseDecode == "yes" ]; then

    # output dir
    mkdir -p $mseDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $mseDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $mseDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $mseDecodeDir/000/$d/$n ]; then
		cat $mseDecodeDir/*/$d/$n/mse_frm > $mseDecodeOutDir/mse_${d}_${n} || exit 1
		numLines=$(wc $mseDecodeOutDir/mse_${d}_${n} | awk '{print $1}')
		numIters=$(wc $mseDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "MSE: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi









if [ $mmDecode == "yes" ]; then

    # output dir
    mkdir -p $mmDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $mmDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $mmDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $mmDecodeDir/000/$d/$n ]; then
		cat $mmDecodeDir/*/$d/$n/data/mm_score > $mmDecodeOutDir/mm_${d}_${n} || exit 1
		numLines=$(wc $mmDecodeOutDir/mm_${d}_${n} | awk '{print $1}')
		numIters=$(wc $mmDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "MM: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi


if [ $mmSmDecode == "yes" ]; then

    # output dir
    mkdir -p $mmSmDecodeOutDir || exit 1

    # iter_list
    iterList=$(ls --color=never $mmSmDecodeDir)
    echo $iterList | sed "s/\ /\\n/g" > $mmSmDecodeOutDir/iter_list || exit 1
    
    # get results
    for d in $subset; do
	for n in $noise; do
	    if [ -d $mmSmDecodeDir/000/$d/$n ]; then
		cat $mmSmDecodeDir/*/$d/$n/data/mm_score > $mmSmDecodeOutDir/mm_${d}_${n} || exit 1
		numLines=$(wc $mmSmDecodeOutDir/mm_${d}_${n} | awk '{print $1}')
		numIters=$(wc $mmSmDecodeOutDir/iter_list | awk '{print $1}')
		[ $numLines != $numIters ] && echo "MM-SM: Iterations mismatch $d $n!" && exit 1
	    fi
	done
    done    
fi

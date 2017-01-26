#!/bin/bash

# This script get results and stats (fn fn_mb frm)
# note: fn and fn_mb come together as fn_decode option
# assumption: every layer should have same iters for each results

# output results options
frm_acc_decode="yes"
fn_decode="yes"

adaptid= #"_minibatch19306"
dnnid=  # "dnn-512"
outdir=
subset=
noise=
layer_list=  # self-define or "ls" the exp_dir

. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# experiment directory
exp_dir=exp/${dnnid}_${noise}${adaptid}
echo "experiment dir: $exp_dir"
[ -z ${layer_list} ] && layer_list=$(ls --color=never $exp_dir) 

if [ $fn_decode == "yes" ]; then

    # output dir
    mkdir -p $outdir/fn_decode    
    # initialize
    fn_results=$outdir/fn_decode/fn_layer_$(echo ${layer_list} | tr ' ' '-')
    fn_mb_results=$outdir/fn_decode/fn-mb_layer_$(echo ${layer_list} | tr ' ' '-')
    # iterate over layer list
    for layer in ${layer_list[@]}; do
       
	fn_dir=${exp_dir}/${layer}/postAdapt-dnn/fn_decode
	# make iter list (vertical)
	if [ ! -f $outdir/fn_decode/iter_list ]; then
            iter_list=$(ls --color=never ${fn_dir})
	    echo $iter_list | sed "s/\ /\\n/g" > $outdir/fn_decode/iter_list
	fi
	# make layer list (horizontal)
	[ ! -f $outdir/fn_decode/layer_list ] &&  echo $layer_list | sed "s/\ /\\n/g" > $outdir/fn_decode/layer_list

	# fn
	cat $fn_dir/*/${subset}/${noise}/fn-loss | awk '{print $3}' > fn.tmp
	if [ -e $fn_results ]; then
	    paste -d',' $fn_results fn.tmp > fn.results.tmp
	else
	    mv fn.tmp fn.results.tmp
	fi
	mv fn.results.tmp $fn_results

	# fn mb19306
	cat $fn_dir/*/${subset}/${noise}/mb19306/fn-loss > fn_mb.tmp
	if [ -e $fn_mb_results ]; then
	    paste -d',' $fn_mb_results fn_mb.tmp > fn_mb.results.tmp
	else
	    mv fn_mb.tmp fn_mb.results.tmp
	fi
	mv fn_mb.results.tmp $fn_mb_results
    done
    rm fn*.tmp
fi


if [ $frm_acc_decode == "yes" ]; then

    # output dir
    mkdir -p $outdir/frm_decode    
    # initialize
    frm_results=$outdir/frm_decode/frm_layer_$(echo ${layer_list} | tr ' ' '-')
    # iterate over layer list
    for layer in ${layer_list[@]}; do
       
	frm_dir=${exp_dir}/${layer}/postAdapt-dnn/frm_decode
	# make iter list (vertical)
	if [ ! -f $outdir/frm_decode/iter_list ]; then
            iter_list=$(ls --color=never ${frm_dir})
	    echo $iter_list | sed "s/\ /\\n/g" > $outdir/frm_decode/iter_list
	fi
	# make layer list (horizontal)
	[ ! -f $outdir/frm_decode/layer_list ] &&  echo $layer_list | sed "s/\ /\\n/g" > $outdir/frm_decode/layer_list
	# frm-acc
	cat $frm_dir/*/${subset}/${noise}/frm_acc  > frm.tmp
	if [ -e $frm_results ]; then
	    paste -d',' $frm_results frm.tmp > frm.results.tmp
	else
	    mv frm.tmp frm.results.tmp
	fi
	mv frm.results.tmp $frm_results
    done
    rm frm*.tmp
fi

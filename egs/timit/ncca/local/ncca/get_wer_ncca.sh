#!/bin/bash

# run decoding on data for specific iteration and specific layer adaptation 

# CONFIG
decode_nj=20
data_fbank_allnoise=data-fbank13-allnoise
gmmdir=exp/mono
adaptid=  #"_minibatch19306"
dnnid=  #"dnn-512"
subset=  
noise= 
first_components=
iter= ## 003 format 

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

exp_dir=exp/${dnnid}_${noise}${adaptid}/${first_components}/postAdapt-dnn
echo "experiment dir (DNN Dir): $exp_dir"

outdir=$exp_dir/nnets_decode/${iter}
[ ! -d $outdir ] && echo "Missing decoding mlp dir! $outdir" && exit 1

mlp=$outdir/final.nnet

local/ncca/decode_ncca.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
		    	  --srcdir ${exp_dir} \
		    	  --nnet $mlp \
    		    	  $gmmdir/graph $data_fbank_allnoise/$subset/$noise $outdir/$subset/$noise || exit 1; 

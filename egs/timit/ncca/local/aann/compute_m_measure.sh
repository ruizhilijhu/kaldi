#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
remove_last_components=4 # remove N last components from the nnet
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),
mlp=

#trnasform nnet out opts
transf_nnet_out_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <tgt-data-dir> <src-data-dir> <nnet-dir> <log-dir> <abs-path-to-bn-feat-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
nndir=$3
logdir=$4
bnfeadir=$5
aligndir=$6

#bnfeadir=`perl -e '($dir,$ped)= @ARGV; if($dir!~m:^/:){ $dir = "$pwd/$dir"; } print $dir; ' $bnfeadir ${PWD}`
######## CONFIGURATION 

# copy the dataset metadata from srcdata.
mkdir -p $data $logdir $bnfeadir || exit 1;
utils/copy_data_dir.sh $srcdata $data; rm $data/feats.scp 2>/dev/null

# make $bnfeadir an absolute pathname.
[ '/' != ${bnfeadir:0:1} ] && bnfeadir=$PWD/$bnfeadir

[ -z $mlp ] && mlp=$nndir/final.nnet

required="$srcdata/feats.scp $mlp $nndir/final.feature_transform"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Concat feature transform with trimmed MLP:
nnet=$bnfeadir/feature_extractor.nnet
nnet-concat $nndir/final.feature_transform "nnet-copy --remove-last-components=$remove_last_components $mlp - |" $nnet 2>$logdir/feature_extractor.log || exit 1
nnet-info $nnet >$data/feature_extractor.nnet-info

echo "Creating bn-feats into $data"

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nndir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  [ -z $ivector ] && echo "Missing --ivector, they were used in training!" && exit 1
  # Get the tool, 
  ivector_append_tool=append-vector-to-feats # default,
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  # Check dims,
  feats_job_1=$(sed 's:JOB:1:g' <(echo $feats))
  dim_raw=$(feat-to-dim "$feats_job_1" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats_job_1 $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  [ $dim_ivec != "$(cat $D/ivector_dim)" ] && \
    echo "Error, i-vector dim. mismatch (expected $(cat $D/ivector_dim), got $dim_ivec in '$ivector')" && \
    exit 1
  # Append to feats,
  feats="$feats $ivector_append_tool ark:- '$ivector' ark:- |"
fi
# From here, it is for M-measure/M-delta calculation
# pri file
mdelta_stats_dir=$bnfeadir/mdelta_stats_dir
if [ $htk_save == false ]; then

# transf_nnet_out_opts="--pdf-to-pseudo-phone=$mdelta_stats_dir/pdf_to_pseudo_phone.txt"

  # generate pdf-to-phone-map
#  utils/phone_post/create_pdf_to_phone_map.sh data/lang $nndir/final.mdl $mdelta_stats_dir

  # # generate pri file
  # ali-to-phones --per-frame \
  #   $aligndir/final.mdl \
  #   "ark:gunzip -c $aligndir/ali.*.gz |" \
  #   ark,t:$mdelta_stats_dir/phones.tra
    
  #   echo "1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80" >$mdelta_stats_dir/dt.list
      
  # perl utils/multi-stream/pm_utils/mtd/calc_pri_dt.pl \
  #   $mdelta_stats_dir/dt.list \
  #   $mdelta_stats_dir/phones.tra >$mdelta_stats_dir/pri



  # Run the forward pass,
#  $cmd JOB=1:$nj $logdir/make_bnfeats.JOB.log \
#    nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
#    transform-nnet-posteriors $transf_nnet_out_opts ark:- ark:- \| \
#    copy-feats ark:- ark,scp:$bnfeadir/raw_bnfea_$name.JOB.ark,$bnfeadir/raw_bnfea_$name.JOB.scp \
#    || exit 1;
  # concatenate the .scp files
#  for ((n=1; n<=nj; n++)); do
#    cat $bnfeadir/raw_bnfea_$name.$n.scp >> $data/feats.scp
#  done

#  $cmd JOB=1:$nj $logdir/compute_mdelta_scores.JOB.log \
#    tmp_dir=\$\(mktemp -d\) '&&' mkdir -p \$tmp_dir '&&' \
#    nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
#    transform-nnet-posteriors $transf_nnet_out_opts ark:- ark,scp:\$tmp_dir/post.JOB.ark,\$tmp_dir/post.JOB.scp '&&' \
#    python utils/multi-stream/pm_utils/compute_mdelta_post.py \$tmp_dir/post.JOB.scp $bnfeadir/mdelta_stats_dir/pri $bnfeadir/mdelta_scores.JOB.pklz '&&' \
#   rm -rf \$tmp_dir/ || exit 1; 

  $cmd JOB=1:$nj $logdir/compute_m-measure_scores.JOB.log \
    tmp_dir=\$\(mktemp -d\) '&&' mkdir -p \$tmp_dir '&&' \
    nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet "$feats" ark,scp:\$tmp_dir/post.JOB.ark,\$tmp_dir/post.JOB.scp '&&' \
    python local/aann/compute_m-measure.py \$tmp_dir/post.JOB.scp $bnfeadir/mmeasure_scores.JOB.pklz '&&' \
   rm -rf \$tmp_dir/ || exit 1;  

 # combine the splits
#  comb_str=""
#  for ((n=1; n<=nj; n++)); do
#    comb_str=$comb_str" "$bnfeadir/mdelta_scores.${n}.pklz
#  done

#  python utils/multi-stream/pm_utils/merge_dicts.py $comb_str $bnfeadir/mdelta_scores.$name.pklz 2>$logdir/merge_dicts.log || exit 1;
#  python utils/multi-stream/pm_utils/dicts2txt.py $bnfeadir/mdelta_scores.$name.pklz $bnfeadir/mdelta_scores.$name.txt 2>$logdir/dicts2txt.log || exit 1;

  comb_str=""
  for ((n=1; n<=nj; n++)); do
    comb_str=$comb_str" "$bnfeadir/mmeasure_scores.${n}.pklz
  done

  python local/aann/merge_dicts.py $comb_str $bnfeadir/mmeasure_scores.$name.pklz 2>$logdir/merge_dicts.log || exit 1;
  python local/aann/dicts2txt.py $bnfeadir/mmeasure_scores.$name.pklz $bnfeadir/mmeasure_scores.$name.txt 2>$logdir/dicts2txt.log || exit 1;


  cat $bnfeadir/mmeasure_scores.$name.txt | tr '[||]' ' ' | awk '{T=0;for(i=7;i<21;i++) T+=$i; print T/15}' > $bnfeadir/mm_utt_score
  awk 'BEGIN{T=0}{T+=$1}END{print T/NR}' $bnfeadir/mm_utt_score > $bnfeadir/mm_score
  # check sentence counts,
#  N0=$(cat $srcdata/feats.scp | wc -l) 
#  N1=$(cat $data/feats.scp | wc -l)
#  [[ "$N0" != "$N1" ]] && echo "$0: sentence-count mismatch, $srcdata $N0, $data $N1" && exit 1
  echo "Succeeded creating MLP-BN features '$data'"

fi

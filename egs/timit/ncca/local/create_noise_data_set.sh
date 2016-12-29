#!/bin/bash

# This script create noise data set

# CONFIG
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=5
stage=0


exp=data-fbank23-allnoise-sub1 # the only output dir in this stage

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# STEPS
echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================
if [ $stage -le 0 ]; then

  dir=$exp # the only output dir in this stage

  noiseList=`awk -F'_||-' '{print $1}' ${dir}/dev/spk2utt|sort -u`

  for noise_condition in ${noiseList[@]}; do
    if [ noise_condition != ' CLEAN' ]; then
      for d in dev test; do
	noiseDir=$dir/${d}_${noise_condition}
	mkdir -p $noiseDir
	cp $dir/$d/cmvn.scp $noiseDir/cmvn.scp
	cp $dir/$d/stm $noiseDir/stm
	cp $dir/$d/glm $noiseDir/glm
	
	cp $dir/$d/spk2utt $noiseDir/spk2utt.tmp
	cp $dir/$d/text $noiseDir/text
	cp $dir/$d/feats.scp $noiseDir/feats.scp
	cp $dir/$d/wav.scp $noiseDir/wav.scp
	grep $noise_condition $noiseDir/spk2utt.tmp > $noiseDir/spk2utt
	rm $noiseDir/spk2utt.tmp
	spk2utt_to_utt2spk.pl $noiseDir/spk2utt > $noiseDir/utt2spk
	utils/fix_data_dir.sh $noiseDir
      done
    fi
  done
  
  # get list
  
  
    
fi
	
echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0

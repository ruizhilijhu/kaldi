#!/bin/bash

# This script does the following
# 0) Data & Lexicon & Language Preparation for CLEAN (trn dev test)
# 1) Data & Lexicon & Language Preparation for AllNoise (dev test)
# 2) Extract MFCC for CLEAN (trn dev test)
# 3) Extract Fbank for CLEAN (trn dev test)
# 4) Extract Fbank for AllNoise (dev test)

# CONFIG
feats_nj=10
stage=0

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1

set -e # Exit immediately if a command exits with a non-zero status.
echo "$0 $@"  # Print the command line for logging

# STEPS
echo ============================================================================
echo "          Data & Lexicon & Language Preparation for CLEAN (trn dev test)  "
echo ============================================================================
outdir=data
if [ $stage -le 0 ]; then

    timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
    local/timit_data_prep.sh $timit || exit 1

    local/timit_prepare_dict.sh

    # Caution below: we remove optional silence by setting "--sil-prob 0.0",
    # in TIMIT the silence appears also as a word in the dictionary and is scored.
    utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
			  data/local/dict "sil" data/local/lang_tmp data/lang
    
    local/timit_format_data.sh    
fi
outdir0=$outdir



echo ============================================================================
echo "         Data & Lexicon & Language Preparation for AllNoise (dev test)    "
echo ============================================================================
outdir=data-fbank13-allnoise
if [ $stage -le 1 ]; then
    data_allnoise_src=/export/b04/ruizhili/kaldi/egs/timit/data_allnoise_src
    for subdir in data-fbank23-allnoise-sub1 data-fbank23-allnoise-sub2 data-fbank23-allnoise-sub3; do
	noiseList=`awk -F'_||-' '{print $1}' ${data_allnoise_src}/${subdir}/dev/spk2utt|sort -u`
	# echo $noiseList
	for noise_condition in ${noiseList[@]}; do
	    for d in dev test; do
  		noiseDir=$outdir/${d}/${noise_condition}
  		mkdir -p $noiseDir
		for f in stm glm text wav.scp; do
		    cp $data_allnoise_src/$subdir/$d/$f $noiseDir/$f			
		done
		cp $data_allnoise_src/$subdir/$d/spk2utt $noiseDir/spk2utt.tmp
  		grep $noise_condition $noiseDir/spk2utt.tmp > $noiseDir/spk2utt
  		rm $noiseDir/spk2utt.tmp
  		spk2utt_to_utt2spk.pl $noiseDir/spk2utt > $noiseDir/utt2spk
  		utils/fix_data_dir.sh $noiseDir
	    done
	done

	# snr set
	noiseList=`awk -F'_' '{print $1}' ${data_allnoise_src}/${subdir}/dev/spk2utt|sort -u`
	# echo $noiseList
	for noise_condition in ${noiseList[@]}; do
	    for d in dev test; do
  		noiseDir=$outdir/${d}/${noise_condition}
  		mkdir -p $noiseDir
		for f in stm glm text wav.scp; do
		    cp $data_allnoise_src/$subdir/$d/$f $noiseDir/$f			
		done
		cp $data_allnoise_src/$subdir/$d/spk2utt $noiseDir/spk2utt.tmp
  		grep $noise_condition $noiseDir/spk2utt.tmp > $noiseDir/spk2utt
  		rm $noiseDir/spk2utt.tmp
  		spk2utt_to_utt2spk.pl $noiseDir/spk2utt > $noiseDir/utt2spk
  		utils/fix_data_dir.sh $noiseDir
	    done
	done
    done
fi
outdir1=$outdir


echo ============================================================================
echo "              Extract MFCC for CLEAN (trn dev test)                       "
echo ============================================================================
outdir=data 
if [ $stage -le 2 ]; then
    # MFCC
    for x in train dev test; do 
	(
	    steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj $outdir/$x $outdir/$x/log $outdir/$x/data || exit 1
	    steps/compute_cmvn_stats.sh $outdir/$x $outdir/$x/log $outdir/$x/data || exit 1
	) & sleep 10
    done
fi
outdir2=$outdir


echo ============================================================================
echo "              Extract Fbank for CLEAN (trn dev test)                      "
echo ============================================================================
outdir=data-fbank13-clean
if [ $stage -le 3 ]; then
    mkdir $outdir
    # fbank
    for x in train dev test; do 
	(
	    cp -r $outdir2/$x $outdir/$x
	    steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj $outdir/$x $outdir/$x/log $outdir/$x/data || exit 1
	    steps/compute_cmvn_stats.sh $outdir/$x $outdir/$x/log $outdir/$x/data || exit 1
	) & sleep 10
    done
fi
outdir3=$outdir


echo ============================================================================
echo "             Extract Fbank for AllNoise (dev test)                        "
echo ============================================================================
outdir=data-fbank13-allnoise
if [ $stage -le 4 ]; then
    noiseList=`ls $outdir/dev`
    for x in ${noiseList[@]}; do
	for d in dev test; do
	    (
		steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj $outdir/$d/$x $outdir/$d/$x/log $outdir/$d/$x/data || exit 1
		steps/compute_cmvn_stats.sh $outdir/$d/$x $outdir/$d/$x/log $outdir/$d/$x/data || exit 1
	    ) & sleep 10
	done
    done
fi
outdir4=$outdir


echo ============================================================================
echo "Finished successfully data preparation"
echo ============================================================================

exit 0

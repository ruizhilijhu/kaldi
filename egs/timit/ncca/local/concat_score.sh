#!/bin/bash

. path.sh
. cmd.sh

dir=/export/b04/ruizhili/kaldi/egs/timit/dnnAdapt_spkCMVN-dnn/exp/co-Act_1
score=fn_utt.txt.ark

prefix=

for d in ${prefix}data-fbank23-allnoise-sub1 ${prefix}data-fbank23-allnoise-sub2 ${prefix}data-fbank23-allnoise-sub3; do
  for s in dev test; do
    cat $dir/$d/$s/$score >> $dir/utt_${s}.txt.ark.tmp  
  done
done

sort -u $dir/utt_dev.txt.ark.tmp > $dir/utt_dev.txt.ark
sort -u $dir/utt_test.txt.ark.tmp > $dir/utt_test.txt.ark

rm $dir/utt_*.txt.ark.tmp

awk '{print $1}' $dir/utt_dev.txt.ark | sort -u > $dir/utt.dev
awk '{print $1}' $dir/utt_test.txt.ark | sort -u > $dir/utt.test

awk 'FNR==NR{a[$1]=$3;next}{print $1,"[",a[$1],"]"}' $dir/utt_dev.txt.ark $dir/utt.dev > $dir/utt_dev
awk 'FNR==NR{a[$1]=$3;next}{print $1,"[",a[$1],"]"}' $dir/utt_test.txt.ark $dir/utt.test > $dir/utt_test

mv $dir/utt_dev $dir/utt_dev.txt.ark
mv $dir/utt_test $dir/utt_test.txt.ark

awk '{split($1, a, "_"); print $1, a[1]}' $dir/utt_dev.txt.ark > $dir/utt2spk.dev
awk '{split($1, a, "_"); print $1, a[1]}' $dir/utt_test.txt.ark > $dir/utt2spk.test

utt2spk_to_spk2utt.pl $dir/utt2spk.dev > $dir/spk2utt.dev
utt2spk_to_spk2utt.pl $dir/utt2spk.test > $dir/spk2utt.test

ivector-mean ark,t:$dir/spk2utt.dev ark,t:$dir/utt_dev.txt.ark ark,t:$dir/condtion_score.dev
ivector-mean ark,t:$dir/spk2utt.test ark,t:$dir/utt_test.txt.ark ark,t:$dir/condtion_score.test


rm $dir/utt.*

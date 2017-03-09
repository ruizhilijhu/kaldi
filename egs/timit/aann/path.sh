export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH:/export/a07/harish/multistream_clean/kaldi/egs/aurora4/s5_ReLU_BlockDropoutMiddle/../../../src/nnetbin
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONPATH=/export/a07/xwang/kaldi-chime4/egs/ami/s5/utils/multi-stream/pm_utils:$PYTHONPATH

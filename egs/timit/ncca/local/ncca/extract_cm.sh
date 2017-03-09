
data_fbank_allnoise="data-fbank13-allnoise"
outDir=  
last_components=0
expDir='exp/mono_dnn-256_babble-00_no_rdn_aann-3hl-512hu_lr1.0e-06/2/postAdapt-dnn'
iter=250
d=test
n=babble-10
nj=4
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

[ -z $outDir ] && outDir=$expDir/cm_decode/$iter/$d/$n
[ ! -d $outDir ] && mkdir -p $outDir
mlp=$expDir/nnets_decode/$iter/final.nnet
local/ncca/make_bn_feats_ncca.sh \
    --nj $nj \
    --cmd "$train_cmd" \
    --remove_last_components $last_components \
    --dnnnet $mlp \
    $outDir $data_fbank_allnoise/$d/$n $expDir $outDir/log $outDir/data || exit 1

numComponents=$(nnet-info $mlp | grep num-components | awk '{print $2}')
first_components=$((numComponents-last_components))

est-coact-mat --binary=false scp:$outDir/feats.scp - | tail -n +2 | sed s'/]$/\ /g' > $outDir/${first_components}.txt.mat




inDir=exp/mono_dnn-256_babble-10_no_rdn_aann-3hl-512hu_lr1.0e-05/2/postAdapt-dnn/nnets_decode
f=final.nnet
outDir=here
[ ! -d $outDir ] && 

list=$(ls $inDir)
echo $list
for i in $list; do
    [ ! -d $outDir/$i/ ] && mkdir -p $outDir/$i
    echo $i
    nnet-info $inDir/$i/$f > $outDir/$i/$f
    
done

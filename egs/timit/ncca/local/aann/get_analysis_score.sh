
dnnId=mono_dnn-256
expDir=exp/analysis_${dnnId}_aann
outDir=RESULTS/analysis_${dnnId}_aann
aannIds=$(ls $expDir)
for a in $aannIds; do
    expIds=$(ls $expDir/$a/exp)
    for e in $expIds; do
	layerIds=$(ls $expDir/$a/exp/$e)
	for l in $layerIds; do
	    subsetIds=$(ls $expDir/$a/exp/$e/$l)
	    for s in $subsetIds;do
		noisesIds=$(ls $expDir/$a/exp/$e/$l/$s)
		for n in $noisesIds;do
		    in=$expDir/$a/exp/$e/$l/$s/$n
		    out=$outDir/$a/exp/$e/$l/$s/$n
		    [ ! -d $out ]&& mkdir -p $out
		    iterList=$(ls --color=never $in)
		    echo $iterList | sed "s/\ /\\n/g" > $out/iter_list || exit 1
		    cat $in/*/score.text > $out/score || exit 1
		done
	    done
	done
    done
    
done

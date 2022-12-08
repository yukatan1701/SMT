SIZE=$1
PROCS=$2
ARG=1
OUTPUT=lsf_$1_$2_$ARG.lsf
FOUT=lomp_${SIZE}_${PROCS}_$ARG

rm -f $OUTPUT
touch $OUTPUT

echo "source /polusfs/setenv/setup.SMPI" >> $OUTPUT
echo "#BSUB -n $PROCS" >> $OUTPUT
echo "#BSUB -W 00:15" >> $OUTPUT
echo "#BSUB -o ${FOUT}.out" >> $OUTPUT
echo "#BSUB -e ${FOUT}.err" >> $OUTPUT
echo '#BSUB -R "affinity[core(4)]"' >> $OUTPUT
echo "mpiexec ./task3_omp $ARG $ARG $ARG $SIZE 5" >> $OUTPUT

#cat $OUTPUT

bsub < $OUTPUT

# tr #BSUB -R "span[hosts=1]"
# tr #BSUB -R "rusage[ut=1]"

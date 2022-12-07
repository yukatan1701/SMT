eps=$1
n=$2

name="task.$(echo "$eps" | tr -d '.').$n"
mpisubmit.pl -p $n -w 00:01 --stdout ${name}.out --stderr ${name}.err ./a.out -- $eps

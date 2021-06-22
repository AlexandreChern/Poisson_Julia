#!/bin/bash
for x in 8 16 32 64 128 256 512 1024
do
	#x = $(( 2 ** $iter))
	#x <<< 2^$iter
	echo "N = " $x
	#time  ./loadsolve -fA data/A_$x.dat -fb data/b_$x.dat -ksp_rtol 1e-10  -ksp_converged_reason \
	#	-ksp_type cg -pc_type ilu
	time  ./loadsolve -fA data/A_$x.dat -fb data/b_$x.dat -ksp_rtol 1e-10  -ksp_converged_reason \
		-ksp_type richardson -pc_type jacobi
	echo "\n"
done


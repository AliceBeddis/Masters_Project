#!/bin/bash

## SIMULATE GENES

MSMS=/Applications/msms/lib/msms.jar

# parameters of the demographic model are taken from Marth et al. Genetics 2004
# table 1, European data, 3 epoch model:
# N3=10,000,N2=2,000,N1=20,000,T2=5,00,T1=3,000
# one epoch: N1=10,000
# two epoch: N2=10,000, N1=140,000, T1=2000
NREF=10000
MARTH3='-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2'
MARTH2='-eN 0.05 1 -eN 0 14'

# other (fixed) paramaters are
LEN=100000
THETA=60 # 1.5 close to 1.44e-8*100000*40000 ## Gravel et al. 2013
RHO=40 # 1e-8*100000*40000
NCHROMS=198 # to match unrelated CEU in 1000G data
NREPL=5000 # nr of replicates per parameter value

# other (variable) parameters are
# start of selection approx 10,000, reference Gerbault et al 2011 phil trans roayl B, using 25 years per generation, this is 400 divided by 4*Nref
SELTIME=`bc <<< 'scale=4; 400/40000'`
SELPOS=`bc <<< 'scale=2; 1/2'` # relative position of selected allele
FREQ=`bc <<< 'scale=4; 1/200'` # frequency of allele at start of selection, 0.5%

# parameter
SELRANGE=`seq 0 10 400` # 2*10000*0.02=400

for SEL in $SELRANGE
do
	echo $SEL
	# 3 epoch model
	java -jar $MSMS -N $NREF -ms $NCHROMS $NREPL -t $THETA -r $RHO $LEN -Sp $SELPOS -SI $SELTIME 1 $FREQ -SAA $(($SEL*2)) -SAa $SEL -Saa 0 -Smark $MARTH3 -seed 1234 | gzip > /Users/alicebeddis/Desktop/Master_Proj/Data/Simulations_3epoch/msms..$SEL..txt.gz
	# 2 epoch model
	java -jar $MSMS -N $NREF -ms $NCHROMS $NREPL -t $THETA -r $RHO $LEN -Sp $SELPOS -SI $SELTIME 1 $FREQ -SAA $(($SEL*2)) -SAa $SEL -Saa 0 -Smark $MARTH2 -seed 1234 | gzip > /Users/alicebeddis/Desktop/Master_Proj/Data/Simulations_2epoch/msms..$SEL..txt.gz
	# 1 epoch model
	java -jar $MSMS -N $NREF -ms $NCHROMS $NREPL -t $THETA -r $RHO $LEN -Sp $SELPOS -SI $SELTIME 1 $FREQ -SAA $(($SEL*2)) -SAa $SEL -Saa 0 -Smark -seed 1234 | gzip > /Users/alicebeddis/Desktop/Master_Proj/Data/Simulations_1epoch/msms..$SEL..txt.gz

done




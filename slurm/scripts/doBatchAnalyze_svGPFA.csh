#!/bin/csh

setenv estInitNumber 1
setenv priorCovRegParam 1e-3
# set arrayOpt=2-15
# set arrayOpt=2-3
# set arrayOpt=4-15
# set arrayOpt=16-25
# set arrayOpt=20,22,23
set arrayOpt=10

sbatch \
--job-name=svGPFA \
--output=../outputs/doBathAnalyze_svGPFA_%A_%a.out \
--error=../outputs/doBathAnalyze_svGPFA_%A_%a.err \
--time=96:00:00 \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=12 \
--mem=20G \
--array=$arrayOpt \
./doAnalyze_svGPFA.sbatch 

#!/bin/csh

cd ../../code/scripts

# Print the task id.
echo "Starting n_latents: " $SLURM_ARRAY_TASK_ID " priorCovRegParam: " $priorCovRegParam " on node " $SLURMD_NODENAME

python -u doEstimateSVGPFA.py $estInitNumber --n_latents=$SLURM_ARRAY_TASK_ID --prior_cov_reg_param=$priorCovRegParam

echo "Done with n_latents: " $SLURM_ARRAY_TASK_ID " priorCovRegParam: " $priorCovRegParam " on node " $SLURMD_NODENAME

cd -

done:
 exit 0


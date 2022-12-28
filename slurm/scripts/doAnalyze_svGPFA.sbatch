#!/bin/csh

cd ../../code/scripts

# Print the task id.

if ( $?common_n_ind_points ) then
    echo "Starting n_latents: " $SLURM_ARRAY_TASK_ID "common_n_ind_point: " $common_n_ind_points " on node " $SLURMD_NODENAME
    python -u doEstimateSVGPFA.py $estInitNumber --n_latents=$SLURM_ARRAY_TASK_ID --common_n_ind_points=$common_n_ind_points
    echo "Done with n_latents: " $SLURM_ARRAY_TASK_ID "common_n_ind_point: " $common_n_ind_points " on node " $SLURMD_NODENAME
else
    echo "Starting n_latents: " $SLURM_ARRAY_TASK_ID " on node " $SLURMD_NODENAME
    python -u doEstimateSVGPFA.py $estInitNumber --n_latents=$SLURM_ARRAY_TASK_ID
    echo "Done with n_latents: " $SLURM_ARRAY_TASK_ID " on node " $SLURMD_NODENAME
endif


cd -

done:
 exit 0


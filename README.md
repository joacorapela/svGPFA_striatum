# Scripts to estimate [svGPFA](http://www.github.com/joacorapela/svGPFA) model to ephys recording from the striatum

Please refer to [code/scripts](code/scripts).

To estimate and svGPFA model you could follow these steps:

1. Create epochs for all trials of striatum and motor cortex using [doEpochSpikesTimes.py](code/scripts/doEpochSpikesTimes.py).

2. Create an `ini` file specifying initial values and options for the estimation of an svGPFA model. You could modify this example [00000017_estimation_metaData.ini](init/00000017_estimation_metaData.ini). For details on the entries on this file please refer to the [parameters sections](https://joacorapela.github.io/svGPFA/params.html) of the [svGPFA documentation](https://joacorapela.github.io/svGPFA/index.html).

3. Create a text file containing the trial numbers you want to use to estimate an svGPFA model. This file should contain one trial number per line. For example the file [results/trialsIDsOfPerfectSecuences.csv](results/trialsIDsOfPerfectSecuences.csv) specifies the trials numbers of perfect sequences.

4. Estimate an svGPFA model by changing to the `code/scripts` directory of the repo and running the following command:

```python
python doEstimageSVGPFA.py 17 --trials_ids_filename ../../results/trialsIDsOfPerfectSecuences.csv

```

where 17 refers to the number of the `ini` file and `--trials_ids_filename` provides the path to the file created in step 3 above.

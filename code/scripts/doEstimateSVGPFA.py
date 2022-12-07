import sys
import os.path
import random
import math
import torch
import pickle
import argparse
import configparser
import pandas as pd
import numpy as np

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_init_number", help="estimation init number",
                        type=int)
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=10)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=9)
    parser.add_argument("--max_trial_duration",
                        help="maximum trial duration (secs)",
                        type=float, default=5.0)
    parser.add_argument("--min_neuron_trials_avg_firing_rate",
                        help="min trials-averaged firing rate to keep a neuron",
                        type=float, default=0.1)
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filename",
                        type=str,
                        default="../../results/spikes_times_epochedFirst2In.pickle")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../init/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    n_latents = args.n_latents
    common_n_ind_points = args.common_n_ind_points
    max_trial_duration = args.max_trial_duration 
    min_neuron_trials_avg_firing_rate = args.min_neuron_trials_avg_firing_rate
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = np.array(load_res["trials_start_times"])
    trials_end_times = np.array(load_res["trials_end_times"])
    n_trials = len(spikes_times)
    trials_indices = np.arange(n_trials)
    n_neurons = len(spikes_times[0])
    neurons_indices = np.arange(n_neurons)

    trials_durations = trials_end_times - trials_start_times
    spikes_times, neurons_indices = \
        gcnu_common.utils.neuralDataAnalysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, neurons_indices=neurons_indices,
            trials_durations = trials_durations,
            min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)

    spikes_times, trials_indices = \
        gcnu_common.utils.neuralDataAnalysis.removeTrialsLongerThanThr(
            spikes_times=spikes_times, trials_indices=trials_indices,
            trials_durations=trials_durations,
            max_trial_duration=max_trial_duration)

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_start_times = trials_start_times[trials_indices]
    trials_end_times = trials_end_times[trials_indices]

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #   build config file parameters specification
    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials,
            strings_dict=strings_dict, args_info=args_info)
    #    build default parameter specificiations
    default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
        common_n_ind_points=common_n_ind_points)
    #    finally, get the parameters from the dynamic,
    #    configuration file and default parameter specifications
    params, kernels_types, = \
        svGPFA.utils.initUtils.getParamsAndKernelsTypes(
            n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times,
            dynamic_params_spec=dynamic_params_spec,
            config_file_params_spec=config_file_params_spec,
            default_params_spec=default_params_spec)

    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            estim_res_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    modelSaveFilename = model_save_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
        linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setParamsAndData(
        measurements=spikes_times,
        initial_params=params["initial_params"],
        eLLCalculationParams=params["ell_calculation_params"],
        priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

    # save estimated values
    estim_res_config = configparser.ConfigParser()
    estim_res_config["data_params"] = {
        "trials_indices": trials_indices,
        "neurons_indices": neurons_indices,
        "nLatents": n_latents,
        "max_trial_duration": max_trial_duration,
        "min_neuron_trials_avg_firing_rate": min_neuron_trials_avg_firing_rate}
    estim_res_config["optim_params"] = params["optim_params"]
    estim_res_config["estimation_params"] = {"est_init_number":
                                             est_init_number}
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)

    # maximize lower bound
    def getKernelParams(model):
        kernelParams = model.getKernelsParams()
        print("kernelParams", kernelParams)
        return kernelParams[0]

    # maximize lower bound
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=params["optim_params"],
                      method=params["optim_params"]["optim_method"],
                      getIterationModelParamsFn=getKernelParams)

    resultsToSave = {"neurons_indices": neurons_indices,
                     "lowerBoundHist": lowerBoundHist,
                     "elapsedTimeHist": elapsedTimeHist,
                     "terminationInfo": terminationInfo,
                     "iterationModelParams": iterationsModelParams,
                     "spikes_times": spikes_times,
                     "trials_indices": trials_indices,
                     "trials_start_times": trials_start_times,
                     "trials_end_times": trials_end_times,
                     "model": model}
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
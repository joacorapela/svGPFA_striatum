
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import pickle
import argparse

import gcnu_common.stats.pointProcesses.tests
import svGPFA.utils.miscUtils
import svGPFA.plot.plotUtilsPlotly
import striatumUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--latent_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--latents_to_3D_plot", help="latents to plot in 3D plot",
                        type=str, default="[0,1,2]")
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int, default=0)
    parser.add_argument("--trial_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--n_time_steps_CIF",
                        help="number of time steps to plot for CIF",
                        type=int, default=1000)
    parser.add_argument("--trials_timing_info_filename",
                        help="trials' timing info filename",
                        type=str,
                        default="../../results/trials_timing_info.csv")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    latent_to_plot = args.latent_to_plot
    latents_to_3D_plot = [int(str) for str in args.latents_to_3D_plot[1:-1].split(",")]
    neuron_to_plot = args.neuron_to_plot
    trial_to_plot = args.trial_to_plot
    ksTestGamma = args.ksTestGamma
    n_time_steps_CIF = args.n_time_steps_CIF
    trials_timing_info_filename = args.trials_timing_info_filename

    modelSaveFilename = "../../results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVSIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "../../figures/{:08d}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatentsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    latents_to_3D_plot_str = "".join(str(i)+"_" for i in latents_to_3D_plot)
    orthonormalizedLatents3DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_3D_plot_str)
    embeddingsFigFilenamePattern = "../../figures/{:08d}_embedding_neuron{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
    orthonormalizedEmbeddingParamsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_embedding_params.{{:s}}".format(estResNumber)
#     CIFFigFilenamePattern = "../../figures/{:08d}_CIF_trial{:03d}_neuron{:03d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
#     CIFimageFigFilenamePattern = "../../figures/{:08d}_CIFimage_neuron{:03d}_sortedBy{:s}.{{:s}}".format(estResNumber, neuron_to_plot, column_name)
    CIFsOneNeuronAllTrialsFigFilenamePattern = "../../figures/{:08d}_intensityFunctionOneNeuronAllTrials_neuron{:03d}.{{:s}}".format(estResNumber, neuron_to_plot)
    ksTestTimeRescalingNumericalCorrectionFigFilenamePattern = "../../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    rocFigFilenamePattern = "../../figures/{:08d}_predictive_analysis_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    kernelsParamsFigFilenamePattern = "../../figures/{:08d}_kernels_params.{{:s}}".format(estResNumber)

    trials_timing_info = pd.read_csv(trials_timing_info_filename)

    with open(modelSaveFilename, "rb") as f:
        estResults = pickle.load(f)
    spikes_times = estResults["spikes_times"]
    trials_indices = estResults["trials_indices"]
    clusters_ids = estResults["clusters_ids"]
    trials_start_times = estResults["trials_start_times"]
    trials_end_times = estResults["trials_end_times"]
    lowerBoundHist = estResults["lowerBoundHist"]
    elapsedTimeHist = estResults["elapsedTimeHist"]
    model = estResults["model"]
    neurons_indices = estResults["neurons_indices"]

    n_trials = len(trials_indices)
    neuron_to_plot_index = torch.nonzero(torch.tensor(neurons_indices) ==
                                         neuron_to_plot)
    neurons_indices_str = " ".join(str(i) for i in neurons_indices)
    if len(neuron_to_plot_index) == 0:
        raise ValueError("Neuron {:d} is not valid. Valid neurons are ".format(
            neuron_to_plot) + neurons_indices_str)

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=trials_start_times,
        end_times=trials_end_times,
        n_steps=n_time_steps_CIF)

    trials_labels = np.array([str(i) for i in trials_indices])
    n_trials = len(spikes_times)

    marked_events_times, marked_events_colors, marked_events_markers = \
        striatumUtils.buildMarkedEventsInfo(
            trials_timing_info=trials_timing_info,
            trials_indices=trials_indices,
        )

    align_event = np.array([marked_events_times[r][0] \
                            for r in range(n_trials)])

    # plot lower bound history
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot estimated latent across trials
    testMuK, testVarK = model.predictLatents(times=trials_times)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=trials_times.numpy(),
        latentsMeans=testMuK, latentsSTDs=torch.sqrt(testVarK),
        latentToPlot=latent_to_plot,
        trials_labels=trials_labels,
        xlabel="Time (msec)")
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))

    # plot orthonormalized estimated latent across trials
    testMuK, _ = model.predictLatents(times=trials_times)
    test_mu_k_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
    estimatedC, estimatedD = model.getSVEmbeddingParams()
    estimatedC_np = estimatedC.detach().numpy()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
        trials_times=trials_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
        align_event=align_event,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        C=estimatedC_np, trials_indices=trials_indices,
        xlabel="Time (msec)")
    fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
        trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
        C=estimatedC_np, trials_indices=trials_indices,
        latentsToPlot=latents_to_3D_plot,
        align_event=align_event,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
    )
    fig.write_image(orthonormalizedLatents3DFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatents3DFigFilenamePattern.format("html"))

    # plot embedding
    embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
    embeddingMeans = embeddingMeans.detach().numpy()
    embeddingVars = embeddingVars.detach().numpy()
    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
        times=trials_times.numpy(),
        embeddingsMeans=embeddingMeans[:, :, neuron_to_plot],
        embeddingsSTDs=np.sqrt(embeddingVars[:, :, neuron_to_plot]),
        title=title)
    fig.write_image(embeddingsFigFilenamePattern.format("png"))
    fig.write_html(embeddingsFigFilenamePattern.format("html"))

    # calculate expected IF values (for KS test and IF plots)
    with torch.no_grad():
        cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
    cif_values_GOF = cif_values[trial_to_plot][neuron_to_plot]

    # CIF

    # CIFs one neuron all trials
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
        trials_times=trials_times,
        cif_values=cif_values,
        neuron_index=neuron_to_plot,
        spikes_times=spikes_times,
        trials_indices=trials_indices,
        align_event=align_event,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
    )
    fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("png"))
    fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("html"))

    trial_times_GOF = trials_times[trial_to_plot, :, 0]
    spikes_times_GOF = spikes_times[trial_to_plot][neuron_to_plot]
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
        trial_to_plot, neuron_to_plot, len(spikes_times_GOF))

    # plot KS test time rescaling (numerical correction)
    if len(spikes_times_GOF) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF, cif_times=trial_times_GOF, cif_values=cif_values_GOF, gamma=ksTestGamma)
        fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
    fig.write_image(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("png"))
    fig.write_html(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("html"))

    # ROC predictive analysis
    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_GOF,
        cif_times=trial_times_GOF,
        cif_values=cif_values_GOF)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
        fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    fig.write_image(rocFigFilenamePattern.format("png"))
    fig.write_html(rocFigFilenamePattern.format("html"))

    # plot orthonormalized embedding parameters
    hovertemplate = "value: %{y}<br>" + \
                    "neuron index: %{x}<br>" + \
                    "%{text}"
    text = [f"cluster_id: {cluster_id}" for cluster_id in clusters_ids]
    estimatedC, estimatedD = model.getSVEmbeddingParams()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
        C=estimatedC.numpy(), d=estimatedD.numpy(),
        hovertemplate=hovertemplate, text=text)
    fig.write_image(
        orthonormalizedEmbeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(
        orthonormalizedEmbeddingParamsFigFilenamePattern.format("html"))

    breakpoint()

    # plot kernel parameters
    kernelsParams = model.getKernelsParams()
    kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
    fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
        kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

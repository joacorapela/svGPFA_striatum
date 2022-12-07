
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

import svGPFA.plot.plotUtilsPlotly
import striatumUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number",
                        type=int)
    parser.add_argument("--n_time_steps_CIF", help="number of stime steps in "
                        "CIF plots", type=int, default=100)
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--trials_timing_info_filename",
                        help="trials' timing info filename",
                        type=str,
                        default="../../results/trials_timing_info.csv")
    parser.add_argument("--orthonormalized_latents_fig_filename_pattern",
                        help="figure filename for an orthonormalized latent",
                        type=str,
                        default="../../figures/{:08d}_orthonormalized_estimatedLatent_latent{:03d}.{:s}")

    args = parser.parse_args()
    est_res_number = args.est_res_number
    n_time_steps_CIF = args.n_time_steps_CIF
    model_save_filename_pattern = args.model_save_filename_pattern
    trials_timing_info_filename = args.trials_timing_info_filename
    orthonormalized_latents_fig_filename_pattern = \
        args.orthonormalized_latents_fig_filename_pattern

    trials_timing_info = pd.read_csv(trials_timing_info_filename)

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    model = estResults["model"]
    trials_indices = estResults["trials_indices"]
    n_trials = len(trials_indices)
    trials_start_times = estResults["trials_start_times"]
    trials_end_times = estResults["trials_end_times"]

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=np.squeeze(trials_start_times),
        end_times=np.squeeze(trials_end_times),
        n_steps=n_time_steps_CIF)

    test_mu_k, _ = model.predictLatents(times=trials_times)
    n_latents = test_mu_k[0].shape[1]
    test_mu_k_np = [test_mu_k[r].detach().numpy()
                    for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()

    marked_events_times, marked_events_colors, marked_events_markers = \
        striatumUtils.buildMarkedEventsInfo(
            trials_timing_info=trials_timing_info,
            trials_indices=trials_indices,
        )

    align_event = np.array([marked_events_times[r][0] \
                            for r in range(n_trials)])

    for latent_to_plot in range(n_latents):
        print(f"Processing latent {latent_to_plot}")
        fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
            trials_times=trials_times, latentsMeans=test_mu_k_np,
            latentToPlot=latent_to_plot,
            align_event=align_event,
            marked_events_times=marked_events_times,
            marked_events_colors=marked_events_colors,
            marked_events_markers=marked_events_markers,
            C=estimatedC_np, trials_indices=trials_indices,
            xlabel="Time (msec)")
        fig.write_image(
            orthonormalized_latents_fig_filename_pattern.format(est_res_number, latent_to_plot, "png"))
        fig.write_html(
            orthonormalized_latents_fig_filename_pattern.format(est_res_number, latent_to_plot, "html"))


if __name__ == "__main__":
    main(sys.argv)


import sys
import argparse
import pandas as pd
import plotly.graph_objs as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_lower_bounds_filename", type=str,
                        help="models lower bounds filename",
                        default="../outputs/modelsLowerBounds.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename",
                        default="../figures/modelSelection.{:s}")
    args = parser.parse_args()

    models_lower_bounds_filename = args.models_lower_bounds_filename
    fig_filename_pattern = args.fig_filename_pattern

    models_lower_bounds = pd.read_csv(models_lower_bounds_filename, sep=",")
    fig = go.Figure()
    trace = go.Scatter(x=models_lower_bounds["nLatents"],
                       y=models_lower_bounds["lower_bound"],
                       mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Number of Latents")
    fig.update_yaxes(title_text="Lower Bound")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)

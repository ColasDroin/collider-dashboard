#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc


# Import functions
import plot

#################### Filling scheme Layout ####################


def return_filling_scheme_layout(array_b1, array_b2):
    scheme_layout = dmc.Stack(
        children=[
            dmc.Center(
                dmc.Alert(
                    (
                        "I may add a plot displaying the number of long-ranges and head-on"
                        " interaction for each bunch is it's deemed relevant."
                    ),
                    title="Alert!",
                    style={"width": "70%", "margin-top": "10px"},
                ),
            ),
            dcc.Loading(
                dcc.Graph(
                    id="filling-scheme-graph",
                    mathjax=True,
                    config={
                        "displayModeBar": False,
                        "scrollZoom": True,
                        "responsive": True,
                        "displaylogo": False,
                    },
                    figure=plot.return_plot_filling_scheme(array_b1, array_b2),
                    style={"height": "30vh", "width": "100%", "margin": "10 auto"},
                ),
                type="circle",
            ),
        ]
    )
    return scheme_layout

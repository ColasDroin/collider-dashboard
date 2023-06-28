#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc


# Import functions
import plot

#################### Filling scheme Layout ####################


def return_filling_scheme_layout(array_b1, array_b2, i_bunch_b1, i_bunch_b2):
    scheme_layout = dmc.Stack(
        children=[
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
                    figure=plot.return_plot_filling_scheme(
                        array_b1, array_b2, i_bunch_b1, i_bunch_b2
                    ),
                    style={"height": "20vh", "width": "100%", "margin": "auto"},
                ),
                type="circle",
            ),
        ]
    )
    return scheme_layout

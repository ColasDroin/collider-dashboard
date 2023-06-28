#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc

# Import functions
import plot

#################### Optics Layout ####################


def return_optics_layout():
    optics_layout = dmc.Stack(
        children=[
            dcc.Loading(
                dcc.Graph(
                    id="LHC-2D-near-IP",
                    mathjax=True,
                    config={
                        "displayModeBar": False,
                        "scrollZoom": True,
                        "responsive": True,
                        "displaylogo": False,
                    },
                    style={"height": "90vh", "width": "100%", "margin": "auto"},
                ),
                type="circle",
                color="cyan",
            ),
        ]
    )
    return optics_layout

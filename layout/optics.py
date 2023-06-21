#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc

# Import functions
import plot

#################### Optics Layout ####################


def return_optics_layout(tw_b1, tw_b2, df_sv_b1, df_elements_corrected):
    optics_layout = dmc.Center(
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
                figure=plot.return_plot_optics(tw_b1, tw_b2, df_sv_b1, df_elements_corrected),
                style={"height": "90vh", "width": "100%", "margin": "auto"},
            ),
            type="circle",
            style={"height": "100%", "width": "100%", "margin": "auto"},
            parent_style={"height": "100%", "width": "100%", "margin": "auto"},
        ),
    )
    return optics_layout

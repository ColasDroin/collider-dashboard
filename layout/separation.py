#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc

# Import functions
import plot

#################### Separation Layout ####################


def return_separation_layout(dic_sep_IPs):
    separation_layout = dmc.Center(
        dcc.Loading(
            dcc.Graph(
                id="beam-separation",
                mathjax=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": True,
                    "responsive": True,
                    "displaylogo": False,
                },
                figure=plot.return_plot_separation(dic_sep_IPs),
                style={"height": "90vh", "width": "100%", "margin": "auto"},
            ),
            type="circle",
            style={"height": "100%", "width": "100%", "margin": "auto"},
            parent_style={"height": "100%", "width": "100%", "margin": "auto"},
        ),
    )
    return separation_layout

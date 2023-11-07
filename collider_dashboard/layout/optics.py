# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc, html

# Import local functions
from ..backend import plot


# ==================================================================================================
# --- Optics layout
# ==================================================================================================
def return_optics_layout(dic_with_bb):
    """
    Returns the layout for the optics page of the simulation dashboard.

    Args:
        dic_with_bb (dict): A dictionary of collider (including beam-beam) observables.

    Returns:
        html.Div: A Div representing the layout for the optics tab.
    """
    optics_layout = html.Div(
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
                    figure=plot.return_plot_optics(
                        dic_with_bb["df_tw_b1"],
                        dic_with_bb["df_tw_b2"],
                        dic_with_bb["df_sv_b1"],
                        dic_with_bb["df_elements_corrected"],
                        empty=True,
                    ),
                    style={"height": "90vh", "width": "100%", "margin": "auto"},
                ),
                type="circle",
                color="cyan",
            ),
            dmc.NumberInput(
                label="Vertical zoom",
                description="Vertical zoom level",
                id="vertical-zoom-optics",
                value=0,
                min=-14,
                max=14,
                step=1,
                style={
                    "width": 150,
                    "position": "absolute",
                    "bottom": "3%",
                    "right": "3%",
                },
            ),
        ],
        # style={"height": "100vh", "width": "100%", "margin": "auto"},
    )
    return optics_layout

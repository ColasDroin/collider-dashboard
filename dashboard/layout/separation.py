# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc


# ==================================================================================================
# --- Separation Layout
# ==================================================================================================
def return_separation_layout():
    """
    Returns the layout for the beam separation dashboard page.

    Returns:
        separation_layout : dash.development.base_component.Component
            The layout for the beam separation dashboard page.
    """
    separation_layout = (
        dmc.Center(
            dmc.Stack(
                children=[
                    dmc.Center(
                        children=[
                            dmc.Group(
                                children=[
                                    dmc.Text("Beam-beam: "),
                                    dmc.ChipGroup(
                                        [
                                            dmc.Chip(
                                                x,
                                                value=x,
                                                variant="outline",
                                                color="cyan",
                                            )
                                            for x in ["On", "Off"]
                                        ],
                                        id="chips-sep-bb",
                                        value="Off",
                                        mb=0,
                                    ),
                                    dmc.Space(),
                                    dmc.Text("Separation plane: "),
                                    dmc.ChipGroup(
                                        [
                                            dmc.Chip(
                                                x,
                                                value=x,
                                                variant="outline",
                                                color="cyan",
                                            )
                                            for x in ["v", "h", "||v+h||"]
                                        ],
                                        id="chips-sep",
                                        value="v",
                                        mb=0,
                                    ),
                                ],
                                pt=5,
                            ),
                        ],
                    ),
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
                            style={"height": "90vh", "width": "100%", "margin": "auto"},
                        ),
                        type="circle",
                        color="cyan",
                        style={"height": "100%", "width": "100%", "margin": "auto"},
                        parent_style={
                            "height": "100%",
                            "width": "100%",
                            "margin": "auto",
                        },
                    ),
                ],
                style={"width": "100%", "margin": "auto"},
            )
        ),
    )
    return separation_layout

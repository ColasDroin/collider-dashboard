# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc


# ==================================================================================================
# --- 3D separation Layout
# ==================================================================================================
def return_3D_separation_layout():
    """
    Returns the layout for the 3D beam separation dashboard.

    Returns:
        separation_layout : dash_bootstrap_components.Center
            Layout for the 3D beam separation dashboard.
    """
    separation_layout = (
        dmc.Center(
            dmc.Stack(
                children=[
                    dmc.Center(
                        children=[
                            dmc.Group(
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
                                                id="chips-sep-bb-3D",
                                                value="Off",
                                                mb=0,
                                            ),
                                        ],
                                        pt=5,
                                    ),
                                    dmc.Space(w="xl"),
                                    dmc.Group(
                                        children=[
                                            dmc.Text("IP: "),
                                            dmc.ChipGroup(
                                                [
                                                    dmc.Chip(
                                                        x,
                                                        value=x,
                                                        variant="outline",
                                                        color="cyan",
                                                    )
                                                    for x in ["IP 1", "IP 2", "IP 5", "IP 8"]
                                                ],
                                                id="chips-ip-3D",
                                                value="IP 1",
                                                mb=0,
                                            ),
                                        ],
                                        pt=5,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="beam-separation-3D",
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

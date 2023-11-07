# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc


# ==================================================================================================
# --- Footprint Layout
# ==================================================================================================
def return_footprint_layout():
    """
    Returns the layout for the footprint section of the simulation dashboard.

    Returns:
        footprint_layout : dash.development.base_component.Component
            The layout for the footprint section of the simulation dashboard.

    """
    footprint_layout = (
        dmc.Center(
            dmc.Stack(
                children=[
                    dcc.Loading(
                        children=[
                            dmc.Group(
                                children=[
                                    dcc.Graph(
                                        id="footprint-without-bb-b1",
                                        mathjax=True,
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                            "responsive": True,
                                            "displaylogo": False,
                                        },
                                        style={
                                            "height": "45vh",
                                            "width": "45%",
                                            "margin": "auto",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="footprint-with-bb-b1",
                                        mathjax=True,
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                            "responsive": True,
                                            "displaylogo": False,
                                        },
                                        style={
                                            "height": "45vh",
                                            "width": "45%",
                                            "margin": "auto",
                                        },
                                    ),
                                ],
                            ),
                            dmc.Group(
                                children=[
                                    dcc.Graph(
                                        id="footprint-without-bb-b2",
                                        mathjax=True,
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                            "responsive": True,
                                            "displaylogo": False,
                                        },
                                        style={
                                            "height": "45vh",
                                            "width": "45%",
                                            "margin": "auto",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="footprint-with-bb-b2",
                                        mathjax=True,
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                            "responsive": True,
                                            "displaylogo": False,
                                        },
                                        style={
                                            "height": "45vh",
                                            "width": "45%",
                                            "margin": "auto",
                                        },
                                    ),
                                ],
                            ),
                        ],
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
    return footprint_layout

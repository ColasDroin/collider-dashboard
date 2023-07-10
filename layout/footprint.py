#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import dcc

# Import functions
import plot

#################### Footprint Layout ####################


def return_footprint_layout():
    footprint_layout = (
        dmc.Center(
            dmc.Stack(
                children=[
                    dcc.Loading(
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
                                    style={"height": "70vh", "width": "45%", "margin": "auto"},
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
                                    style={"height": "70vh", "width": "45%", "margin": "auto"},
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
                                    style={"height": "70vh", "width": "45%", "margin": "auto"},
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
                                    style={"height": "70vh", "width": "45%", "margin": "auto"},
                                ),
                            ],
                        ),
                        type="circle",
                        color="cyan",
                        style={"height": "100%", "width": "100%", "margin": "auto"},
                        parent_style={"height": "100%", "width": "100%", "margin": "auto"},
                    ),
                ],
                style={"width": "100%", "margin": "auto"},
            )
        ),
    )
    return footprint_layout

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc, html


# ==================================================================================================
# --- Survey layout
# ==================================================================================================
def return_survey_layout():
    """
    Returns the layout for the survey page of the simulation dashboard.

    Returns:
        A Dash HTML div element representing the survey layout.
    """
    survey_layout = html.Div(
        children=[
            dmc.Center(
                dmc.Stack(
                    children=[
                        dmc.Center(
                            children=[
                                dmc.Group(
                                    children=[
                                        dmc.Text("Sectors to display: "),
                                        dmc.ChipGroup(
                                            [
                                                dmc.Chip(
                                                    x,
                                                    value=x,
                                                    variant="outline",
                                                    color="cyan",
                                                )
                                                for x in ["8-2", "2-4", "4-6", "6-8"]
                                            ],
                                            id="chips-ip",
                                            value=["4-6"],
                                            multiple=True,
                                            mb=0,
                                        ),
                                    ],
                                    pt=10,
                                ),
                            ],
                        ),
                        dcc.Loading(
                            children=dcc.Graph(
                                id="LHC-layout",
                                mathjax=True,
                                config={
                                    "displayModeBar": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                    "displaylogo": False,
                                },
                                style={
                                    "height": "90vh",
                                    "width": "100%",
                                    "margin": "auto",
                                },
                            ),
                            type="circle",
                            color="cyan",
                        ),
                    ],
                    style={"width": "100%", "margin": "auto"},
                )
            ),
            dmc.Drawer(
                title="Element information",
                id="drawer-magnets",
                padding="md",
                transition="rotate-left",
                transitionDuration=20,
                zIndex=10000,
                transitionTimingFunction="ease",
                children=dmc.Card(
                    children=[
                        dmc.Group(
                            [
                                dmc.Text(
                                    id="title-element",
                                    children="Element",
                                    weight=500,
                                ),
                                dmc.Badge(
                                    id="type-element",
                                    children="Dipole",
                                    color="blue",
                                    variant="light",
                                ),
                            ],
                            position="apart",
                            mt="md",
                            mb="xs",
                        ),
                        html.Div(
                            id="text-element",
                            children=[
                                dmc.Text(
                                    id="initial-text",
                                    children=(
                                        "Please click on a multipole or an"
                                        " interaction point to get the"
                                        " corresponding knob information."
                                    ),
                                    size="sm",
                                    color="dimmed",
                                ),
                            ],
                        ),
                    ],
                    withBorder=True,
                    shadow="sm",
                    radius="md",
                    style={"width": "100%"},
                ),
            ),
        ],
        style={"width": "100%", "margin": "auto"},
    )
    return survey_layout

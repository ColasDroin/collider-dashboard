#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import html, dcc

#################### Tables Layout ####################


def return_tables_layout():
    layout = html.Div(
        children=[
            dmc.Center(
                dmc.SegmentedControl(
                    id="segmented-data-table",
                    data=[
                        "Twiss table beam 1",
                        "Survey table beam 1",
                        "Twiss table beam 2",
                        "Survey table beam 2",
                    ],
                    radius="md",
                    mt=10,
                    value="Twiss table beam 1",
                    color="cyan",
                ),
            ),
            dcc.Loading(
                html.Div(id="placeholder-data-table"),
                type="circle",
                style={"margin-top": "100px"},
                color="cyan",
            ),
        ],
        style={"width": "90%", "margin": "auto"},
    )
    return layout

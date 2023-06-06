#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import html, dcc

#################### Tables Layout ####################


def return_tables_layout():
    layout = html.Div(
        children=[
            dmc.Center(
                dmc.Alert(
                    (
                        "The datatables are slow as they are"
                        " heavy to download from the server. If"
                        " we want to keep this feature, I will"
                        " try to implement a lazy loading,"
                        " sorting and filtering in the backend"
                        " to speed things up."
                    ),
                    title="Alert!",
                    style={
                        "width": "70%",
                        "margin-top": "10px",
                    },
                ),
            ),
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
            dcc.Loading(html.Div(id="placeholder-data-table"), type="circle"),
        ],
        style={"width": "90%", "margin": "auto"},
    )
    return layout

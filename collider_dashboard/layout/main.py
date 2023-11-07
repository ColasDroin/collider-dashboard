# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash import dcc, html

# Import local packages
from .header import return_header_layout


# ==================================================================================================
# --- App layout
# ==================================================================================================
def return_app_layout():
    """
    Returns the overall layout for the app.

    Returns:
        layout (dash.html.Div): App layout.
    """
    # function code here
    layout = html.Div(
        style={"width": "90%", "margin": "auto"},
        children=[
            dcc.Location(id="url", refresh=True),
            return_header_layout(),
            dmc.Center(
                children=[
                    html.Div(
                        id="main-div",
                        style={"width": "100%", "margin": "auto"},
                        children=[
                            html.Div(id="placeholder-tabs"),
                        ],
                    ),
                ],
                style={"margin-top": "80px"},
            ),
        ],
    )

    # Dark theme
    layout = dmc.MantineProvider(
        withGlobalStyles=True,
        theme={"colorScheme": "dark"},
        children=layout,
    )
    return layout

# ==================================================================================================
# --- Imports
# ==================================================================================================
import dash_mantine_components as dmc


# ==================================================================================================
# --- Configuration layout
# ==================================================================================================
def return_knobs_layout(dic_knob_str):
    """
    Returns a Dash layout for displaying knob information.

    Args:
        dic_knob_str (str): A dictionnary containing the knob names (key) and information (values).

    Returns:
        dash.development.base_component.Component: A Dash layout component.
    """
    l_knobs = list(dic_knob_str.keys())
    if dic_knob_str is not None:
        configuration_layout = dmc.Stack(
            children=[
                dmc.Center(
                    dmc.Select(
                        id="select-knob_info",
                        data=l_knobs,
                        searchable=True,
                        nothingFound="No options found",
                        style={"width": 400},
                        value=l_knobs[0],
                    )
                ),
                dmc.Center(
                    children=[
                        dmc.Prism(
                            id="prism-knob_info",
                            language="python",
                            children="",
                            style={
                                "height": "90vh",
                                "overflowY": "auto",
                                "width": "80%",
                            },
                        ),
                    ],
                ),
            ],
        )

    else:
        configuration_layout = dmc.Center(
            dmc.Alert(
                "Configuration can't be displayed as no configuration file was provided.",
                title="No configuration !",
                style={"margin": "auto"},
            ),
            style={"width": "70%", "margin": "auto"},
        )

    return configuration_layout

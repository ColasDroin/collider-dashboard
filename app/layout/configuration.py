# ==================================================================================================
# --- Imports
# ==================================================================================================
import dash_mantine_components as dmc

# ==================================================================================================
# --- Configuration layout
# ==================================================================================================

def return_configuration_layout(configuration_str, path_collider):
    if configuration_str is not None:
        configuration_layout = dmc.Stack(
            children=[
                dmc.Center(dmc.Text("Configuration path: " + path_collider)),
                dmc.Center(
                    children=[
                        dmc.Prism(
                            language="yaml",
                            children=configuration_str,
                            style={"height": "90vh", "overflowY": "auto", "width": "80%"},
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

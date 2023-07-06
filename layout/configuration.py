#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc


#################### Configuration layout ####################


def return_configuration_layout(path_configuration):
    if path_configuration is not None:
        # Load configuration file
        with open(path_configuration, "r") as file:
            configuration_str = file.read()

        configuration_layout = dmc.Stack(
            children=[
                dmc.Center(dmc.Text("Configuration path: " + path_configuration)),
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
        configuration_layout = dmc.Text(
            "No configuration file provided", size="xl", style={"margin": "auto"}
        )

    return configuration_layout

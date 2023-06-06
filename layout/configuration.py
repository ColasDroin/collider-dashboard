#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc


#################### Configuration layout ####################

def return_configuration_layout(path_configuration):
    # Load configuration file
    with open(path_configuration, "r") as file:
        configuration_str = file.read()

    configuration_layout = dmc.Center(
        dmc.Prism(
            language="yaml",
            children=configuration_str,
            style={"height": "90vh", "overflowY": "auto", "width": "80%"},
        )
    )

    return configuration_layout


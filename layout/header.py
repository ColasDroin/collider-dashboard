#################### Imports ####################
# Import standard libraries
import os
import dash_mantine_components as dmc
from dash_iconify import DashIconify


#################### Functions to load collider choices ####################
def set_collider_dropdown_options():
    l_data = []
    for x in sorted(os.listdir("temp/")):
        try:
            id_collider = int(x.split("optics_")[1].split("_xtrack")[0].split("_")[-1]) + 23
            data = {
                "value": os.path.join("temp", x),
                "label": x.split("optics_")[1].split("_xtrack")[0][:-2] + f"{id_collider}",
            }
            l_data.append(data)
        except:
            pass
    return l_data


def return_initial_value(l_data):
    return l_data[0]["value"]


l_data = set_collider_dropdown_options()
initial_value = return_initial_value(l_data)


#################### Header Layout ####################
def return_header_layout():
    def create_header_link(icon, href, size=22, color="cyan"):
        return dmc.Anchor(
            dmc.ThemeIcon(
                DashIconify(
                    icon=icon,
                    width=size,
                ),
                variant="outline",
                radius=30,
                size=36,
                color=color,
            ),
            href=href,
            target="_blank",
        )

    header = dmc.Header(
        height=70,
        fixed=True,
        px=25,
        children=[
            dmc.Stack(
                justify="center",
                style={"height": 70, "width": "100%"},
                children=dmc.Group(
                    position="apart",
                    style={"width": "100%"},
                    children=[
                        dmc.Text(
                            "Simulation dashboard",
                            size=30,
                            color="cyan",
                            weight="bold",
                        ),
                        dmc.Group(
                            children=[
                                dmc.Text("Preloaded collider: "),
                                dmc.Select(
                                    id="select-preloaded-collider",
                                    data=l_data,
                                    value=initial_value,
                                    searchable=True,
                                    nothingFound="No options found",
                                    # style={"width": 200},
                                ),
                            ],
                        ),
                        dmc.SegmentedControl(
                            id="tab-titles",
                            value="display-configuration",
                            data=[
                                {"value": "display-configuration", "label": "Configuration"},
                                {"value": "display-twiss", "label": "Twiss tables"},
                                {"value": "display-scheme", "label": "Filling scheme"},
                                {"value": "display-sanity", "label": "Sanity checks"},
                                {"value": "display-separation", "label": "Beam-beam separation"},
                                {"value": "display-footprint", "label": "Footprint"},
                                {"value": "display-optics", "label": "Optics"},
                                {"value": "display-survey", "label": "Survey"},
                            ],
                            # color="cyan",
                            style={"margin-right": "5%"},
                            fullWidth=True,
                            size="md",
                            color="cyan",
                        ),
                        dmc.Group(
                            position="right",
                            spacing="xl",
                            children=[
                                create_header_link(
                                    "radix-icons:github-logo",
                                    "https://github.com/ColasDroin/simulation-dashboard",
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )
    return header

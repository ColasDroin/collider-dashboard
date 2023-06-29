#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash_iconify import DashIconify

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
                        dmc.SegmentedControl(
                            id="tab-titles",
                            value="display-configuration",
                            data=[
                                {"value": "display-configuration", "label": "Configuration"},
                                {"value": "display-twiss", "label": "Twiss tables"},
                                {"value": "display-scheme", "label": "Filling scheme"},
                                {"value": "display-sanity", "label": "Sanity checks"},
                                {"value": "display-separation", "label": "Beam-beam separation"},
                                {"value": "display-optics", "label": "Optics"},
                                {"value": "display-survey", "label": "Survey"},
                            ],
                            # color="cyan",
                            # style={"margin-right": "10%"},
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

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
from dash_iconify import DashIconify


# ==================================================================================================
# --- Header layout
# ==================================================================================================
def return_header_layout():
    """
    Returns the layout for the header of the simulation dashboard.

    Returns:
        dmc.Header: The header layout for the simulation dashboard.
    """

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
        px=20,
        children=[
            dmc.Stack(
                justify="center",
                align="center",
                style={"height": 70, "width": "100%"},
                children=dmc.Group(
                    position="apart",
                    style={"width": "100%"},
                    children=[
                        dmc.Text(
                            "ColBoard",
                            size=25,
                            color="cyan",
                            weight="bold",
                        ),
                        dmc.SegmentedControl(
                            id="tab-titles",
                            value="display-configuration",
                            data=[
                                {
                                    "value": "display-configuration",
                                    "label": "Configuration",
                                },
                                {"value": "display-knobs", "label": "Knobs"},
                                {"value": "display-twiss", "label": "Twiss"},
                                {"value": "display-scheme", "label": "Scheme"},
                                {"value": "display-sanity", "label": "Sanity checks"},
                                {"value": "display-separation", "label": "Separation"},
                                {
                                    "value": "display-3D-separation",
                                    "label": "3D separation",
                                },
                                {"value": "display-footprint", "label": "Footprint"},
                                {"value": "display-optics", "label": "Optics"},
                                {"value": "display-survey", "label": "Survey"},
                            ],
                            # color="cyan",
                            # style={"margin-right": "5%"},
                            fullWidth=True,
                            size="md",
                            color="cyan",
                        ),
                        dmc.Group(
                            position="right",
                            # spacing="xl",
                            children=[
                                create_header_link(
                                    "radix-icons:github-logo",
                                    "https://github.com/ColasDroin/simulation-dashboard",
                                ),
                            ],
                            mb=5,
                        ),
                    ],
                ),
            ),
        ],
    )
    return header

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
import numpy as np
from dash import html


# ==================================================================================================
# --- Sanity checks Layout
# ==================================================================================================
def return_sanity_layout(
    dic_tw_b1,
    dic_tw_b2,
    l_lumi,
    array_b1,
    array_b2,
    polarity_alice,
    polarity_lhcb,
    energy,
    cross_section_PU = 81e-27,
):
    """
    Returns the layout for the sanity check page of the dashboard.

    Args:
        dic_tw_b1 : dict
            A twiss dictionary for beam 1.
        dic_tw_b2 : dict
            A twiss dictionary for beam 2.
        l_lumi : list
            A list containing the luminosities for each interaction point.
        array_b1 : numpy.ndarray
            Beam schedule for beam 1.
        array_b2 : numpy.ndarray
            Beam schedule for beam 2.
        polarity_alice : str
            The polarity of the ALICE detector (+1 or -1).
        polarity_lhcb : str
            The polarity of the LHCb detector (+1 or -1).
        energy : float
            The energy of the beam.

    Returns:
        layout : dash.development.base_component.Component
            The layout for the sanity check page of the dashboard.
    """

    # Ensure that the polarities are defined
    polarity_alice = polarity_alice if polarity_alice is not None else "N/A"
    polarity_lhcb = polarity_lhcb if polarity_lhcb is not None else "N/A"

    # Check general observables (tune, chroma, etc.)
    header_1 = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Beam"),
                    html.Th("Qx"),
                    html.Th("Qy"),
                    html.Th("dQx"),
                    html.Th("dQy"),
                    html.Th("Linear coupling"),
                    # html.Th("Momentum compaction factor"),
                ]
            )
        )
    ]

    row1 = html.Tr(
        [
            html.Td("1"),
            html.Td(f'{dic_tw_b1["qx"]:.5f}'),
            html.Td(f'{dic_tw_b1["qy"]:.5f}'),
            html.Td(f'{dic_tw_b1["dqx"]:.2f}'),
            html.Td(f'{dic_tw_b1["dqy"]:.2f}'),
            html.Td(f'{dic_tw_b1["c_minus"]:.4f}'),
        ]
    )
    row2 = html.Tr(
        [
            html.Td("2"),
            html.Td(f'{dic_tw_b2["qx"]:.5f}'),
            html.Td(f'{dic_tw_b2["qy"]:.5f}'),
            html.Td(f'{dic_tw_b2["dqx"]:.2f}'),
            html.Td(f'{dic_tw_b2["dqy"]:.2f}'),
            html.Td(f'{dic_tw_b2["c_minus"]:.4f}'),
        ]
    )
    body_1 = [html.Tbody([row1, row2])]
    table_1 = dmc.Table(header_1 + body_1)

    # Check IP-specific observables (crossing angle, beta functions, etc.)
    header_2 = [
        html.Thead(
            html.Tr(
                [
                    html.Th("IP"),
                    html.Th("s [m]"),
                    html.Th("x [mm]"),
                    html.Th("px [µrad]"),
                    html.Th("y [mm]"),
                    html.Th("py [µrad]"),
                    html.Th("betx [cm]"),
                    html.Th("bety [cm]"),
                    html.Th("dx_zeta [µrad]"),
                    html.Th("dy_zeta [µrad]"),
                    html.Th("dpx_zeta [µrad/m]"),
                    html.Th("dpy_zeta [µrad/m]"),
                ]
            )
        )
    ]
    l_rows_1 = []
    l_rows_2 = []
    for ip in [1, 2, 5, 8]:
        row_values_1 = dic_tw_b1[f"ip{ip}"]
        row_values_2 = dic_tw_b2[f"ip{ip}"]

        l_rows_1.append(
            html.Tr(
                [
                    html.Td(row_values_1[0]),
                    html.Td(f"{row_values_1[1]:.3f}"),
                    html.Td(f"{(row_values_1[2]*1e3):.3f}"),
                    html.Td(f"{(row_values_1[3]*1e6):.3f}"),
                    html.Td(f"{(row_values_1[4]*1e3):.3f}"),
                    html.Td(f"{(row_values_1[5]*1e6):.3f}"),
                    html.Td(f"{(row_values_1[6]*1e2):.3f}"),
                    html.Td(f"{(row_values_1[7]*1e2):.3f}"),
                    html.Td(f"{(row_values_1[8]*1e6):.3f}"),
                    html.Td(f"{(row_values_1[9]*1e6):.3f}"),
                    html.Td(f"{(row_values_1[10]*1e6):.3f}"),
                    html.Td(f"{(row_values_1[11]*1e6):.3f}"),
                ]
            )
        )         

        l_rows_2.append(
            html.Tr(
                [
                    html.Td(row_values_2[0]),
                    html.Td(f"{row_values_2[1]:.3f}"),
                    html.Td(f"{(row_values_2[2]*1e3):.3f}"),
                    html.Td(f"{(row_values_2[3]*1e6):.3f}"),
                    html.Td(f"{(row_values_2[4]*1e3):.3f}"),
                    html.Td(f"{(row_values_2[5]*1e6):.3f}"),
                    html.Td(f"{(row_values_2[6]*1e2):.3f}"),
                    html.Td(f"{(row_values_2[7]*1e2):.3f}"),
                    html.Td(f"{(row_values_2[8]*1e6):.3f}"),
                    html.Td(f"{(row_values_2[9]*1e6):.3f}"),
                    html.Td(f"{(row_values_2[10]*1e6):.3f}"),
                    html.Td(f"{(row_values_2[11]*1e6):.3f}"),
                ]
            )
        )

    body_2 = [html.Tbody(l_rows_1)]
    body_3 = [html.Tbody(l_rows_2)]
    table_2 = dmc.Table(header_2 + body_2)
    table_3 = dmc.Table(header_2 + body_3)

    if l_lumi is not None:
        # Luminosities
        header_3 = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("IP 1 [cm-2 s-1]"),
                        html.Th("IP 2 [cm-2 s-1]"),
                        html.Th("IP 5 [cm-2 s-1]"),
                        html.Th("IP 8 [cm-2 s-1]"),
                    ]
                )
            )
        ]

        row_lumi = html.Tr(
            [
                (
                    html.Td(
                        f"{l_lumi[0]:.3e}",
                        style={"font-weight": "bold", "color": "red"},
                    )
                    if l_lumi[0] > 5e34
                    else html.Td(f"{l_lumi[0]:.3e}")
                ),
                html.Td(f"{l_lumi[1]:.3e}"),
                (
                    html.Td(
                        f"{l_lumi[2]:.3e}",
                        style={"font-weight": "bold", "color": "red"},
                    )
                    if l_lumi[2] > 5e34
                    else html.Td(f"{l_lumi[2]:.3e}")
                ),
                html.Td(f"{l_lumi[3]:.3e}"),
            ]
        )
        body_4 = [html.Tbody([row_lumi])]
        table_4 = dmc.Table(header_3 + body_4)

        # Assert that the arrays have the required length, and do the convolution to get number of collisions
        assert len(array_b1) == len(array_b2) == 3564
        n_collisions_ip1_and_5 = array_b1 @ array_b2
        n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
        n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2
        l_n_collisions = [
            n_collisions_ip1_and_5,
            n_collisions_ip2,
            n_collisions_ip1_and_5,
            n_collisions_ip8,
        ]
        n_turn_per_second = 1 / dic_tw_b1["T_rev0"]
        l_PU = [
            lumi / n_col * cross_section_PU / n_turn_per_second
            for lumi, n_col in zip(l_lumi, l_n_collisions)
        ]

        # PU
        header_4 = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("IP 1 [counts]"),
                        html.Th("IP 2 [counts]"),
                        html.Th("IP 5 [counts]"),
                        html.Th("IP 8 [counts]"),
                    ]
                )
            )
        ]

        # Table
        row_PU = html.Tr(
            [
                (
                    html.Td(f"{l_PU[0]:.1f}", style={"font-weight": "bold", "color": "red"})
                    if l_PU[0] > 160
                    else html.Td(f"{l_PU[0]:.1f}")
                ),
                html.Td(f"{l_PU[1]:.4f}"),
                (
                    html.Td(f"{l_PU[2]:.1f}", style={"font-weight": "bold", "color": "red"})
                    if l_PU[2] > 160
                    else html.Td(f"{l_PU[2]:.1f}")
                ),
                html.Td(f"{l_PU[3]:.4f}"),
            ]
        )
        body_5 = [html.Tbody([row_PU])]
        table_5 = dmc.Table(header_4 + body_5)
    else:
        table_4 = dmc.Text(
            (
                "No luminosities could be computed since no configuration was provided with the"
                " collider"
            ),
            size="xl",
            style={"margin": "auto"},
            color="tomato",
        )
        table_5 = dmc.Text(
            "No pile-up could be computed since no configuration was provided with the collider",
            size="xl",
            style={"margin": "auto"},
            color="tomato",
        )

    # Others
    header_5 = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Polarity Alice"),
                    html.Th("Polarity LHCb"),
                    html.Th("Energy"),
                ]
            )
        )
    ]

    row_other = html.Tr(
        [
            html.Td(str(polarity_alice)),
            html.Td(str(polarity_lhcb)),
            html.Td(f"{energy:.5f}"),
        ]
    )
    body_6 = [html.Tbody([row_other])]
    table_6 = dmc.Table(header_5 + body_6)

    return dmc.Stack(
        children=[
            dmc.Group(
                children=[
                    dmc.Text("General observables", size="xl", style={"margin": "auto"}),
                    table_1,
                ],
                mb=10,
                style={"width": "100%"},
            ),
            dmc.Group(
                children=[
                    dmc.Text("Beam 1 observables at IPs", size="xl", style={"margin": "auto"}),
                    table_2,
                ],
                mb=10,
                style={"width": "100%"},
            ),
            dmc.Group(
                children=[
                    dmc.Text("Beam 2 observables at IPs", size="xl", style={"margin": "auto"}),
                    table_3,
                ],
                mb=10,
                style={"width": "100%"},
            ),
            dmc.Stack(
                children=[
                    dmc.Text("Luminosities", size="xl", style={"margin": "auto"}),
                    table_4,
                ],
                mb=10,
                style={"width": "100%"},
            ),
            dmc.Stack(
                children=[
                    dmc.Text("Pile-up", size="xl", style={"margin": "auto"}),
                    table_5,
                ],
                mb=10,
                style={"width": "100%"},
            ),
            dmc.Group(
                children=[
                    dmc.Text("Other properties", size="xl", style={"margin": "auto"}),
                    table_6,
                ],
                mb=10,
                style={"width": "100%"},
            ),
        ],
        style={"width": "90%", "margin": "auto"},
    )

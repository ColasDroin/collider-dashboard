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


def return_general_observables_header():
    return [
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


def return_general_observables_values(dic_tw):
    return html.Tr(
        [
            html.Td("1"),
            html.Td(f'{dic_tw["qx"]:.5f}'),
            html.Td(f'{dic_tw["qy"]:.5f}'),
            html.Td(f'{dic_tw["dqx"]:.2f}'),
            html.Td(f'{dic_tw["dqy"]:.2f}'),
            html.Td(f'{dic_tw["c_minus"]:.4f}'),
        ]
    )


def return_general_observables_layout(dic_tw_b1, dic_tw_b2):
    header = return_general_observables_header()
    row1 = return_general_observables_values(dic_tw_b1)
    row2 = return_general_observables_values(dic_tw_b2)
    body = [html.Tbody([row1, row2])]
    return dmc.Table(header + body)


def return_IP_specific_observables_header():
    return [
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


def return_IP_specific_observables_values(dic_tw):
    l_rows = []
    for ip in [1, 2, 5, 8]:
        row_values = dic_tw[f"ip{ip}"]

        # Get html objects for crossing angle and crabs
        px_html = html.Td(f"{(row_values[3]*1e6):.3f}")
        py_html = html.Td(f"{(row_values[5]*1e6):.3f}")
        dx_zeta_html = html.Td(f"{(row_values[8]*1e6):.3f}")
        dy_zeta_html = html.Td(f"{(row_values[9]*1e6):.3f}")

        # Check if the optics is flat, and if yes ensure that the beta is large in the same plane
        # as the crossing angle
        if row_values[6] / row_values[7] > 1.2 or row_values[6] / row_values[7] < 0.8:
            large_plane = "x" if row_values[6] > row_values[7] else "y"
            large_angle = "x" if abs(row_values[3]) > abs(row_values[5]) else "y"
            if large_angle != large_plane:
                px_html = html.Td(
                    f"{(row_values[3]*1e6):.3f}", style={"color": "red", "font-weight": "bold"}
                )
                py_html = html.Td(
                    f"{(row_values[5]*1e6):.3f}", style={"color": "red", "font-weight": "bold"}
                )

            # Check if the crabs are on and if the large crab is in the same plane as the crossing angle
            crab_on = max(abs(row_values[8]), abs(row_values[9])) > 50 * 1e-6
            large_crab = "x" if abs(row_values[8]) > abs(row_values[9]) else "y"
            if crab_on and large_crab != large_plane:
                dx_zeta_html = html.Td(
                    f"{(row_values[8]*1e6):.3f}",
                    style={"color": "red", "font-weight": "bold"},
                )
                dy_zeta_html = html.Td(
                    f"{(row_values[9]*1e6):.3f}",
                    style={"color": "red", "font-weight": "bold"},
                )

        l_rows.append(
            html.Tr(
                [
                    html.Td(row_values[0]),
                    html.Td(f"{row_values[1]:.3f}"),
                    html.Td(f"{(row_values[2]*1e3):.3f}"),
                    px_html,
                    html.Td(f"{(row_values[4]*1e3):.3f}"),
                    py_html,
                    html.Td(f"{(row_values[6]*1e2):.3f}"),
                    html.Td(f"{(row_values[7]*1e2):.3f}"),
                    dx_zeta_html,
                    dy_zeta_html,
                    html.Td(f"{(row_values[10]*1e6):.3f}"),
                    html.Td(f"{(row_values[11]*1e6):.3f}"),
                ]
            )
        )

    return l_rows


def return_IP_specific_observables_layout(dic_tw_b1, dic_tw_b2):
    header = return_IP_specific_observables_header()
    body_1 = [html.Tbody(return_IP_specific_observables_values(dic_tw_b1))]
    body_2 = [html.Tbody(return_IP_specific_observables_values(dic_tw_b2))]
    table_1 = dmc.Table(header + body_1)
    table_2 = dmc.Table(header + body_2)
    return table_1, table_2


def return_luminosities_header():
    return [
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


def return_luminosities_values(l_lumi):
    return html.Tr(
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


def return_luminosities_layout(l_lumi):
    header = return_luminosities_header()
    row = return_luminosities_values(l_lumi)
    body = [html.Tbody([row])]
    return dmc.Table(header + body)


def compute_l_PU(l_lumi, array_b1, array_b2, cross_section_PU, dic_tw):
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
    n_turn_per_second = 1 / dic_tw["T_rev0"]
    return [
        lumi / n_col * cross_section_PU / n_turn_per_second
        for lumi, n_col in zip(l_lumi, l_n_collisions)
    ]


def return_PU_header():
    return [
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


def return_PU_values(l_PU):
    return html.Tr(
        [
            (
                html.Td(
                    f"{l_PU[0]:.1f}",
                    style={"font-weight": "bold", "color": "red"},
                )
                if l_PU[0] > 160
                else html.Td(f"{l_PU[0]:.1f}")
            ),
            html.Td(f"{l_PU[1]:.4f}"),
            (
                html.Td(
                    f"{l_PU[2]:.1f}",
                    style={"font-weight": "bold", "color": "red"},
                )
                if l_PU[2] > 160
                else html.Td(f"{l_PU[2]:.1f}")
            ),
            html.Td(f"{l_PU[3]:.4f}"),
        ]
    )


def return_PU_layout(l_PU):
    header = return_PU_header()
    row = return_PU_values(l_PU)
    body = [html.Tbody([row])]
    return dmc.Table(header + body)


def return_polarity_and_energy_header():
    return [
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


def return_polarity_and_energy_values(polarity_alice, polarity_lhcb, energy):
    return html.Tr(
        [
            html.Td(str(polarity_alice)),
            html.Td(str(polarity_lhcb)),
            html.Td(f"{energy:.5f}"),
        ]
    )


def return_power_and_energy_layout(polarity_alice, polarity_lhcb, energy):
    header = return_polarity_and_energy_header()
    row = return_polarity_and_energy_values(polarity_alice, polarity_lhcb, energy)
    body = [html.Tbody([row])]
    return dmc.Table(header + body)


def return_sanity_layout_tables(
    dic_tw_b1,
    dic_tw_b2,
    l_lumi,
    array_b1,
    array_b2,
    polarity_alice,
    polarity_lhcb,
    energy,
    cross_section_PU=81e-27,
):
    """
    Returns the tables for the sanity check page of the dashboard.

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

    # Get general observables (tune, chroma, etc.)
    table_1 = return_general_observables_layout(dic_tw_b1, dic_tw_b2)

    # Check IP-specific observables (crossing angle, beta functions, etc.)
    table_2, table_3 = return_IP_specific_observables_layout(dic_tw_b1, dic_tw_b2)

    # Luminosities and pile-up
    if l_lumi is not None:
        table_4 = return_luminosities_layout(l_lumi)

        # PU
        l_PU = compute_l_PU(l_lumi, array_b1, array_b2, cross_section_PU, dic_tw_b1)
        table_5 = return_PU_layout(l_PU)
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

    # Polarity and energy
    table_6 = return_power_and_energy_layout(polarity_alice, polarity_lhcb, energy)

    return table_1, table_2, table_3, table_4, table_5, table_6


def return_sanity_layout_global(
    dic_tw_b1,
    dic_tw_b2,
    l_lumi,
    array_b1,
    array_b2,
    polarity_alice,
    polarity_lhcb,
    energy,
    cross_section_PU=81e-27,
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
    table_1, table_2, table_3, table_4, table_5, table_6 = return_sanity_layout_tables(
        dic_tw_b1,
        dic_tw_b2,
        l_lumi,
        array_b1,
        array_b2,
        polarity_alice,
        polarity_lhcb,
        energy,
        cross_section_PU,
    )

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

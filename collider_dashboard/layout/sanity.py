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


def return_general_observables_values(dic_tw, beam_index):
    return html.Tr(
        [
            html.Td(f"{beam_index}"),
            html.Td(f'{dic_tw["qx"]:.5f}'),
            html.Td(f'{dic_tw["qy"]:.5f}'),
            html.Td(f'{dic_tw["dqx"]:.2f}'),
            html.Td(f'{dic_tw["dqy"]:.2f}'),
            html.Td(f'{dic_tw["c_minus"]:.4f}'),
        ]
    )


def return_general_observables_layout(dic_tw_b1, dic_tw_b2):
    header = return_general_observables_header()
    row1 = return_general_observables_values(dic_tw_b1, 1)
    row2 = return_general_observables_values(dic_tw_b2, 2)
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


def check_for_flat_optics(px, py, betx, bety, dx_zeta, dy_zeta):
    # Get html objects for crossing angle and crabs
    px_html = html.Td(f"{(px*1e6):.3f}")
    py_html = html.Td(f"{(py*1e6):.3f}")
    dx_zeta_html = html.Td(f"{(dx_zeta*1e6):.3f}")
    dy_zeta_html = html.Td(f"{(dy_zeta*1e6):.3f}")
    betx_html = html.Td(f"{(betx*1e2):.3f}")
    bety_html = html.Td(f"{(bety*1e2):.3f}")

    # Check if the optics is flat, and if yes ensure that the beta is large in the same plane
    # as the crossing angle
    if betx / bety > 1.2 or betx / bety < 0.8:
        large_plane = "x" if betx > bety else "y"
        large_angle = "x" if abs(px) > abs(py) else "y"
        if large_angle != large_plane:
            px_html = html.Td(f"{(px*1e6):.3f}", style={"color": "red", "font-weight": "bold"})
            py_html = html.Td(f"{(py*1e6):.3f}", style={"color": "red", "font-weight": "bold"})

        # Check if the crabs are on and if the large crab is in the same plane as the crossing angle
        crab_on = max(abs(dx_zeta), abs(dy_zeta)) > 50 * 1e-6
        large_crab = "x" if abs(dx_zeta) > abs(dy_zeta) else "y"
        if crab_on and large_crab != large_plane:
            dx_zeta_html = html.Td(
                f"{(dx_zeta*1e6):.3f}",
                style={"color": "red", "font-weight": "bold"},
            )
            dy_zeta_html = html.Td(
                f"{(dy_zeta*1e6):.3f}",
                style={"color": "red", "font-weight": "bold"},
            )

        if large_angle != large_plane or (crab_on and large_crab != large_plane):
            betx_html = html.Td(f"{(betx*1e2):.3f}", style={"color": "red", "font-weight": "bold"})
            bety_html = html.Td(f"{(bety*1e2):.3f}", style={"color": "red", "font-weight": "bold"})
    return px_html, py_html, dx_zeta_html, dy_zeta_html, betx_html, bety_html


def return_IP_specific_observables_values(dic_tw):
    l_rows = []
    for ip in [1, 2, 5, 8]:
        row_values = dic_tw[f"ip{ip}"]
        ip_val, s, x, px, y, py, betx, bety, dx_zeta, dy_zeta, dpx_zeta, dpy_zeta = row_values

        # Check if the optics is flat, and if yes ensure that the beta is large in the same plane
        # as the crossing angle
        px_html, py_html, dx_zeta_html, dy_zeta_html, betx_html, bety_html = check_for_flat_optics(
            px, py, betx, bety, dx_zeta, dy_zeta
        )

        l_rows.append(
            html.Tr(
                [
                    html.Td(ip_val),
                    html.Td(f"{s:.3f}"),
                    html.Td(f"{(x*1e3):.3f}"),
                    px_html,
                    html.Td(f"{(y*1e3):.3f}"),
                    py_html,
                    betx_html,
                    bety_html,
                    dx_zeta_html,
                    dy_zeta_html,
                    html.Td(f"{(dpx_zeta*1e6):.3f}"),
                    html.Td(f"{(dpy_zeta*1e6):.3f}"),
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


def return_final_sanity_layout(table_1, table_2, table_3, table_4, table_5, table_6):
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


def compute_alert_beta_beating(dic_with_bb, dic_without_bb):
    # Ensure the beta-beating is below 10%
    alert_hbeat = None
    alert_vbeat = None
    for ip in [1, 2, 5, 8]:
        betx_b1_with_bb = dic_with_bb["dic_tw_b1"][f"ip{ip}"][6]
        bety_b1_with_bb = dic_with_bb["dic_tw_b1"][f"ip{ip}"][7]
        betx_b1_without_bb = dic_without_bb["dic_tw_b1"][f"ip{ip}"][6]
        bety_b1_without_bb = dic_without_bb["dic_tw_b1"][f"ip{ip}"][7]

        if abs(betx_b1_with_bb - betx_b1_without_bb) / betx_b1_without_bb > 0.1:
            alert_hbeat = dmc.Text(
                "The horizontal relative beta-beating due to beam-beam is above 10% for beam 1 in at least one IP, please check the configuration",
                size="l",
                style={"margin": "auto"},
                color="tomato",
            )
        if abs(bety_b1_with_bb - bety_b1_without_bb) / bety_b1_without_bb > 0.1:
            alert_vbeat = dmc.Text(
                "The vertical relative beta-beating due to beam-beam is above 10% for beam 1 in at least one IP, please check the configuration",
                size="l",
                style={"margin": "auto"},
                color="tomato",
            )

    return alert_hbeat, alert_vbeat


def return_sanity_layout_global(
    dic_with_bb,
    dic_without_bb,
):
    # First ensure that the beta-beat is below 10%
    alert_hbeat, alert_vbeat = compute_alert_beta_beating(dic_with_bb, dic_without_bb)

    l_layout = []
    for dic in [dic_with_bb, dic_without_bb]:
        # Get tables
        table_1, table_2, table_3, table_4, table_5, table_6 = return_sanity_layout_tables(
            dic["dic_tw_b1"],
            dic["dic_tw_b2"],
            dic["l_lumi"],
            dic["array_b1"],
            dic["array_b2"],
            dic["polarity_alice"],
            dic["polarity_lhcb"],
            dic["energy"],
            dic["cross_section"],
        )

        # Return the final layout
        layout = return_final_sanity_layout(
            table_1,
            table_2,
            table_3,
            table_4,
            table_5,
            table_6,
        )

        # Add the alert at the top if needed
        if alert_hbeat is not None:
            layout.children.insert(0, alert_hbeat)
        if alert_vbeat is not None:
            layout.children.insert(0, alert_vbeat)

        # Add the layout to the list
        l_layout.append(layout)

    return l_layout

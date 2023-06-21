#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import html

import xtrack as xt


#################### Sanity checks Layout ####################


def return_sanity_layout(dic_tw_b1, dic_tw_b2, l_lumi):
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
                    html.Th("Momentum compaction factor"),
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
            html.Td(f'{dic_tw_b1["momentum_compaction_factor"]:.4f}'),
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
            html.Td(f'{dic_tw_b2["momentum_compaction_factor"]:.4f}'),
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
                    html.Th("x [m]"),
                    html.Th("px"),
                    html.Th("y [m]"),
                    html.Th("py"),
                    html.Th("betx [m]"),
                    html.Th("bety [m]"),
                ]
            )
        )
    ]
    l_rows_1 = []
    l_rows_2 = []
    for ip in [1, 2, 5, 8]:
        row_values_1 = dic_tw_b1.rows[f"ip{ip}"]
        row_values_2 = dic_tw_b2.rows[f"ip{ip}"]

        l_rows_1.append(
            html.Tr(
                [
                    html.Td(row_values_1[0]),
                    html.Td(f"{row_values_1[1]:.3f}"),
                    html.Td(f"{row_values_1[2]:.4e}"),
                    html.Td(f"{row_values_1[3]:.4e}"),
                    html.Td(f"{row_values_1[4]:.4e}"),
                    html.Td(f"{row_values_1[5]:.4e}"),
                    html.Td(f"{row_values_1[6]:.4e}"),
                    html.Td(f"{row_values_1[7]:.4e}"),
                ]
            )
        )

        l_rows_2.append(
            html.Tr(
                [
                    html.Td(row_values_2[0]),
                    html.Td(f"{row_values_2[1]:.3f}"),
                    html.Td(f"{row_values_2[2]:.4e}"),
                    html.Td(f"{row_values_2[3]:.4e}"),
                    html.Td(f"{row_values_2[4]:.4e}"),
                    html.Td(f"{row_values_2[5]:.4e}"),
                    html.Td(f"{row_values_2[6]:.4e}"),
                    html.Td(f"{row_values_2[7]:.4e}"),
                ]
            )
        )

    body_2 = [html.Tbody(l_rows_1)]
    body_3 = [html.Tbody(l_rows_2)]
    table_2 = dmc.Table(header_2 + body_2)
    table_3 = dmc.Table(header_2 + body_3)

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
            html.Td(f"{l_lumi[0]:.3e}"),
            html.Td(f"{l_lumi[1]:.3e}"),
            html.Td(f"{l_lumi[2]:.3e}"),
            html.Td(f"{l_lumi[3]:.3e}"),
        ]
    )
    body_4 = [html.Tbody([row_lumi])]
    table_4 = dmc.Table(header_3 + body_4)

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
            dmc.Group(
                children=[
                    dmc.Text("Luminosities", size="xl", style={"margin": "auto"}),
                    table_4,
                ],
                mb=10,
                style={"width": "100%"},
            ),
        ],
        style={"width": "90%", "margin": "auto"},
    )

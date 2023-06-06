#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import html

import xtrack as xt



#################### Sanity checks Layout ####################

def return_sanity_layout(tw_b1, tw_b2, l_ncollisions, num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z):
    
    # Get number of collisions per IP
    n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8 = l_ncollisions
    
    
    # Check general observables (tune, chroma, etc.)
    header_1 = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Beam"),
                    html.Th("Tune"),
                    html.Th("Chromaticity"),
                    html.Th("Linear coupling"),
                    html.Th("Momentum compaction factor"),
                ]
            )
        )
    ]

    row1 = html.Tr(
        [
            html.Td("1"),
            html.Td(f'{tw_b1["qx"]:.5f}'),
            html.Td(f'{tw_b1["dqx"]:.2f}'),
            html.Td(f'{tw_b1["c_minus"]:.2f}'),
            html.Td(f'{tw_b1["momentum_compaction_factor"]:.2f}'),
        ]
    )
    row2 = html.Tr(
        [
            html.Td("2"),
            html.Td(f'{tw_b2["qx"]:.5f}'),
            html.Td(f'{tw_b2["dqx"]:.2f}'),
            html.Td(f'{tw_b2["c_minus"]:.2f}'),
            html.Td(f'{tw_b2["momentum_compaction_factor"]:.2f}'),
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
                    html.Th("s"),
                    html.Th("x"),
                    html.Th("px"),
                    html.Th("y"),
                    html.Th("py"),
                    html.Th("betx"),
                    html.Th("bety"),
                ]
            )
        )
    ]
    l_rows_1 = []
    l_rows_2 = []
    for ip in [1, 2, 5, 8]:
        row_values_1 = (
            tw_b1.rows[f"ip{ip}"]
            .cols["s", "x", "px", "y", "py", "betx", "bety"]
            .to_pandas()
            .to_numpy()
            .squeeze()
        )

        row_values_2 = (
            tw_b2.rows[f"ip{ip}"]
            .cols["s", "x", "px", "y", "py", "betx", "bety"]
            .to_pandas()
            .to_numpy()
            .squeeze()
        )

        l_rows_1.append(
            html.Tr(
                [
                    html.Td(row_values_1[0]),
                    html.Td(f"{row_values_1[1]:.3f}"),
                    html.Td(f"{row_values_1[2]:.4f}"),
                    html.Td(f"{row_values_1[3]:.5f}"),
                    html.Td(f"{row_values_1[4]:.4f}"),
                    html.Td(f"{row_values_1[5]:.5f}"),
                    html.Td(f"{row_values_1[6]:.3f}"),
                    html.Td(f"{row_values_1[7]:.3f}"),
                ]
            )
        )

        l_rows_2.append(
            html.Tr(
                [
                    html.Td(row_values_2[0]),
                    html.Td(f"{row_values_2[1]:.3f}"),
                    html.Td(f"{row_values_2[2]:.4f}"),
                    html.Td(f"{row_values_2[3]:.5f}"),
                    html.Td(f"{row_values_2[4]:.4f}"),
                    html.Td(f"{row_values_2[5]:.5f}"),
                    html.Td(f"{row_values_2[6]:.3f}"),
                    html.Td(f"{row_values_2[7]:.3f}"),
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
                    html.Th("IP 1"),
                    html.Th("IP 2"),
                    html.Th("IP 5"),
                    html.Th("IP 8"),
                ]
            )
        )
    ]

    l_lumi = []
    for ip, n_col in zip(
        [1, 2, 5, 8],
        [n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip1_and_5, n_collisions_ip8],
    ):
        l_lumi.append(
            xt.lumi.luminosity_from_twiss(
                n_colliding_bunches=n_col,
                num_particles_per_bunch=num_particles_per_bunch,
                ip_name="ip" + str(ip),
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                sigma_z=sigma_z,
                twiss_b1=tw_b1,
                twiss_b2=tw_b2,
                crab=False,
            )
        )
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


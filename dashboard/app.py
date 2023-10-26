# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import from standard library
import logging

# Import third-party packages
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update

# Import initialization and plotting functions
from .backend import init, plot

# Import layout functions
from .layout.configuration import return_configuration_layout
from .layout.filling import return_filling_scheme_layout
from .layout.footprint import return_footprint_layout
from .layout.header import return_header_layout
from .layout.optics import return_optics_layout
from .layout.sanity import return_sanity_layout
from .layout.separation import return_separation_layout
from .layout.separation_3D import return_3D_separation_layout
from .layout.survey import return_survey_layout
from .layout.tables import return_tables_layout

#################### Load global variables ####################
logging.info("Loading global variables")
# Load dashboard variables
path_config = None
path_collider = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/test_dump/base_collider/xtrack_0000/collider.json"
path_job = path_collider.split("/final_collider.json")[0]
dic_without_bb, dic_with_bb, initial_pickle_path = init.init_from_collider(
    path_collider, load_global_variables_from_pickle=False
)

#################### App ####################
logging.info("Defining app")
app = Dash(
    __name__,
    title="Dashboard for current simulation",
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
    ],
    # suppress_callback_exceptions=True,
)
server = app.server

# ==================================================================================================
# --- App layout
# ==================================================================================================
logging.info("Defining app layout")


def build_layout():
    layout = html.Div(
        style={"width": "90%", "margin": "auto"},
        children=[
            dcc.Location(id="url", refresh=True),
            return_header_layout(),
            dmc.Center(
                children=[
                    html.Div(
                        id="main-div",
                        style={"width": "100%", "margin": "auto"},
                        children=[
                            html.Div(id="placeholder-tabs"),
                        ],
                    ),
                ],
                style={"margin-top": "80px"},
            ),
        ],
    )

    # Dark theme
    layout = dmc.MantineProvider(
        withGlobalStyles=True,
        theme={"colorScheme": "dark"},
        children=layout,
    )
    return layout


app.layout = build_layout()


# ==================================================================================================
# --- App callbacks
# ==================================================================================================


@app.callback(
    Output("group-collider-dropdown", "style"),
    Input("main-div", "style"),
)
def update_preloaded_collider_value_at_launch_style(_):
    if not ACTIVATE_COLLIDER_DROPDOWN:
        return {"display": "none"}
    return no_update


@app.callback(Output("placeholder-tabs", "children"), Input("tab-titles", "value"))
def select_tab(value):
    match value:
        case "display-configuration":
            return return_configuration_layout(
                dic_with_bb["configuration_str"], path_job
            )
        case "display-twiss":
            return return_tables_layout()
        case "display-scheme":
            return return_filling_scheme_layout()
        case "display-separation":
            return return_separation_layout()
        case "display-3D-separation":
            return return_3D_separation_layout()
        case "display-footprint":
            return return_footprint_layout()
        case "display-sanity":
            sanity_after_beam_beam = return_sanity_layout(
                dic_with_bb["dic_tw_b1"],
                dic_with_bb["dic_tw_b2"],
                dic_with_bb["l_lumi"],
                dic_with_bb["array_b1"],
                dic_with_bb["array_b2"],
                dic_with_bb["polarity_alice"],
                dic_with_bb["polarity_lhcb"],
                dic_with_bb["energy"],
            )
            sanity_before_beam_beam = return_sanity_layout(
                dic_without_bb["dic_tw_b1"],
                dic_without_bb["dic_tw_b2"],
                dic_without_bb["l_lumi"],
                dic_without_bb["array_b1"],
                dic_without_bb["array_b2"],
                dic_without_bb["polarity_alice"],
                dic_without_bb["polarity_lhcb"],
                dic_with_bb["energy"],
            )
            tabs_sanity = dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.Tab(
                                "Before beam-beam",
                                value="sanity-before-beam-beam",
                                style={"font-size": "1.1rem"},
                            ),
                            dmc.Tab(
                                "After beam beam",
                                value="sanity-after-beam-beam",
                                style={"font-size": "1.1rem"},
                            ),
                        ],
                        position="center",
                    ),
                    dmc.TabsPanel(
                        sanity_before_beam_beam, value="sanity-before-beam-beam"
                    ),
                    dmc.TabsPanel(
                        sanity_after_beam_beam, value="sanity-after-beam-beam"
                    ),
                ],
                color="cyan",
                value="sanity-after-beam-beam",
            )
            return tabs_sanity

        case "display-optics":
            return return_optics_layout(dic_with_bb)
        case "display-survey":
            return return_survey_layout()
        case _:
            return return_configuration_layout(
                dic_with_bb["configuration_str"], path_job
            )


@app.callback(
    Output("placeholder-data-table", "children"), Input("segmented-data-table", "value")
)
def select_data_table(value):
    match value:
        case "Twiss table beam 1":
            return dic_with_bb["table_tw_b1"]
        case "Survey table beam 1":
            return dic_with_bb["table_sv_b1"]
        case "Twiss table beam 2":
            return dic_with_bb["table_tw_b2"]
        case "Survey table beam 2":
            return dic_with_bb["table_sv_b2"]
        case _:
            return dic_with_bb["table_tw_b1"]


@app.callback(
    Output("LHC-layout", "figure"),
    Input("chips-ip", "value"),
)
def update_graph_LHC_layout(l_values):
    l_indices_to_keep = []
    for val in l_values:
        str_ind_1, str_ind_2 = val.split("-")
        # Get indices of elements to keep (# ! implemented only for beam 1)
        l_indices_to_keep.extend(
            plot.get_indices_of_interest(
                dic_with_bb["df_tw_b1"], "ip" + str_ind_1, "ip" + str_ind_2
            )
        )

    fig = plot.return_plot_lattice_with_tracking(
        dic_with_bb["df_sv_b1"],
        dic_with_bb["df_elements_corrected"],
        dic_with_bb["df_tw_b1"],
        df_sv_2=dic_with_bb["df_sv_b2"],
        df_tw_2=dic_with_bb["df_tw_b2"],
        l_indices_to_keep=l_indices_to_keep,
    )

    return fig


@app.callback(
    Output("filling-scheme-graph", "figure"),
    Output("filling-scheme-graph", "style"),
    Output("filling-scheme-alert", "style"),
    Input("tab-titles", "value"),
)
def update_graph_filling(value):
    if value == "display-scheme":
        if dic_with_bb["array_b1"] is not None:
            return (
                plot.return_plot_filling_scheme(
                    dic_with_bb["array_b1"],
                    dic_with_bb["array_b2"],
                    dic_with_bb["i_bunch_b1"],
                    dic_with_bb["i_bunch_b2"],
                    dic_with_bb["bbs"],
                ),
                {"height": "90vh", "width": "100%", "margin": "auto"},
                {"margin": "auto", "display": "none"},
            )

        else:
            return (
                go.Figure(),
                {
                    "height": "90vh",
                    "width": "100%",
                    "margin": "auto",
                    "display": "none",
                },
                {"margin": "auto"},
            )

    else:
        return no_update


@app.callback(
    Output("LHC-2D-near-IP", "figure"),
    Input("tab-titles", "value"),
    Input("vertical-zoom-optics", "value"),
)
def update_graph_optics(tab_value, zoom_value):
    if tab_value == "display-optics":
        fig = plot.return_plot_optics(
            dic_with_bb["df_tw_b1"],
            dic_with_bb["df_tw_b2"],
            dic_with_bb["df_sv_b1"],
            dic_with_bb["df_elements_corrected"],
        )

        factor = 2**-zoom_value
        fig.update_yaxes(
            title_text=r"$\beta_{x,y}[m]$", range=[0, 10000 * factor * 2], row=2, col=1
        )
        fig.update_yaxes(
            title_text=r"(Closed orbit)$_{x,y}[m]$",
            range=[-0.03 * factor, 0.03 * factor],
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title_text=r"$D_{x,y}[m]$", range=[-3 * factor, 3 * factor], row=4, col=1
        )

        return fig
    else:
        return no_update


@app.callback(
    Output("beam-separation", "figure"),
    Input("chips-sep", "value"),
    Input("chips-sep-bb", "value"),
)
def update_graph_separation(value, bb):
    if bb == "On":
        dic = dic_with_bb
    elif bb == "Off":
        dic = dic_without_bb
    else:
        raise ValueError("bb should be either On or Off")

    if value == "v" or value == "h":
        fig = plot.return_plot_separation(
            dic["dic_separation_ip"], "x" if value == "h" else "y"
        )
    elif value == "||v+h||":
        fig = plot.return_plot_separation(dic["dic_separation_ip"], "xy")
    else:
        raise ValueError("value should be either v, h or ||v+h||")
    return fig


@app.callback(
    Output("beam-separation-3D", "figure"),
    Input("chips-sep-bb-3D", "value"),
)
def update_graph_separation_3D(bb):
    if bb == "On":
        dic = dic_with_bb
    elif bb == "Off":
        dic = dic_without_bb
    else:
        raise ValueError("bb should be either On or Off")

    fig = plot.return_plot_separation_3D(dic["dic_position_ip"])

    return fig


@app.callback(
    Output("footprint-without-bb-b1", "figure"),
    Output("footprint-without-bb-b2", "figure"),
    Output("footprint-with-bb-b1", "figure"),
    Output("footprint-with-bb-b2", "figure"),
    Input("tab-titles", "value"),
)
def update_graph_footprint(value):
    if value == "display-footprint":
        if dic_without_bb["i_bunch_b1"] is not None:
            title_without_bb_b1 = (
                "Tune footprint without beam-beam effects for beam 1 and bunch "
                + str(dic_without_bb["i_bunch_b1"])
            )
            title_without_bb_b2 = (
                "Tune footprint without beam-beam effects for beam 2 and bunch "
                + str(dic_without_bb["i_bunch_b2"])
            )
            title_with_bb_b1 = (
                "Tune footprint with beam-beam effects for beam 1 and bunch "
                + str(dic_with_bb["i_bunch_b1"])
            )
            title_with_bb_b2 = (
                "Tune footprint with beam-beam effects for beam 2 and bunch "
                + str(dic_with_bb["i_bunch_b2"])
            )
        else:
            title_without_bb_b1 = "Tune footprint without beam-beam effects for beam 1 (bunch number unknown)"
            title_without_bb_b2 = "Tune footprint without beam-beam effects for beam 2 (bunch number unknown)"
            title_with_bb_b1 = "Tune footprint with beam-beam effects for beam 1 (bunch number unknown)"
            title_with_bb_b2 = "Tune footprint with beam-beam effects for beam 2 (bunch number unknown)"

        return [
            plot.return_plot_footprint(
                dic_without_bb["footprint_b1"],
                dic_without_bb["dic_tw_b1"]["qx"],
                dic_without_bb["dic_tw_b1"]["qy"],
                title=title_without_bb_b1,
            ),
            plot.return_plot_footprint(
                dic_without_bb["footprint_b2"],
                dic_without_bb["dic_tw_b2"]["qx"],
                dic_without_bb["dic_tw_b2"]["qy"],
                title=title_without_bb_b2,
            ),
            plot.return_plot_footprint(
                dic_with_bb["footprint_b1"],
                dic_without_bb["dic_tw_b1"]["qx"],
                dic_without_bb["dic_tw_b1"]["qy"],
                title=title_with_bb_b1,
            ),
            plot.return_plot_footprint(
                dic_with_bb["footprint_b2"],
                dic_without_bb["dic_tw_b2"]["qx"],
                dic_without_bb["dic_tw_b2"]["qy"],
                title=title_with_bb_b2,
            ),
        ]
    else:
        return no_update


#################### Launch app ####################
# if __name__ == "__main__":
#     logging.info("Launching app")
#     app.run_server(debug=False, host="0.0.0.0", port=8084)


# Run with gunicorn dashboard:server -b :8000
# Run silently with nohup gunicorn dashboard:server -b :8080 &
# Kill with pkill gunicorn

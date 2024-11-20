"""Module containing the callbacks for the app."""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import third-party packages
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, no_update

# Import initialization and plotting functions
from .backend import plot

# Import layout functions
from .layout import (
    return_3D_separation_layout,
    return_configuration_layout,
    return_filling_scheme_layout,
    return_footprint_layout,
    return_knobs_layout,
    return_optics_layout,
    return_sanity_layout_global,
    return_separation_layout,
    return_survey_layout,
    return_tables_layout,
)


# ==================================================================================================
# --- Functions used by callbacks
# ==================================================================================================
def return_tabs_sanity(dic_with_bb, dic_without_bb):
    sanity_after_beam_beam, sanity_before_beam_beam = return_sanity_layout_global(
        dic_with_bb, dic_without_bb
    )

    return dmc.Tabs(
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
            dmc.TabsPanel(sanity_before_beam_beam, value="sanity-before-beam-beam"),
            dmc.TabsPanel(sanity_after_beam_beam, value="sanity-after-beam-beam"),
        ],
        color="cyan",
        value="sanity-after-beam-beam",
    )


# ==================================================================================================
# --- App callbacks
# ==================================================================================================


def all_callbacks(app, dic_with_bb, dic_without_bb, path_collider):
    @app.callback(Output("placeholder-tabs", "children"), Input("tab-titles", "value"))
    def select_tab(value):
        match value:
            case "display-configuration":
                return return_configuration_layout(dic_with_bb["configuration_str"], path_collider)
            case "display-knobs":
                return return_knobs_layout(dic_with_bb["dic_knob_str"])
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
                return return_tabs_sanity(dic_with_bb, dic_without_bb)
            case "display-optics":
                return return_optics_layout(dic_with_bb)
            case "display-survey":
                return return_survey_layout()
            case _:
                return return_configuration_layout(dic_with_bb["configuration_str"], path_collider)

    @app.callback(
            Output("prism-knob_info", "children"),
            Input("select-knob_info", "value"),
        )
    def select_knob_info(knob):
        return dic_with_bb["dic_knob_str"][knob] if knob is not None else no_update

    @app.callback(
        Output("placeholder-data-table", "children"),
        Input("segmented-data-table", "value"),
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
                    dic_with_bb["df_tw_b1"], f"ip{str_ind_1}", f"ip{str_ind_2}"
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
                title_text=r"$\beta_{x,y}[m]$",
                range=[0, 10000 * factor * 2],
                row=2,
                col=1,
            )
            fig.update_yaxes(
                title_text=r"(Closed orbit)$_{x,y}[m]$",
                range=[-0.03 * factor, 0.03 * factor],
                row=3,
                col=1,
            )
            fig.update_yaxes(
                title_text=r"$D_{x,y}[m]$",
                range=[-3 * factor, 3 * factor],
                row=4,
                col=1,
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

        if value in ["v", "h"]:
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
        Input("chips-ip-3D", "value"),
    )
    def update_graph_separation_3D(bb, ip):
        if bb == "On":
            dic = dic_with_bb
        elif bb == "Off":
            dic = dic_without_bb
        else:
            raise ValueError("bb should be either On or Off")

        fig = plot.return_plot_separation_3D(dic["dic_position_ip"], ip.lower().replace(" ", ""))

        return fig

    @app.callback(
            Output("footprint-without-bb-b1", "figure"),
            Output("footprint-without-bb-b2", "figure"),
            Output("footprint-with-bb-b1", "figure"),
            Output("footprint-with-bb-b2", "figure"),
            Input("tab-titles", "value"),
        )
    def update_graph_footprint(value):
        if value != "display-footprint":
            return no_update
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
            title_without_bb_b1 = (
                "Tune footprint without beam-beam effects for beam 1 (bunch number unknown)"
            )
            title_without_bb_b2 = (
                "Tune footprint without beam-beam effects for beam 2 (bunch number unknown)"
            )
            title_with_bb_b1 = (
                "Tune footprint with beam-beam effects for beam 1 (bunch number unknown)"
            )
            title_with_bb_b2 = (
                "Tune footprint with beam-beam effects for beam 2 (bunch number unknown)"
            )

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

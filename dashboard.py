#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash import Dash, html, Input, Output
import sys

# Import initialization and plotting functions
import init
import plot

# Import layout functions
from layout.configuration import return_configuration_layout
from layout.filling import return_filling_scheme_layout
from layout.optics import return_optics_layout
from layout.sanity import return_sanity_layout
from layout.survey import return_survey_layout
from layout.header import return_header_layout
from layout.tables import return_tables_layout
from layout.separation import return_separation_layout


#################### Load global variables ####################

# Get the path to the config file from the command line, or use the default one
if len(sys.argv) > 1:
    path_config = sys.argv[1]
else:
    # path_config = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/2024_test/base_collider/xtrack_0004/config.yaml"
    path_config = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma15_eol_bbb_2228/base_collider/xtrack_0000/config.yaml"

dic_before_bb, dic_after_bb = init.init(path_config, build_collider=False, load_from_pickle=True)

#################### App ####################
app = Dash(
    __name__,
    title="Dashboard for current simulation",
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
    ],
    # suppress_callback_exceptions=True,
)
server = app.server

#################### App Layout ####################

layout = html.Div(
    style={"width": "90%", "margin": "auto"},
    children=[
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
            style={"margin-top": "75px"},
        ),
    ],
)

# Dark theme
layout = dmc.MantineProvider(
    withGlobalStyles=True,
    theme={"colorScheme": "dark"},
    children=layout,
)

app.layout = layout


#################### App Callbacks ####################


@app.callback(Output("placeholder-tabs", "children"), Input("tab-titles", "value"))
def select_tab(value):
    match value:
        case "display-configuration":
            return return_configuration_layout(path_config)
        case "display-twiss":
            return return_tables_layout()
        case "display-scheme":
            return return_filling_scheme_layout(
                dic_after_bb["array_b1"],
                dic_after_bb["array_b2"],
                dic_after_bb["i_bunch_b1"],
                dic_after_bb["i_bunch_b2"],
                dic_after_bb["bbs"],
            )
        case "display-separation":
            return return_separation_layout(dic_before_bb["dic_sep_IPs"]["v"])
        case "display-sanity":
            sanity_after_beam_beam = return_sanity_layout(
                dic_after_bb["dic_tw_b1"],
                dic_after_bb["dic_tw_b2"],
                dic_after_bb["l_lumi"],
                dic_after_bb["array_b1"],
                dic_after_bb["array_b2"],
            )
            sanity_before_beam_beam = return_sanity_layout(
                dic_before_bb["dic_tw_b1"],
                dic_before_bb["dic_tw_b2"],
                dic_before_bb["l_lumi"],
                dic_before_bb["array_b1"],
                dic_before_bb["array_b2"],
            )
            tabs_sanity = dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.Tab("Before beam-beam", value="sanity-before-beam-beam"),
                            dmc.Tab("After beam beam", value="sanity-after-beam-beam"),
                        ],
                        position="center",
                    ),
                    dmc.TabsPanel(sanity_before_beam_beam, value="sanity-before-beam-beam"),
                    dmc.TabsPanel(sanity_after_beam_beam, value="sanity-after-beam-beam"),
                ],
                color="cyan",
                value="sanity-after-beam-beam",
            )
            return tabs_sanity

        case "display-optics":
            return return_optics_layout(
                dic_after_bb["df_tw_b1"],
                dic_after_bb["df_tw_b2"],
                dic_after_bb["df_sv_b1"],
                dic_after_bb["df_elements_corrected"],
            )
        case "display-survey":
            return return_survey_layout()
        case _:
            return return_configuration_layout(path_config)


@app.callback(Output("placeholder-data-table", "children"), Input("segmented-data-table", "value"))
def select_data_table(value):
    match value:
        case "Twiss table beam 1":
            return dic_after_bb["table_tw_b1"]
        case "Survey table beam 1":
            return dic_after_bb["table_sv_b1"]
        case "Twiss table beam 2":
            return dic_after_bb["table_tw_b2"]
        case "Survey table beam 2":
            return dic_after_bb["table_sv_b2"]
        case _:
            return dic_after_bb["table_tw_b1"]


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
                dic_after_bb["df_tw_b1"], "ip" + str_ind_1, "ip" + str_ind_2
            )
        )

    fig = plot.return_plot_lattice_with_tracking(
        dic_after_bb["df_sv_b1"],
        dic_after_bb["df_elements_corrected"],
        dic_after_bb["df_tw_b1"],
        df_sv_2=dic_after_bb["df_sv_b2"],
        df_tw_2=dic_after_bb["df_tw_b2"],
        l_indices_to_keep=l_indices_to_keep,
    )

    return fig


@app.callback(
    Output("beam-separation", "figure"),
    Input("chips-sep", "value"),
)
def update_graph_LHC_layout(value):
    fig = plot.return_plot_separation(dic_before_bb["dic_sep_IPs"][value])
    return fig


# ! Uncomment this function once I find out how to store collider elements
# @app.callback(
#     Output("text-element", "children"),
#     Output("title-element", "children"),
#     Output("type-element", "children"),
#     Output("drawer-magnets", "opened"),
#     Input("LHC-layout", "clickData"),
#     prevent_initial_call=False,
# )
# def update_text_graph_LHC_2D(clickData):
#     if clickData is not None:
#         if "customdata" in clickData["points"][0]:
#             name = clickData["points"][0]["customdata"]
#             if name.startswith("mb"):
#                 type_text = "Dipole"
#                 try:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name]
#                         .knl[0]
#                         ._expr._get_dependencies()
#                     )
#                 except:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name + "..1"]
#                         .knl[0]
#                         ._expr._get_dependencies()
#                     )
#             elif name.startswith("mq"):
#                 type_text = "Quadrupole"
#                 try:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name]
#                         .knl[1]
#                         ._expr._get_dependencies()
#                     )
#                 except:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name + "..1"]
#                         .knl[1]
#                         ._expr._get_dependencies()
#                     )
#             elif name.startswith("ms"):
#                 type_text = "Sextupole"
#                 try:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name]
#                         .knl[2]
#                         ._expr._get_dependencies()
#                     )
#                 except:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name + "..1"]
#                         .knl[2]
#                         ._expr._get_dependencies()
#                     )
#             elif name.startswith("mo"):
#                 type_text = "Octupole"
#                 try:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name]
#                         .knl[3]
#                         ._expr._get_dependencies()
#                     )
#                 except:
#                     set_var = (
#                         collider_after_beam_beam.lhcb1.element_refs[name + "..1"]
#                         .knl[3]
#                         ._expr._get_dependencies()
#                     )

#             text = []
#             for var in set_var:
#                 name_var = str(var).split("'")[1]
#                 val = collider_after_beam_beam.lhcb1.vars[name_var]._get_value()
#                 expr = collider_after_beam_beam.lhcb1.vars[name_var]._expr
#                 if expr is not None:
#                     dependencies = collider_after_beam_beam.lhcb1.vars[
#                         name_var
#                     ]._expr._get_dependencies()
#                 else:
#                     dependencies = "No dependencies"
#                     expr = "No expression"
#                 targets = collider_after_beam_beam.lhcb1.vars[name_var]._find_dependant_targets()

#                 text.append(dmc.Text("Name: ", weight=500))
#                 text.append(dmc.Text(name_var, size="sm"))
#                 text.append(dmc.Text("Element value: ", weight=500))
#                 text.append(dmc.Text(str(val), size="sm"))
#                 text.append(dmc.Text("Expression: ", weight=500))
#                 text.append(dmc.Text(str(expr), size="sm"))
#                 text.append(dmc.Text("Dependencies: ", weight=500))
#                 text.append(dmc.Text(str(dependencies), size="sm"))
#                 text.append(dmc.Text("Targets: ", weight=500))
#                 if len(targets) > 10:
#                     text.append(
#                         dmc.Text(str(targets[:10]), size="sm"),
#                     )
#                     text.append(dmc.Text("...", size="sm"))
#                 else:
#                     text.append(dmc.Text(str(targets), size="sm"))

#             return text, name, type_text, True

#     return (
#         dmc.Text("Please click on a multipole to get the corresponding knob information."),
#         dmc.Text("Click !"),
#         dmc.Text("Undefined type"),
#         False,
#     )


#################### Launch app ####################
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)


# Run with gunicorn app:server -b :8000
# Run silently with nohup gunicorn app:server -b :8000 &
# Kill with pkill gunicorn

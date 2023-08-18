# ! This class needs to be cleaned up and documented !
# ==================================================================================================
# --- Imports
# ==================================================================================================import numpy as np
import pandas as pd
import xtrack as xt
import numpy as np
from dash import dash_table
from dash.dash_table.Format import Format, Scheme
import pickle
import os
import copy
import logging
import json

# Module to compute beam-beam schedule
import fillingpatterns as fp

# Import collider and twiss functions
from modules.collider_check.collider_check import ColliderCheck
from modules.build_collider.build_collider import BuildCollider

# ==================================================================================================
# --- Functions initialize all global variables
# ==================================================================================================

"""Thid module initializes all global variables from a collider json, potentially embedding a 
configuration file.
"""


def init_from_collider(path_collider, load_global_variables_from_pickle=False):
    """Initialize the app variables from a given collider json file. All features related to the
    configuration will be deactivated."""

    # Path to the pickle dictionnaries (for loading and saving)
    path_pickle = "temp/" + path_collider.replace("/", "_") + "t_dic_var.pkl"

    # Try to load the dictionnaries of variables from pickle
    if load_global_variables_from_pickle:
        # Check that the pickle file exists
        if not os.path.isfile(path_pickle):
            raise ValueError("The pickle file does not exist.")
        with open(path_pickle, "rb") as f:
            dic_without_bb, dic_with_bb = pickle.load(f)
        print("Returning global variables from pickle file.")
        return dic_without_bb, dic_with_bb, path_pickle

    else:
        # Rebuild collider
        # ! This should be updated when metadata is hanlded better
        with open(path_collider, "r") as fid:
            collider_dict = json.load(fid)
        if "config_yaml" in collider_dict:
            print("A configuration has been found in the collider file. Using it.")
            config = collider_dict["config_yaml"]["config_collider"]
        else:
            print(
                "Warning, you provided a collider file without a configuration. Some features of"
                " the dashboard will be missing."
            )
            config = None
        collider = xt.Multiline.from_dict(collider_dict)
        collider.build_trackers()

        # Build collider before bb
        collider_without_bb = xt.Multiline.from_dict(collider_dict)
        collider_without_bb.build_trackers()
        collider_without_bb.vars["beambeam_scale"] = 0

        # ! This should be updated when metadata is hanlded better
        # Add configuration to collider as metadata
        if config is not None and not hasattr(collider, "metadata"):
            collider.metadata = config
            collider_without_bb.metadata = config

        # Compute collider checks
        collider_check_with_bb = ColliderCheck(collider)
        collider_check_without_bb = ColliderCheck(collider_without_bb)

        # Compute global variables
        dic_without_bb, dic_with_bb = compute_global_variables_from_collider_checks(
            collider_check_with_bb,
            collider_check_without_bb,
            path_pickle=path_pickle,
        )

        return dic_without_bb, dic_with_bb, path_pickle


def compute_global_variables_from_collider_checks(
    collider_check_after_beam_beam, collider_check_without_beam_beam, path_pickle=None
):
    # Get the global variables before and after the beam-beam
    dic_with_bb = initialize_global_variables(
        collider_check_after_beam_beam, compute_footprint=True
    )
    dic_without_bb = initialize_global_variables(
        collider_check_without_beam_beam, compute_footprint=True
    )

    if path_pickle is not None:
        # Dump the dictionnaries in a pickle file
        print("Dumping global variables in a pickle file.")
        with open(path_pickle, "wb") as f:
            pickle.dump((dic_without_bb, dic_with_bb), f)

    return dic_without_bb, dic_with_bb


def initialize_global_variables(collider_check, compute_footprint=True):
    """Initialize global variables from a collider check object."""

    # Get luminosity at each IP
    if collider_check.configuration is not None:
        l_lumi = [collider_check.return_luminosity(IP=x) for x in [1, 2, 5, 8]]

        # Get the beams schemes
        array_b1 = collider_check.array_b1
        array_b2 = collider_check.array_b2

        # Get the bunches selected for tracking
        i_bunch_b1 = collider_check.i_bunch_b1
        i_bunch_b2 = collider_check.i_bunch_b2

        # Get emittances
        nemitt_x = collider_check.nemitt_x
        nemitt_y = collider_check.nemitt_y

        # Get the beam-beam schedule
        patt = fp.FillingPattern.from_json(collider_check.path_filling_scheme)
        patt.compute_beam_beam_schedule(n_lr_per_side=26)
        bbs = patt.b1.bb_schedule

        # Get polarity Alice and LHCb
        polarity_alice, polarity_lhcb = collider_check.return_polarity_ip_2_8()

        # Get configuration
        configuration_str = collider_check.configuration_str

    else:
        l_lumi = None
        array_b1 = None
        array_b2 = None
        i_bunch_b1 = None
        i_bunch_b2 = None
        bbs = None
        polarity_alice = None
        polarity_lhcb = None
        configuration_str = None
        # Get emittance for the computation of the normalized separation
        logging.warning("No configuration file provided, using default values for emittances.")
        nemitt_x = 2.2e-6
        nemitt_y = 2.2e-6

    # Get elements of the line (only done for b1, should be identical for b2)
    df_elements = return_dataframe_elements_from_line(collider_check.collider.lhcb1)

    # Get twiss and survey for both lines
    tw_b1, df_sv_b1, df_tw_b1 = (
        collider_check.tw_b1,
        collider_check.df_sv_b1,
        collider_check.df_tw_b1,
    )

    tw_b2, sv_b2 = collider_check.tw_b2.reverse(), collider_check.sv_b2.reverse()
    df_tw_b2, df_sv_b2 = tw_b2.to_pandas(), sv_b2.to_pandas()

    # Correct df elements for thin lens approximation
    df_elements_corrected = return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw_b1)

    # Get corresponding data tables
    table_sv_b1 = return_data_table(df_sv_b1, "id-df-sv-b1-after-bb", twiss=False)
    table_tw_b1 = return_data_table(df_tw_b1, "id-df-tw-b1-after-bb", twiss=True)
    table_sv_b2 = return_data_table(df_sv_b2, "id-df-sv-b2-after-bb", twiss=False)
    table_tw_b2 = return_data_table(df_tw_b2, "id-df-tw-b2-after-bb", twiss=True)

    # Get the twiss dictionnary (tune, chroma, etc + twiss at IPs)
    dic_tw_b1 = return_twiss_dic(tw_b1)
    dic_tw_b2 = return_twiss_dic(tw_b2)

    # Get the dictionnary to plot separation
    dic_separation_ip = {
        f"ip{ip}": collider_check.compute_separation_variables(ip=f"ip{ip}") for ip in [1, 2, 5, 8]
    }

    # Convert the twiss variables in dic_separation_ip to pandas dataframe so that it can be saved in a pickle file
    for ip in [1, 2, 5, 8]:
        for variable_to_convert in [
            "twiss_filtered",
            "survey_filtered",
            "s",
            "dx_meter",
            "dy_meter",
            "dx_sig",
            "dy_sig",
        ]:
            if variable_to_convert == "twiss_filtered" or variable_to_convert == "survey_filtered":
                dic_separation_ip[f"ip{ip}"][variable_to_convert]["b1"] = dic_separation_ip[
                    f"ip{ip}"
                ][variable_to_convert]["b1"].to_pandas()
                dic_separation_ip[f"ip{ip}"][variable_to_convert]["b2"] = dic_separation_ip[
                    f"ip{ip}"
                ][variable_to_convert]["b2"].to_pandas()
            else:
                dic_separation_ip[f"ip{ip}"][variable_to_convert] = np.array(
                    dic_separation_ip[f"ip{ip}"][variable_to_convert], dtype=np.float64
                )

    # Get the footprint only if bb is on
    if compute_footprint:
        array_qx1, array_qy1 = return_footprint(
            collider_check.collider, nemitt_x, beam="lhcb1", n_turns=2000
        )
        array_qx2, array_qy2 = return_footprint(
            collider_check.collider, nemitt_x, beam="lhcb2", n_turns=2000
        )
    else:
        array_qx1 = np.array([])
        array_qy1 = np.array([])
        array_qx2 = np.array([])
        array_qy2 = np.array([])

    # Store everything in a dictionnary
    dic_global_var = {
        "l_lumi": l_lumi,
        "dic_tw_b1": dic_tw_b1,
        "dic_tw_b2": dic_tw_b2,
        # "dic_sep_IPs": dic_sep_IPs,
        # "dic_bb_ho_IPs": dic_bb_ho_IPs,
        "dic_separation_ip": dic_separation_ip,
        "df_sv_b1": df_sv_b1,
        "df_sv_b2": df_sv_b2,
        "df_tw_b1": df_tw_b1,
        "df_tw_b2": df_tw_b2,
        "df_elements_corrected": df_elements_corrected,
        "table_sv_b1": table_sv_b1,
        "table_tw_b1": table_tw_b1,
        "table_sv_b2": table_sv_b2,
        "table_tw_b2": table_tw_b2,
        "array_b1": array_b1,
        "array_b2": array_b2,
        "i_bunch_b1": i_bunch_b1,
        "i_bunch_b2": i_bunch_b2,
        "bbs": bbs,
        "footprint_b1": (array_qx1, array_qy1),
        "footprint_b2": (array_qx2, array_qy2),
        "polarity_alice": polarity_alice,
        "polarity_lhcb": polarity_lhcb,
        "configuration_str": configuration_str,
    }

    return dic_global_var


# ==================================================================================================
# --- Functions to load dashboard variables
# ==================================================================================================
def return_dataframe_elements_from_line(line):
    # Build a dataframe with the elements of the lines
    df_elements = pd.DataFrame([x.to_dict() for x in line.elements])
    return df_elements


def return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw):
    """Correct the dataframe of elements for thin lens approximation."""
    df_elements_corrected = df_elements.copy(deep=True)

    # Add all thin lenses (length + strength)
    for i, row in df_tw.iterrows():
        # Correct for thin lens approximation and weird duplicates
        if ".." in row["name"] and "f" not in row["name"].split("..")[1]:
            name = row["name"].split("..")[0]
            try:
                index = df_tw[df_tw.name == name].index[0]
            except IndexError:
                print(f"IndexError trying to correct slicing for {name}")
                continue

            # Add length
            if np.isnan(df_elements_corrected.loc[index]["length"]):
                df_elements_corrected.at[index, "length"] = 0.0
            df_elements_corrected.at[index, "length"] += df_elements.loc[i]["length"]

            # Add strength
            if np.isnan(df_elements_corrected.loc[index]["knl"]).all():
                df_elements_corrected.at[index, "knl"] = (
                    np.array([0.0] * df_elements.loc[i]["knl"].shape[0], dtype=np.float64)
                    if type(df_elements.loc[i]["knl"]) != float
                    else 0.0
                )
            df_elements_corrected.at[index, "knl"] = (
                df_elements_corrected.loc[index, "knl"] + np.array(df_elements.loc[i]["knl"])
                if type(df_elements.loc[i]["knl"]) != float
                else df_elements.loc[i]["knl"]
            )

            # Replace order
            df_elements_corrected.at[index, "_order"] = df_elements.loc[i]["_order"]

            # Drop row
            df_elements_corrected.drop(i, inplace=True)

    return df_elements_corrected


def return_twiss_dic(tw):
    # Init empty dic
    dic_tw = {}

    # Load main observables
    dic_tw["qx"] = tw["qx"]
    dic_tw["qy"] = tw["qy"]
    dic_tw["dqx"] = tw["dqx"]
    dic_tw["dqy"] = tw["dqy"]
    dic_tw["c_minus"] = tw["c_minus"]
    dic_tw["momentum_compaction_factor"] = tw["momentum_compaction_factor"]
    dic_tw["T_rev0"] = tw["T_rev0"]

    # Load observables at IPs
    for ip in [1, 2, 5, 8]:
        dic_tw["ip" + str(ip)] = (
            tw.rows[f"ip{ip}"]
            .cols["s", "x", "px", "y", "py", "betx", "bety"]
            .to_pandas()
            .to_numpy()
            .squeeze()
        )

    return dic_tw


# def return_bb_ho_dic(df_tw_b1, df_tw_b2, collider):
#     # Find elements at extremities of each IP
#     # IP1 : mqy.4l1.b1 to mqy.4r1.b1
#     # IP2 : mqy.b5l2.b1 to mqy.b4r2.b1
#     # IP5 : mqy.4l5.b1 to mqy.4r5.b1
#     # IP8 : mqy.b4l8.b1 to mqy.b4r8.b1
#     dic_bb_ho_IPs = {"lhcb1": {"sv": {}, "tw": {}}, "lhcb2": {"sv": {}, "tw": {}}}
#     for beam, df_tw in zip(["lhcb1", "lhcb2"], [df_tw_b1, df_tw_b2]):
#         for ip, el_start, el_end in zip(
#             ["ip1", "ip2", "ip5", "ip8"],
#             ["mqy.4l1", "mqy.b4l2", "mqy.4l5", "mqy.b4l8"],
#             ["mqy.4r1", "mqy.b4r2", "mqy.4r5", "mqy.b4r8"],
#         ):
#             # Change element name for current beam
#             el_start = el_start + "." + beam[3:]
#             el_end = el_end + "." + beam[3:]

#             # # Recompute survey from ip
#             if beam == "lhcb1":
#                 df_sv = collider[beam].survey(element0=ip).to_pandas()
#             else:
#                 df_sv = collider[beam].survey(element0=ip).reverse().to_pandas()

#             # Get twiss and sv between start and end element
#             idx_element_start_tw = df_tw.index[df_tw.name == el_start].tolist()[0]
#             idx_element_end_tw = df_tw.index[df_tw.name == el_end].tolist()[0]
#             idx_element_start_sv = df_sv.index[df_sv.name == el_start].tolist()[0]
#             idx_element_end_sv = df_sv.index[df_sv.name == el_end].tolist()[0]

#             # Get dataframe of elements between s_start and s_end
#             dic_bb_ho_IPs[beam]["sv"][ip] = copy.deepcopy(
#                 df_sv.iloc[idx_element_start_sv : idx_element_end_sv + 1]
#             )
#             dic_bb_ho_IPs[beam]["tw"][ip] = copy.deepcopy(
#                 df_tw.iloc[idx_element_start_tw : idx_element_end_tw + 1]
#             )

#     # Delete all .b1 and .b2 from element names
#     for ip in ["ip1", "ip2", "ip5", "ip8"]:
#         dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "name"] = [
#             el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["sv"][ip].name
#         ]
#         dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "name"] = [
#             el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["sv"][ip].name
#         ]
#         dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "name"] = [
#             el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["tw"][ip].name
#         ]
#         dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "name"] = [
#             el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["tw"][ip].name
#         ]

#     for ip in ["ip1", "ip2", "ip5", "ip8"]:
#         # Get intersection of names in twiss and survey
#         s_intersection = (
#             set(dic_bb_ho_IPs["lhcb2"]["sv"][ip].name)
#             .intersection(set(dic_bb_ho_IPs["lhcb1"]["sv"][ip].name))
#             .intersection(set(dic_bb_ho_IPs["lhcb2"]["tw"][ip].name))
#             .intersection(set(dic_bb_ho_IPs["lhcb1"]["tw"][ip].name))
#         )

#         # Clean dataframes in both beams so that they are comparable
#         for beam in ["lhcb1", "lhcb2"]:
#             # Remove all rows whose name is not in both beams
#             dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
#                 dic_bb_ho_IPs[beam]["sv"][ip].name.isin(s_intersection)
#             ]
#             dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
#                 dic_bb_ho_IPs[beam]["tw"][ip].name.isin(s_intersection)
#             ]

#             # Remove all elements whose name contains '..'
#             for i in range(1, 6):
#                 dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
#                     ~dic_bb_ho_IPs[beam]["sv"][ip].name.str.endswith(f"..{i}")
#                 ]
#                 dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
#                     ~dic_bb_ho_IPs[beam]["tw"][ip].name.str.endswith(f"..{i}")
#                 ]

#         # Center s around IP for beam 1
#         dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"] = (
#             dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"]
#             - dic_bb_ho_IPs["lhcb1"]["sv"][ip][
#                 dic_bb_ho_IPs["lhcb1"]["sv"][ip].name == ip
#             ].s.to_numpy()
#         )
#         dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"] = (
#             dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"]
#             - dic_bb_ho_IPs["lhcb1"]["tw"][ip][
#                 dic_bb_ho_IPs["lhcb1"]["tw"][ip].name == ip
#             ].s.to_numpy()
#         )

#         # Set the s of beam 1 as reference for all dataframes
#         dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["sv"][ip].s.to_numpy()
#         dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["tw"][ip].s.to_numpy()

#         # Only keep bb_ho and bb_lr elements
#         for beam in ["lhcb1", "lhcb2"]:
#             dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
#                 dic_bb_ho_IPs[beam]["sv"][ip].name.str.contains(f"bb_ho|bb_lr")
#             ]
#             dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
#                 dic_bb_ho_IPs[beam]["tw"][ip].name.str.contains(f"bb_ho|bb_lr")
#             ]

#     return dic_bb_ho_IPs


# def return_separation_dic(dic_bb_ho_IPs, tw_b1, nemitt_x, nemitt_y, energy):
#     dic_sep_IPs = {"v": {}, "h": {}}

#     for idx, n_ip in enumerate([1, 2, 5, 8]):
#         # s doesn't depend on plane
#         s = dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].s

#         # Horizontal separation
#         x = abs(
#             dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].x
#             - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].x.to_numpy()
#         )
#         n_emitt = nemitt_x / energy
#         sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].betx * n_emitt) ** 0.5
#         xing = float(tw_b1.rows[f"ip{n_ip}"]["px"])
#         beta = float(tw_b1.rows[f"ip{n_ip}"]["betx"])
#         sep_survey = abs(
#             dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].X
#             - dic_bb_ho_IPs["lhcb2"]["sv"][f"ip{n_ip}"].X.to_numpy()
#         )
#         sep = xing * 2 * np.sqrt(beta / n_emitt)

#         # Store everyting in dic
#         dic_sep_IPs["h"][f"ip{n_ip}"] = {
#             "s": s,
#             "x": x,
#             "sep": sep,
#             "sep_survey": sep_survey,
#             "sigma": sigma,
#         }

#         # Vertical separation
#         x = abs(
#             dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].y
#             - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].y.to_numpy()
#         )
#         n_emitt = nemitt_y / 7000
#         sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].bety * n_emitt) ** 0.5
#         xing = abs(float(tw_b1.rows[f"ip{n_ip}"]["py"]))
#         beta = float(tw_b1.rows[f"ip{n_ip}"]["bety"])
#         sep_survey = 0
#         sep = xing * 2 * np.sqrt(beta / n_emitt)

#         # Store everyting in dic
#         dic_sep_IPs["v"][f"ip{n_ip}"] = {
#             "s": s,
#             "x": x,
#             "sep": sep,
#             "sep_survey": sep_survey,
#             "sigma": sigma,
#         }
#     return dic_sep_IPs


# ==================================================================================================
# --- Functions to build data tables
# ==================================================================================================
def return_data_table(df, id_table, twiss=True):
    if twiss:
        df = df.drop(["W_matrix"], axis=1)
        idx_column_name = 0
    else:
        idx_column_name = 6

    # Change order of columns such that name is first
    df = df[["name"] + [col for col in df.columns if col != "name"]]

    table = (
        dash_table.DataTable(
            id=id_table,
            columns=[
                (
                    {
                        "name": i,
                        "id": i,
                        "deletable": False,
                        "type": "numeric",
                        "format": Format(precision=6, scheme=Scheme.decimal_or_exponent),
                    }
                    if idx != idx_column_name
                    else {"name": i, "id": i, "deletable": False}
                )
                for idx, i in enumerate(df.columns)
            ],
            data=df.to_dict("records"),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable=False,
            row_deletable=False,
            # page_action="none",
            # fixed_rows={"headers": True, "data": 0},
            # fixed_columns={"headers": True, "data": 1},
            virtualization=False,
            page_size=25,
            style_table={
                # "height": "100%",
                "maxHeight": "75vh",
                "margin-x": "auto",
                "margin-top": "20px",
                "overflowY": "auto",
                "overflowX": "auto",
                "minWidth": "98%",
                "padding": "1em",
            },
            style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white", "padding": "1em"},
            style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            style_filter={"backgroundColor": "rgb(70, 70, 70)"},  # , "color": "white"},
            style_cell={"font-family": "sans-serif", "minWidth": 95},
        ),
    )
    return table


def return_footprint(collider, emittance, beam="lhcb1", n_turns=2000):
    fp_polar_xm = collider[beam].get_footprint(
        nemitt_x=emittance,
        nemitt_y=emittance,
        n_turns=n_turns,
        linear_rescale_on_knobs=[xt.LinearRescale(knob_name="beambeam_scale", v0=0.0, dv=0.05)],
        freeze_longitudinal=True,
    )

    qx = fp_polar_xm.qx
    qy = fp_polar_xm.qy

    return qx, qy

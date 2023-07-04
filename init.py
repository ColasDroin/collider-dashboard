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

# Module to compute beam-beam schedule
import fillingpatterns as fp

# Import collider and twiss functions
from modules.twiss_check.twiss_check import TwissCheck, BuildCollider


# ==================================================================================================
# --- Functions initialize all global variables
# ==================================================================================================
def init(path_config, build_collider=False, load_from_pickle=False):
    """Initialize the app variables from a given collider configuration file."""

    # Path to the pickle dictionnaries (for loading and saving)
    path_pickle = (
        "temp/"
        + path_config.split("/scans/")[1].split("config.yaml")[0].replace("/", "_")
        + "t_dic_var.pkl"
    )

    # Try to load the dictionnaries of variables from pickle
    if load_from_pickle:
        # Raise error if a collider must be built
        if build_collider:
            raise ValueError("If load_from_pickle is True, build_collider must be False.")

        # Check that the pickle file exists
        if not os.path.isfile(path_pickle):
            raise ValueError("The pickle file does not exist.")
        with open(path_pickle, "rb") as f:
            dic_before_bb, dic_after_bb = pickle.load(f)
        print("Returning global variables from pickle file.")
        return dic_before_bb, dic_after_bb

    else:
        # If a collider is being built, explictely set the paths to None
        if build_collider:
            path_collider = None
            path_collider_before_bb = None
        else:
            # Get the path to the collider object
            path_collider = (
                "temp/"
                + path_config.split("/scans/")[1].split("config.yaml")[0].replace("/", "_")
                + "collider.json"
            )
            # Also get a path to the collider after beam-beam object
            path_collider_before_bb = path_collider.replace(".json", "_before_bb.json")

        # Load the global variables for the final collider
        # (if build_collider is True, a collider object is stored in temp folder)
        twiss_check_after_beam_beam, twiss_check_before_beam_beam = initialize_both_twiss_checks(
            path_config,
            path_collider=path_collider,
            path_collider_before_bb=path_collider_before_bb,
            build_collider=build_collider,
        )

        # Get the global variables before and after the beam-beam
        dic_after_bb = initialize_global_variables(
            twiss_check_after_beam_beam, compute_footprint=True
        )
        dic_before_bb = initialize_global_variables(
            twiss_check_before_beam_beam, compute_footprint=False
        )

        # Dump the dictionnaries in a pickle file
        print("Dumping global variables in a pickle file.")
        with open(path_pickle, "wb") as f:
            pickle.dump((dic_before_bb, dic_after_bb), f)

        return dic_before_bb, dic_after_bb


def initialize_both_twiss_checks(
    path_config, path_collider=None, path_collider_before_bb=None, build_collider=True
):
    """Initialize all twiss_check object from a collider or path to a collider."""
    if build_collider:
        if path_collider is not None or path_collider_before_bb is not None:
            raise ValueError(
                "If build_collider is True, path_collider and path_collider_before_bb must not be"
                " provided. If you want to use a collider from a json file, set build_collider to"
                " False."
            )

        # Build collider from config file
        build_collider = BuildCollider(path_config)

        # Dump collider
        path_collider, path_collider_before_bb = build_collider.dump_collider(
            prefix="temp/", dump_before_bb=True
        )

        # Do Twiss check after bb with the collider dumped previously
        twiss_check_after_bb = TwissCheck(
            path_config,
            path_collider=None,
            collider=build_collider.collider,
        )

        # Same before bb
        twiss_check_before_bb = TwissCheck(
            path_config,
            path_collider=None,
            collider=build_collider.collider_before_bb,
        )

    elif path_collider is not None and path_collider_before_bb is not None:
        # Do Twiss check, reloading the collider from a json file
        twiss_check_after_bb = TwissCheck(path_config, path_collider=path_collider, collider=None)
        twiss_check_before_bb = TwissCheck(
            path_config, path_collider=path_collider_before_bb, collider=None
        )
    else:
        raise ValueError(
            "If build_collider is False, path_collider and path_collider_before_bb must be"
            " provided."
        )

    # Finally, return the two twiss_check objects
    return twiss_check_after_bb, twiss_check_before_bb


def initialize_global_variables(twiss_check, compute_footprint=True):
    """Initialize global variables, from a collider with beam-beam set."""
    if twiss_check.collider is None:
        raise ValueError("The collider must be provided in the twiss_check object.")

    # Get luminosity at each IP
    l_lumi = [twiss_check.return_luminosity(IP=x) for x in [1, 2, 5, 8]]

    # Get collider and twiss variables (can't do it from twiss_check as corrections must be applied)
    (
        collider,
        tw_b1,
        sv_b1,
        df_sv_b1,
        df_tw_b1,
        tw_b2,
        sv_b2,
        df_sv_b2,
        df_tw_b2,
        df_elements_corrected,
    ) = return_all_loaded_variables(collider=twiss_check.collider)

    # Get corresponding data tables
    table_sv_b1 = return_data_table(df_sv_b1, "id-df-sv-b1-after-bb", twiss=False)
    table_tw_b1 = return_data_table(df_tw_b1, "id-df-tw-b1-after-bb", twiss=True)
    table_sv_b2 = return_data_table(df_sv_b2, "id-df-sv-b2-after-bb", twiss=False)
    table_tw_b2 = return_data_table(df_tw_b2, "id-df-tw-b2-after-bb", twiss=True)

    # Get the twiss dictionnary (tune, chroma, etc + twiss at IPs)
    dic_tw_b1 = return_twiss_dic(tw_b1)
    dic_tw_b2 = return_twiss_dic(tw_b2)

    # Get the beams schemes
    array_b1 = twiss_check.array_b1
    array_b2 = twiss_check.array_b2

    # get the bunches selected for tracking
    i_bunch_b1 = twiss_check.i_bunch_b1
    i_bunch_b2 = twiss_check.i_bunch_b2

    # Get the dictionnary to plot separation
    dic_bb_ho_IPs = return_bb_ho_dic(df_tw_b1, df_tw_b2, collider)
    dic_sep_IPs = return_separation_dic(dic_bb_ho_IPs, twiss_check, tw_b1)

    # Get the beam-beam schedule
    patt = fp.FillingPattern.from_json(twiss_check.path_filling_scheme)
    patt.compute_beam_beam_schedule(n_lr_per_side=26)
    bbs = patt.b1.bb_schedule

    # Get the footprint only if bb is on
    if compute_footprint:
        array_qx, array_qy = return_footprint(
            collider, twiss_check.nemitt_x, beam="lhcb1", n_turns=2000
        )
    else:
        array_qx = np.array([])
        array_qy = np.array([])

    # Store everything in a dictionnary
    dic_global_var = {
        "l_lumi": l_lumi,
        "dic_tw_b1": dic_tw_b1,
        "dic_tw_b2": dic_tw_b2,
        "dic_sep_IPs": dic_sep_IPs,
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
        "footprint": (array_qx, array_qy),
    }

    return dic_global_var


# ==================================================================================================
# --- Functions to load dashboard variables
# ==================================================================================================
def return_dataframe_elements_from_line(line):
    # Build a dataframe with the elements of the lines
    df_elements = pd.DataFrame([x.to_dict() for x in line.elements])
    return df_elements


def return_survey_and_twiss_dataframes_from_line(line, correct_s_axis=False):
    """Return the survey and twiss dataframes from a line."""

    # Get Twiss and survey
    tw = line.twiss()
    sv = line.survey()

    # Correct s-axis if required
    if correct_s_axis:
        tw = tw.reverse()
        sv = sv.reverse()

    # Convert to dataframe
    df_tw = tw.to_pandas()
    df_sv = sv.to_pandas()

    return tw, sv, df_sv, df_tw


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
                print(f"IndexError for {name}")
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

            # print(df_elements)
            # print(df_elements_corrected)

            # Replace order
            df_elements_corrected.at[index, "_order"] = df_elements.loc[i]["_order"]

            # Drop row
            df_elements_corrected.drop(i, inplace=True)

    return df_elements_corrected


def return_all_loaded_variables(collider):
    """Return all loaded variables if they are not already loaded."""

    if collider is None:
        raise ValueError("Either collider or collider_path must be provided")

    # Get elements of the line (only done for b1, should be identical for b2)
    df_elements = return_dataframe_elements_from_line(collider.lhcb1)

    # Compute twiss and survey for both lines
    tw_b1, sv_b1, df_sv_b1, df_tw_b1 = return_survey_and_twiss_dataframes_from_line(
        collider.lhcb1, correct_s_axis=False
    )

    tw_b2, sv_b2, df_sv_b2, df_tw_b2 = return_survey_and_twiss_dataframes_from_line(
        collider.lhcb2, correct_s_axis=True
    )

    # Correct df elements for thin lens approximation
    df_elements_corrected = return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw_b1)

    # Return all variables
    return (
        collider,
        tw_b1,
        sv_b1,
        df_sv_b1,
        df_tw_b1,
        tw_b2,
        sv_b2,
        df_sv_b2,
        df_tw_b2,
        df_elements_corrected,
    )


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


def return_bb_ho_dic(df_tw_b1, df_tw_b2, collider):
    # Find elements at extremities of each IP
    # IP1 : mqy.4l1.b1 to mqy.4r1.b1
    # IP2 : mqy.b5l2.b1 to mqy.b4r2.b1
    # IP5 : mqy.4l5.b1 to mqy.4r5.b1
    # IP8 : mqy.b4l8.b1 to mqy.b4r8.b1
    dic_bb_ho_IPs = {"lhcb1": {"sv": {}, "tw": {}}, "lhcb2": {"sv": {}, "tw": {}}}
    for beam, df_tw in zip(["lhcb1", "lhcb2"], [df_tw_b1, df_tw_b2]):
        for ip, el_start, el_end in zip(
            ["ip1", "ip2", "ip5", "ip8"],
            ["mqy.4l1", "mqy.b4l2", "mqy.4l5", "mqy.b4l8"],
            ["mqy.4r1", "mqy.b4r2", "mqy.4r5", "mqy.b4r8"],
        ):
            # Change element name for current beam
            el_start = el_start + "." + beam[3:]
            el_end = el_end + "." + beam[3:]

            # # Recompute survey from ip
            if beam == "lhcb1":
                df_sv = collider[beam].survey(element0=ip).to_pandas()
            else:
                df_sv = collider[beam].survey(element0=ip).reverse().to_pandas()

            # Get twiss and sv between start and end element
            idx_element_start_tw = df_tw.index[df_tw.name == el_start].tolist()[0]
            idx_element_end_tw = df_tw.index[df_tw.name == el_end].tolist()[0]
            idx_element_start_sv = df_sv.index[df_sv.name == el_start].tolist()[0]
            idx_element_end_sv = df_sv.index[df_sv.name == el_end].tolist()[0]

            # Get dataframe of elements between s_start and s_end
            dic_bb_ho_IPs[beam]["sv"][ip] = copy.deepcopy(
                df_sv.iloc[idx_element_start_sv : idx_element_end_sv + 1]
            )
            dic_bb_ho_IPs[beam]["tw"][ip] = copy.deepcopy(
                df_tw.iloc[idx_element_start_tw : idx_element_end_tw + 1]
            )

    # Delete all .b1 and .b2 from element names
    for ip in ["ip1", "ip2", "ip5", "ip8"]:
        dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "name"] = [
            el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["sv"][ip].name
        ]
        dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "name"] = [
            el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["sv"][ip].name
        ]
        dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "name"] = [
            el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["tw"][ip].name
        ]
        dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "name"] = [
            el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["tw"][ip].name
        ]

    for ip in ["ip1", "ip2", "ip5", "ip8"]:
        # Get intersection of names in twiss and survey
        s_intersection = (
            set(dic_bb_ho_IPs["lhcb2"]["sv"][ip].name)
            .intersection(set(dic_bb_ho_IPs["lhcb1"]["sv"][ip].name))
            .intersection(set(dic_bb_ho_IPs["lhcb2"]["tw"][ip].name))
            .intersection(set(dic_bb_ho_IPs["lhcb1"]["tw"][ip].name))
        )

        # Clean dataframes in both beams so that they are comparable
        for beam in ["lhcb1", "lhcb2"]:
            # Remove all rows whose name is not in both beams
            dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                dic_bb_ho_IPs[beam]["sv"][ip].name.isin(s_intersection)
            ]
            dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                dic_bb_ho_IPs[beam]["tw"][ip].name.isin(s_intersection)
            ]

            # Remove all elements whose name contains '..'
            for i in range(1, 6):
                dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                    ~dic_bb_ho_IPs[beam]["sv"][ip].name.str.endswith(f"..{i}")
                ]
                dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                    ~dic_bb_ho_IPs[beam]["tw"][ip].name.str.endswith(f"..{i}")
                ]

        # Center s around IP for beam 1
        dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"] = (
            dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"]
            - dic_bb_ho_IPs["lhcb1"]["sv"][ip][
                dic_bb_ho_IPs["lhcb1"]["sv"][ip].name == ip
            ].s.to_numpy()
        )
        dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"] = (
            dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"]
            - dic_bb_ho_IPs["lhcb1"]["tw"][ip][
                dic_bb_ho_IPs["lhcb1"]["tw"][ip].name == ip
            ].s.to_numpy()
        )

        # Set the s of beam 1 as reference for all dataframes
        dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["sv"][ip].s.to_numpy()
        dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["tw"][ip].s.to_numpy()

        # Only keep bb_ho and bb_lr elements
        for beam in ["lhcb1", "lhcb2"]:
            dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                dic_bb_ho_IPs[beam]["sv"][ip].name.str.contains(f"bb_ho|bb_lr")
            ]
            dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                dic_bb_ho_IPs[beam]["tw"][ip].name.str.contains(f"bb_ho|bb_lr")
            ]

    return dic_bb_ho_IPs


def return_separation_dic(dic_bb_ho_IPs, twiss_check, tw_b1):
    dic_sep_IPs = {"v": {}, "h": {}}
    for idx, n_ip in enumerate([1, 2, 5, 8]):
        # s doesn't depend on plane
        s = dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].s

        # Horizontal separation
        x = abs(
            dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].x
            - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].x.to_numpy()
        )
        n_emitt = twiss_check.nemitt_x / 7000
        sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].betx * n_emitt) ** 0.5
        xing = float(tw_b1.rows[f"ip{n_ip}"]["px"])
        beta = float(tw_b1.rows[f"ip{n_ip}"]["betx"])
        sep_survey = abs(
            dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].X
            - dic_bb_ho_IPs["lhcb2"]["sv"][f"ip{n_ip}"].X.to_numpy()
        )
        sep = xing * 2 * np.sqrt(beta / n_emitt)

        # Store everyting in dic
        dic_sep_IPs["h"][f"ip{n_ip}"] = {
            "s": s,
            "x": x,
            "sep": sep,
            "sep_survey": sep_survey,
            "sigma": sigma,
        }

        # Vertical separation
        x = abs(
            dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].y
            - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].y.to_numpy()
        )
        n_emitt = twiss_check.nemitt_y / 7000
        sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].bety * n_emitt) ** 0.5
        xing = abs(float(tw_b1.rows[f"ip{n_ip}"]["py"]))
        beta = float(tw_b1.rows[f"ip{n_ip}"]["bety"])
        sep_survey = 0
        sep = xing * 2 * np.sqrt(beta / n_emitt)

        # Store everyting in dic
        dic_sep_IPs["v"][f"ip{n_ip}"] = {
            "s": s,
            "x": x,
            "sep": sep,
            "sep_survey": sep_survey,
            "sigma": sigma,
        }
    return dic_sep_IPs


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
                        "format": Format(precision=6, scheme=Scheme.decimal_si_prefix),
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

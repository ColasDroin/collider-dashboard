# ! This class needs to be cleaned up and documented !
# ==================================================================================================
# --- Imports
# ==================================================================================================import numpy as np
import pandas as pd
import xtrack as xt
import numpy as np
from dash import dash_table
from dash.dash_table.Format import Format, Scheme

# Import collider and twiss functions
from modules.twiss_check.twiss_check import TwissCheck, BuildCollider


# ==================================================================================================
# --- Functions initialize all global variables
# ==================================================================================================
def init(path_config, build_collider=False):
    """Initialize the app variables from a given collider configuration file."""

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

    # Get the global variables after the beam-beam
    dic_after_bb = initialize_global_variables(twiss_check_after_beam_beam)

    # Same before beam_beam
    dic_before_bb = initialize_global_variables(twiss_check_before_beam_beam)

    return dic_after_bb, dic_before_bb


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


def initialize_global_variables(twiss_check):
    """Initialize global variables, from a collider with beam-beam set."""
    if twiss_check.collider is None:
        raise ValueError("The collider must be provided in the twiss_check_after_beam_beam object.")

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

    # Store everything in a dictionnary
    dic_global_var = {
        "l_lumi": l_lumi,
        "collider": collider,
        "dic_tw_b1": dic_tw_b1,
        "dic_tw_b2": dic_tw_b2,
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
            index = df_tw[df_tw.name == name].index[0]

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
            df_elements_corrected.at[index, "order"] = df_elements.loc[i]["order"]

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
    # ! TO DEBUG HERE
    dic_tw["qx"] = tw["qx"]
    dic_tw["qy"] = tw["qy"]
    dic_tw["dqx"] = tw["dqx"]
    dic_tw["dqy"] = tw["dqy"]
    dic_tw["c_minus"] = tw["c_minus"]
    dic_tw["momentum_compaction_factor"] = tw["momentum_compaction_factor"]

    # Load observables at IPs
    for ip in [1, 2, 5, 8]:
        dic_tw["ip" + str(ip)] = (
            tw.rows[f"ip{ip}"]
            .cols["s", "x", "px", "y", "py", "betx", "bety"]
            .to_pandas()
            .to_numpy()
            .squeeze()
        )


# ==================================================================================================
# --- Functions to build data tables
# ==================================================================================================
def return_data_table(df, id_table, twiss=True):
    if twiss:
        df = df.drop(["W_matrix"], axis=1)
        idx_column_name = 0
    else:
        idx_column_name = 6
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

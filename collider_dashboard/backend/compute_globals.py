"""Thid module initializes all global variables from a collider json, potentially embedding a
configuration file.
"""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import from standard library
import io
import logging
import os
import pickle
from contextlib import redirect_stdout
from importlib.resources import files

# Third-party packages
import numpy as np
import pandas as pd
import xtrack as xt

# Package to check collider observables
from collider_check import ColliderCheck

# Dash imports
from dash import dash_table
from dash.dash_table.Format import Format, Scheme

# Package to compute beam-beam schedule
from .fillingpatterns import FillingPattern

# ==================================================================================================
# --- Functions initialize all global variables
# ==================================================================================================


def init_from_collider(
    path_collider,
    path_scheme=None,
    force_reload=False,
    ignore_footprint=False,
    simplify_tw=True,
    type_particles=None,
):
    """
    Initializes a collider from a JSON file and computes global variables from collider checks.

    Args:
        path_collider (str): Path to the JSON file containing the collider definition.
        force_reload (bool, optional): If False, tries to to load the global variables
            from a pickle file instead of computing them from scratch. Otherwise, force the
            (re-)computation of the global variables. Defaults to False.
        ignore_footprint (bool, optional): If True, does not compute the footprints to gain loading
            time. Defaults to False.
        simplify_tw (bool, optional): If True, simplifies the Twiss and Survey dataframes by
            removing duplicated elements to speed up computations. Defaults to True.

    Returns:
        tuple: A tuple containing two dictionaries of global variables computed from collider
            checks, one for the collider with beam-beam interactions and one for the collider
            without beam-beam interactions, and the path to the pickle file used to store the global
            variables.
    """

    # Path to the pickle dictionnaries (for loading and saving)
    temp_path = str(files("collider_dashboard")) + "/temp/"
    path_pickle = temp_path + path_collider.replace("/", "_") + "t_dic_var.pkl"

    # Check that the pickle file exists
    if not os.path.isfile(path_pickle) or force_reload:
        print(
            "No data has been recorded for this collider, or force_reload is True. Recomputing the"
            " dashboard collider data now."
        )

        logging.info("Building collider.")
        # Rebuild collider
        collider = xt.Multiline.from_json(path_collider)

        # Make a copy of the collider to load without bb after
        collider_without_bb = xt.Multiline.from_dict(collider.to_dict())

        # Load collider with bb
        collider.build_trackers()

        # Build collider before bb
        collider_without_bb.build_trackers()
        collider_without_bb.vars["beambeam_scale"] = 0

        # Check configuration
        if collider.metadata is not None and collider.metadata != {}:
            logging.info("The collider file contains metadata. Using it.")
        else:
            logging.warning("The collider file does not contain metadata. Using default values.")

        # Compute collider checks
        logging.info("Computing collider checks.")
        collider_check_with_bb = ColliderCheck(
            collider, path_filling_scheme=path_scheme, type_particles=type_particles
        )
        collider_check_without_bb = ColliderCheck(
            collider_without_bb, path_filling_scheme=path_scheme, type_particles=type_particles
        )

        # Compute global variables
        dic_without_bb, dic_with_bb = compute_global_variables_from_collider_checks(
            collider_check_with_bb,
            collider_check_without_bb,
            path_pickle=path_pickle,
            ignore_footprint=ignore_footprint,
            simplify_tw=simplify_tw,
        )

    else:
        print("Some collider data already exists for this path, loading it.")
        logging.info("Returning global variables from pickle file.")
        with open(path_pickle, "rb") as f:
            dic_without_bb, dic_with_bb = pickle.load(f)

    return dic_without_bb, dic_with_bb


def compute_global_variables_from_collider_checks(
    collider_check_after_beam_beam,
    collider_check_without_beam_beam,
    path_pickle=None,
    ignore_footprint=False,
    simplify_tw=True,
):
    """
    Computes global variables with and without beam-beam interaction.

    Args:
        collider_check_after_beam_beam (ColliderCheck): Collider check object including beam-beam
            interaction.
        collider_check_without_beam_beam (ColliderCheck): Collider check object without beam-beam
            interaction.
        path_pickle (str, optional): Path to the pickle file to dump the dictionaries. Defaults to
            None.
        ignore_footprint (bool, optional): If True, does not compute the footprints to gain loading
            time. Defaults to False.
        simplify_tw (bool, optional): If True, simplifies the Twiss and Survey dataframes by
            removing duplicated elements to speed up computations. Defaults to True.


    Returns:
        tuple: A tuple containing two dictionaries, one for global variables with beam-beam
            interaction and one for global variables without beam-beam interaction.
    """
    # Get the global variables before and after the beam-beam
    logging.info("Computing global variables with beam beam.")
    dic_with_bb = initialize_global_variables(
        collider_check_after_beam_beam,
        compute_footprint=not ignore_footprint,
        simplify_tw=simplify_tw,
    )
    logging.info("Computing global variables without beam beam.")
    dic_without_bb = initialize_global_variables(
        collider_check_without_beam_beam,
        compute_footprint=not ignore_footprint,
        simplify_tw=simplify_tw,
    )

    if path_pickle is not None:
        # Dump the dictionnaries in a pickle file, creating the directory if it does not exist
        if not os.path.isdir(os.path.dirname(path_pickle)):
            os.makedirs(os.path.dirname(path_pickle))
        logging.info("Dumping global variables into a pickle file.")
        with open(path_pickle, "wb") as f:
            pickle.dump((dic_without_bb, dic_with_bb), f)

    return dic_without_bb, dic_with_bb


def initialize_global_variables(collider_check, compute_footprint=True, simplify_tw=True):
    """
    Initializes global variables used in the simulation dashboard.

    Args:
        collider_check (ColliderCheck): An instance of the ColliderCheck class containing the
            collider configuration.
        compute_footprint (bool, optional): Whether to compute the collider footprint or not.
            Default is True.
        simplify_tw (bool, optional): If True, simplifies the Twiss and Survey dataframes by
            removing duplicated elements to speed up computations. Defaults to True.

    Returns:
        dict: A dictionary containing the initialized global variables.
    """

    if collider_check.path_filling_scheme is not None:
        # Get number of LR per side
        n_lr_per_side = collider_check.n_lr_per_side

        # Get the beam-beam schedule
        logging.info("Computing beam-beam schedule.")
        patt = FillingPattern.from_json(collider_check.path_filling_scheme)
        patt.compute_beam_beam_schedule(n_lr_per_side=n_lr_per_side)
        bbs = patt.b1.bb_schedule
    else:
        bbs = None

    if collider_check.configuration is not None:
        # Get luminosity at each IP
        logging.info("Computing luminosity at each IP.")
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

        # Get energy
        energy = collider_check.energy
        cross_section = collider_check.cross_section

        # Get polarity Alice and LHCb
        polarity_alice, polarity_lhcb = collider_check.return_polarity_ip_2_8()

        # Get configuration
        configuration_str = collider_check.configuration_str

    else:
        l_lumi = None
        array_b1 = collider_check.array_b1 if hasattr(collider_check, "array_b1") else None
        array_b2 = collider_check.array_b2 if hasattr(collider_check, "array_b2") else None
        i_bunch_b1 = collider_check.i_bunch_b1 if hasattr(collider_check, "i_bunch_b1") else None
        i_bunch_b2 = collider_check.i_bunch_b2 if hasattr(collider_check, "i_bunch_b2") else None
        polarity_alice = None
        polarity_lhcb = None
        configuration_str = None
        energy = None
        cross_section = None

        # Get emittance for the computation of the normalized separation
        logging.warning("No configuration file provided, using default values for emittances.")
        nemitt_x = 2.2e-6
        # nemitt_y = 2.2e-6

        # Assume 7 TeV for protons and 82 * 7 TeV for Pb ions
        if collider_check.type_particles == "proton":
            logging.warning(
                "No configuration file provided, using default values for energy for protons (7 TeV)."
            )
            energy = 7000
        elif collider_check.type_particles == "lead":
            logging.warning(
                "No configuration file provided, using default values for energy for ions (7 Z TeV)."
            )
            energy = 82 * 7000
        else:
            logging.warning("No type of particles provided. Assuming protons for energy.")
            energy = 7000

    # Get elements of the line (only done for b1, should be identical for b2)
    logging.info("Getting beam-beam elements for plotting.")
    df_elements = return_dataframe_elements_from_line(collider_check.collider.lhcb1)

    # Get twiss and survey for both lines
    logging.info("Computing Twiss and Survey.")
    tw_b1, df_sv_b1, df_tw_b1 = (
        collider_check.tw_b1,
        collider_check.df_sv_b1,
        collider_check.df_tw_b1,
    )

    tw_b2, sv_b2 = collider_check.tw_b2.reverse(), collider_check.sv_b2.reverse()
    df_tw_b2, df_sv_b2 = tw_b2.to_pandas(), sv_b2.to_pandas()

    # Correct df elements for thin lens approximation
    logging.info("Correcting beam-beam elements for thin lens approximation.")
    df_elements_corrected = return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw_b1)

    # Get corresponding data tables
    logging.info("Get Twiss and survey datatables.")
    if simplify_tw:
        logging.info("Simplifying Twiss and survey datatables as requested.")

    table_sv_b1, df_sv_b1 = return_data_table(
        df_sv_b1, "id-df-sv-b1-after-bb", twiss=False, simplify_tw=simplify_tw
    )
    table_tw_b1, df_tw_b1 = return_data_table(
        df_tw_b1, "id-df-tw-b1-after-bb", twiss=True, simplify_tw=simplify_tw
    )
    table_sv_b2, df_sv_b2 = return_data_table(
        df_sv_b2, "id-df-sv-b2-after-bb", twiss=False, simplify_tw=simplify_tw
    )
    table_tw_b2, df_tw_b2 = return_data_table(
        df_tw_b2, "id-df-tw-b2-after-bb", twiss=True, simplify_tw=simplify_tw
    )

    # Get the twiss dictionary (tune, chroma, etc + twiss at IPs)
    logging.info("Get Twiss dictionary.")
    dic_tw_b1 = return_twiss_dic(tw_b1)
    dic_tw_b2 = return_twiss_dic(tw_b2)

    # Get the dictionary to plot separation
    logging.info("Computing separation variables")
    dic_separation_ip = {
        f"ip{ip}": collider_check.compute_separation_variables(ip=f"ip{ip}") for ip in [1, 2, 5, 8]
    }
    dic_position_ip = collider_check.return_dic_position_all_ips()

    # Convert the twiss variables in dic_separation_ip to pandas dataframe so that it can be saved in a pickle file
    logging.info("Converting twiss and survey objects to pandas dataframes.")
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
            if variable_to_convert in ["twiss_filtered", "survey_filtered"]:
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
    array_qx1 = np.array([])
    array_qy1 = np.array([])
    array_qx2 = np.array([])
    array_qy2 = np.array([])
    if compute_footprint:
        logging.info("Computing footprints.")
        try:
            array_qx1, array_qy1 = return_footprint(
                collider_check.collider, nemitt_x, beam="lhcb1", n_turns=2000
            )
            array_qx2, array_qy2 = return_footprint(
                collider_check.collider, nemitt_x, beam="lhcb2", n_turns=2000
            )
        except KeyError:
            logging.warning(
                "Could not compute footprint as 'beambeam_scale' is probably non existent."
            )
        except AssertionError:
            logging.warning(AssertionError)

    # Get knobs
    logging.info("Getting knobs.")
    dic_knob_str = compute_knob_str(collider_check)

    # Store everything in a dictionary
    dic_global_var = {
        "l_lumi": l_lumi,
        "dic_tw_b1": dic_tw_b1,
        "dic_tw_b2": dic_tw_b2,
        "dic_separation_ip": dic_separation_ip,
        "dic_position_ip": dic_position_ip,
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
        "energy": energy,
        "cross_section": cross_section,
        "configuration_str": configuration_str,
        "dic_knob_str": dic_knob_str,
    }

    return dic_global_var


# ==================================================================================================
# --- Functions to load dashboard variables
# ==================================================================================================
def return_dataframe_elements_from_line(line):
    """
    Returns a pandas DataFrame containing the elements of a given line object.

    Args:
    line (Line): A Line object containing elements to be extracted into a DataFrame.

    Returns:
    df_elements (pandas.DataFrame): A DataFrame containing the elements of the given line object.
    """
    return pd.DataFrame(
        [
            {
                varname: getattr(x, varname)
                for varname in ["length", "knl", "ksl", "_order"]
                if hasattr(x, varname)
            }
            for x in line.elements
        ]
    )


def return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw):
    """
    Corrects the dataframe of elements for thin lens approximation.

    Args:
        df_elements (pandas.DataFrame): The dataframe of elements to be corrected.
        df_tw (pandas.DataFrame): The corresponding Twiss from which the correction is computed.

    Returns:
        pandas.DataFrame: The corrected dataframe of elements.
    """

    df_elements_corrected = df_elements.copy(deep=True)

    # Get duplicated elements, according to regex, whose name:
    # - does not contain the words "entry" or "exit" or ends with "f".
    # - does not contain a period before the end of the string.
    # - contain exactly two consecutive periods.
    df_tw_duplicated_elements = df_tw[
        df_tw.name.str.contains("^(?!.*(?:entry|exit|[^f]*f[^.]*$)).*\.{2}.*", regex=True)
    ]

    # Get original elements
    df_tw_original_elements = df_tw[
        df_tw.name.isin(df_tw_duplicated_elements.name.str.split("..", regex=False).str[0])
    ]

    # ? Remove drifts for now as it doesn't change the output and make computation faster
    # # Add drifts (original elements end with ..0) using concat
    # df_tw_original_elements = pd.concat(
    #     [
    #         df_tw_original_elements,
    #         df_tw[df_tw.name.str.contains(r"^(?:(?!\.\.|entry|exit).)*$", regex=True)],
    #     ]
    # )

    # Add all thin lenses (length + strength)
    for i, row in df_tw_duplicated_elements.iterrows():
        # Correct for thin lens approximation and weird duplicates
        name = row["name"].split("..")[0]
        try:
            index = df_tw_original_elements[df_tw_original_elements.name == name].index[0]
        except IndexError:
            continue

        # Add length
        if np.isnan(df_elements_corrected.loc[index]["length"]):
            df_elements_corrected.at[index, "length"] = 0.0
        df_elements_corrected.at[index, "length"] += df_elements.loc[i]["length"]

        # Add strength
        if np.isnan(df_elements_corrected.loc[index]["knl"]).all():
            df_elements_corrected.at[index, "knl"] = (
                0.0
                if isinstance(df_elements.loc[i]["knl"], float)
                else np.array([0.0] * df_elements.loc[i]["knl"].shape[0], dtype=np.float64)
            )

        df_elements_corrected.at[index, "knl"] = (
            df_elements.loc[i]["knl"]
            if isinstance(df_elements.loc[i]["knl"], float)
            else df_elements_corrected.loc[index, "knl"] + np.array(df_elements.loc[i]["knl"])
        )

        # Replace order
        df_elements_corrected.at[index, "_order"] = df_elements.loc[i]["_order"]

    # Drop all duplicate rows
    df_elements_corrected.drop(index=df_tw_duplicated_elements.index, inplace=True)

    return df_elements_corrected


def return_twiss_dic(tw):
    """
    Returns a dictionary containing important Twiss parameters.

    Parameters:
    tw (Twiss): A Twiss object for the current line.

    Returns:
    dict: A dictionary containing the following keys:
        - qx (float): Horizontal tune.
        - qy (float): Vertical tune.
        - dqx (float): Horizontal chromaticity.
        - dqy (float): Vertical chromaticity.
        - c_minus (float): Linear coupling.
        - momentum_compaction_factor (float): Momentum compaction factor.
        - T_rev0 (float): Revolution period.
        - ip1 (numpy.ndarray): Momentum, position and beta functions at IP1.
        - ip2 (numpy.ndarray): Momentum, position and beta functions at IP2.
        - ip5 (numpy.ndarray): Momentum, position and beta functions at IP5.
        - ip8 (numpy.ndarray): Momentum, position and beta functions at IP8.
    """
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
        dic_tw[f"ip{str(ip)}"] = (
            tw.rows[f"ip{ip}"]
            .cols[
                "s",
                "x",
                "px",
                "y",
                "py",
                "betx",
                "bety",
                "dx_zeta",
                "dy_zeta",
                "dpx_zeta",
                "dpy_zeta",
            ]
            .to_pandas()
            .to_numpy()
            .squeeze()
        )

    return dic_tw


# ==================================================================================================
# --- Functions to build data tables
# ==================================================================================================
def return_data_table(df, id_table, twiss=True, simplify_tw=True):
    """
    Returns a (stylized) Dash DataTable object containing the twiss/survey data of a given line
        (through the pandas dataframe input).

    Args:
        df (pandas.DataFrame): The DataFrame containing the twiss/survey data to be displayed in
            the table.
        id_table (str): The ID to be assigned to the DataTable object (for Dash callback).
        twiss (bool, optional): Whether or not the DataFrame contains twiss data. Defaults to True.
            If False, it is assumed to contain survey data.
        simplify_tw (bool, optional): If True, simplifies the Twiss and Survey dataframes by
            removing duplicated elements to speed up computations. Defaults to True.

    Returns:
        dash_table.DataTable: The DataTable object populated with the data from the input DataFrame.
    """
    if twiss:
        df = df.drop(["W_matrix"], axis=1)
        idx_column_name = 0
    else:
        idx_column_name = 6

    # Change order of columns such that name is first
    df = df[["name"] + [col for col in df.columns if col != "name"]]

    # Simplify the dataframe removing all duplicated elements, entry and exit
    if simplify_tw:
        df = df[df["name"].str.contains(r"^(?:(?!\.\.|entry|exit).)*$", regex=True)]

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
            virtualization=False,
            page_size=25,
            page_current=0,
            # page_action="custom",
            # filter_action="custom",
            # filter_query="",
            # sort_action="custom",
            # sort_mode="multi",
            # sort_by=[],
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
            style_header={
                "backgroundColor": "rgb(30, 30, 30)",
                "color": "white",
                "padding": "1em",
            },
            style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            style_filter={"backgroundColor": "rgb(70, 70, 70)"},  # , "color": "white"},
            style_cell={"font-family": "sans-serif", "minWidth": 95},
        ),
    )
    return table, df


def return_footprint(collider, emittance, beam="lhcb1", n_turns=2000):
    """
    Calculates the collider footprint, for beam 1 or beam 2.

    Args:
        collider (Collider): A Collider object.
        emittance (float): The emittance of the beam.
        beam (str, optional): The name of the beam. Defaults to "lhcb1".
        n_turns (int, optional): The number of turns to simulate to compute the footprint. Defaults
            to 2000.

    Returns:
        tuple: A tuple containing the detuning (qx and qy values) of the footprint.
    """
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


def compute_knob_str(collider_check):
    whole_str = ""
    l_knobs = []
    with io.StringIO() as buf, redirect_stdout(buf):
        for k in collider_check.collider.vars.keys():
            collider_check.collider.vars[k]._info(limit=None)
            l_knobs.append(k)
            print("****")
        whole_str = buf.getvalue()

    l_knob_str = whole_str.split("****\n")
    return dict(zip(l_knobs, l_knob_str))

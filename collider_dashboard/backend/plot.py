"""Thid module contains all the functions used to produce graphs in the app."""

import itertools

# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import from standard library
import logging
from functools import lru_cache

# Import third-party packages
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# Import local packages
from .resonance import get_working_diagram


# ==================================================================================================
# --- Plotting functions
# ==================================================================================================
def return_radial_background_traces(df_sv):
    """
    Returns a list of 4 scatter traces, representing the (background) radial lines of the collider.
        Useful when plotting the survey.

    Args:
        df_sv (pandas.DataFrame): The survey DataFrame containing the X and Z coordinates of the
            beam pipe.

    Returns:
        list: A list of plotly.graph_objs.Scattergl objects representing the radial traces.
    """
    # Add 4 radial lines, each parametrized with a different set of x1, x2, y1, y2
    l_traces_background = []
    for x1, x2, y1, y2 in [
        [np.mean(df_sv["X"]), np.mean(df_sv["X"]), -50000, 50000],
        [-50000, 50000, 0, 0],
        [-50000 + np.mean(df_sv["X"]), 50000 + np.mean(df_sv["X"]), -50000, 50000],
        [-50000 + np.mean(df_sv["X"]), 50000 + np.mean(df_sv["X"]), 50000, -50000],
    ]:
        l_traces_background.append(
            go.Scattergl(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                name="Drift space",
                line_color="lightgrey",
                line_width=1,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Return result in a list readable by plotly.add_traces()
    return l_traces_background


def return_beam_pipe_trace(df_sv):
    """
    Returns a Plotly trace representing the beam pipe in the survey.

    Args:
        df_sv (pandas.DataFrame): The survey DataFrame containing the X and Z coordinates of the
            beam pipe.

    Returns:
        plotly.graph_objs.Scattergl: The Plotly trace containing the beam coordinates.
    """
    return go.Scattergl(
        x=df_sv["X"],
        y=df_sv["Z"],
        mode="lines",
        name="Drift space",
        line_color="darkgrey",
        line_width=3,
        hoverinfo="skip",
        showlegend=False,
    )


def return_multipole_trace(
    df_elements,
    df_sv,
    order,
    strength_magnification_factor=5000,
    add_ghost_trace=True,
    l_indices_to_keep=None,
    flat=False,
    xaxis=None,
    yaxis=None,
):
    """
    Returns a list of multipole traces (represented as a rectangle).

    Args:
        df_elements (pandas.DataFrame): A DataFrame containing the elements of the accelerator
            lattice.
        df_sv (pandas.DataFrame): DataFrame containing the survey data.
        order (int): Order of the multipole to represent.
        strength_magnification_factor (float, optional): Factor by which to (visually) magnify the
            strength values. Defaults to 5000.
        add_ghost_trace (bool, optional): Whether to add a ghost trace for legend purposes. Defaults
            to True.
        l_indices_to_keep (list, optional): List of element indices to keep (not to represent the
            whole survey). Defaults to None.
        flat (bool, optional): Whether to return a flat (1D) trace. Defaults to False.
        xaxis (dict, optional): Dictionary containing the x-axis range. Defaults to None.
        yaxis (dict, optional): Dictionary containing y-axis range. Defaults to None.

    Returns:
        plotly.graph_objs._scatter.Scatter: A trace representing a multipole of a given order.
    """
    if flat:
        return return_flat_multipole_trace(
            df_elements,
            df_sv,
            order,
            strength_magnification_factor=strength_magnification_factor,
            add_ghost_trace=add_ghost_trace,
            l_indices_to_keep=l_indices_to_keep,
            xaxis=xaxis,
            yaxis=yaxis,
        )
    else:
        return return_circular_multipole_trace(
            df_elements,
            df_sv,
            order,
            strength_magnification_factor=strength_magnification_factor,
            add_ghost_trace=add_ghost_trace,
            l_indices_to_keep=l_indices_to_keep,
        )


def return_circular_multipole_trace(
    df_elements,
    df_sv,
    order,
    strength_magnification_factor=5000,
    add_ghost_trace=True,
    l_indices_to_keep=None,
):
    """
    Returns a list of multipole traces (represented as a rectangle) in a 2D graph.

    Args:
        df_elements (pandas.DataFrame): A DataFrame containing the elements of the accelerator
            lattice.
        df_sv (pandas.DataFrame): 1 DataFrame containing the survey data.
        order (int): The order of the multipoles to plot (0 for dipoles, 1 for quadrupoles, etc.).
        strength_magnification_factor (float, optional): Factor by which to (visually) magnify the
            strength values. Defaults to 5000.
        add_ghost_trace (bool, optional): Whether to add a ghost trace for legend purposes. Defaults
            to True.
        l_indices_to_keep (list, optional): List of element indices to keep (not to represent the
            whole survey). Defaults to None.

    Returns:
        list: A list of plotly traces representing the multipoles to represent.
    """

    # Get corresponding colors and name for the multipoles
    if order == 0:
        color = px.colors.qualitative.Plotly[0]
        name = "Dipoles"
    elif order == 1:
        color = px.colors.qualitative.Plotly[1]
        name = "Quadrupoles"
    elif order == 2:
        color = px.colors.qualitative.Plotly[-1]
        name = "Sextupoles"
        strength_magnification_factor = strength_magnification_factor / 2
    elif order == 3:
        color = px.colors.qualitative.Plotly[2]
        name = "Octupoles"
        strength_magnification_factor = strength_magnification_factor / 40
    else:
        raise ValueError("The order of the multipole is not recognized.")

    # Function to filter magnet strength
    def return_correct_strength(x):
        try:
            return x[order]
        except Exception:
            return float(x)

    # Get strength of all multipoles of the requested order
    s_knl = df_elements[df_elements._order == order]["knl"].apply(return_correct_strength)

    # Remove zero-strength dipoles and magnify
    s_knl = s_knl[s_knl != 0] * strength_magnification_factor

    # Filter out indices outside of the range if needed
    if l_indices_to_keep is not None:
        s_knl = s_knl[s_knl.index.isin(l_indices_to_keep)]

    # Get corresponding lengths
    s_lengths = df_elements.loc[s_knl.index].length

    # Ghost trace for legend if requested
    if add_ghost_trace:
        ghost_trace = go.Scattergl(
            x=[200000, 200001],
            y=[0, 0],
            mode="lines",
            line=dict(color=color, width=5),
            showlegend=True,
            name=name,
            legendgroup=name,
            # visible="legendonly",
        )

    # Add all multipoles at once, merge them by line width
    dic_trace = {}
    for i, row in df_sv.loc[s_knl.index].iterrows():
        width = (
            np.ceil(s_lengths[i]) if not np.isnan(s_lengths[i]) or np.ceil(s_lengths[i]) == 0 else 1
        )

        if width in dic_trace:
            dic_trace[width]["x"].extend(
                [row["X"], row["X"] + s_knl[i] * np.cos(row["theta"]), None]
            )
            dic_trace[width]["y"].extend(
                [row["Z"], row["Z"] - s_knl[i] * np.sin(row["theta"]), None]
            )
            dic_trace[width]["customdata"].extend([row["name"], row["name"], None])
        else:
            dic_trace[width] = {
                "x": [row["X"], row["X"] + s_knl[i] * np.cos(row["theta"]), None],
                "y": [row["Z"], row["Z"] - s_knl[i] * np.sin(row["theta"]), None],
                "customdata": [row["name"], row["name"], None],
                "mode": "lines",
                "line": dict(
                    color=color,
                    width=width,
                ),
                "showlegend": False,
                "name": row["name"],
                "legendgroup": name,
                "hovertemplate": "Magnet: %{customdata}" + "<extra></extra>",
            }

    l_traces = [go.Scattergl(**dic_trace[width]) for width in dic_trace]

    # Return result in a list readable by plotly.add_traces()
    return [ghost_trace] + l_traces if add_ghost_trace else l_traces


def return_flat_multipole_trace(
    df_elements,
    df_sv,
    order,
    strength_magnification_factor=5000,
    add_ghost_trace=True,
    l_indices_to_keep=None,
    xaxis=None,
    yaxis=None,
):
    """
    Returns a list of multipole traces (each represented as a rectangle) in a 1D graph.

    Args:
        df_elements (pandas.DataFrame): A DataFrame containing the elements of the accelerator
            lattice.
        df_sv (pandas.DataFrame): DataFrame containing the survey data.
        order (int): Order of the multipole to represent.
        strength_magnification_factor (float, optional): Factor by which to (visually) magnify the
            strength values. Defaults to 5000.
        add_ghost_trace (bool, optional): Whether to add a ghost trace for legend purposes. Defaults
            to True.
        l_indices_to_keep (list, optional): List of element indices to keep (not to represent the
            whole survey). Defaults to None.
        xaxis (dict, optional): Dictionary containing the x-axis range. Defaults to None.
        yaxis (dict, optional): Dictionary containing y-axis range. Defaults to None.

    Returns:
        list: List of plotly traces for all multipoles of the given order.
    """

    # Get corresponding colors and name for the multipoles
    if order == 0:
        color = px.colors.qualitative.Plotly[0]
        name = "Dipoles"
        strength_magnification_factor = strength_magnification_factor * 20
    elif order == 1:
        color = px.colors.qualitative.Plotly[1]
        name = "Quadrupoles"
        strength_magnification_factor = strength_magnification_factor * 6
    elif order == 2:
        color = px.colors.qualitative.Plotly[-1]
        name = "Sextupoles"
        strength_magnification_factor = strength_magnification_factor * 2
    elif order == 3:
        color = px.colors.qualitative.Plotly[2]
        name = "Octupoles"
        strength_magnification_factor = strength_magnification_factor / 4
    else:
        raise ValueError("The order of the multipole is not recognized.")

    # Function to filter magnet strength
    def return_correct_strength(x):
        try:
            return x[order]
        except Exception:
            return float(x)

    # Get strength of all multipoles of the requested order
    s_knl = df_elements[df_elements._order == order]["knl"].apply(return_correct_strength)

    # Remove zero-strength dipoles and magnify
    s_knl = s_knl[s_knl != 0] * strength_magnification_factor

    # Filter out indices outside of the range if needed
    if l_indices_to_keep is not None:
        s_knl = s_knl[s_knl.index.isin(l_indices_to_keep)]

    # Get corresponding lengths
    s_lengths = df_elements.loc[s_knl.index].length

    # Ghost trace for legend if requested
    if add_ghost_trace:
        if xaxis is not None and yaxis is not None:
            ghost_trace = go.Scattergl(
                x=[200000, 200001],
                y=[0, 0],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=True,
                name=name,
                legendgroup=name,
                xaxis=xaxis,
                yaxis=yaxis,
                # visible="legendonly",
            )
        else:
            ghost_trace = go.Scattergl(
                x=[200000, 200001],
                y=[0, 0],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=True,
                name=name,
                legendgroup=name,
            )

    # Add all multipoles at once, merge them by line width
    dic_trace = {}
    for i, row in df_sv.loc[s_knl.index].iterrows():
        width = (
            np.ceil(s_lengths[i]) if not np.isnan(s_lengths[i]) or np.ceil(s_lengths[i]) == 0 else 1
        )
        if width in dic_trace:
            dic_trace[width]["x"].extend([row["s"], row["s"], None])
            dic_trace[width]["y"].extend([0, s_knl[i], None])
            dic_trace[width]["customdata"].extend([row["name"], row["name"], None])
        else:
            if xaxis is not None and yaxis is not None:
                dic_trace[width] = {
                    "x": [row["s"], row["s"], None],
                    "y": [0, s_knl[i], None],
                    "customdata": [row["name"], row["name"], None],
                    "mode": "lines",
                    "line": dict(
                        color=color,
                        width=width,
                    ),
                    "showlegend": False,
                    "name": row["name"],
                    "legendgroup": name,
                    "hovertemplate": "Magnet: %{customdata}" + "<extra></extra>",
                    "xaxis": xaxis,
                    "yaxis": yaxis,
                }
            else:
                dic_trace[width] = {
                    "x": [row["s"], row["s"], None],
                    "y": [0, s_knl[i], None],
                    "customdata": [row["name"], row["name"], None],
                    "mode": "lines",
                    "line": dict(
                        color=color,
                        width=width,
                    ),
                    "showlegend": False,
                    "name": row["name"],
                    "legendgroup": name,
                    "hovertemplate": "Magnet: %{customdata}" + "<extra></extra>",
                }

    l_traces = [go.Scattergl(**dic_trace[width]) for width in dic_trace]

    # Return result in a list readable by plotly.add_traces()
    return [ghost_trace] + l_traces if add_ghost_trace else l_traces


def return_IP_trace(df_sv, add_ghost_trace=True):
    """
    Args:
        df_sv (pandas.DataFrame): The dataframe containing the survey data.
        add_ghost_trace (bool, optional): Whether to add a ghost trace for the IP elements
            (for the legend). Defaults to True.

    Rreturns:
        list of plotly.graph_objs._scattergl.Scattergl: The list of plotly traces representing
            the IP elements.
    """
    # Get dataframe containing only IP elements
    df_ip = df_sv[df_sv["name"].str.startswith("ip")]

    # Ghost trace for legend if requested
    if add_ghost_trace:
        ghost_trace = go.Scattergl(
            x=[200000, 200000],
            y=[0, 0],
            mode="markers",
            # marker_symbol=218,
            marker_line_color="midnightblue",
            marker_color="grey",
            marker_line_width=2,
            marker_size=15,
            showlegend=True,
            name="IP",
            legendgroup="IP",
            # visible="legendonly",
        )

    # Add all IP at once
    l_traces = [
        go.Scattergl(
            mode="markers",
            x=df_ip["X"],
            y=df_ip["Z"],
            customdata=df_ip["name"],
            # marker_symbol=218,
            marker_line_color="midnightblue",
            marker_color="grey",
            marker_line_width=2,
            marker_size=15,
            name="IP",
            showlegend=False,
            legendgroup="IP",
            hovertemplate="IP: %{customdata}" + "<extra></extra>",
        )
    ]

    # Return result in a list readable by plotly.add_traces()
    return [ghost_trace] + l_traces if add_ghost_trace else l_traces


def return_optic_trace(
    df_sv,
    df_tw,
    type_trace,
    beam_2=False,
    l_indices_to_keep=None,
):
    """
    Returns a plotly scatter trace for a given type of optics data.

    Args:
        df_sv (pandas.DataFrame): DataFrame containing the survey data.
        df_tw (pandas.DataFrame): DataFrame containing the twiss data.
        type_trace (str): Type of trace to plot. Must be one of "betax", "bety", "dx", "dy", "x",
            or "y".
        beam_2 (bool, optional): Whether to plot the trace for beam 2. Defaults to False.
        l_indices_to_keep (list, optional): List of indices to keep, if traces arenot represented
            all along the survey. Defaults to None.

    Returns:
        plotly.graph_objs.Scattergl: Scatter trace for the given type of trace to plot.
    """
    # Get the right twiss dataframe and plotting parameters
    match type_trace:
        case "betax":
            magnification_factor = 1.0
            tw_name = "betx"
            name = r"$\beta_{x2}^{0.8}$" if beam_2 else r"$\beta_{x1}^{0.8}$"
            color = px.colors.qualitative.Plotly[3]
            dash = "dash" if beam_2 else None
            exponent = 0.8

        case "bety":
            magnification_factor = 1.0
            tw_name = "bety"
            name = r"$\beta_{y2}^{0.8}$" if beam_2 else r"$\beta_{y1}^{0.8}$"
            color = px.colors.qualitative.Plotly[4]
            dash = "dash" if beam_2 else None
            exponent = 0.8

        case "dx":
            magnification_factor = 100
            tw_name = "dx"
            name = r"$100D_{x2}$" if beam_2 else r"$100D_{x1}$"
            color = px.colors.qualitative.Plotly[5]
            dash = "dash" if beam_2 else None
            exponent = 1.0

        case "dy":
            magnification_factor = 100
            tw_name = "dy"
            name = r"$100D_{y2}$" if beam_2 else r"$100D_{y1}$"
            color = px.colors.qualitative.Plotly[6]
            dash = "dash" if beam_2 else None
            exponent = 1.0

        case "x":
            magnification_factor = 100000
            tw_name = "x"
            name = r"$10^5x_2$" if beam_2 else r"$10^5x_1$"
            color = px.colors.qualitative.Plotly[7]
            dash = "dash" if beam_2 else None
            exponent = 1.0

        case "y":
            magnification_factor = 100000
            tw_name = "y"
            name = r"$10^5y_2$" if beam_2 else r"$10^5y_1$"
            color = px.colors.qualitative.Plotly[8]
            dash = "dash" if beam_2 else None
            exponent = 1.0

        case _:
            raise ValueError("The type of trace is not recognized.")

    # Correct for circular projection depending if x-coordinate has been reversed or not
    correction = -1 if beam_2 else 1
    # Only keep requested indices
    if l_indices_to_keep is not None:
        df_sv_temp = df_sv[df_sv.index.isin(l_indices_to_keep)]
        df_tw_temp = df_tw[df_tw.index.isin(l_indices_to_keep)]
    else:
        df_sv_temp = df_sv
        df_tw_temp = df_tw

    # Return the trace
    return go.Scattergl(
        x=[None]
        + list(
            df_sv_temp["X"]
            + df_tw_temp[tw_name] ** exponent
            * correction
            * magnification_factor
            * np.cos(df_sv_temp["theta"])
        )
        + [None],
        y=[None]
        + list(
            df_sv_temp["Z"]
            - df_tw_temp[tw_name] ** exponent * magnification_factor * np.sin(df_sv_temp["theta"])
        )
        + [None],
        mode="lines",
        line=dict(color=color, width=2, dash=dash),
        showlegend=True,
        name=name,
        # visible="legendonly" if hide_optics_traces_initially else True,
        visible=True if (not beam_2 and "bet" in type_trace) else "legendonly",
    )


def add_multipoles_to_fig(
    fig,
    df_elements,
    df_sv,
    l_indices_to_keep,
    add_dipoles,
    add_quadrupoles,
    add_sextupoles,
    add_octupoles,
    flat=False,
    row=None,
    col=None,
    xaxis=None,
    yaxis=None,
):
    """
    Add multipole traces to a given plotly figure.

    Args:
        fig (plotly.graph_objs.Figure): The figure to which the multipole traces will be added.
        df_elements (pandas.DataFrame): DataFrame containing the accelerator elements.
        df_sv (pandas.DataFrame): DataFrame containing the accelerator survey.
        l_indices_to_keep (list of int): List of indices of the elements to keep in the plot (to
            only represent part of the survey).
        add_dipoles (bool): Whether to add dipoles to the plot.
        add_quadrupoles (bool): Whether to add quadrupoles to the plot.
        add_sextupoles (bool): Whether to add sextupoles to the plot.
        add_octupoles (bool): Whether to add octupoles to the plot.
        flat (bool, optional): Whether to make a 1D or 2D plot (default is False, i.e. 2D plot).
        row (int, optional): The row index of the subplot to which the traces will be added
            (default is None).
        col (int, optional): The column index of the subplot to which the traces will be added
            (default is None).
        xaxis (str, optional): The x-axis type of the plot (default is None).
        yaxis (str, optional): The y-axis type of the plot (default is None).

    Returns:
        plotly.graph_objs.Figure: The updated figure with the multipole traces added.
    """

    for order, add in zip(
        [0, 1, 2, 3], [add_dipoles, add_quadrupoles, add_sextupoles, add_octupoles]
    ):
        if add:
            if row is None or col is None:
                fig.add_traces(
                    return_multipole_trace(
                        df_elements,
                        df_sv,
                        order=order,
                        strength_magnification_factor=5000,
                        l_indices_to_keep=l_indices_to_keep,
                        flat=flat,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

            else:
                l_traces = return_multipole_trace(
                    df_elements,
                    df_sv,
                    order=order,
                    strength_magnification_factor=5000,
                    l_indices_to_keep=l_indices_to_keep,
                    flat=flat,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
                for trace in l_traces:
                    fig.add_trace(
                        trace,
                        row=row,
                        col=col,
                    )
    return fig


def add_optics_to_fig(
    fig,
    plot_horizontal_betatron,
    plot_vertical_betatron,
    plot_horizontal_dispersion,
    plot_vertical_dispersion,
    plot_horizontal_position,
    plot_vertical_position,
    df_sv,
    df_tw,
    beam_2=False,
    l_indices_to_keep=None,
):
    """
    Add optics traces to a plotly figure.

    Args:
        fig (plotly.graph_objs._figure.Figure): The figure to which the traces will be added.
        plot_horizontal_betatron (bool): Whether to add the horizontal betatron trace.
        plot_vertical_betatron (bool): Whether to add the vertical betatron trace.
        plot_horizontal_dispersion (bool): Whether to add the horizontal dispersion trace.
        plot_vertical_dispersion (bool): Whether to add the vertical dispersion trace.
        plot_horizontal_position (bool): Whether to add the horizontal position trace.
        plot_vertical_position (bool): Whether to add the vertical position trace.
        df_sv (pandas.DataFrame): The survey data.
        df_tw (pandas.DataFrame): The twiss data.
        beam_2 (bool, optional): Whether to represent the second beam. Defaults to False.
        l_indices_to_keep (list of int, optional): The indices of the elements to keep (to only
            represent part of the survey). Defaults to None.

    Returns:
        plotly.graph_objs._figure.Figure: The figure with the added traces.
    """

    # Add horizontal betatron if requested
    if plot_horizontal_betatron:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="betax",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    # Add vertical betatron if requested
    if plot_vertical_betatron:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="bety",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    # Add horizontal dispersion if requested
    if plot_horizontal_dispersion:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="dx",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    # Add vertical dispersion if requested
    if plot_vertical_dispersion:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="dy",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    # Add horizontal position if requested
    if plot_horizontal_position:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="x",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    # Add vertical position if requested
    if plot_vertical_position:
        fig.add_trace(
            return_optic_trace(
                df_sv,
                df_tw,
                type_trace="y",
                beam_2=beam_2,
                l_indices_to_keep=l_indices_to_keep,
            )
        )

    return fig


def return_plot_lattice_with_tracking(
    df_sv,
    df_elements,
    df_tw,
    df_sv_2=None,
    df_tw_2=None,
    add_dipoles=True,
    add_quadrupoles=True,
    add_sextupoles=True,
    add_octupoles=True,
    add_IP=True,
    l_indices_to_keep=None,
    plot_horizontal_betatron=True,
    plot_vertical_betatron=True,
    plot_horizontal_dispersion=True,
    plot_vertical_dispersion=True,
    plot_horizontal_position=True,
    plot_vertical_position=True,
    add_optics_beam_2=True,
):
    """
    Returns a Plotly figure showing the lattice of a particle accelerator, with beam tracking data.

    Args:
        df_sv (pandas.DataFrame): Dataframe containing the accelerator survey for beam 1.
        df_elements (pandas.DataFrame): Dataframe containing the accelerator elements.
        df_tw (pandas.DataFrame): Dataframe containing the twiss parameters of the accelerators for
            beam 1.
        df_sv_2 (pandas.DataFrame, optional): Dataframe containing the accelerator survey for beam 2.
            Defaults to None.
        df_tw_2 (pandas.DataFrame, optional): Dataframe containing the twiss parameters of the
            accelerators for beam 2. Defaults to None.
        add_dipoles (bool, optional): Whether to add dipoles to the plot. Defaults to True.
        add_quadrupoles (bool, optional): Whether to add quadrupoles to the plot. Defaults to True.
        add_sextupoles (bool, optional): Whether to add sextupoles to the plot. Defaults to True.
        add_octupoles (bool, optional): Whether to add octupoles to the plot. Defaults to True.
        add_IP (bool, optional): Whether to add the interaction point to the plot. Defaults to True.
        l_indices_to_keep (list of int, optional): List of indices of the elements to keep in the
            plot. Defaults to None (keep all elements).
        plot_horizontal_betatron (bool, optional): Whether to plot the horizontal betatron function.
            Defaults to True.
        plot_vertical_betatron (bool, optional): Whether to plot the vertical betatron function.
            Defaults to True.
        plot_horizontal_dispersion (bool, optional): Whether to plot the horizontal dispersion.
            Defaults to True.
        plot_vertical_dispersion (bool, optional): Whether to plot the vertical dispersion.
            Defaults to True.
        plot_horizontal_position (bool, optional): Whether to plot the horizontal position.
            Defaults to True.
        plot_vertical_position (bool, optional): Whether to plot the vertical position.
            Defaults to True.
        add_optics_beam_2 (bool, optional): Whether to add optics traces for the second beam.
            Defaults to True.

    Returns:
        plotly.graph_objs._figure.Figure: The Plotly figure object representing the lattice with
            beam tracking data.
    """

    # Center X coordinate (otherwise conversion to polar coordinates is not possible)
    # X_centered = df_sv["X"] - np.mean(df_sv["X"])

    # Get corresponding angle
    # l_theta = np.arctan2(df_sv["Z"], X_centered)

    # Build plotly figure
    fig = go.Figure()

    # Add lines to the bakckground delimit octants
    fig.add_traces(return_radial_background_traces(df_sv))

    # Add beam pipe
    fig.add_trace(return_beam_pipe_trace(df_sv))

    # Add multipoles
    fig = add_multipoles_to_fig(
        fig,
        df_elements,
        df_sv,
        l_indices_to_keep,
        add_dipoles,
        add_quadrupoles,
        add_sextupoles,
        add_octupoles,
        flat=False,
    )

    # Add IP if requested
    if add_IP:
        fig.add_traces(return_IP_trace(df_sv))

    # Add optics traces for beam_1
    fig = add_optics_to_fig(
        fig,
        plot_horizontal_betatron,
        plot_vertical_betatron,
        plot_horizontal_dispersion,
        plot_vertical_dispersion,
        plot_horizontal_position,
        plot_vertical_position,
        df_sv,
        df_tw,
        beam_2=False,
        l_indices_to_keep=l_indices_to_keep,
    )

    # Add optics traces for beam_2 if requested
    if add_optics_beam_2:
        if df_sv_2 is None or df_tw_2 is None:
            print("Warning: df_sv_2 or df_tw_2 is None, beam_2 optics will not be plotted")
        else:
            fig = add_optics_to_fig(
                fig,
                plot_horizontal_betatron,
                plot_vertical_betatron,
                plot_horizontal_dispersion,
                plot_vertical_dispersion,
                plot_horizontal_position,
                plot_vertical_position,
                df_sv_2,
                df_tw_2,
                beam_2=True,
                l_indices_to_keep=l_indices_to_keep,
            )

    # Set general layout for figure
    fig.update_layout(
        # title_text="Survey of the current simulation",
        # title_x=0.5,
        # title_xanchor="center",
        showlegend=True,
        xaxis_range=[df_sv["X"].min() - 300, df_sv["X"].max() + 300],
        yaxis_range=[df_sv["Z"].min() - 300, df_sv["Z"].max() + 300],
        xaxis_showgrid=True,
        xaxis_showticklabels=False,
        yaxis_showgrid=True,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        yaxis_showticklabels=False,
        legend_tracegroupgap=30,
        # width=1000,
        # height=1000,
        margin=dict(l=10, r=10, b=10, t=10, pad=10),
        dragmode="pan",
    )

    # Make background transparent
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def add_scatter_trace(
    fig,
    x,
    y,
    name,
    row,
    col,
    xaxis,
    yaxis,
    visible=None,
    color=None,
    legendgroup=None,
    dashed=False,
    opacity=0.8,
):
    """
    Add a scatter trace to a plotly figure object.

    Args:
        fig (plotly.graph_objs.Figure): The figure object to add the trace to.
        x (list): The x-axis data for the trace.
        y (list): The y-axis data for the trace.
        name (str): The name of the trace.
        row (int): The row number of the subplot to add the trace to.
        col (int): The column number of the subplot to add the trace to.
        xaxis (str): The x-axis to use for the trace.
        yaxis (str): The y-axis to use for the trace.
        visible (str, optional): The visibility of the trace. Defaults to None.
        color (str, optional): The color of the trace. Defaults to None.
        legendgroup (str, optional): The legend group of the trace. Defaults to None.
        dashed (bool, optional): Whether to use a dashed line for the trace. Defaults to False.
        opacity (float, optional): The opacity of the trace. Defaults to 0.8.

    Returns:
        plotly.graph_objs.Figure: The updated figure object.
    """
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            showlegend=True,
            name=name,
            xaxis=xaxis,
            yaxis=yaxis,
            visible=visible,
            opacity=opacity,
            line=dict(color=color, dash="dash") if dashed else dict(color=color),
            # Deactivate legendgroup for now as it doesn't work
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )
    return fig


def return_plot_optics(df_tw_b1, df_tw_b2, df_sv, df_elements, empty=False):
    """
    Returns a Plotly figure object with subplots for magnet traces, beta functions, position
        functions, and dispersion functions.

    Args:
        df_tw_b1 (pandas.DataFrame): DataFrame containing Twiss parameters for beam 1.
        df_tw_b2 (pandas.DataFrame): DataFrame containing Twiss parameters for beam 2.
        df_sv (pandas.DataFrame): DataFrame containing survey data.
        df_elements (pandas.DataFrame): DataFrame containing element data.
        empty (bool, optional): If True, returns an empty figure. Defaults to False.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object with subplots for magnet elements,
            beta functions, position functions, and dispersion functions.
    """
    # Build figure
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    if not empty:
        # Add traces for magnets
        fig = add_multipoles_to_fig(
            fig,
            df_elements,
            df_sv,
            l_indices_to_keep=None,
            add_dipoles=True,
            add_quadrupoles=True,
            add_sextupoles=True,
            add_octupoles=True,
            flat=True,
            row=1,
            col=1,
            xaxis="x",
            yaxis="y",
        )

        # Add traces for beta functions
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["betx"],
            r"$\beta_{x_1}$",
            2,
            1,
            "x",
            "y",
            color="cyan",
            legendgroup="beta-1",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["bety"],
            r"$\beta_{y_1}$",
            2,
            1,
            "x",
            "y",
            # visible="legendonly",
            color="cyan",
            legendgroup="beta-2",
            dashed=True,
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["betx"],
            r"$\beta_{x_2}$",
            2,
            1,
            "x",
            "y",
            # visible="legendonly",
            color="tomato",
            legendgroup="beta-3",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["bety"],
            r"$\beta_{y_2}$",
            2,
            1,
            "x",
            "y",
            # visible="legendonly",
            color="tomato",
            legendgroup="beta-4",
            dashed=True,
        )

        # Add traces for position functions
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["x"],
            r"$x_1$",
            3,
            1,
            "x",
            "y2",
            color="cyan",
            legendgroup="position-1",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["y"],
            r"$y_1$",
            3,
            1,
            "x",
            "y2",
            # visible="legendonly",
            color="cyan",
            legendgroup="position-2",
            dashed=True,
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["x"],
            r"$x_2$",
            3,
            1,
            "x",
            "y2",
            # visible="legendonly",
            color="tomato",
            legendgroup="position-3",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["y"],
            r"$y_2$",
            3,
            1,
            "x",
            "y2",
            # visible="legendonly",
            color="tomato",
            legendgroup="position-4",
            dashed=True,
        )

        # Add traces for dispersion functions
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["dx"],
            r"$D_{x_1}$",
            4,
            1,
            "x",
            "y3",
            color="cyan",
            legendgroup="dispersion-1",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b1["s"],
            df_tw_b1["dy"],
            r"$D_{y_1}$",
            4,
            1,
            "x",
            "y3",
            # visible="legendonly",
            color="cyan",
            legendgroup="dispersion-2",
            dashed=True,
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["dx"],
            r"$D_{x_2}$",
            4,
            1,
            "x",
            "y3",
            # visible="legendonly",
            color="tomato",
            legendgroup="dispersion-3",
        )
        fig = add_scatter_trace(
            fig,
            df_tw_b2["s"],
            df_tw_b2["dy"],
            r"$D_{y_2}$",
            4,
            1,
            "x",
            "y3",
            # visible="legendonly",
            color="tomato",
            legendgroup="dispersion-4",
            dashed=True,
        )

        # Add horizontal lines for all ips
        for ip in [1, 2, 5, 8]:
            fig.add_vline(
                x=float(df_tw_b1[df_tw_b1["name"] == f"ip{str(ip)}"]["s"].iloc[0]),
                line_width=1,
                line_dash="dash",
                line_color="pink",
                annotation_text=f"IP {ip}",
                annotation_position="top right",
            )

    fig.update_layout(
        showlegend=True,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        legend_tracegroupgap=30,
        dragmode="pan",
        uirevision="Don't change",
        margin=dict(l=10, r=10, b=10, t=10, pad=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Update yaxis properties
    fig.update_xaxes(range=[0, df_tw_b1["s"].iloc[-1] + 1])
    fig.update_yaxes(title_text=r"$\beta_{x,y}[m]$", range=[0, 10000], row=2, col=1)
    fig.update_yaxes(title_text=r"(Closed orbit)$_{x,y}[m]$", range=[-0.03, 0.03], row=3, col=1)
    fig.update_yaxes(title_text=r"$D_{x,y}[m]$", range=[-3, 3], row=4, col=1)
    fig.update_xaxes(title_text=r"$s[m]$", row=4, col=1)
    fig.update_yaxes(fixedrange=True)

    return fig


def return_plot_filling_scheme(array_b1, array_b2, i_bunch_b1, i_bunch_b2, beam_beam_schedule):
    """
    Returns a Plotly figure object representing the filling scheme and number of Long-Range/Head-on.

    Args:
        array_b1 (numpy.ndarray): Filling scheme for beam 1.
        array_b2 (numpy.ndarray): Filling scheme for beam 2.
        i_bunch_b1 (int): Index of the selected bunch for tracking in beam 1.
        i_bunch_b2 (int): Index of the selected bunch for tracking in beam 2.
        beam_beam_schedule (pandas.DataFrame): Dataframe containing the beam-beam schedule information.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object representing the filling scheme and
            number of beam-beam interactions.
    """
    # ! i_bunch_b2 is not used for now

    # Get indices of slots filled with bunches
    non_zero_indices_b1 = np.nonzero(array_b1)[0]
    non_zero_indices_b2 = np.nonzero(array_b2)[0]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    # Add the filling scheme for beam 1
    fig.add_trace(
        go.Scattergl(
            x=non_zero_indices_b1,
            y=array_b1[non_zero_indices_b1],
            mode="markers",
            marker=dict(color="cyan", size=5),
            # name="Beam 1",
            xaxis="x",
            yaxis="y",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add the filling scheme for beam 2
    fig.add_trace(
        go.Scattergl(
            x=non_zero_indices_b2,
            y=array_b2[non_zero_indices_b2] * 2,
            mode="markers",
            marker=dict(color="tomato", size=5),
            # name="Beam 2",
            xaxis="x",
            yaxis="y",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Compute the number of LR in each experiment
    bbs = beam_beam_schedule
    series_collide_atlas_cms = bbs[bbs["collides in ATLAS/CMS"]]["# of LR in ATLAS/CMS"]
    series_not_collide_atlas_cms = bbs[~bbs["collides in ATLAS/CMS"]]["# of LR in ATLAS/CMS"]
    series_collide_lhcb = bbs[bbs["collides in LHCB"]]["# of LR in LHCB"]
    series_not_collide_lhcb = bbs[~bbs["collides in LHCB"]]["# of LR in LHCB"]
    series_collide_alice = bbs[bbs["collides in ALICE"]]["# of LR in ALICE"]
    series_not_collide_alice = bbs[~bbs["collides in ALICE"]]["# of LR in ALICE"]

    # Add the plot for the number of LR in each experiment
    for row, series_collide, series_not_collide in zip(
        [2, 3, 4],
        [series_collide_atlas_cms, series_collide_lhcb, series_collide_alice],
        [
            series_not_collide_atlas_cms,
            series_not_collide_lhcb,
            series_not_collide_alice,
        ],
    ):
        fig.add_trace(
            go.Scattergl(
                x=series_collide.index,
                y=series_collide,
                xaxis="x",
                yaxis="y2",
                mode="markers",
                marker=dict(
                    color="orange",
                    size=5,
                ),
                legendgroup="colliding",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=series_not_collide.index,
                y=series_not_collide,
                xaxis="x",
                yaxis="y2",
                mode="markers",
                marker=dict(
                    color="teal",
                    size=5,
                ),
                legendgroup="not_colliding",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    # Traces for legend
    fig.add_trace(
        go.Scattergl(
            x=series_collide_atlas_cms.index[:1],
            y=series_collide_atlas_cms.values[:1],
            xaxis="x",
            yaxis="y2",
            mode="markers",
            marker=dict(color="orange", size=5),
            legendgroup="colliding",
            name="Colliding in IP",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=series_not_collide_atlas_cms.index[:1],
            y=series_not_collide_atlas_cms.values[:1],
            xaxis="x",
            yaxis="y2",
            mode="markers",
            marker=dict(color="teal", size=5),
            legendgroup="not_colliding",
            name="Not colliding in IP",
        ),
        row=2,
        col=1,
    )

    # Add a vertical line (in all subplots) to indicate the bunch selected for tracking
    fig.add_vline(
        x=i_bunch_b1,
        line_width=1,
        line_dash="dash",
        line_color="white",
        annotation_text="Selected bunch",
        annotation_position="top right",
    )

    # Update yaxis properties
    fig.update_xaxes(range=[0, non_zero_indices_b1[-1] + 1])
    fig.update_yaxes(
        title_text=r"Beam",
        range=[0.7, 2.3],
        row=1,
        col=1,
        tickmode="linear",
        tick0=1,
        dtick=1,
        fixedrange=True,
    )
    fig.update_yaxes(title_text=r"#LR in Atlas/CMS", range=[0, 52], row=2, col=1, fixedrange=True)
    fig.update_yaxes(title_text=r"#LR in LHCb", range=[0, 52], row=3, col=1, fixedrange=True)
    fig.update_yaxes(title_text=r"#LR in ALice", range=[0, 52], row=4, col=1, fixedrange=True)
    fig.update_xaxes(title_text=r"25ns slot", row=4, col=1)
    fig.update_yaxes(fixedrange=True)

    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        dragmode="pan",
        uirevision="Don't change",
        margin=dict(l=20, r=20, b=10, t=30, pad=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_x=1,
        legend_y=0.5,
    )
    # fig.update_layout(yaxis=dict(tickmode="linear", tick0=1, dtick=1))

    return fig


def get_indices_of_interest(df_tw, element_1, element_2):
    """
    Return the indices between the two elements provided.

    Args:
        df_tw (pandas.DataFrame): DataFrame containing twiss data.
        element_1 (str): The name of the first element of interest.
        element_2 (str): The name of the second element of interest.

    Returns:
        list: A list of element indices located between element_1 and element_2 (excluded).
    """
    idx_1 = df_tw.loc[df_tw["name"] == element_1].index[0]
    idx_2 = df_tw.loc[df_tw["name"] == element_2].index[0]
    if idx_2 < idx_1:
        return df_tw.loc[: idx_2 + 1].index.union(df_tw.loc[idx_1:].index)
    return df_tw.loc[idx_1 : idx_2 + 1].index


def return_plot_separation(dic_separation_ip, plane):
    """
    Returns a plotly figure object representing beam-beam separation at the different IPs.

    Args:
        dic_separation_ip (dict): A dictionary containing separation data for the different IPs.
        plane (str): The plane for which separation data is to be plotted. Can be "x", "y", or "xy".

    Returns:
        go.Figure: A plotly figure object representing beam-beam separation at the different IPs.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("IP 1", "IP 2", "IP 5", "IP 8"),
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        horizontal_spacing=0.2,
    )
    for idx, n_ip in enumerate([1, 2, 5, 8]):
        s = dic_separation_ip[f"ip{n_ip}"]["s"]
        l_scale_strength = dic_separation_ip[f"ip{n_ip}"]["l_scale_strength"]
        if plane in ["x", "y"]:
            sep = np.abs(dic_separation_ip[f"ip{n_ip}"][f"d{plane}_meter"])
            sep_sigma = np.abs(dic_separation_ip[f"ip{n_ip}"][f"d{plane}_sig"])
        elif plane == "xy":
            sep = np.abs(
                np.sqrt(
                    dic_separation_ip[f"ip{n_ip}"]["dx_meter"] ** 2
                    + dic_separation_ip[f"ip{n_ip}"]["dy_meter"] ** 2
                )
            )
            sep_sigma = np.abs(
                np.sqrt(
                    dic_separation_ip[f"ip{n_ip}"]["dx_sig"] ** 2
                    + dic_separation_ip[f"ip{n_ip}"]["dy_sig"] ** 2
                )
            )
        else:
            raise ValueError("plane should be 'x', 'y', or 'xy'")

        if plane in ["x", "y"]:
            # Do the plot
            fig.add_trace(
                go.Scatter(
                    x=s,
                    y=sep,
                    name=f"Separation at ip {str(n_ip)}",
                    legendgroup=f" IP {str(n_ip)}",
                    mode="lines+markers",
                    line=dict(color="coral", width=1),
                    marker=dict(opacity=l_scale_strength),
                ),
                row=idx // 2 + 1,
                col=idx % 2 + 1,
                secondary_y=False,
            )

        # Plot normalized separation in any case
        fig.add_trace(
            go.Scatter(
                x=s,
                y=sep_sigma,
                name=f"Normalized separation at ip {str(n_ip)}",
                legendgroup=f" IP {str(n_ip)}",
                mode="lines+markers",
                line=dict(color="cyan", width=1),
                marker=dict(opacity=l_scale_strength),
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
            secondary_y=plane in ["y", "x"],
        )

        # fig.add_trace(
        #     go.Scatter(
        #         x=s,
        #         y=[sep] * len(s),
        #         name="Inner normalized separation at ip " + str(n_ip),
        #         legendgroup=" IP " + str(n_ip),
        #         mode="lines+text",
        #         textposition="top left",
        #         line=dict(color="white", width=1, dash="dash"),
        #         text=[""] * (len(s) - 1) + ["Inner normalized separation"],
        #     ),
        #     row=idx // 2 + 1,
        #     col=idx % 2 + 1,
        #     secondary_y=True,
        # )

    for row, column in itertools.product(range(1, 3), range(1, 3)):
        fig.update_yaxes(
            title_text=r"$\textrm{B-B separation }[m]$",
            row=row,
            col=column,
            linecolor="coral",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=r"$\textrm{B-B separation }[\sigma]$",
            row=row,
            col=column,
            linecolor="cyan",
            secondary_y=plane in ["y", "x"],
        )
        fig.update_xaxes(title_text=r"$s [m]$", row=row, col=column)

    # fig.update_yaxes(range=[0, 0.25], row = 1, col = 1, secondary_y= False)
    # Use white theme for graph, centered title
    fig.update_layout(
        template="plotly_dark",
        title="Beam-beam separation at the different IPs",
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        showlegend=False,
    )

    return fig


def return_plot_separation_3D(dic_position_ip, ip="ip1"):
    """
    Returns a 3D Plotly figure showing the beam-beam separation at different interaction points.

    Args:
        dic_position_ip (dict): A dictionary containing the beam position data for different IPs and
            beams.
        ip (str, optional): The interaction point to plot. Defaults to "ip1".

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object containing the 3D beam-beam
            separation plot.
    """
    logging.info("Starting computation 3D beam-beam separation")

    fig = make_subplots(
        rows=1,
        cols=1,
        # subplot_titles="IP 1",
        specs=[
            [{"type": "scatter3d"}],
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    logging.info("Starting ip " + str(ip))
    for beam, color in zip(["lhcb1", "lhcb2"], ["teal", "tomato"]):
        s = dic_position_ip[beam]["tw"][ip]["s"].to_numpy()
        x = dic_position_ip[beam]["tw"][ip]["x"].to_numpy()
        X = dic_position_ip[beam]["sv"][ip]["X"].to_numpy()
        y = dic_position_ip[beam]["tw"][ip]["y"].to_numpy()
        # Y = dic_position_ip[beam]["sv"][ip]["Y"].to_numpy()
        bx = dic_position_ip[beam]["tw"][ip]["betx"].to_numpy()
        by = dic_position_ip[beam]["tw"][ip]["bety"].to_numpy()
        w = np.sqrt((bx + by) / 2)

        if beam == "lhcb2":
            s = s[::-1]
            x = x[::-1]
            X = X[::-1]
            y = y[::-1]
            bx = bx[::-1]
            by = by[::-1]
            w = w[::-1]

        for i in range(0, int(s.shape[0] / 2) - 20, 10):
            fig.add_trace(
                go.Scatter3d(
                    x=s[i - 5 : i + 6],
                    y=x[i - 5 : i + 6] + X[i - 5 : i + 6],
                    z=y[i - 5 : i + 6],
                    mode="lines",
                    line=dict(color=color, width=w[i]),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        for i in range(int(s.shape[0] / 2) + 20, s.shape[0], 10):
            fig.add_trace(
                go.Scatter3d(
                    x=s[i - 5 : i + 6],
                    y=x[i - 5 : i + 6] + X[i - 5 : i + 6],
                    z=y[i - 5 : i + 6],
                    mode="lines",
                    line=dict(color=color, width=w[i]),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        for i in range(int(s.shape[0] / 2) - 18, int(s.shape[0] / 2) + 19):
            fig.add_trace(
                go.Scatter3d(
                    x=s[i : i + 3],
                    y=x[i : i + 3] + X[i : i + 3],
                    z=y[i : i + 3],
                    mode="lines",
                    line=dict(color=color, width=w[i + 1]),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    range_s = (s[0] - 5, s[-1] + 5)
    range_x = range_y = (
        np.min((np.min((x + X)), np.min(y))) - 0.01,
        np.max((np.max((x + X)), np.max(y))) + 0.01,
    )
    fig.update_scenes(
        xaxis=dict(title="s[m]", range=range_s),
        yaxis=dict(title="x[m]", range=range_x),
        zaxis=dict(title="y[m]", range=range_y),
        aspectmode="cube",
        row=1,
        col=1,
    )
    fig.update_layout(
        autosize=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="3D Beam-beam separation",
        title_x=0.5,
        title_y=0.95,
        margin=dict(l=20, r=20, b=10, t=30),  # , pad=10),
    )
    logging.info("Returning 3D beam-beam separation figure")
    return fig


@lru_cache(maxsize=20)
def compute_footprint_mesh():
    """Compute the mesh for the footprint plot."""
    return get_working_diagram(
        Qx_range=[0.28, 0.35],
        Qy_range=[0.28, 0.35],
        order=12,
        color="white",
        alpha=0.1,
    )


def return_plot_footprint(t_array_footprint, qx, qy, title, plot_filtered_web=False):
    """
    Returns a Plotly figure object representing the footprint of a particle beam in the Qx-Qy plane.

    Args:
        t_array_footprint (tuple of two 1D arrays): The Qx and Qy coordinates of the footprint mesh.
        qx (float): The horizontal tune of the beam.
        qy (float): The vertical tune of the beam.
        title (str): The title of the plot.
        plot_filtered_web (bool, optional): Whether to plot the filtered footprint mesh. Defaults to False.

    Returns:
        plotly.graph_objs._figure.Figure: The Plotly figure object representing the footprint plot.
    """
    logging.info("Starting computation footprint figure")
    palette = sns.color_palette("Spectral", 10).as_hex()
    array_qx, array_qy = t_array_footprint
    fig = go.Figure()
    if array_qx.shape[0] > 0:
        # Get resonance lines first
        fig.add_traces(compute_footprint_mesh())

        if plot_filtered_web:
            # Filter the footprint mesh
            for x, y in zip(array_qx, array_qy):
                # Insert additional None when dx or dy is too big
                # to avoid connecting the lines
                x_temp = np.insert(x, np.where(np.abs(np.diff(x)) > 0.003)[0] + 1, None)
                y_temp = np.insert(y, np.where(np.abs(np.diff(x)) > 0.003)[0] + 1, None)
                x_temp = np.insert(x_temp, np.where(np.abs(np.diff(y)) > 0.003)[0] + 1, None)
                y_temp = np.insert(y_temp, np.where(np.abs(np.diff(y)) > 0.003)[0] + 1, None)
                fig.add_trace(
                    go.Scattergl(
                        x=x_temp,
                        y=y_temp,
                        line_color="whitesmoke",
                        opacity=0.3,
                    )
                )
        for idx, (x, y) in enumerate(zip(array_qx.T, array_qy.T)):
            if plot_filtered_web:
                x_temp = np.insert(x, np.where(np.abs(np.diff(x)) > 0.003)[0] + 1, None)
                y_temp = np.insert(y, np.where(np.abs(np.diff(x)) > 0.003)[0] + 1, None)
                x_temp = np.insert(x_temp, np.where(np.abs(np.diff(y)) > 0.003)[0] + 1, None)
                y_temp = np.insert(y_temp, np.where(np.abs(np.diff(y)) > 0.003)[0] + 1, None)
            x_temp = x
            y_temp = y
            fig.add_trace(
                go.Scattergl(
                    x=x_temp,
                    y=y_temp,
                    line_color=palette[9 - idx],
                    mode="markers",
                ),
            )

        fig.add_trace(
            go.Scattergl(
                x=[qx % 1],
                y=[qy % 1],
                mode="markers",
                marker_color="white",
                marker_size=10,
                marker_symbol="x",
            )
        )

        fig.update_layout(
            xaxis=dict(
                range=[
                    np.percentile(array_qx, 10) - 0.001,
                    np.percentile(array_qy, 90) + 0.001,
                ],
            ),
            yaxis=dict(
                range=[
                    np.percentile(array_qx, 10) - 0.003,
                    np.percentile(array_qy, 90) + 0.003,
                ],
            ),
        )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Qx",
        yaxis_title="Qy",
        # width=500,
        # height=500,
        showlegend=False,
        margin=dict(l=20, r=20, b=10, t=30, pad=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
    )

    logging.info("Returning footprint figure")
    return fig

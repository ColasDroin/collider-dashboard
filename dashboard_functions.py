# ==================================================================================================
# --- Imports
# ==================================================================================================import numpy as np
import pandas as pd
import xtrack as xt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots
from dash import dash_table
from dash.dash_table.Format import Format, Scheme, Trim


# ==================================================================================================
# --- Functions to load dashboard variables
# ==================================================================================================
def return_dataframe_elements_from_line(line):
    # Build a dataframe with the elements of the lines
    df_elements = pd.DataFrame([x.to_dict() for x in line.elements])
    return df_elements


def return_survey_and_twiss_dataframes_from_line(line, correct_x_axis=True):
    """Return the survey and twiss dataframes from a line."""
    # Get survey dataframes
    df_sv = line.survey().to_pandas()

    # Get Twiss dataframes
    tw = line.twiss()
    df_tw = tw.to_pandas()

    # Reverse x-axis if requested
    if correct_x_axis:
        df_sv["X"] = -df_sv["X"]
        df_tw["x"] = -df_tw["x"]

    return tw, df_sv, df_tw


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


def return_all_loaded_variables(collider_path=None, collider=None):
    """Return all loaded variables if they are not already loaded."""

    if collider is None and collider_path is not None:
        # Rebuild line (can't be pickled, most likely because of struct and multiprocessing)
        collider = xt.Multiline.from_json(collider_path)

    elif collider is None and collider_path is None:
        raise ValueError("Either collider or collider_path must be provided")

    # Build tracker
    collider.build_trackers()

    # Get elements of the line (only done for b1, should be identical for b2)
    df_elements = return_dataframe_elements_from_line(collider.lhcb1)

    # Compute twiss and survey for both lines
    tw_b1, df_sv_b1, df_tw_b1 = return_survey_and_twiss_dataframes_from_line(
        collider.lhcb1, correct_x_axis=True
    )
    tw_b2, df_sv_b2, df_tw_b2 = return_survey_and_twiss_dataframes_from_line(
        collider.lhcb2, correct_x_axis=False
    )

    # Correct df elements for thin lens approximation
    df_elements_corrected = return_dataframe_corrected_for_thin_lens_approx(df_elements, df_tw_b1)

    # Return all variables
    return collider, tw_b1, df_sv_b1, df_tw_b1, tw_b2, df_sv_b2, df_tw_b2, df_elements_corrected


def get_indices_of_interest(df_tw, element_1, element_2):
    """Return the indices of the elements of interest."""
    idx_1 = df_tw.loc[df_tw["name"] == element_1].index[0]
    idx_2 = df_tw.loc[df_tw["name"] == element_2].index[0]
    if idx_2 < idx_1:
        return list(range(0, idx_2)) + list(range(idx_1, len(df_tw)))
    return list(range(idx_1, idx_2))


# ==================================================================================================
# --- Plotting functions
# ==================================================================================================
def return_radial_background_traces(df_sv):
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
    # Return a Plotly trace containing the beam pipe
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
    if flat:
        return return_flat_multipole_trace(
            df_elements,
            df_sv,
            order,
            strength_magnification_factor=5000,
            add_ghost_trace=True,
            l_indices_to_keep=None,
            xaxis=xaxis,
            yaxis=yaxis,
        )
    else:
        return return_circular_multipole_trace(
            df_elements,
            df_sv,
            order,
            strength_magnification_factor=5000,
            add_ghost_trace=True,
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
        strength_magnification_factor = strength_magnification_factor / 10

    # Get strength of all multipoles of the requested order
    s_knl = df_elements[df_elements.order == order]["knl"].apply(lambda x: x[order])

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
                [row["Z"], row["Z"] + s_knl[i] * np.sin(row["theta"]), None]
            )
            dic_trace[width]["customdata"].extend([row["name"], row["name"], None])
        else:
            dic_trace[width] = {
                "x": [row["X"], row["X"] + s_knl[i] * np.cos(row["theta"]), None],
                "y": [row["Z"], row["Z"] + s_knl[i] * np.sin(row["theta"]), None],
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
        strength_magnification_factor = strength_magnification_factor / 2

    # Get strength of all multipoles of the requested order
    s_knl = df_elements[df_elements.order == order]["knl"].apply(lambda x: x[order])

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
    hide_optics_traces_initially=True,
    beam_2=False,
    l_indices_to_keep=None,
):
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
            print("The type of trace is not recognized.")

    # Correct for circular projection depending if x-coordinate has been reversed or not
    if beam_2:
        correction = -1
    else:
        correction = 1

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
            - df_tw_temp[tw_name] ** exponent
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
    for order, add in zip(
        [0, 1, 2, 3], [add_dipoles, add_quadrupoles, add_sextupoles, add_octupoles]
    ):
        # Add multipole if requested
        if add:
            if row is not None and col is not None:
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
                    fig.append_trace(
                        trace,
                        row=row,
                        col=col,
                    )
            else:
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
    # Add horizontal betatron if requested
    if plot_horizontal_betatron:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="betax", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
            )
        )

    # Add vertical betatron if requested
    if plot_vertical_betatron:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="bety", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
            )
        )

    # Add horizontal dispersion if requested
    if plot_horizontal_dispersion:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="dx", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
            )
        )

    # Add vertical dispersion if requested
    if plot_vertical_dispersion:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="dy", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
            )
        )

    # Add horizontal position if requested
    if plot_horizontal_position:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="x", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
            )
        )

    # Add vertical position if requested
    if plot_vertical_position:
        fig.add_trace(
            return_optic_trace(
                df_sv, df_tw, type_trace="y", beam_2=beam_2, l_indices_to_keep=l_indices_to_keep
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
    plot_horizontal_momentum=True,
    plot_vertical_momentum=True,
    hide_optics_traces_initially=True,
    add_optics_beam_2=True,
):
    # Center X coordinate (otherwise conversion to polar coordinates is not possible)
    X_centered = df_sv["X"] - np.mean(df_sv["X"])

    # Get corresponding angle
    l_theta = np.arctan2(df_sv["Z"], X_centered)

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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
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
    fig.append_trace(
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
            line=dict(color=color) if not dashed else dict(color=color, dash="dash"),
            # Deactivate legendgroup for now as it doesn't work
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )
    return fig


def return_plot_optics(
    tw_b1,
    tw_b2,
    df_sv,
    df_elements,
):
    # Build figure
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

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
        tw_b1["s"],
        tw_b1["betx"],
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
        tw_b1["s"],
        tw_b1["bety"],
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
        tw_b2["s"],
        tw_b2["betx"],
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
        tw_b2["s"],
        tw_b2["bety"],
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
        tw_b1["s"],
        tw_b1["x"],
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
        tw_b1["s"],
        tw_b1["y"],
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
        tw_b2["s"],
        tw_b2["x"],
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
        tw_b2["s"],
        tw_b2["y"],
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
        tw_b1["s"],
        tw_b1["dx"],
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
        tw_b1["s"],
        tw_b1["dy"],
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
        tw_b2["s"],
        tw_b2["dx"],
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
        tw_b2["s"],
        tw_b2["dy"],
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

    # Add horizontal lines for ip1 and ip5
    fig.add_vline(
        x=float(tw_b1.rows["ip1"].cols["s"].to_pandas().s),
        line_width=1,
        line_dash="dash",
        line_color="pink",
        annotation_text="IP 1",
        annotation_position="top right",
    )

    fig.add_vline(
        x=float(tw_b1.rows["ip5"].cols["s"].to_pandas().s),
        line_width=1,
        line_dash="dash",
        line_color="pink",
        annotation_text="IP 5",
        annotation_position="top right",
    )

    # Update overall layout
    title_1 = (
        r"$q_{x_{1}} = "
        + f'{tw_b1["qx"]:.5f}'
        + r"\hspace{0.5cm}"
        + r" q_{y_{1}} = "
        + f'{tw_b1["qy"]:.5f}'
        + r"\hspace{0.5cm}"
        + r"Q'_{x_{1}} = "
        + f'{tw_b1["dqx"]:.2f}'
        + r"\hspace{0.5cm}"
        + r" Q'_{y_{1}} = "
        + f'{tw_b1["dqy"]:.2f}'
        + r"\hspace{0.5cm}"
        # + r" \gamma_{tr_{1}} = "
        # + f'{1/np.sqrt(tw_b1["momentum_compaction_factor"]):.2f}'
    )
    title_2 = (
        r"\\ "
        + r"q_{x_{2}} = "
        + f'{tw_b2["qx"]:.5f}'
        + r"\hspace{0.5cm}"
        + r" q_{y_{2}} = "
        + f'{tw_b2["qy"]:.5f}'
        + r"\hspace{0.5cm}"
        + r"Q'_{x_{2}} = "
        + f'{tw_b2["dqx"]:.2f}'
        + r"\hspace{0.5cm}"
        + r" Q'_{y_{2}} = "
        + f'{tw_b2["dqy"]:.2f}'
        + r"\hspace{0.5cm}"
        # + r" \gamma_{tr_{2}} = "
        # + f'{1/np.sqrt(tw_b2["momentum_compaction_factor"]):.2f}'
        + r"\\ $"
    )
    title = title_1 + title_2

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_xanchor="center",
        showlegend=True,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        # xaxis_title=r'$s$',
        # yaxis_title=r'$[m]$',
        # width=1000,
        # height=1000,
        legend_tracegroupgap=30,
        dragmode="pan",
        uirevision="Don't change",
    )

    # Make background transparent
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    # Update yaxis properties
    fig.update_xaxes(range=[0, tw_b1["s"][-1] + 1])
    fig.update_yaxes(title_text=r"$\beta_{x,y}$ [m]", range=[0, 10000], row=2, col=1)
    fig.update_yaxes(title_text=r"(Closed orbit)$_{x,y}$ [m]", range=[-0.03, 0.03], row=3, col=1)
    fig.update_yaxes(title_text=r"$D_{x,y}$ [m]", range=[-3, 3], row=4, col=1)
    fig.update_xaxes(title_text=r"$s$", row=4, col=1)
    fig.update_yaxes(fixedrange=True)

    return fig


def return_plot_filling_scheme(array_b1, array_b2):
    # Color filling scheme with blue and red
    array_b1_colored = np.array(
        [[30, 144, 255, 200] if x != 0 else [255, 255, 255, 0] for x in array_b1], dtype=np.uint8
    )
    array_b2_colored = np.array(
        [[238, 75, 43, 200] if x != 0 else [255, 255, 255, 0] for x in array_b2], dtype=np.uint8
    )

    # Convert to matrix
    mat = np.stack((array_b1_colored, array_b2_colored), dtype=np.uint8)

    # Plot filling scheme with plotly
    fig = go.Figure(go.Image(z=mat, colormodel="rgba256"))

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=200,
        tickvals=[0, 1],
        ticktext=["Beam 1 ", "Beam 2 "],
        constrain="domain",
        range=[0, 1],
    )

    fig.update_layout(
        xaxis=dict(title="Bucket number"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # dragmode="zoom",
        title_text="Filling scheme for the current simulation",
        title_x=0.5,
        title_xanchor="center",
    )

    return fig


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
                        "format": Format(precision=2, scheme=Scheme.decimal_si_prefix),
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
            page_size=30,
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


# ==================================================================================================
# --- Functions to build CSS styling
# ==================================================================================================
def build_CSS():
    if not os.path.exists("assets"):
        os.makedirs("assets")

    # Build css
    string_css = """
    
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner input:not([type=radio]):not([type=checkbox]) {
        color: #F8F8FF;
    }
    
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner .dash-filter > div input[type="text"] {
        color: #F8F8FF;
    }
    
    #datatable-twiss .dash-filter .input {color: white}



    .dash-spreadsheet.dash-freeze-top, .dash-spreadsheet.dash-virtualized 
        { max-height: inherit !important; }

    .dash-table-container {max-height: calc(80vh - 225px);}

    ._dash-loading {
    margin: auto;
    color: transparent;
    width: 2rem;
    height: 2rem;
    text-align: center;
    }

    ._dash-loading::after {
    font-size: 60px;
    content: "";
    display: inline-block;
    width: 1em;
    height: 1em;
    color: #000;
    border-radius: 50%;
    margin: 72px auto;
    vertical-align: text-bottom;
    -webkit-animation: load6 1.7s infinite ease, round 1.7s infinite ease;
    animation: load6 1.7s infinite ease, round 1.7s infinite ease;
    margin-top: 2rem;
    }

    @-webkit-keyframes load6 {
    0% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    5%,
    95% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    10%,
    59% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.087em -0.825em 0 -0.42em,
        -0.173em -0.812em 0 -0.44em, -0.256em -0.789em 0 -0.46em,
        -0.297em -0.775em 0 -0.477em;
    }
    20% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.338em -0.758em 0 -0.42em,
        -0.555em -0.617em 0 -0.44em, -0.671em -0.488em 0 -0.46em,
        -0.749em -0.34em 0 -0.477em;
    }
    38% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.377em -0.74em 0 -0.42em,
        -0.645em -0.522em 0 -0.44em, -0.775em -0.297em 0 -0.46em,
        -0.82em -0.09em 0 -0.477em;
    }
    100% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    }
    @keyframes load6 {
    0% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    5%,
    95% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    10%,
    59% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.087em -0.825em 0 -0.42em,
        -0.173em -0.812em 0 -0.44em, -0.256em -0.789em 0 -0.46em,
        -0.297em -0.775em 0 -0.477em;
    }
    20% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.338em -0.758em 0 -0.42em,
        -0.555em -0.617em 0 -0.44em, -0.671em -0.488em 0 -0.46em,
        -0.749em -0.34em 0 -0.477em;
    }
    38% {
        box-shadow: 0 -0.83em 0 -0.4em, -0.377em -0.74em 0 -0.42em,
        -0.645em -0.522em 0 -0.44em, -0.775em -0.297em 0 -0.46em,
        -0.82em -0.09em 0 -0.477em;
    }
    100% {
        box-shadow: 0 -0.83em 0 -0.4em, 0 -0.83em 0 -0.42em, 0 -0.83em 0 -0.44em,
        0 -0.83em 0 -0.46em, 0 -0.83em 0 -0.477em;
    }
    }
    @-webkit-keyframes round {
    0% {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
    100% {
        -webkit-transform: rotate(360deg);
        transform: rotate(360deg);
    }
    }
    @keyframes round {
    0% {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
    100% {
        -webkit-transform: rotate(360deg);
        transform: rotate(360deg);
    }
    }
    """

    with open("assets/style.css", "w") as file:
        file.write(string_css)

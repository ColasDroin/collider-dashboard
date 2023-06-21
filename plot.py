# ==================================================================================================
# --- Imports
# ==================================================================================================import numpy as np
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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

    # Add horizontal lines for all ips
    for ip in [1, 2, 5, 8]:
        fig.add_vline(
            x=float(tw_b1.rows["ip" + str(ip)].cols["s"].to_pandas().s),
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
        # xaxis_title=r'$s$',
        # yaxis_title=r'$[m]$',
        # width=1000,
        # height=1000,
        legend_tracegroupgap=30,
        dragmode="pan",
        uirevision="Don't change",
        margin=dict(l=10, r=10, b=10, t=10, pad=10),
    )

    # Make background transparent
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    # Update yaxis properties
    fig.update_xaxes(range=[0, tw_b1["s"][-1] + 1])
    fig.update_yaxes(title_text=r"$\beta_{x,y}[m]$", range=[0, 10000], row=2, col=1)
    fig.update_yaxes(title_text=r"(Closed orbit)$_{x,y}[m]$", range=[-0.03, 0.03], row=3, col=1)
    fig.update_yaxes(title_text=r"$D_{x,y}[m]$", range=[-3, 3], row=4, col=1)
    fig.update_xaxes(title_text=r"$s[m]$", row=4, col=1)
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
        dragmode="pan",
    )

    return fig


def get_indices_of_interest(df_tw, element_1, element_2):
    """Return the indices of the elements of interest."""
    idx_1 = df_tw.loc[df_tw["name"] == element_1].index[0]
    idx_2 = df_tw.loc[df_tw["name"] == element_2].index[0]
    if idx_2 < idx_1:
        return list(range(0, idx_2)) + list(range(idx_1, len(df_tw)))
    return list(range(idx_1, idx_2))

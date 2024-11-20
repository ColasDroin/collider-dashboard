"""Module to build the dashboard app."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import from standard library
import logging

# Import third-party packages
from dash import Dash

# Import local functions
from .backend import compute_globals
from .callbacks import all_callbacks
from .layout import return_app_layout

# ==================================================================================================
# --- Load global variables and build app
# ==================================================================================================


def build_app(
    path_collider,
    path_scheme=None,
    force_reload=False,
    ignore_footprint=False,
    simplify_tw=True,
    type_particles=None,
):
    # Load dashboard variables
    dic_without_bb, dic_with_bb = compute_globals.init_from_collider(
        path_collider,
        path_scheme=path_scheme,
        force_reload=force_reload,
        ignore_footprint=ignore_footprint,
        simplify_tw=simplify_tw,
        type_particles=type_particles,
    )

    #################### App ####################
    logging.info("Defining app")
    app = Dash(
        __name__,
        title="Collider dashboard",
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
        ],
        suppress_callback_exceptions=False,
    )
    logging.info("Defining app layout")
    app.layout = return_app_layout()
    all_callbacks(app, dic_with_bb, dic_without_bb, path_collider)
    return app, app.server

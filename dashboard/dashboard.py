# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import from standard library
import logging

# Import third-party packages
from dash import Dash

# Import local functions
from .backend import init
from .callbacks import all_callbacks
from .layout.main import return_app_layout

# ==================================================================================================
# --- Load global variables and build app
# ==================================================================================================


def build_app():
    # Load dashboard variables
    path_collider = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/test_dump/base_collider/xtrack_0000/collider.json"
    path_job = path_collider.split("/final_collider.json")[0]
    dic_without_bb, dic_with_bb, initial_pickle_path = init.init_from_collider(
        path_collider, load_global_variables_from_pickle=True
    )

    #################### App ####################
    logging.info("Defining app")
    app = Dash(
        __name__,
        title="Dashboard for current simulation",
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
        ],
        suppress_callback_exceptions=False,
    )
    logging.info("Defining app layout")
    app.layout = return_app_layout()
    all_callbacks(app, dic_with_bb, dic_without_bb, path_job)
    return app

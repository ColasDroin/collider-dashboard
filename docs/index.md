# Welcome to the Simboard documentation.

Please use the navigation bar on the left, or the search field on top to navigate through the documentation.

## Installation

The dashboard can be installed from PyPI with pip:

```bash
pip install simulation-dashboard
```

This will install the required packages and build the application.


## Usage

The simplest way to use the dashboard is to run the package from the command line, providing the path of a collider json file:

```bash
python -m simulation-dashboard path_to_collider.json
```

After computing some temporary variables (this may take a while the first time), this will deploy a local server and open the dashboard in a browser window.

Alternatively, one can import the dashboard as a module and use it in a custom script:

```python
# my-awesome-dashboard.py

from simulation_dashboard import build_app
app, server = build_app(path_to_collider.json)
```

The dashboard can then be deployed on a remote server, e.g. with gunicorn (here on port 8080):

```bash
gunicorn my-awesome-dashboard:server -b :8080
```

Note that, as the dashboard deals with global variables, it is not thread-safe. It is therefore recommended to run it with a single worker (it's the case by default).


## Project repository

Please visit the [project webpage](https://github.com/ColasDroin/simulation-dashboard) for more information about the project.

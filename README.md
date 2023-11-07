# Simulation Dashboard

A Dash application to visualize the observables and parameters of a collider built and configured with Xsuite.

## Installation

The dashboard can be installed from PyPI with pip:

```bash
pip install simulation-dashboard
```

This will install the required packages and build the application.


## Usage

For personal usage, the simplest way to use the dashboard is to run the package from the command line, providing the path of a collider json file:

```bash
python -m simulation-dashboard --path path_to_collider.json
```

If needed, one can specify the port on which the dashboard will be deployed:

```bash
python -m simulation-dashboard --path path_to_collider.json --port 8080
```

After computing some temporary variables (this may take a while the first time), this will deploy a local server and open the dashboard in a browser window.

Alternatively, one can import the dashboard as a module and use it in a custom script:

```python
# my-awesome-dashboard.py

from simulation_dashboard import build_app
app, server = build_app(path_to_collider.json, port=8080)
```

The dashboard can then be deployed on a remote server, e.g. with gunicorn:

```bash
gunicorn my-awesome-dashboard:server -b :8080
```

Note that, as the dashboard deals with global variables, it is not thread-safe. It is therefore recommended to run it with a single worker (it's the case by default).


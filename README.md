# Collider Dashboard

A Dash application to visualize the observables and parameters of a collider built and configured with Xsuite.

## Installation

The dashboard can be installed from PyPI with pip:

```bash
pip install collider-dashboard
```

This will install the required packages and build the application. If you haven't done it already, it is recommended to prebuild the Xsuite kernel to gain some computation time:

```bash
xsuite-prebuild
```

## Usage

For personal usage, the simplest way to use the dashboard is to run the package as a development server from the command line, providing a few arguments:

```bash
python -m collider_dashboard --collider-path path_to_collider.json --filling-path path_to_scheme.json --port 8080 --force-reload --ignore-footprint --simplify --debug
```

- `--collider-path`, or `-c`, sets the path to the collider configuration file. Default value to the path of a dummy collider used for testing.
- `--filling-path`, or `-f`, sets the path to the filling scheme, instead of using the one in the collider configuration file. Optional.
- `--port`, or `-p`, sets the port on which the dashboard will be deployed. Default value to `8080``.
- `--force-reload`, or `-r`,  sets a boolean indicating whether the collider dashboard data should be reloaded if already existing. Optional.
- `--ignore-footprint`, or `-i`, sets a boolean indicating whether the footprint should be ignored to gain computation time. Optional.
- `--simplify`, or `-s`, sets a boolean indicating whether the Twiss/Survey tables should be simplified (remove duplicates and entry/exit elements) to gain computation time. Recommended but optional.
- `--debug`, or `-d`, sets a boolean indicating whether the dashboard should be run in debug mode. Optional.

After computing some temporary variables (this may take a while the first time), this will deploy a local server and open the dashboard in a browser window.

Alternatively, one can import the dashboard as a module and use it in a custom script:

```python
# my-awesome-dashboard.py

from collider_dashboard import build_app
app, server = build_app(path_to_collider.json, port=8080 force_reload=False, ignore_footprint=False, debug = False, simplify_tw=True)
```

The dashboard can then be deployed e.g. with gunicorn:

```bash
gunicorn my-awesome-dashboard:server -b :8080
```

Note that, as the dashboard deals with global variables, it is not thread-safe. It is therefore recommended to run it with a single worker (it's the case by default).

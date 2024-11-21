# Collider Dashboard

<p align="center">
    <a href="https://opensource.org/license/mit/">
    <img src="https://badgen.net/static/license/MIT/blue">
    </a>
    <a href="https://python-poetry.org/">
    <img src="https://badgen.net/static/Package%20Manager/Poetry/orange">
    </a>
    <a href="https://dash.plotly.com/">
    <img src="https://badgen.net/static/Dash/2.18.2/green">
    </a>
    <a href="https://black.readthedocs.io/en/stable/">
    <img src="https://badgen.net/static/Code%20style/Black/black">
    </a>
    <img src="https://badgen.net/github/release/colasdroin/collider-dashboard">
    <img src="https://badgen.net/github/commits/colasdroin/collider-dashboard">
</p>

A Dash application to visualize the observables and parameters of a collider built and configured with Xsuite.

## Installation

The dashboard can be installed from PyPI with pip:

```bash
pip install collider-dashboard
```

This will install the required packages and build the application. If you haven't done it already, it is recommended to prebuild the Xsuite kernel to gain some computation time:

```bash
xsuite-prebuild regenerate
```

## Usage

For personal usage, the simplest way to use the dashboard is to run the package as a development server from the command line, providing a few arguments:

```bash
python -m collider_dashboard --collider-path path_to_collider.json --filling-path path_to_scheme.json --port 8080 --force-reload --ignore-footprint --full-twiss --type-particles proton --debug
```

- `--collider-path`, or `-c`, sets the path to the collider configuration file. Default value to the path of a dummy collider used for testing.
- `--filling-path`, or `-f`, sets the path to the filling scheme, instead of using the one in the collider configuration file (which _must be_ absolute). Optional.
- `--port`, or `-p`, sets the port on which the dashboard will be deployed. Default value to `8080``.
- `--force-reload`, or `-r`,  sets a boolean indicating whether the collider dashboard data should be reloaded if already existing. Optional.
- `--ignore-footprint`, or `-i`, sets a boolean indicating whether the footprint should be ignored to gain computation time. Optional.
- `--full-twiss`, or `-t`, sets a boolean indicating whether the Twiss/Survey tables should be computed fully (not removing duplicates and entry/exit elements), at the expense of computation time. Optional.
- `--type-particles`, or `-a`, sets the type of particles to be used for the collider. Default value is inferred from the config is present in the metadata of the collider. Otherwise, must be provided as "proton" or "lead".
- `--debug`, or `-d`, sets a boolean indicating whether the dashboard should be run in debug mode. Optional.

After computing some temporary variables (this may take a while the first time), this will deploy a local server and open the dashboard in a browser window.

Alternatively, one can import the dashboard as a module and use it in a custom script:

```python
# my-awesome-dashboard.py

from collider_dashboard import build_app
app, server = build_app(path_to_collider.json, 
                        path_scheme=path_to_scheme.json, 
                        port=8080, 
                        force_reload=False, 
                        ignore_footprint=False, 
                        debug = False, 
                        simplify_tw=True
                        type_particles='proton'
                )
```

The dashboard can then be deployed, for instance, with gunicorn:

```bash
gunicorn my-awesome-dashboard:server -b :8080
```

Note that, as the dashboard deals with global variables, it is not thread-safe. It is therefore recommended to run it with a single worker (it's the case by default).

## Collider metadata

The dashboard will work with or without a configuration file embedded as metadata in the collider json file. If the metadata is present, the dashboard will use it to infer the type of particles, the filling scheme path, etc. Otherwise, some data and tabs might not be available. Note that the filling scheme tab will be available if the filling scheme is provided as an argument, even if the metadata is not present.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

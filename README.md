# PLUNE - PLotting Utility N Exporter

## Info
A utility for visualizing GROND logs. Converts raw binary logs to JSON for portability.

## Setup
1. Install [protobuf](https://github.com/protocolbuffers/protobuf/releases/)
2. Generate python files: `protoc --python_out generated --pyi_out generated log_message.proto header_message.proto`
3. Install protobuf for python:
    `pip install protobuf`

## Graph Settings
The desired graphs to be displayed are defined in `graphs.json`. An example is already provided, but adding new logs easy!
1. Add a new item to the `"figures"` array
2. Add a x_axis, a list of y_axes, and a title:
```
    {
      "x_axis": "control_cycle_start_s",
      "y_axis": [
        "engine_rpm",
        "secondary_rpm",
        "target_rpm"
      ],
      "title": "Engine, Secondary, Target RPM"
    }
```

## Usage
### `./plune.py`
Run the webserver. Displays all JSON logs in `logs/` (must convert binary logs first).
### `./plune.py -c, --convert`
Convert all raw binary logs to JSON files. By deafult, `raw_logs/*.bin` are converted to JSON files in `logs/`.
### `./plune.py -e, --export`
Export all JSON logs to HTML files for offline viewing. By default exports all `logs/*.json` into `graphs/`.
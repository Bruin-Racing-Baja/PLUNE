#!/usr/bin/env python
import argparse
import json
import os
from typing import List, Tuple
from uuid import uuid4

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from dash.dependencies import ALL, MATCH
from dash_extensions.enrich import (
    DashProxy,
    NoOutputTransform,
    Serverside,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)
from plotly_resampler import FigureResampler
from trace_updater import TraceUpdater

from utils import log_parser, odrive_utils

raw_log_dir = "raw_logs"
log_dir = "logs"
export_dir = "graphs"
graph_info_file = "graphs.json"

# with open(graph_info_file, "r") as file:
# graph_info_json = json.loads(file.read())
# raw_paths = log_parser.getFilesByExtension(raw_log_dir, "bin")


app = DashProxy(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    transforms=[ServersideOutputTransform(), TriggerTransform(), NoOutputTransform()],
)


def serveLayout():
    with open(graph_info_file, "r") as file:
        graph_info = json.loads(file.read())
    log_paths = log_parser.getFilesByExtension(log_dir, "tar")

    header_layout = [
        # Page title
        html.Div(
            id="title-container",
            children=[
                html.H1(
                    id="title",
                ),
            ],
        ),
        # File selector
        dcc.Dropdown(
            id="file-selector",
            options=log_paths,
            value=log_paths[-1] if len(log_paths) > 0 else None,
        ),
        # Log data
        dcc.Loading(
            html.Div(
                id="header-data-container",
                children=[
                    dbc.Textarea(id="description-input", debounce=True),
                    dbc.Table(
                        id="log-header-table", bordered=True, hover=True, striped=True
                    ),
                    dbc.Table(
                        id="odrive-error-table", bordered=True, hover=True, striped=True
                    ),
                ],
            )
        ),
    ]

    return html.Div(
        [
            dcc.Store(id="graph-info", data=graph_info),
            html.Div(id="header", children=header_layout),
            dcc.Store(id="graph-data"),
            html.Div(
                id="graph-container",
            ),
        ]
    )


app.layout = serveLayout()


@app.callback(
    Input("description-input", "value"),
    State("file-selector", "value"),
    prevent_initial_call=True,
)
def onDescriptionChanged(description: str, path: str) -> None:
    log_parser.updateTarDescription(path, description)


@app.callback(
    Output({"type": "store", "uid": MATCH}, "data"),
    Output({"type": "dynamic-graph", "uid": MATCH}, "figure"),
    Trigger({"type": "interval", "uid": MATCH, "index": ALL}, "n_intervals"),
    State({"type": "interval", "uid": MATCH, "index": ALL}, "id"),
    State("graph-data", "data"),
    State("graph-info", "data"),
    prevent_initial_call=True,
)
def constructGraphs(interval_id, data, graph_info_json) -> FigureResampler:
    idx = int(interval_id[0]["index"])
    df = data
    graph_info = graph_info_json["figures"][idx]
    if graph_info.get("disabled", False):
        return (None), (None)

    fig = FigureResampler(go.Figure(), default_n_shown_samples=2_000)

    if not set(graph_info["y_axis"]).issubset(df.columns):
        print(f'Column(s) Missing: Skipping "{graph_info["title"]}"')
        return (None), (None)

    for y_axis in graph_info["y_axis"]:
        fig.add_trace(
            go.Scattergl(name=y_axis),
            hf_x=df[graph_info["x_axis"]],
            hf_y=df[y_axis],
        )

    fig.update_layout(
        title=graph_info["title"],
        xaxis_title=graph_info["x_axis"],
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return Serverside(fig), fig


@app.callback(
    Output({"type": "dynamic-updater", "uid": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "uid": MATCH}, "relayoutData"),
    State({"type": "store", "uid": MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data(relayoutdata)
    return no_update


@app.callback(
    Output("title", "children"),
    Output("odrive-error-table", "children"),
    Output("log-header-table", "children"),
    Output("description-input", "value"),
    Output("graph-data", "data"),
    Output("graph-container", "children"),
    Input("graph-info", "data"),
    Input("file-selector", "value"),
)
def onGraphInfoChanged(graph_info, path):
    if path == None:
        return no_update

    # Create list of empty graphs and corresponding resampler components
    dynamic_graphs = []
    for idx in range(len(graph_info["figures"])):
        uid = str(uuid4())
        dynamic_graphs.append(
            html.Div(
                children=[
                    dcc.Graph(id={"type": "dynamic-graph", "uid": uid}),
                    dcc.Loading(dcc.Store(id={"type": "store", "uid": uid})),
                    TraceUpdater(
                        id={"type": "dynamic-updater", "uid": uid}, gdID=f"{uid}"
                    ),
                    dcc.Interval(
                        id={"type": "interval", "uid": uid, "index": str(idx)},
                        max_intervals=1,
                        interval=1,
                    ),
                ],
            )
        )

    header, df = log_parser.loadTar(path)

    log_parser.postProcessDataframe(df)

    title = header["header"]["timestampHuman"]
    if title == "":
        title = "Timestamp Missing"

    uid = str(uuid4())

    odrive_error_table_data = getODriveErrorDict(df)
    odrive_error_table = dictToTable(
        ("Timestamp", "ODrive Errors"), odrive_error_table_data
    )

    header_table = dictToTable(("Constant", "Value"), header["header"])

    return (
        title,
        odrive_error_table,
        header_table,
        header["description"],
        Serverside(df),
        dynamic_graphs,
    )


def getODriveErrorDict(df: pd.DataFrame) -> dict:
    odrive_errors = odrive_utils.getODriveErrors(df)
    if len(odrive_errors) == 0:
        return None

    odrive_error_dict = {}

    for odrive_error in odrive_errors:
        timestamp_str = f"{odrive_error.timestamp:.03f}"
        odrive_error_dict[timestamp_str] = "\n".join(odrive_error.names)

    return odrive_error_dict


def dictToTable(header: Tuple[str, str], data: dict) -> list:
    table_header = [html.Thead(html.Tr([html.Th(head) for head in header]))]
    table_rows = [
        html.Tr([html.Td(key), html.Td(value)]) for key, value in data.items()
    ]
    table_body = [html.Tbody(table_rows)]
    return table_header + table_body


def exportGraphs():
    pass


if __name__ == "__main__":
    # hi cal poly slo
    parser = argparse.ArgumentParser(description="PLotting Utility N Exporter")

    parser.add_argument(
        "-e", "--export", action="store_true", help="export all logs to html graphs"
    )
    parser.add_argument(
        "-c", "--convert", action="store_true", help="convert all logs to json files"
    )
    parser.add_argument("-l", "--clean", action="store_true", help="clean log files")
    parser.add_argument("-t", "--test", action="store_true", help="testing")

    args = parser.parse_args()

    if args.export:
        exportGraphs()
    else:
        app.run_server(debug=True)
    """
    elif args.convert:
        num_paths = len(raw_paths)
        for n, raw_path in enumerate(raw_paths):
            file_ext = os.path.splitext(raw_path)[1]
            filename = os.path.splitext(os.path.basename(raw_path))[0]
            json_path = os.path.join(json_log_dir, f"{filename}.json")
            if os.path.isfile(json_path):
                continue
            log_data = log_parser.loadBinary(raw_path)
            print(f"Converting ({n+1}/{num_paths}): {raw_path} -> {json_path}")
            # log_parser.dumpLogDataToJson(json_path, log_data)
    elif args.clean:
        num_paths = len(raw_paths)
        for n, raw_path in enumerate(raw_paths):
            file_ext = os.path.splitext(raw_path)[1]
            filename = os.path.splitext(os.path.basename(raw_path))[0]
            json_path = os.path.join(json_log_dir, f"{filename}.json")
            log_data = log_parser.loadBinary(raw_path)
            if (
                np.max(log_data.df["engine_rpm"]) < 300
                and (
                    np.max(log_data.df["shadow_count"])
                    - np.min(log_data.df["shadow_count"])
                )
                < 300
            ):
                print(raw_path)
            if len(log_data.df) == 0:
                print(raw_path)
    """

"""
@callback(
    [Output("export-loading-spinner", "children"), Input("export-graphs", "n_clicks")],
    prevent_initial_call=True,
)
def onExportButtonClicked(n_clicks):
    exportGraphs()
    return (None,)

@callback(
    [
        Output("description-input", "style"),
        Input("description-input", "value"),
        Input("description-save", "n_clicks"),
        State("description-input", "style"),
        State("file-selection", "value"),
    ],
    prevent_initial_call=True,
)
def onDescriptionSaveClicked(description, n_clicks, description_style, path):
    global description_changed
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "description-input":
        if description_changed:
            description_style["backgroundColor"] = "#fccfcf"
        description_changed = True
    elif trigger_id == "description-save":
        with open(path, "r+") as file:
            json_obj = json.load(file)
            file.seek(0)
            if not "metadata" in json_obj:
                json_obj["metadata"] = {}
            json_obj["metadata"]["description"] = description
            json.dump(json_obj, file)
            file.truncate()
        description_style["backgroundColor"] = "white"
    return (description_style,)


def exportGraphs() -> None:
    num_paths = len(json_paths)
    for n, json_path in enumerate(json_paths):
        log_data = log_parser.loadJson(json_path, post_process=True)
        filename = os.path.splitext(os.path.basename(json_path))[0]
        html_path = os.path.join(export_dir, f"{filename}.html")
        print(f"Exporting ({n+1}/{num_paths}): {json_path} -> {html_path}")
        log_parser.dumpLogDataToHTML(html_path, graph_info_json, log_data)
"""

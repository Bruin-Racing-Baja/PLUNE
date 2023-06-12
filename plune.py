#!/usr/bin/env python
import argparse
import json
import os
from typing import List
import numpy as np

import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dash_table,
    dcc,
    html,
    no_update,
)
from google.protobuf import json_format

from generated_protos import header_message_pb2
from utils import log_parser, odrive_utils
from header import header_layout

raw_log_dir = "raw_logs"
json_log_dir = "logs"
export_dir = "graphs"
graph_info_file = "graphs.json"

with open(graph_info_file, "r") as file:
    graph_info_json = json.loads(file.read())
raw_paths = log_parser.getFilesByExtension(raw_log_dir, "bin")
json_paths = log_parser.getFilesByExtension(json_log_dir, "json")


import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div(
    [
        html.Div(id="header", children=header_layout),
        html.Div(
            [],
            id="graphs",
        ),
    ]
)


def exportGraphs() -> None:
    num_paths = len(json_paths)
    for n, json_path in enumerate(json_paths):
        log_data = log_parser.loadJson(json_path, post_process=True)
        filename = os.path.splitext(os.path.basename(json_path))[0]
        html_path = os.path.join(export_dir, f"{filename}.html")
        print(f"Exporting ({n+1}/{num_paths}): {json_path} -> {html_path}")
        log_parser.dumpLogDataToHTML(html_path, graph_info_json, log_data)


def createODriveErrorTable(df: pd.DataFrame) -> dash_table.DataTable:
    odrive_errors = odrive_utils.getODriveErrors(df)
    if len(odrive_errors) == 0:
        return None

    odrive_error_dict = {}
    for odrive_error in odrive_errors:
        timestamp_str = f"{odrive_error.timestamp:.03f}"
        if not timestamp_str in odrive_error_dict:
            odrive_error_dict[timestamp_str] = "\n".join(odrive_error.names)
        else:
            odrive_error_dict[timestamp_str] += "\n" + "\n".join(odrive_error.names)

    odrive_error_rows = [
        {"timestamp": timestamp, "errors": errors}
        for timestamp, errors in odrive_error_dict.items()
    ]

    odrive_error_table = dash_table.DataTable(
        id="datatable",
        columns=[
            {"name": "TIMESTAMP", "id": "timestamp"},
            {"name": "ODRIVE ERRORS", "id": "errors"},
        ],
        data=odrive_error_rows,
    )
    return odrive_error_table


def createHeaderTable(header: header_message_pb2.HeaderMessage) -> dash_table.DataTable:
    table_data = [
        {"constant": field.name, "value": getattr(header, field.name)}
        for field in header_message_pb2.HeaderMessage.DESCRIPTOR.fields
    ]
    header_table = dash_table.DataTable(
        columns=[
            {"name": "CONSTANT", "id": "constant"},
            {"name": "VALUE", "id": "value"},
        ],
        data=table_data,
    )
    return header_table


"""
@callback(
    [Output("export-loading-spinner", "children"), Input("export-graphs", "n_clicks")],
    prevent_initial_call=True,
)
def onExportButtonClicked(n_clicks):
    exportGraphs()
    return (None,)
"""

description_changed = False


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


@callback(
    [
        Output("title", "children"),
        Output("graphs", "children"),
        Output("odrive-errors", "children"),
        Output("log-header", "children"),
        Output("description-input", "value"),
        Input("file-selection", "value"),
    ]
)
def onLogSelection(path):
    if path == None:
        return no_update

    log_data = log_parser.loadJson(path)
    header = log_data.header
    df = log_data.df
    description = log_data.description

    log_parser.postProcessLogData(log_data)

    title = log_data.header.timestamp_human
    if title == "":
        title = "Timestamp Missing"

    figures = log_parser.createFigures(graph_info_json, log_data)
    graphs = [dcc.Graph(figure=fig) for fig in figures]

    odrive_error_table = createODriveErrorTable(df)

    header_table = createHeaderTable(header)

    global description_changed
    description_changed = False
    return title, graphs, [odrive_error_table], header_table, description


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
            log_parser.dumpLogDataToJson(json_path, log_data)
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
    else:
        app.run_server(debug=True)

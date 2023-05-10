#!/usr/bin/env python
import argparse
import json
import os
from typing import List, NamedTuple

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update

from utils import log_parser, odrive_utils
from google.protobuf import json_format

graph_info_file = "graphs.json"
paths = log_parser.getFilesByExtension("logs", "bin")


app = Dash(__name__)


app.layout = html.Div(
    [
        dcc.Loading(
            id="title-loading",
            children=html.Div(
                children=[
                    html.H1(
                        children="Loading...", style={"textAlign": "center"}, id="title"
                    ),
                    html.H4(
                        children="Loading...",
                        style={"textAlign": "center"},
                        id="subtitle",
                    ),
                ]
            ),
        ),
        dcc.Loading(
            id="export-loading",
            children=[
                html.Div(
                    [
                        html.Div(id="export-loading-spinner"),
                    ]
                )
            ],
            type="cube",
        ),
        dcc.Dropdown(paths, paths[-1], id="file-selection", style={"width": "50%"}),
        html.Button("Export Graphs", id="export-graphs"),
        html.Div(
            "",
            id="gains",
        ),
        html.Div(
            dash_table.DataTable(),
            id="odrive-errors",
        ),
        html.Div(
            [],
            id="graphs",
        ),
    ]
)


@callback(
    [
        Output("export-loading-spinner", "children"),
        Input("export-graphs", "n_clicks"),
    ]
)
def onExportButtonClicked(n_clicks):
    if n_clicks == None:
        return (None,)
    log_parser.exportGraphsToHTML(paths, graph_info_file, silent=True)
    log_parser.exportTuningGraphs(paths)
    return (None,)

import pandas as pd
import time
@callback(
    [
        Output("title", "children"),
        Output("subtitle", "children"),
        Output("graphs", "children"),
        Output("odrive-errors", "children"),
        Output("gains", "children"),
        Input("file-selection", "value"),
    ]
)
def onFileSelection(path):
    if path == None:
        return no_update

    log_data = log_parser.loadBinary(path)
    header = log_data.header
    df = log_data.df

    log_parser.postProcessLogData(log_data)

    title = log_data.header.timestamp_human
    if title == "":
        title = "Timestamp Missing"

    graphs = []
    with open(graph_info_file) as file:
        graph_json_data = json.load(file)
        for graph_info in graph_json_data["figures"]:
            if not set(graph_info["y_axis"]).issubset(df.columns):
                print(f'Column(s) Missing: Skipping "{graph_info["title"]}" for {path}')
                continue

            traces = [
                go.Scatter(x=df[graph_info["x_axis"]], y=df[y_axis], name=y_axis)
                for y_axis in graph_info["y_axis"]
            ]

            fig = go.Figure(traces)
            fig.update_layout(
                title=graph_info["title"],
                xaxis_title=graph_info["x_axis"],
                showlegend=True,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            graphs.append(dcc.Graph(figure=fig))

    odrive_errors = odrive_utils.getODriveErrors(df)
    if len(odrive_errors) != 0:
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
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Errors", "id": "errors"},
            ],
            data=odrive_error_rows,
            style_cell={"whiteSpace": "pre-line", "textAlign": "left"},
            style_header={"backgroundColor": "#ff9999", "fontWeight": "bold"},
            fill_width=False,
        )
    else:
        odrive_error_table = None

    return title, path, graphs, [odrive_error_table], json_format.MessageToJson(header)


import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLotting Utility N Exporter")

    parser.add_argument(
        "-e", "--export", action="store_true", help="export all logs to html graphs"
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="testing"
    )

    args = parser.parse_args()

    if args.export:
        log_parser.exportGraphsToHTML(paths, graph_info_file)
    if args.test:

        np.set_printoptions(suppress=True)
        all_stats = []
        i = 0
        for path in paths[-50:]:
            print(i)
            orig_size = os.path.getsize(path)/1e6
            log_data = log_parser.loadBinary(path)
            header = log_data.header
            df = log_data.df
            log_parser.postProcessLogData(log_data)
            formats, stats = log_parser.dumpLogDataToJson("out", log_data)
            stats = np.array(stats)
            stats[:,0] /= orig_size
            all_stats.append(stats)
            i += 1

        x = np.array(all_stats)
        mean_times = np.mean(x[:,:,0], axis=0)
        mean_memories = np.mean(x[:,:,1], axis=0)
        print(formats)
        print(mean_times)
        print(mean_memories)
        
    else:
        app.run_server(debug=True)

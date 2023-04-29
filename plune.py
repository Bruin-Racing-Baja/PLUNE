#!/usr/bin/env python
import argparse
import json
import os
from typing import List, NamedTuple

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update

from utils import log_parser, odrive_utils

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
    log_parser.exportGraphsToHTML(paths, graph_info_file)
    return (None,)


@callback(
    [
        Output("title", "children"),
        Output("subtitle", "children"),
        Output("graphs", "children"),
        Output("odrive-errors", "children"),
        Input("file-selection", "value"),
    ]
)
def onFileSelection(path):
    if path == None:
        return no_update

    header, df = log_parser.loadBinary(path)
    log_parser.postProcessDataframe(df)

    title = header.timestamp_human
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
            print(len(traces))

            fig = go.Figure(traces)
            fig.update_layout(
                title=graph_info["title"],
                xaxis_title=graph_info["x_axis"],
                showlegend=True,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            graphs.append(dcc.Graph(figure=fig))
    print(len(graphs))

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

    return title, path, graphs, [odrive_error_table]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLotting Utility N Exporter")

    parser.add_argument(
        "-e", "--export", action="store_true", help="export all logs to html graphs"
    )

    args = parser.parse_args()

    if args.export:
        log_parser.exportGraphsToHTML(paths, graph_info_file)
    else:
        app.run_server(debug=True)

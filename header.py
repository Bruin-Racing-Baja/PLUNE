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
import dash_bootstrap_components as dbc
from utils import log_parser
import json

raw_log_dir = "raw_logs"
json_log_dir = "logs"
export_dir = "graphs"
graph_info_file = "graphs.json"

with open(graph_info_file, "r") as file:
    graph_info_json = json.loads(file.read())

raw_paths = log_parser.getFilesByExtension(raw_log_dir, "bin")
json_paths = log_parser.getFilesByExtension(json_log_dir, "json")


header_layout = [
    # Page title
    html.Div(
        id="title-container",
        children=dcc.Loading(
            id="title-loading",
            children=html.Div(
                children=[
                    html.H1(
                        children="Loading...",
                        style={"textAlign": "center"},
                        id="title",
                    ),
                ]
            ),
        ),
    ),
    # File selection
    html.Div(
        id="file-selection-container",
        children=dcc.Dropdown(
            id="file-selection",
            options=json_paths,
            value=json_paths[-1] if len(json_paths) > 0 else None
            # style={"width": "50%"},
        ),
    ),
    # Data
    html.Div(
        id="header-data-container",
        children=[
            html.Div(
                id="description-container",
                children=[
                    dbc.Textarea(
                        id="description-input",
                    ),
                    dbc.Button(
                        "💾",
                        id="description-save",
                    ),
                ],
            ),
            html.Div(
                dash_table.DataTable(),
                id="log-header",
            ),
            html.Div(
                dash_table.DataTable(),
                id="odrive-errors",
            ),
        ],
    ),
]

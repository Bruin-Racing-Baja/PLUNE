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


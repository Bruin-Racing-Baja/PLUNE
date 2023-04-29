import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from google.protobuf import json_format

from generated_protos import header_message_pb2, log_message_pb2

header_message_id = 0
log_message_id = 1


def getFilesByExtension(log_dir: str, ext: str) -> List[str]:
    log_dir_contents = os.listdir(log_dir)
    txt_paths = []
    for potential_file in log_dir_contents:
        potential_file_path = os.path.join(log_dir, potential_file)
        potential_file_ext = os.path.splitext(potential_file)[1]
        if os.path.isfile(potential_file_path) and potential_file_ext == f".{ext}":
            txt_paths.append(potential_file_path)
    txt_paths.sort()
    return txt_paths


def filterFilesBySize(paths: List[str], size_kb: int) -> List[str]:
    return [path for path in paths if os.path.getsize(path) >= size_kb * 1e3]


def loadBinary(path: str) -> Tuple[header_message_pb2.HeaderMessage, pd.DataFrame]:
    header_message = None
    df = None
    with open(path, "rb") as file:
        all_rows = []
        header_message = header_message_pb2.HeaderMessage()
        while True:
            message_type_raw = file.read(1)
            if message_type_raw == b"":
                break
            message_length_raw = file.read(4)

            message_type = int(message_type_raw, 16)
            message_length = int(message_length_raw, 16)
            message = file.read(message_length)

            if message_type == header_message_id:
                header_message.ParseFromString(message)
            elif message_type == log_message_id:
                message_type = log_message_pb2.LogMessage.DESCRIPTOR
                log_message = log_message_pb2.LogMessage()
                log_message.ParseFromString(message)
                row_values = []
                for field in log_message_pb2.LogMessage.DESCRIPTOR.fields:
                    row_values.append(getattr(log_message, field.name))
                all_rows.append(row_values)
        columns = [field.name for field in log_message_pb2.LogMessage.DESCRIPTOR.fields]
        df = pd.DataFrame(all_rows, columns=columns)
    return header_message, df


def dumpDataframeToBinary(
    path: str, header: header_message_pb2.HeaderMessage, df: pd.DataFrame
) -> None:
    with open(path, "wb") as bin_file:
        serialized_header_message = header.SerializeToString()
        message_type = header_message_id
        message_length = len(serialized_header_message)
        delimiter = f"{message_type:01X}{message_length:04X}"

        bin_file.write(delimiter.encode())
        bin_file.write(serialized_header_message)

        rows = df.to_dict(orient="records")
        log_message = log_message_pb2.LogMessage()
        for row in rows:
            json_format.ParseDict(row, log_message)
            serialized_log_message = log_message.SerializeToString()

            message_type = log_message_id
            message_length = len(serialized_log_message)
            delimiter = f"{message_type:01X}{message_length:04X}"

            bin_file.write(delimiter.encode())
            bin_file.write(serialized_log_message)


def dumpFiguresToHTML(figs: List[go.Figure], path: str, offline: bool = False):
    with open(path, "w") as file:
        file.write("<html><head></head><body>\n")
        for fig in figs:
            tmp_layout = fig.layout
            fig.update_layout(font={"size": 20})
            inner_html = (
                fig.to_html(include_plotlyjs=(True if offline else "cdn"))
                .split("<body>")[1]
                .split("</body>")[0]
            )
            file.write(inner_html)
            fig.layout = tmp_layout
        file.write("</body></html>\n")


def appendNormalizedSeries(df: pd.DataFrame) -> None:
    for col in df.columns:
        col_norm = f"norm_{col}"
        if col.startswith("norm_") or col_norm in df:
            continue
        cur_series = df[col]
        if np.issubdtype(cur_series.dtype, np.number):
            if np.max(cur_series) == np.min(cur_series):
                df[col_norm] = 0.0
            else:
                df[col_norm] = (cur_series - np.min(cur_series)) / (
                    np.max(cur_series) - np.min(cur_series)
                )


def postProcessDataframe(df: pd.DataFrame) -> None:
    wheel_diameter = 23
    pitch_angle = 5
    encoder_cpr = 8192
    wheel_to_secondary_ratio = (57 / 18) * (45 / 17)
    df["control_cycle_start_s"] = df["control_cycle_start_us"] / 1e6
    df["control_cycle_dt_s"] = df["control_cycle_dt_us"] / 1e6

    df["secondary_rpm"] = df["wheel_rpm"] * wheel_to_secondary_ratio
    df["wheel_mph"] = (df["wheel_rpm"] * wheel_diameter * np.pi) / (12 * 5280) * 60

    df["vehicle_position_feet"] = (
        np.cumsum(df["wheel_mph"] * 5280 * df["control_cycle_dt_s"]) / 3600
    )

    df["actuator_position_mm"] = -df["shadow_count"] / encoder_cpr * pitch_angle

    df["shift_ratio"] = df["secondary_rpm"] / df["engine_rpm"]
    df["shift_ratio"] = df["shift_ratio"].clip(lower=0.2, upper=2)

    appendNormalizedSeries(df)


def trimDataframe(
    df: pd.DataFrame, start_s: float = 0, end_s: float = float("inf")
) -> pd.DataFrame:
    df = df.loc[
        (start_s < df["control_cycle_start_s"]) & (df["control_cycle_start_s"] < end_s)
    ]
    return df


def exportGraphsToHTML(paths, graph_info_file, export_dir="graphs", silent=False):
    for path in paths:
        header, df = loadBinary(path)
        postProcessDataframe(df)
        figures = []
        with open(graph_info_file) as file:
            graph_json_data = json.load(file)
            for graph_info in graph_json_data["figures"]:
                if not set(graph_info["y_axis"]).issubset(df.columns):
                    print(
                        f'Column(s) Missing: Skipping "{graph_info["title"]}" for {path}'
                    )
                    continue
                traces = [
                    go.Scatter(x=df[graph_info["x_axis"]], y=df[y_axis], name=y_axis)
                    for y_axis in graph_info["y_axis"]
                ]
                figure = go.Figure(traces)
                figure.update_layout(
                    title=graph_info["title"],
                    xaxis_title=graph_info["x_axis"],
                    showlegend=True,
                )
                figures.append(figure)
        filename_without_ext = os.path.splitext(os.path.basename(path))[0]
        html_path = os.path.join(export_dir, f"{filename_without_ext}.html")
        if not silent:
            print(f"Exporting {path} -> {html_path}")
        dumpFiguresToHTML(figures, html_path)

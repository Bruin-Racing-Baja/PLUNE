import json
import os
from typing import List, Tuple
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from google.protobuf import json_format

from generated_protos import header_message_pb2, log_message_pb2
import io
from google.protobuf import message as _message
from enum import IntEnum


class MessageID(IntEnum):
    NONE = -1
    HEADER = 0
    LOG = 1


class LogData:
    def __init__(
        self,
        header: header_message_pb2.HeaderMessage = header_message_pb2.HeaderMessage(),
        df: pd.DataFrame = pd.DataFrame(),
    ):
        self.header = header
        self.df = df


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


def nextDelimitedMessage(
    buffer: io.BufferedIOBase,
) -> Tuple[_message.Message, MessageID]:
    raw_message_id = buffer.read(1)
    if raw_message_id == b"":
        return None, MessageID.NONE
    message_id = MessageID(int(raw_message_id, 16))

    raw_message_length = buffer.read(4)
    message_length = int(raw_message_length, 16)

    raw_message = buffer.read(message_length)
    if message_id == MessageID.HEADER:
        message = header_message_pb2.HeaderMessage()
    elif message_id == MessageID.LOG:
        message = log_message_pb2.LogMessage()
    message.ParseFromString(raw_message)

    return message, message_id


def loadBinary(path: str) -> LogData:
    log_data = LogData()
    columns = [field.name for field in log_message_pb2.LogMessage.DESCRIPTOR.fields]
    rows = []
    with open(path, "rb") as file:
        while True:
            message, message_id = nextDelimitedMessage(file)

            if message_id == MessageID.NONE:
                break
            elif message_id == MessageID.HEADER:
                log_data.header = message
            elif message_id == MessageID.LOG:
                row_values = [None] * len(columns)
                for idx, col in enumerate(columns):
                    row_values[idx] = getattr(message, col)
                rows.append(row_values)
    log_data.df = pd.DataFrame(rows, columns=columns)
    return log_data


from time import process_time
import pyarrow as pa
import pyarrow.parquet as pq

def toParquet(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.parquet"
    pq.write_table(pa.Table.from_pandas(df), filename)
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s

def toJson(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.json"
    df.to_json(filename, orient="split")
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s

def toPickle(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.pickle"
    df.to_pickle(filename)
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s

def toNumpy(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.npy"
    np.save(filename, df.to_numpy())
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s

def toFeather(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.feather"
    df.to_feather(filename)
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s

def toCSV(prefix: str, df: pd.DataFrame):
    start_s = process_time()
    filename = f"{prefix}.csv"
    df.to_csv(filename)
    end_s = process_time()
    return os.path.getsize(filename)/1e6,end_s-start_s


def dumpLogDataToJson(filename: str, log_data: LogData):
    with open(filename, "w") as file:
        df = log_data.df
        formats = ["parquet", "json", "pickle", "numpy", "feather", "csv"]
        stats = []
        stats.append(toParquet(filename, df))
        stats.append(toJson(filename, df))
        stats.append(toPickle(filename, df))
        stats.append(toNumpy(filename, df))
        stats.append(toFeather(filename, df))
        stats.append(toCSV(filename, df))

        #json_data = log_data.df.to_json("out.json", orient="split")
        #a = json.JSONDecoder().raw_decode(json_data)[0]
        #json_obj = {
            #"header": json_format.MessageToJson(log_data.header   ),
            #"dataframe": a
        #}
        #print(type(json_obj["dataframe"]))
        #file.write(log_data.df.to_json(orient="split"))
    return formats, stats


def loadJsonToDataframe(filename: str) -> pd.DataFrame:
    with open(filename, "r") as file:
        json_data = file.read()
        df = pd.read_json(json_data, orient="split")


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


def dumpFiguresToHTML(
    figs: List[go.Figure],
    header: header_message_pb2.HeaderMessage,
    path: str,
    offline: bool = False,
):
    with open(path, "w") as file:
        header_info = ""
        if header != None:
            header_json = json_format.MessageToJson(header)
            header_json_formatted = html.escape(header_json).replace("\n", "<br>")
            header_info = f"<div style='font-size: 28px;'>{header_json_formatted}</div>"
        file.write(f"<html><head></head><body>{header_info}\n")
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


def postProcessLogData(log_data: LogData):
    wheel_diameter = 23
    pitch_angle = 5
    encoder_cpr = 8192
    wheel_to_secondary_ratio = (57 / 18) * (45 / 17)
    df = log_data.df
    df["control_cycle_start_s"] = df["control_cycle_start_us"] / 1e6
    if "last_control_cycle_stop_us" in df.columns:
        df["last_control_cycle_stop_s"] = df["last_control_cycle_stop_us"] / 1e6
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
        dumpFiguresToHTML(figures, header, html_path)


def exportTuningGraphs(paths, export_path="graphs/tuning.html", silent=False):
    figures = []
    for path in paths:
        header, df = loadBinary(path)
        postProcessDataframe(df)

        x_axis = "control_cycle_start_s"
        y_axises = ["engine_rpm", "secondary_rpm", "target_rpm"]
        title = f"{header.timestamp_human} (KP = {header.p_gain:.06f}, KD = {header.d_gain:.06f})"
        traces = [
            go.Scatter(x=df[x_axis], y=df[y_axis], name=y_axis) for y_axis in y_axises
        ]
        figure = go.Figure(traces)
        figure.update_layout(
            title=title,
            xaxis_title=x_axis,
            showlegend=True,
        )
        figures.append(figure)

    dumpFiguresToHTML(figures, None, export_path)

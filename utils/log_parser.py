import io
import json
import os
import tarfile
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.signal
from google.protobuf import json_format
from google.protobuf import message as _message

from generated_protos import operation_header_pb2, control_function_state_pb2#, error_state_pb2

pd.options.mode.chained_assignment = None


class MessageID(IntEnum):
    NONE = -1
    HEADER = 0
    LOG = 1

pb2_messages = {
    MessageID.NONE: None,
    MessageID.HEADER: operation_header_pb2.OperationHeader,
    MessageID.LOG: control_function_state_pb2.ControlFunctionState,
}


class LogData:
    def __init__(self, header: dict = None, post: dict = None, df: pd.DataFrame = None):
        self.header = header
        self.post = post
        self.df = df


def getFilesByExtension(dir: str, ext: str) -> List[str]:
    dir_contents = os.listdir(dir)
    files = []
    for file_in_dir in dir_contents:
        file_path = os.path.join(dir, file_in_dir)
        file_ext = os.path.splitext(file_path)[1]
        if os.path.isfile(file_path) and file_ext == f".{ext}":
            files.append(file_path)
    files.sort()
    return files

def filterFilesBySize(filenames: List[str], size_kb: int) -> List[str]:
    return [
        filename
        for filename in filenames
        if os.path.getsize(filename) >= size_kb * 1e3
    ]


def isHexAscii(c):
    return (ord("0") <= c and c <= ord("9")) or (ord("A") <= c and c <= ord("F"))

## Protobuf helpers ##
def nextDelimitedMessage(
    buffer: io.BufferedIOBase,
) -> Tuple[_message.Message | None, MessageID]:
    raw_message_id = buffer.read(1)

    if raw_message_id == b"":
        return None, MessageID.NONE

    message_id = MessageID(int(raw_message_id, 16))

    raw_message_length = buffer.read(4)
    message_length = int(raw_message_length, 16)

    raw_message = buffer.read(message_length)

    message = pb2_messages[message_id]()
    try:
        message.ParseFromString(raw_message)
    except _message.DecodeError:
        buffer.seek(-message_length, io.SEEK_CUR)
        raw_message = buffer.read(message_length)
        return None, None

    return message, message_id


def loadBinary(path: str) -> LogData:
    print(path)
    log_data = LogData()
    columns = [field.name for field in control_function_state_pb2.ControlFunctionState.DESCRIPTOR.fields]
    rows = []
    with open(path, "rb") as file:
        i = 0
        while True:
            message, message_id = nextDelimitedMessage(file)

            if message_id == None:
                posd = [0, 0, 0, 0,0]

                while True:
                    posd = posd[1:]
                    byte_read = file.read(1)
                    if byte_read == b"":
                        break
                    posd.append(int.from_bytes(byte_read))
                    if posd[0] == ord("1") and posd[1] == ord("0") and posd[2] == ord("0") and isHexAscii(posd[3]) and isHexAscii(posd[4]):
                        print(f"HERE: {i}")
                        file.seek(-5, io.SEEK_CUR)
                        break
            if message != None:
                if message_id == MessageID.HEADER:
                    log_data.header = json_format.MessageToDict(message)
                if message_id == MessageID.LOG:
                    row_values = [None] * len(columns)
                    for idx, col in enumerate(columns):
                        row_values[idx] = getattr(message, col)
                      
                    rows.append(row_values)
            elif message_id != None:
                break
            i += 1
        log_data.df = pd.DataFrame(rows, columns=columns)
    return log_data


## Log loaders ##
"""
def loadJson(filename: str) -> LogData:
    # TODO speed this up
    log_data = LogData()
    with open(filename, "r") as file:
        json_obj = json.load(file)
        if "dataframe" in json_obj:
            log_data.df = pd.DataFrame(**json_obj["dataframe"])
        if "header" in json_obj:
            json_format.ParseDict(json_obj["header"], log_data.header)
        if "metadata" in json_obj:
            log_data.description = json_obj["metadata"].get("description", "")
    return log_data
"""


def loadTar(filename: str) -> LogData:
    log_data = LogData()
    with tarfile.open(filename, "r") as tar:
        tar_members = tar.getnames()

        if "header.json" in tar_members:
            header_file = tar.extractfile("header.json")
            log_data.header = json.loads(header_file.read().decode("utf-8"))
        if "post.json" in tar_members:
            post_file = tar.extractfile("post.json")
            log_data.post = json.loads(post_file.read().decode("utf-8"))
        if "dataframe.parquet" in tar_members:
            dataframe_file = tar.extractfile("dataframe.parquet")
            log_data.df = pd.read_parquet(io.BytesIO(dataframe_file.read()))
            #print(log_data.header)

    return log_data


def dumpTar(filename: str, log_data: LogData) -> None:
    parquet_buffer = io.BytesIO()
    log_data.df.to_parquet(parquet_buffer)
    parquet_buffer.seek(0)

    header_json = json.dumps(log_data.header)
    post_json = json.dumps(log_data.post)
    if post_json == None:
        header_json = {}

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        parquet_info = tarfile.TarInfo("dataframe.parquet")
        parquet_info.size = len(parquet_buffer.getvalue())
        tar.addfile(parquet_info, parquet_buffer)

        header_json_info = tarfile.TarInfo("header.json")
        header_json_buffer = io.BytesIO(header_json.encode())
        header_json_info.size = len(header_json_buffer.getvalue())
        tar.addfile(header_json_info, header_json_buffer)

        post_json_info = tarfile.TarInfo("post.json")
        post_json_buffer = io.BytesIO(post_json.encode())
        post_json_info.size = len(post_json_buffer.getvalue())
        tar.addfile(post_json_info, post_json_buffer)

    tar_buffer.seek(0)

    with open(filename, "wb") as f:
        f.write(tar_buffer.getvalue())


"""
def dumpLogDataToJson(filename: str, log_data: LogData):
    # TODO speed this up
    with open(filename, "w") as file:
        json_obj = {
            "header": json.loads(json_format.MessageToJson(log_data.header)),
            "dataframe": json.loads(log_data.df.to_json(orient="split", index=False)),
            "metadata": {"description": log_data.description},
        }
        json.dump(json_obj, file, separators=(",", ":"))
"""


def appendNormalizedSeries(df: pd.DataFrame) -> None:
    for col in df.columns:
        col_norm = f"norm_{col}"
        if col.startswith("norm_") or col_norm in df:
            continue
        cur_series = df[col]
        if np.issubdtype(cur_series.dtype, bool):
            values = cur_series.values.astype(int)
            if np.max(values) == np.min(values):
                df[col_norm] = 0.0
            else:
                df[col_norm] = (values - np.min(values)) / (
                    np.max(values) - np.min(values)
                )

        if np.issubdtype(cur_series.dtype, np.number):
            if np.max(cur_series) == np.min(cur_series):
                df[col_norm] = 0.0
            else:
                df[col_norm] = (cur_series - np.min(cur_series)) / (
                    np.max(cur_series) - np.min(cur_series)
                )
    df['vel_p'] = df['engine_rpm_error'] * df['p_term']
    df['vel_d'] = df['engine_rpm_derror'] * df['d_term']


def trimDataframe(
    df: pd.DataFrame, start_s: float = 0, end_s: float = float("inf")
) -> pd.DataFrame:
    df.drop(df[(df["cycle_start_s"] <= start_s) | (df["cycle_start_s"] >= end_s)].index, inplace=True)

def paddedDiff(x):
    diff = np.diff(x)
    return np.concatenate(([diff[0]], diff))

def postProcessDataframe(df: pd.DataFrame):
    wheel_diameter = 23
    pitch_angle = 2
    actuator_belt_ratio = 1.5  # ballscrew:motor
    encoder_cpr = 8192
    wheel_to_secondary_ratio = (57 / 18) * (45 / 17)
   # print(df.columns)
    df["cycle_start_s"] = df["cycle_start_us"] / 1e6
    diff = df["cycle_start_s"].diff()
    df["inbound_limit_switch"] = df["inbound_limit_switch"].astype(int)
    df["outbound_limit_switch"] = df["outbound_limit_switch"].astype(int)
    df["engage_limit_switch"] = df["engage_limit_switch"].astype(int)

    idxs = diff[diff < 0].index
    for idx in idxs:
        df["cycle_start_us"].iloc[idx:] += 2**32 - 1
        # df["control_cycle_stop_us"].iloc[idx:] += 2**32 - 1
        df["cycle_start_s"].iloc[idx:] += (2**32 - 1) / 1e6

    # df["control_cycle_dt_s"] = df["control_cycle_dt_us"] / 1e6
    #need_len = df["engine_rpm"].values.shape[0]
    #arr = df["engine_rpm"].values[::10]
    #diff = np.diff(arr)
    #diff2 = np.repeat(diff, 10)
    #cur_len = diff2.shape[0]
    #padded_arr = np.pad(diff2, (0, need_len - cur_len), 'constant', constant_values=(0,))
    #print(padded_arr.shape)
    #df["engine_rpm"] = padded_arr


    trimDataframe(df, 5)
    ###df["filt_engine_rpm_derror"] = paddedDiff(df["filtered_engine_rpm"] - df["target_rpm"])
    #df["filt_engine_rpm_derror"] = paddedDiff(df["filt_engine_rpm_error"]) * 10
    #df["d_term"] = df["engine_rpm_derror"] *0.008;
    #df["p_term"] = df["engine_rpm_error"] *0.04;
    df["all_term"] = df["p_term"] + df["d_term"];
    df["wheel_mph"] = df["filtered_secondary_rpm"] / 8.978 * 6.842484e-02
    b, a = scipy.signal.butter(1, 2, fs=100)

    
    #df["throttle_filt"] = scipy.signal.lfilter(b, a, scipy.signal.medfilt(df["throttle"],5))
    b, a = scipy.signal.butter(1, 0.1, fs=100)
    df["brake_filt"] = scipy.signal.lfilter(b, a, df["brake"])
    df["filtered_engine_rpm"] = np.clip(df["filtered_engine_rpm"], -100, 5000)
    #df["d_throttle"] = paddedDiff(df["throttle_filt"])
    #df["filtered_engine_rpm"] = df["filtered_engine_rpm"].to_numpy()[np.where(np.abs(df["filtered_engine_rpm"].to_numpy()) >4500 )]

#    df["secondary_rpm"] = df["wheel_rpm"] * wheel_to_secondary_ratio
#    df["wheel_mph"] = (df["wheel_rpm"] * wheel_diameter * np.pi) / (12 * 5280) * 60
#    b, a = scipy.signal.butter(12, 1.2, fs=50)
#    df["wheel_mph"] = scipy.signal.filtfilt(b, a, df["wheel_mph"])
#
#    df["vehicle_position_feet"] = (
#        np.cumsum(df["wheel_mph"] * 5280 * df["control_cycle_dt_s"]) / 3600
#    )
#
#    df["actuator_position_inches"] = (
#        -df["shadow_count"]
#        / encoder_cpr
#        * pitch_angle
#        * actuator_belt_ratio
#        / 2.54
#        / 10
#    )
#    df["motor_position_rot"] = df["shadow_count"] / encoder_cpr
#
#    df["shift_ratio"] = df["filtered_engine_rpm"] / df["filtered_secondary_rpm"]
#    df["shift_ratio"] = df["shift_ratio"].clip(lower=0.2, upper=2)
#
    p = 0.04
    d = 0.002

    df["sim_velocity_command_p"] = p * (df["filtered_engine_rpm"] - df["target_rpm"])
    df["sim_velocity_command_d"] = np.minimum(d * df["filtered_engine_rpm"], 0)
#
    df["sim_velocity_command"] = (
        df["sim_velocity_command_p"] + df["sim_velocity_command_d"]
    )
    print(df["sim_velocity_command"])

#
#    df["control_cycle_execution_time_us"] = (
#        df["control_cycle_stop_us"].shift(-1)
#    ) - df["cycle_start_us"]

    appendNormalizedSeries(df)


def updateTarDescription(tar_path: str, description: str) -> None:
    tar_buffer = io.BytesIO()

    with tarfile.open(tar_path, "r") as original_tar:
        with tarfile.open(fileobj=tar_buffer, mode="w") as new_tar:
            for tarinfo in original_tar:
                if tarinfo.name == "post.json":
                    post_file = original_tar.extractfile(tarinfo)
                    post_data = json.load(post_file)
                    if post_data == None:
                        post_data = {}

                    post_data["description"] = description

                    updated_post = json.dumps(post_data).encode("utf-8")

                    updated_post_info = tarinfo
                    updated_post_info.size = len(updated_post)

                    new_tar.addfile(updated_post_info, io.BytesIO(updated_post))
                else:
                    file_content = original_tar.extractfile(tarinfo).read()
                    new_tar.addfile(tarinfo, io.BytesIO(file_content))

    tar_buffer.seek(0)
    with open(tar_path, "wb") as new_tar:
        new_tar.write(tar_buffer.getvalue())


"""
def figuresToHTML(
    figures: List[go.Figure],
    offline: bool = False,
) -> str:
    inner_htmls = []
    for fig in figures:
        fig.update_layout(font={"size": 20})
        inner_htmls.append(
            fig.to_html(include_plotlyjs=(True if offline else "cdn"))
            .split("<body>")[1]
            .split("</body>")[0]
        )
    return "".join(inner_htmls)


def headerToHTML(header: header_message_pb2.HeaderMessage) -> str:
    inner_html = "<table border=1>"
    for field in header_message_pb2.HeaderMessage.DESCRIPTOR.fields:
        key = field.name
        value = getattr(header, key)
        inner_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    inner_html += "</table>"
    return inner_html


def dumpLogDataToHTML(filename: str, graph_info_json: dict, log_data: LogData) -> None:
    figures = createFigures(graph_info_json, log_data)
    figures_html = figuresToHTML(figures)
    header_html = headerToHTML(log_data.header)
    with open(filename, "w") as file:
        file.write("<html><head></head><body>")
        file.write(header_html)
        file.write(figures_html)
        file.write("</body></html>")


def createFigures(graph_info_json: dict, log_data: LogData) -> List[go.Figure]:
    df = log_data.df
    figures = []
    for graph_info in graph_info_json["figures"]:
        if graph_info.get("disabled", False):
            continue
        if not set(graph_info["y_axis"]).issubset(df.columns):
            print(f'Column(s) Missing: Skipping "{graph_info["title"]}"')
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
        figures.append(fig)
    return figures

def exportTuningGraphs(paths, export_path="graphs/tuning.html", silent=False):
   figures = []
   for path in paths:
       log_data = loadBinary(path)
       postProcessLogData(log_data)

       df = log_data.df
       header = log_data.header

       x_axis = "cycle_start_s"
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
"""

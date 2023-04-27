import os

import log_message_pb2
import pandas as pd
from google.protobuf import json_format

header_message_id = 0
log_message_id = 1

csv_cols = [
    "control_cycle_dt_us",
    "voltage",
    "last_heartbeat_ms",
    "wheel_rpm",
    "engine_rpm",
    "target_rpm",
    "velocity_command",
    "unclamped_velocity_command",
    "shadow_count",
    "inbound_estop",
    "outbound_estop",
    "iq_measured",
    "flushed",
    "wheel_count",
    "engine_count",
    "iq_setpoint",
    "control_cycle_start_us",
    "control_cycle_stop_us",
    "odrive_current",
    "axis_error",
    "motor_error",
    "encoder_error",
]
csv_cols_no_unclamp = [col for col in csv_cols if col != "unclamped_velocity_command"]


def convertCSVToBinary(path):
    df = pd.read_csv(path, skiprows=1, header=None)
    if len(df.columns) == 21:
        df.columns = csv_cols_no_unclamp
    else:
        df.columns = csv_cols
    df["inbound_estop"] = df["inbound_estop"].astype(bool)
    df["outbound_estop"] = df["outbound_estop"].astype(bool)
    df = df.drop("flushed", axis=1)

    path_without_ext = os.path.splitext(path)[0]
    with open(f"{path_without_ext}.bin", "wb") as bin_file:
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


def parseCSVFile(path):
    try:
        df = pd.read_csv(path, skiprows=1, header=None, names=csv_cols)
    except:
        return None, None
    return None, pd.Dataframe(df)

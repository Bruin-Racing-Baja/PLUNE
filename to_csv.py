#!/usr/bin/env python
import io
import os
import datetime
from enum import IntEnum
from typing import List, Tuple

from google.protobuf import json_format
from google.protobuf import message as _message
import pandas as pd

from generated_protos import (
    operation_header_pb2,
    control_function_state_pb2,
)

import sys

class MessageID(IntEnum):
    NONE = -1
    HEADER = 0
    LOG = 1


pb2_messages = {
    MessageID.NONE: None,
    MessageID.HEADER: operation_header_pb2.OperationHeader,
    MessageID.LOG: control_function_state_pb2.ControlFunctionState,
}

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
        filename for filename in filenames if os.path.getsize(filename) >= size_kb * 1e3
    ]

def nextDelimitedMessage(
    buffer: io.BufferedIOBase,
) -> Tuple[_message.Message | None, MessageID | None]:
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
        return None, None

    return message, message_id


def loadBinary(path: str):
    columns = [
        field.name
        for field in control_function_state_pb2.ControlFunctionState.DESCRIPTOR.fields
    ]
    rows = []
    header = None
    df = None
    with open(path, "rb") as file:
        while True:
            message, message_id = nextDelimitedMessage(file)
            if message != None:
                if message_id == MessageID.HEADER:
                    header = json_format.MessageToDict(message)
                if message_id == MessageID.LOG:
                    row_values = [None] * len(columns)
                    for idx, col in enumerate(columns):
                        row_values[idx] = getattr(message, col)
                    rows.append(row_values)
            elif message_id != None:
                break
        df = pd.DataFrame(rows, columns=columns)
    return header, df


def dumpCSV(header: dict, df: pd.DataFrame):
    human_timestamp = datetime.datetime.fromtimestamp(header["timestamp"]).strftime('%Y-%m-%d_%H-%M-%S')
    df.to_csv(f"logs/{human_timestamp}.csv")

if __name__ == "__main__":
    header, df = loadBinary(sys.argv[1])
    dumpCSV(header, df)

# PLUNE - PLotting Utility N Exporter
## Instructions
1. Install [protobuf](https://github.com/protocolbuffers/protobuf/releases/)
2. Generate python files: `protoc --python_out generated --pyi_out generated log_message.proto header_message.proto`
3. Install protobuf for python:
    `pip install protobuf`
4. Run:
    `./plune.py`
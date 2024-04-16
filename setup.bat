@echo off
set output_path="generated_protos"

mkdir %output_path%
protoc --python_out %output_path% --pyi_out %output_path% --proto_path PIER23 PIER23/*.proto

mkdir raw_logs logs graph

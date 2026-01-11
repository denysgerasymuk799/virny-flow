protoc --python_out ../ --grpc_out ../ alpine_meadow/common/proto/*.proto --plugin=protoc-gen-grpc=/usr/local/bin/grpc_python_plugin

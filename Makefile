all: python/tensorboard/proto

clean:
	rm tensorboard/src/*_pb2.py

python/tensorboard/proto:
	protoc proto_src/*.proto --python_out=tensorboard/src

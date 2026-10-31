import json
import socket
import socketserver
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

model_path = "./ufo-model.pkl"
try:
    with open(model_path, "rb") as f:
        model: LogisticRegression = pickle.load(f)
except OSError:
    print("model file not found")
    exit(1)


from sklearn.preprocessing import LabelEncoder

encoder_path = "./encoder.pkl"
try:
    with open(encoder_path, "rb") as f:
        encoder: LabelEncoder = pickle.load(f)
except OSError:
    print("encoder file not found")


# req_data = {
#     "seconds": 1,
#     "longitude": 30,
#     "latitude": 20,
# }


class PredictRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.model = model
        self.encoder = encoder
        print(f"the type of mode; {type(model)}")
        print(f"the type of encoder: {type(encoder)}")
        super().__init__(request, client_address, server)
        return

    def setup(self):
        print("predict handler setup")
        return

    def handle(self):
        input_feature = self.load_feature()

        output = self.model.predict(input_feature)
        print(f"shape of output: {output.shape}")
        print(output)

        res = self.encoder.inverse_transform(output)
        print(res)

    def load_feature(self) -> pd.DataFrame:

        blen = self.request.recv(4)
        print(f"length of byte data: {len(blen)}")
        print("byte data: ", blen)
        length = int.from_bytes(blen)
        print(f"the length of data: {length}")

        byte_data: bytes = self.request.recv(length)

        data_dict: dict = json.loads(byte_data.decode())

        return pd.DataFrame([data_dict])

    def finish(self):
        super().finish()


class PingRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)
        return

    def setup(self):
        print("ping handlder setting up")
        super().setup()

    def handle(self):

        try:
            while True:
                data = self.request.recv(1024)
                print(data.decode())
                # self.request.sendall(f"your data is {data.decode()}".encode())
                self.request.sendall("hello from server".encode())
        finally:
            print("connect closed by client")
            return

    def finish(self):
        super().finish()


socket_path = "/tmp/py-predict.sock"

if os.path.exists(socket_path):
    os.unlink(socket_path)

srv = socketserver.UnixStreamServer(socket_path, PredictRequestHandler)

print(f"server addr: {srv.server_address}")
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    srv.server_close()
    os.unlink(socket_path)

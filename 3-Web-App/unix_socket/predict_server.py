import json
import socketserver
import os
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

model_path = "./ufo-model.pkl"
try:
    with open(model_path, "rb") as f:
        model: LogisticRegression = pickle.load(f)
except OSError:
    print("model file not found")
    sys.exit(1)


from sklearn.preprocessing import LabelEncoder

encoder_path = "./encoder.pkl"
try:
    with open(encoder_path, "rb") as f:
        encoder: LabelEncoder = pickle.load(f)
except OSError:
    print("encoder file not found")


# req_data = {
#     "Seconds": 1,
#     "Longitude": 30,
#     "Latitude": 20,
# }

# from ['us', nan, 'gb', 'ca', 'au', 'de']
# to ["Australia", "Canada", "Germany", "UK", "US", "Unknown"]
countryMap = {""}


class PredictRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.model = model
        self.encoder = encoder
        super().__init__(request, client_address, server)

    def setup(self):
        print("predict handler setup")

    def handle(self):
        input_feature = self.load_feature()

        output = self.model.predict(input_feature)
        # it will be (1, )
        # print(f"shape of output: {output.shape}")
        res = self.encoder.inverse_transform(output)

        self.write_res(res)

    def write_res(self, result):
        # the result is (1, ) shape ndarray
        res_data = {"Country": result[0]}

        byte_data = json.dumps(res_data)

        # send the length first
        self.request.send(len(byte_data).to_bytes(length=4))
        self.request.send(byte_data.encode())

    def load_feature(self) -> pd.DataFrame:

        blen = self.request.recv(4)
        length = int.from_bytes(blen)

        byte_data: bytes = self.request.recv(length)

        data_dict: dict = json.loads(byte_data.decode())

        return pd.DataFrame([data_dict])

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

# Copyright 2021 Asuhariet Ygvar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applhttp://127.0.0.1:5000/icable law or agreed to in writing, software
# distributed under the Lihttp://127.0.0.1:5000/cense is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

import sys
import onnxruntime
import numpy as np
from flask import Flask
from flask_restful import reqparse, Api, Resource, request
import werkzeug
from PIL import Image

app = Flask(__name__)
api = Api(app)

class Hashing(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        # get image
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        print(args)
        image_file = args['image']

        print(type(image_file))

        # Load ONNX model
        session = onnxruntime.InferenceSession("Model/model.onnx")

        # Load output hash matrix
        seed1 = open("Model/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
        seed1 = np.frombuffer(seed1, dtype=np.float32)
        seed1 = seed1.reshape([96, 128])

        # Preprocess image
        image1 = Image.open(image_file)
        image = image1.convert('RGB')
        image = image.resize([360, 360])
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

        # Run model
        inputs = {session.get_inputs()[0].name: arr}
        outs = session.run(None, inputs)

        # Convert model output to hex hash
        hash_output = seed1.dot(outs[0].flatten())
        hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
        hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

        hash_dict = {
            "Neural Hash": hash_hex
        }
        return hash_dict

api.add_resource(Hashing, "/hashing")

if __name__ == "__main__":
    app.run(debug=True)
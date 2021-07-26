import json
import os
import pdb
import tempfile

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import layers
"""
step 0: pull the latest devel image by: docker pull tensorflow/serving:latest-devel-gpu
step 1: save model to a specific place
step 2 (optional): freeze the model
step 3: run
    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=fashion_model \
        --model_base_path=your/path/to/model
step 4: open another terminal, go to the container, and run predict



extra: after step 1, can use 'saved_model_cli show --dir your/path/to/model --all' to see some specs like below, this can
help with customized arguments

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 224, 224, 3)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1000)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
"""


def my_resnet50():
    inputs = layers.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    outputs = layers.Dense(1000, activation='softmax')(x)
    # outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def save_model():
    model = my_resnet50()
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "1")
    tf.keras.models.save_model(model,
                               model_dir,
                               include_optimizer=False,
                               save_format=None,
                               signatures=None,
                               options=None)
    print("saved model to {}".format(model_dir))


def prediction():
    data = json.dumps({"signature_name": "serving_default", "instances": np.random.rand(1, 224, 224, 3).tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    pdb.set_trace()


prediction()

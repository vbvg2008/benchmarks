import json
import os
import pdb
import tempfile

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

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


def freeze_graph():
    model = my_resnet50()
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="/data/Xiaomeng/frozen_graph/",
                      name="simple_frozen_graph.pb",
                      as_text=False)


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_freeze_graph():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("/data/Xiaomeng/frozen_graph/simple_frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def, inputs=["x:0"], outputs=["Identity:0"], print_graph=True)
    # Get predictions for test images
    test_images = np.random.rand(1, 224, 224, 3).astype('float32')
    frozen_graph_predictions = frozen_func(x=tf.constant(test_images))[0]


def prediction():
    data = json.dumps({"signature_name": "serving_default", "instances": np.random.rand(1, 224, 224, 3).tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    pdb.set_trace()


prediction()
# load_freeze_graph()

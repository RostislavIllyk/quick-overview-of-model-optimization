import onnx

from onnx_tf.backend import prepare


# run under tf version 1.15

output_path = './new_models/ppp.pb'
input_path = './new_models/pruned_tflite.onnx'



onnx_model = onnx.load(input_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(output_path)  # export the model
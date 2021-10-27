import tflite2onnx

tflite_path = './new_models/pruned_tflite.tflite'
onnx_path = './new_models/pruned_tflite.onnx'

tflite2onnx.convert(tflite_path, onnx_path)
import tensorflow as tf

# Convert the model

model_reloaded = tf.keras.models.load_model('base')
converter = tf.lite.TFLiteConverter.from_keras_model(model_reloaded)
tflite_model = converter.convert()
# Save the model.
with open('base_tflite.tflite', 'wb') as f:
  f.write(tflite_model)



# Create float TFLite model.
float_converter = tf.lite.TFLiteConverter.from_keras_model(model_reloaded)
float_tflite_model = float_converter.convert()
# Save the model.
with open('quantization_tflite.tflite', 'wb') as f:
  f.write(float_tflite_model)
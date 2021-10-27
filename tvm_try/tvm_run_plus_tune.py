from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


image_path = '38.jpg'
#image_path = '514.jpg'

# ================  Эти модели загрузились но не запустились ==================
#        tflite_model_file = 'only_16_bit_activations_with_8_bit_weights.tflite'
#        tflite_model_file = 'post_training_dynamic_range_quantization.tflite'
#        tflite_model_file = 'quantized_and_pruned_tflite.tflite'
#        tflite_model_file = 'sparsity_clustered_model_tflite_file.tflite'


# ================  Эти модели загрузились и запустились ==================
#tflite_model_file = 'post_training_float16_quantization_model.tflite'
#tflite_model_file = 'aware_quantized_tflite_model.tflite'
#tflite_model_file = 'actually_quantized_model_for_the_TFLite_backend.tflite'


# ========  Эта модель (простейшая) загрузились, запустились и оптимизировалась
tflite_model_file = 'pruned_tflite.tflite'








tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)




resized_image = Image.open(image_path).resize((180, 180))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("float32")

# Add a dimension to the image so that we have NHWC format layout
image_data = np.expand_dims(image_data, axis=0)

# Preprocess image as described here:
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
print("input", image_data.shape)


# TFLite input tensor name, shape and type
input_tensor = "input_1"
input_shape = (1, 180, 180, 3)
input_dtype = "float32"

# Parse TFLite model and convert it to a Relay module
from tvm import relay, transform

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

# Build the module against to x86 CPU
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)




import tvm
from tvm.contrib import graph_executor as runtime


# Create a runtime executor module
module = runtime.GraphModule(lib["default"](tvm.cpu()))

# Feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# Run
module.run()

# Get output
tvm_output = module.get_output(0).numpy()


print(tvm_output)

score = tvm_output[0][0]

print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)


#===============================================================================
#
#                                      Tuning
#
#===============================================================================

import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)


print('======================================================================')
print()
print()












from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
from tvm.contrib import graph_executor

# Set up some basic parameters for the runner. The runner takes compiled code
# that is generated with a specific set of parameters and measures the
# performance of it. ``number`` specifies the number of different
# configurations that we will test, while ``repeat`` specifies how many
# measurements we will take of each configuration. ``min_repeat_ms`` is a value
# that specifies how long need to run configuration test. If the number of
# repeats falls under this time, it will be increased. This option is necessary
# for accurate tuning on GPUs, and is not required for CPU tuning. Setting this
# value to 0 disables it. The ``timeout`` places an upper limit on how long to
# run training code for each tested configuration.

number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

# Create a simple structure for holding tuning options. We use an XGBoost
# algorithim for guiding the search. For a production job, you will want to set
# the number of trials to be larger than the value of 10 used here. For CPU we
# recommend 1500, for GPU 3000-4000. The number of trials required can depend
# on the particular model and processor, so it's worth spending some time
# evaluating performance across a range of values to find the best balance
# between tuning time and model optimization. Because running tuning is time
# intensive we set number of trials to 10, but do not recommend a value this
# small. The ``early_stopping`` parameter is the minimum number of trails to
# run before a condition that stops the search early can be applied. The
# measure option indicates where trial code will be built, and where it will be
# run. In this case, we're using the ``LocalRunner`` we just created and a
# ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# the tuning data to.

tuning_option = {
    "tuner": "xgb",
    "trials": 10,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "net-autotuning.json",
}



# begin by extracting the taks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)


print('ready to run tune:')

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

print('ready to run apply_history_best:')



with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


print('ready to run :')

dtype = "float32"
module.set_input(input_tensor, tvm.nd.array(image_data))
module.run()

tvm_output = module.get_output(0).numpy()



print(tvm_output)

score = tvm_output[0][0]

print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
) 
    
    
    
import timeit

timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))






# save the graph, lib and params into separate files

new_arc_name = tflite_model_file[:-7]+"__deploy_lib.tar"
lib.export_library(new_arc_name)






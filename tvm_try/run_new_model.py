# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:29:56 2021

@author: rost_
"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import tvm
from tvm.contrib import graph_executor




path_lib = "pruned_tflite__deploy_lib.tar"
image_path = '38.jpg'
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


data = image_data


dev = tvm.cpu()


# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)


module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()


print(out_deploy)

score = out_deploy[0][0]

print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)



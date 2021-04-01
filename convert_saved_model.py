import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# def my_input_fn():
#     # Let's assume a network with 2 input tensors. We generate 3 sets
#     # of dummy input data:
#     input_shapes = [[(1, 16), (2, 16)], # min and max range for 1st input list
#                     [(2, 32), (4, 32)], # min and max range for 2nd list of two tensors
#                     [(4, 32), (8, 32)]] # 3rd input list
#     for shapes in input_shapes:
#         # return a list of input tensors
#         yield [np.zeros(x).astype(np.float32) for x in shapes]


# Conversion Parameters 
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP16)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='./saved_models',
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Optionally, build TensorRT engines before deployment to save time at runtime
# Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
# converter.build(input_fn=my_input_fn)

# Save the model to the disk 
converter.save('./converted_models')
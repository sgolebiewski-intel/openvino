Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `tensorflow-training-openvino
notebook <tensorflow-training-openvino.ipynb>`__, to improve inference
speed. Quantization is performed with `Post-training Quantization with
NNCF <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__.
A custom dataloader and metric will be defined, and accuracy and
performance will be computed for the original IR model and the quantized
model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Preparation <#Preparation>`__

   -  `Imports <#Imports>`__

-  `Post-training Quantization with
   NNCF <#Post-training-Quantization-with-NNCF>`__

   -  `Select inference device <#Select-inference-device>`__

-  `Compare Metrics <#Compare-Metrics>`__
-  `Run Inference on Quantized
   Model <#Run-Inference-on-Quantized-Model>`__
-  `Compare Inference Speed <#Compare-Inference-Speed>`__

Preparation
-----------

`back to top ⬆️ <#Table-of-contents:>`__

The notebook requires that the training notebook has been run and that
the Intermediate Representation (IR) models are created. If the IR
models do not exist, running the next cell will run the training
notebook. This will take a while.

.. code:: ipython3

    import platform
    
    %pip install -q Pillow numpy tqdm nncf "openvino>=2023.1"
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"
    
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version <= '3.8'"
    %pip install -q tf_keras


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import os
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    
    import tensorflow as tf
    
    model_xml = Path("model/flower/flower_ir.xml")
    dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    )
    data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))
    
    if not model_xml.exists():
        print("Executing training notebook. This will take a while...")
        %run tensorflow-training-openvino.ipynb


.. parsed-literal::

    2024-04-10 00:32:53.652685: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-10 00:32:53.687594: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-10 00:32:54.282056: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    3670


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-04-10 00:33:23.320723: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-10 00:33:23.320754: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-04-10 00:33:23.320758: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-04-10 00:33:23.320891: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-04-10 00:33:23.320907: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-04-10 00:33:23.320910: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-04-10 00:33:23.635471: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:33:23.635730: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_28.png


.. parsed-literal::

    2024-04-10 00:33:24.567491: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:33:24.567733: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:33:24.701288: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:33:24.701590: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-04-10 00:33:25.523959: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:33:25.524264: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_33.png


.. parsed-literal::

    Model: "sequential_2"


.. parsed-literal::

    _________________________________________________________________


.. parsed-literal::

     Layer (type)                Output Shape              Param #   


.. parsed-literal::

    =================================================================


.. parsed-literal::

     sequential_1 (Sequential)   (None, 180, 180, 3)       0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056   


.. parsed-literal::

                                                                     


.. parsed-literal::

     outputs (Dense)             (None, 5)                 645       


.. parsed-literal::

                                                                     


.. parsed-literal::

    =================================================================


.. parsed-literal::

    Total params: 3,989,285


.. parsed-literal::

    Trainable params: 3,989,285


.. parsed-literal::

    Non-trainable params: 0


.. parsed-literal::

    _________________________________________________________________


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-04-10 00:33:26.543289: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:33:26.543637: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

     1/92 [..............................] - ETA: 1:31 - loss: 1.6141 - accuracy: 0.1250

.. parsed-literal::

     2/92 [..............................] - ETA: 6s - loss: 1.8616 - accuracy: 0.2031  

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 2.0526 - accuracy: 0.2083

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.9439 - accuracy: 0.2031

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.8691 - accuracy: 0.2188

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.8279 - accuracy: 0.2083

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 1.8086 - accuracy: 0.2143

.. parsed-literal::

     8/92 [=>............................] - ETA: 5s - loss: 1.7869 - accuracy: 0.2266

.. parsed-literal::

     9/92 [=>............................] - ETA: 5s - loss: 1.7722 - accuracy: 0.2292

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.7570 - accuracy: 0.2250

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.7389 - accuracy: 0.2415

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.7262 - accuracy: 0.2370

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.7143 - accuracy: 0.2356

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.7039 - accuracy: 0.2254

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.6918 - accuracy: 0.2417

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.6809 - accuracy: 0.2559

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.6708 - accuracy: 0.2647

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.6617 - accuracy: 0.2656

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.6513 - accuracy: 0.2747

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.6334 - accuracy: 0.2828

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.6266 - accuracy: 0.2842

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.6144 - accuracy: 0.2898

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 1.6076 - accuracy: 0.2948

.. parsed-literal::

    24/92 [======>.......................] - ETA: 4s - loss: 1.5990 - accuracy: 0.2956

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.5907 - accuracy: 0.3025

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.5845 - accuracy: 0.3053

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.5774 - accuracy: 0.3079

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.5678 - accuracy: 0.3114

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 1.5586 - accuracy: 0.3147

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.5479 - accuracy: 0.3219

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.5444 - accuracy: 0.3226

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.5322 - accuracy: 0.3301

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.5251 - accuracy: 0.3362

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.5253 - accuracy: 0.3392

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.5187 - accuracy: 0.3429

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.5125 - accuracy: 0.3455

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.5096 - accuracy: 0.3454

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.5040 - accuracy: 0.3487

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.5012 - accuracy: 0.3478

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.4927 - accuracy: 0.3539

.. parsed-literal::

    41/92 [============>.................] - ETA: 3s - loss: 1.4857 - accuracy: 0.3582

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.4799 - accuracy: 0.3624

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.4782 - accuracy: 0.3641

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.4777 - accuracy: 0.3643

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.4703 - accuracy: 0.3674

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.4691 - accuracy: 0.3648

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.4638 - accuracy: 0.3664

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.4579 - accuracy: 0.3672

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.4525 - accuracy: 0.3693

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.4457 - accuracy: 0.3713

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.4384 - accuracy: 0.3750

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.4332 - accuracy: 0.3732

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.4331 - accuracy: 0.3715

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.4262 - accuracy: 0.3756

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.4207 - accuracy: 0.3778

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.4142 - accuracy: 0.3800

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.4101 - accuracy: 0.3810

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 1.4066 - accuracy: 0.3825

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.3987 - accuracy: 0.3872

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.3927 - accuracy: 0.3896

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.3903 - accuracy: 0.3929

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.3858 - accuracy: 0.3962

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.3845 - accuracy: 0.3938

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.3791 - accuracy: 0.3965

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.3747 - accuracy: 0.3976

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.3694 - accuracy: 0.3991

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.3697 - accuracy: 0.3988

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.3685 - accuracy: 0.4026

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.3695 - accuracy: 0.4031

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.3665 - accuracy: 0.4059

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.3649 - accuracy: 0.4072

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.3668 - accuracy: 0.4072

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.3664 - accuracy: 0.4085

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 1.3664 - accuracy: 0.4081

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.3623 - accuracy: 0.4105

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.3615 - accuracy: 0.4113

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.3598 - accuracy: 0.4125

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.3586 - accuracy: 0.4132

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 1.3611 - accuracy: 0.4131

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 1.3598 - accuracy: 0.4146

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.3589 - accuracy: 0.4145

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.3556 - accuracy: 0.4174

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.3548 - accuracy: 0.4165

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.3518 - accuracy: 0.4179

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.3499 - accuracy: 0.4196

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.3485 - accuracy: 0.4220

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.3447 - accuracy: 0.4236

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.3416 - accuracy: 0.4259

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.3422 - accuracy: 0.4264

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.3392 - accuracy: 0.4272

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.3351 - accuracy: 0.4287

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.3325 - accuracy: 0.4302

.. parsed-literal::

    2024-04-10 00:33:32.870982: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:33:32.871279: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    92/92 [==============================] - 7s 66ms/step - loss: 1.3325 - accuracy: 0.4302 - val_loss: 1.1643 - val_accuracy: 0.5232


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 1.1792 - accuracy: 0.5312

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.0816 - accuracy: 0.5938

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0731 - accuracy: 0.5729

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.0362 - accuracy: 0.5859

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0439 - accuracy: 0.5750

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.0711 - accuracy: 0.5677

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 1.0489 - accuracy: 0.5848

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 1.0452 - accuracy: 0.5898

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.0472 - accuracy: 0.5799

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.0233 - accuracy: 0.5938

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.0209 - accuracy: 0.5909

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.0215 - accuracy: 0.5859

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.0114 - accuracy: 0.5889

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.0510 - accuracy: 0.5804

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.0549 - accuracy: 0.5750

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.0554 - accuracy: 0.5684

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.0459 - accuracy: 0.5735

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.0491 - accuracy: 0.5747

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.0546 - accuracy: 0.5674

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.0494 - accuracy: 0.5734

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.0492 - accuracy: 0.5744

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.0589 - accuracy: 0.5668

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 1.0583 - accuracy: 0.5666

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 1.0627 - accuracy: 0.5638

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.0765 - accuracy: 0.5575

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.0710 - accuracy: 0.5613

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.0702 - accuracy: 0.5602

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.0698 - accuracy: 0.5603

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 1.0672 - accuracy: 0.5603

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.0691 - accuracy: 0.5562

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.0631 - accuracy: 0.5595

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.0602 - accuracy: 0.5635

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.0611 - accuracy: 0.5616

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.0655 - accuracy: 0.5597

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.0594 - accuracy: 0.5607

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.0707 - accuracy: 0.5573

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.0725 - accuracy: 0.5574

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.0702 - accuracy: 0.5584

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.0713 - accuracy: 0.5585

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.0730 - accuracy: 0.5562

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 1.0729 - accuracy: 0.5549

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.0683 - accuracy: 0.5580

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.0683 - accuracy: 0.5603

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.0720 - accuracy: 0.5589

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.0723 - accuracy: 0.5597

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.0737 - accuracy: 0.5605

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.0693 - accuracy: 0.5605

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.0686 - accuracy: 0.5592

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.0700 - accuracy: 0.5619

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.0689 - accuracy: 0.5619

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.0696 - accuracy: 0.5613

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.0668 - accuracy: 0.5625

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.0686 - accuracy: 0.5619

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.0657 - accuracy: 0.5619

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.0622 - accuracy: 0.5648

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.0591 - accuracy: 0.5658

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.0590 - accuracy: 0.5652

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 1.0592 - accuracy: 0.5647

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.0603 - accuracy: 0.5646

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.0577 - accuracy: 0.5661

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.0591 - accuracy: 0.5681

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.0569 - accuracy: 0.5670

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.0540 - accuracy: 0.5689

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.0495 - accuracy: 0.5713

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.0467 - accuracy: 0.5736

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.0475 - accuracy: 0.5724

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.0478 - accuracy: 0.5704

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.0474 - accuracy: 0.5689

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.0476 - accuracy: 0.5684

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.0451 - accuracy: 0.5701

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.0440 - accuracy: 0.5695

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.0449 - accuracy: 0.5694

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.0447 - accuracy: 0.5706

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 1.0484 - accuracy: 0.5697

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.0510 - accuracy: 0.5688

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.0497 - accuracy: 0.5687

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.0481 - accuracy: 0.5694

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.0550 - accuracy: 0.5681

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 1.0520 - accuracy: 0.5704

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.0494 - accuracy: 0.5720

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.0500 - accuracy: 0.5726

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.0497 - accuracy: 0.5736

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.0494 - accuracy: 0.5743

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.0468 - accuracy: 0.5760

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.0479 - accuracy: 0.5751

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.0497 - accuracy: 0.5742

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.0536 - accuracy: 0.5719

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.0545 - accuracy: 0.5715

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.0552 - accuracy: 0.5707

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.0546 - accuracy: 0.5713

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.0553 - accuracy: 0.5708

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 1.0553 - accuracy: 0.5708 - val_loss: 0.9767 - val_accuracy: 0.6226


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.9725 - accuracy: 0.6250

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.0703 - accuracy: 0.5469

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0467 - accuracy: 0.5938

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.9935 - accuracy: 0.6250

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0048 - accuracy: 0.6438

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.9923 - accuracy: 0.6354

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 1.0038 - accuracy: 0.6161

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 1.0189 - accuracy: 0.6094

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.0325 - accuracy: 0.5972

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.0198 - accuracy: 0.6031

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.0432 - accuracy: 0.5994

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.0338 - accuracy: 0.6068

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.0228 - accuracy: 0.6082

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.0278 - accuracy: 0.6027

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.0069 - accuracy: 0.6167

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.9947 - accuracy: 0.6191

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.9942 - accuracy: 0.6140

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.0011 - accuracy: 0.6042

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.9971 - accuracy: 0.6053

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.9901 - accuracy: 0.6078

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.0149 - accuracy: 0.5982

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.0203 - accuracy: 0.6009

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 1.0141 - accuracy: 0.6005

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 1.0110 - accuracy: 0.6016

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.0014 - accuracy: 0.6075

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.9975 - accuracy: 0.6106

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.9927 - accuracy: 0.6123

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.9899 - accuracy: 0.6150

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.9821 - accuracy: 0.6196

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.9786 - accuracy: 0.6250

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.9732 - accuracy: 0.6270

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.9748 - accuracy: 0.6240

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.9782 - accuracy: 0.6231

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.9773 - accuracy: 0.6241

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.9816 - accuracy: 0.6223

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.9778 - accuracy: 0.6233

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.9777 - accuracy: 0.6225

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.9813 - accuracy: 0.6234

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.9762 - accuracy: 0.6274

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.9839 - accuracy: 0.6227

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.9797 - accuracy: 0.6235

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.9788 - accuracy: 0.6243

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.9763 - accuracy: 0.6264

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.9784 - accuracy: 0.6250

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.9733 - accuracy: 0.6264

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.9710 - accuracy: 0.6250

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.9742 - accuracy: 0.6204

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.9736 - accuracy: 0.6212

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.9742 - accuracy: 0.6206

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.9709 - accuracy: 0.6188

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.9695 - accuracy: 0.6184

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.9701 - accuracy: 0.6185

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.9778 - accuracy: 0.6145

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.9768 - accuracy: 0.6153

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.9780 - accuracy: 0.6138

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.9794 - accuracy: 0.6129

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.9805 - accuracy: 0.6142

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.9814 - accuracy: 0.6144

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.9810 - accuracy: 0.6145

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.9813 - accuracy: 0.6142

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.9824 - accuracy: 0.6149

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.9855 - accuracy: 0.6140

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.9896 - accuracy: 0.6118

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.9913 - accuracy: 0.6120

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.9913 - accuracy: 0.6112

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.9910 - accuracy: 0.6119

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.9912 - accuracy: 0.6116

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.9892 - accuracy: 0.6123

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.9873 - accuracy: 0.6138

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.9872 - accuracy: 0.6140

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.9862 - accuracy: 0.6132

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.9849 - accuracy: 0.6134

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.9845 - accuracy: 0.6140

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.9825 - accuracy: 0.6137

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.9823 - accuracy: 0.6139

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.9825 - accuracy: 0.6132

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.9831 - accuracy: 0.6121

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.9793 - accuracy: 0.6135

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.9744 - accuracy: 0.6172

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.9722 - accuracy: 0.6173

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.9698 - accuracy: 0.6166

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.9699 - accuracy: 0.6148

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.9661 - accuracy: 0.6172

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.9652 - accuracy: 0.6176

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.9662 - accuracy: 0.6170

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.9663 - accuracy: 0.6167

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.9675 - accuracy: 0.6161

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.9696 - accuracy: 0.6162

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.9682 - accuracy: 0.6166

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.9674 - accuracy: 0.6164

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.9664 - accuracy: 0.6168

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.9664 - accuracy: 0.6168 - val_loss: 0.9166 - val_accuracy: 0.6253


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8287 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7570 - accuracy: 0.7500

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.8145 - accuracy: 0.7083

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.8705 - accuracy: 0.6875

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8931 - accuracy: 0.6687

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.8819 - accuracy: 0.6667

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.8992 - accuracy: 0.6473

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.8736 - accuracy: 0.6602

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.8777 - accuracy: 0.6562

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.9016 - accuracy: 0.6438

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.9067 - accuracy: 0.6392

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.9290 - accuracy: 0.6354

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.9220 - accuracy: 0.6394

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.9117 - accuracy: 0.6451

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.8987 - accuracy: 0.6542

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.8920 - accuracy: 0.6602

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.8897 - accuracy: 0.6618

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.8839 - accuracy: 0.6597

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.8785 - accuracy: 0.6595

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.8751 - accuracy: 0.6578

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.8871 - accuracy: 0.6548

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.8824 - accuracy: 0.6562

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.8833 - accuracy: 0.6590

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.8861 - accuracy: 0.6641

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.8867 - accuracy: 0.6600

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.8910 - accuracy: 0.6575

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.8889 - accuracy: 0.6574

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.8932 - accuracy: 0.6551

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.8930 - accuracy: 0.6552

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.8902 - accuracy: 0.6552

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.8924 - accuracy: 0.6532

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.8942 - accuracy: 0.6523

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.8878 - accuracy: 0.6553

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.8970 - accuracy: 0.6507

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.8936 - accuracy: 0.6518

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.8935 - accuracy: 0.6510

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.8982 - accuracy: 0.6478

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.8935 - accuracy: 0.6480

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.8897 - accuracy: 0.6514

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.8908 - accuracy: 0.6516

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.8883 - accuracy: 0.6532

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8799 - accuracy: 0.6577

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.8820 - accuracy: 0.6570

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8829 - accuracy: 0.6591

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8811 - accuracy: 0.6583

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8804 - accuracy: 0.6590

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8849 - accuracy: 0.6562

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.8952 - accuracy: 0.6517

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8972 - accuracy: 0.6505

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.8951 - accuracy: 0.6506

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.8942 - accuracy: 0.6513

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.8916 - accuracy: 0.6514

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.8899 - accuracy: 0.6509

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.8901 - accuracy: 0.6518

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.8943 - accuracy: 0.6530

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.8928 - accuracy: 0.6542

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.8901 - accuracy: 0.6564

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.8902 - accuracy: 0.6559

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.8907 - accuracy: 0.6564

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.8929 - accuracy: 0.6553

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8924 - accuracy: 0.6569

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.8893 - accuracy: 0.6589

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.8938 - accuracy: 0.6559

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.8948 - accuracy: 0.6564

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.8966 - accuracy: 0.6549

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.8955 - accuracy: 0.6554

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.8954 - accuracy: 0.6568

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.8937 - accuracy: 0.6573

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.8930 - accuracy: 0.6564

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.8965 - accuracy: 0.6559

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.8981 - accuracy: 0.6555

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.8954 - accuracy: 0.6568

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.8958 - accuracy: 0.6564

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.8936 - accuracy: 0.6580

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.8950 - accuracy: 0.6592

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.8939 - accuracy: 0.6596

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.8920 - accuracy: 0.6592

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8933 - accuracy: 0.6587

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.8925 - accuracy: 0.6591

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8932 - accuracy: 0.6594

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8937 - accuracy: 0.6575

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8954 - accuracy: 0.6556

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8965 - accuracy: 0.6545

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8964 - accuracy: 0.6530

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8936 - accuracy: 0.6542

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8933 - accuracy: 0.6542

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8915 - accuracy: 0.6549

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8899 - accuracy: 0.6553

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8890 - accuracy: 0.6556

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8871 - accuracy: 0.6560

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8881 - accuracy: 0.6557

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.8881 - accuracy: 0.6557 - val_loss: 0.9263 - val_accuracy: 0.6308


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8184 - accuracy: 0.6562

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.9063 - accuracy: 0.6406

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.8869 - accuracy: 0.6354

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.8348 - accuracy: 0.6484

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8013 - accuracy: 0.6750

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.7963 - accuracy: 0.6771

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7836 - accuracy: 0.6830

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7837 - accuracy: 0.6836

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.8137 - accuracy: 0.6701

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.8064 - accuracy: 0.6687

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.8133 - accuracy: 0.6733

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.8215 - accuracy: 0.6693

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.8179 - accuracy: 0.6731

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.8369 - accuracy: 0.6652

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.8347 - accuracy: 0.6646

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.8348 - accuracy: 0.6602

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.8264 - accuracy: 0.6654

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.8251 - accuracy: 0.6684

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.8187 - accuracy: 0.6727

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.8142 - accuracy: 0.6766

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.8129 - accuracy: 0.6771

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.8226 - accuracy: 0.6705

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.8262 - accuracy: 0.6685

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.8450 - accuracy: 0.6562

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.8453 - accuracy: 0.6562

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.8439 - accuracy: 0.6538

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.8383 - accuracy: 0.6586

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.8338 - accuracy: 0.6618

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.8269 - accuracy: 0.6638

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.8265 - accuracy: 0.6625

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.8241 - accuracy: 0.6623

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.8144 - accuracy: 0.6680

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.8172 - accuracy: 0.6667

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.8167 - accuracy: 0.6673

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.8115 - accuracy: 0.6687

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.8081 - accuracy: 0.6701

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.8083 - accuracy: 0.6698

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.8061 - accuracy: 0.6702

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.8045 - accuracy: 0.6707

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.8042 - accuracy: 0.6734

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.8083 - accuracy: 0.6715

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8120 - accuracy: 0.6682

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.8089 - accuracy: 0.6693

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8055 - accuracy: 0.6719

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8099 - accuracy: 0.6701

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8074 - accuracy: 0.6712

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8067 - accuracy: 0.6722

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.8065 - accuracy: 0.6732

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8079 - accuracy: 0.6709

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.8082 - accuracy: 0.6718

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.8129 - accuracy: 0.6703

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.8159 - accuracy: 0.6688

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.8153 - accuracy: 0.6686

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.8146 - accuracy: 0.6701

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.8140 - accuracy: 0.6704

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.8199 - accuracy: 0.6685

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.8196 - accuracy: 0.6699

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.8196 - accuracy: 0.6713

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.8263 - accuracy: 0.6689

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.8254 - accuracy: 0.6698

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8247 - accuracy: 0.6700

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.8274 - accuracy: 0.6693

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.8264 - accuracy: 0.6706

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.8245 - accuracy: 0.6704

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.8232 - accuracy: 0.6711

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.8255 - accuracy: 0.6709

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.8261 - accuracy: 0.6702

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.8287 - accuracy: 0.6700

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.8290 - accuracy: 0.6707

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.8308 - accuracy: 0.6709

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.8284 - accuracy: 0.6725

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.8282 - accuracy: 0.6731

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.8276 - accuracy: 0.6720

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.8260 - accuracy: 0.6739

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.8233 - accuracy: 0.6753

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.8211 - accuracy: 0.6767

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.8223 - accuracy: 0.6768

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8213 - accuracy: 0.6766

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.8183 - accuracy: 0.6779

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8187 - accuracy: 0.6780

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8201 - accuracy: 0.6778

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8214 - accuracy: 0.6771

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8223 - accuracy: 0.6772

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8236 - accuracy: 0.6755

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8241 - accuracy: 0.6760

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8224 - accuracy: 0.6769

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8230 - accuracy: 0.6763

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8210 - accuracy: 0.6771

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8245 - accuracy: 0.6755

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8258 - accuracy: 0.6756

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8239 - accuracy: 0.6761

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.8239 - accuracy: 0.6761 - val_loss: 0.8647 - val_accuracy: 0.6757


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.5892 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6658 - accuracy: 0.7344

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6392 - accuracy: 0.7708

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7720 - accuracy: 0.7188

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8001 - accuracy: 0.7000

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.8401 - accuracy: 0.6719

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.8187 - accuracy: 0.6741

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.8040 - accuracy: 0.6758

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.8197 - accuracy: 0.6736

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.8094 - accuracy: 0.6719

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7908 - accuracy: 0.6790

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7883 - accuracy: 0.6797

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7900 - accuracy: 0.6803

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7832 - accuracy: 0.6853

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7787 - accuracy: 0.6833

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7754 - accuracy: 0.6934

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7729 - accuracy: 0.6985

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7893 - accuracy: 0.6910

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7879 - accuracy: 0.6891

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7876 - accuracy: 0.6891

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7864 - accuracy: 0.6890

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7918 - accuracy: 0.6818

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.7817 - accuracy: 0.6834

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7833 - accuracy: 0.6810

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7786 - accuracy: 0.6825

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7790 - accuracy: 0.6815

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7884 - accuracy: 0.6840

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7950 - accuracy: 0.6797

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7911 - accuracy: 0.6832

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7911 - accuracy: 0.6823

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7882 - accuracy: 0.6835

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7851 - accuracy: 0.6836

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7837 - accuracy: 0.6837

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7825 - accuracy: 0.6847

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7984 - accuracy: 0.6759

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.8015 - accuracy: 0.6753

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7993 - accuracy: 0.6765

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.8027 - accuracy: 0.6752

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.8031 - accuracy: 0.6747

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.8020 - accuracy: 0.6734

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.8031 - accuracy: 0.6730

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8050 - accuracy: 0.6749

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.8056 - accuracy: 0.6751

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8060 - accuracy: 0.6761

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8106 - accuracy: 0.6757

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8109 - accuracy: 0.6773

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8060 - accuracy: 0.6789

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.8084 - accuracy: 0.6764

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8028 - accuracy: 0.6805

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.8027 - accuracy: 0.6825

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.8008 - accuracy: 0.6824

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7964 - accuracy: 0.6848

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7954 - accuracy: 0.6860

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.8001 - accuracy: 0.6826

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7971 - accuracy: 0.6844

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7964 - accuracy: 0.6861

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7955 - accuracy: 0.6872

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7954 - accuracy: 0.6883

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.8002 - accuracy: 0.6862

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.8041 - accuracy: 0.6842

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8010 - accuracy: 0.6867

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7967 - accuracy: 0.6892

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7944 - accuracy: 0.6897

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7927 - accuracy: 0.6902

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7975 - accuracy: 0.6892

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7967 - accuracy: 0.6887

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7941 - accuracy: 0.6896

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7926 - accuracy: 0.6891

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7904 - accuracy: 0.6904

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7904 - accuracy: 0.6908

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7950 - accuracy: 0.6886

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7950 - accuracy: 0.6881

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7929 - accuracy: 0.6907

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7938 - accuracy: 0.6906

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7946 - accuracy: 0.6906

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7939 - accuracy: 0.6906

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7922 - accuracy: 0.6909

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7938 - accuracy: 0.6905

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7963 - accuracy: 0.6904

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7974 - accuracy: 0.6892

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7962 - accuracy: 0.6900

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7941 - accuracy: 0.6911

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7955 - accuracy: 0.6896

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7954 - accuracy: 0.6899

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7954 - accuracy: 0.6913

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7968 - accuracy: 0.6913

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7991 - accuracy: 0.6905

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7983 - accuracy: 0.6905

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7943 - accuracy: 0.6925

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7962 - accuracy: 0.6921

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7966 - accuracy: 0.6918

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7966 - accuracy: 0.6918 - val_loss: 0.8035 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8062 - accuracy: 0.6562

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.8096 - accuracy: 0.6875

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7425 - accuracy: 0.6875

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7706 - accuracy: 0.6953

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7846 - accuracy: 0.6938

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.7628 - accuracy: 0.7083

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7491 - accuracy: 0.7098

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7312 - accuracy: 0.7227

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7203 - accuracy: 0.7326

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7357 - accuracy: 0.7219

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7256 - accuracy: 0.7330

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7355 - accuracy: 0.7240

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7347 - accuracy: 0.7212

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7185 - accuracy: 0.7277

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7256 - accuracy: 0.7250

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7156 - accuracy: 0.7266

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7288 - accuracy: 0.7206

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7458 - accuracy: 0.7153

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7378 - accuracy: 0.7171

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7359 - accuracy: 0.7188

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7315 - accuracy: 0.7188

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7371 - accuracy: 0.7145

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.7304 - accuracy: 0.7174

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7230 - accuracy: 0.7227

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7179 - accuracy: 0.7237

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7072 - accuracy: 0.7308

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6995 - accuracy: 0.7350

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6949 - accuracy: 0.7366

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6929 - accuracy: 0.7349

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6926 - accuracy: 0.7365

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6857 - accuracy: 0.7409

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7009 - accuracy: 0.7334

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7181 - accuracy: 0.7263

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7170 - accuracy: 0.7261

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7180 - accuracy: 0.7250

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7250 - accuracy: 0.7205

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7225 - accuracy: 0.7213

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7259 - accuracy: 0.7196

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7232 - accuracy: 0.7212

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.7268 - accuracy: 0.7211

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7331 - accuracy: 0.7180

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7314 - accuracy: 0.7188

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7308 - accuracy: 0.7180

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7315 - accuracy: 0.7173

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7279 - accuracy: 0.7188

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7299 - accuracy: 0.7174

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7329 - accuracy: 0.7181

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7309 - accuracy: 0.7188

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7341 - accuracy: 0.7168

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7336 - accuracy: 0.7169

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7305 - accuracy: 0.7188

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7355 - accuracy: 0.7157

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7364 - accuracy: 0.7152

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7330 - accuracy: 0.7159

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7283 - accuracy: 0.7176

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7313 - accuracy: 0.7154

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7330 - accuracy: 0.7144

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7347 - accuracy: 0.7138

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7360 - accuracy: 0.7129

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7358 - accuracy: 0.7135

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7358 - accuracy: 0.7136

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7393 - accuracy: 0.7117

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7417 - accuracy: 0.7118

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7416 - accuracy: 0.7114

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7414 - accuracy: 0.7120

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7422 - accuracy: 0.7121

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7395 - accuracy: 0.7136

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7400 - accuracy: 0.7118

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7409 - accuracy: 0.7110

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7397 - accuracy: 0.7116

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7421 - accuracy: 0.7095

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7404 - accuracy: 0.7109

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7437 - accuracy: 0.7089

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7487 - accuracy: 0.7069

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7478 - accuracy: 0.7075

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7467 - accuracy: 0.7089

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7493 - accuracy: 0.7082

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7500 - accuracy: 0.7087

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7522 - accuracy: 0.7077

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7530 - accuracy: 0.7082

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7541 - accuracy: 0.7072

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7561 - accuracy: 0.7066

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7561 - accuracy: 0.7063

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7561 - accuracy: 0.7065

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7561 - accuracy: 0.7074

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7587 - accuracy: 0.7061

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7569 - accuracy: 0.7080

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7558 - accuracy: 0.7085

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7556 - accuracy: 0.7086

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7556 - accuracy: 0.7090

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7567 - accuracy: 0.7088

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.7567 - accuracy: 0.7088 - val_loss: 0.8197 - val_accuracy: 0.6853


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.9049 - accuracy: 0.6250

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7205 - accuracy: 0.7656

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6883 - accuracy: 0.7292

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7258 - accuracy: 0.7344

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7145 - accuracy: 0.7437

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.6980 - accuracy: 0.7500

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6926 - accuracy: 0.7411

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7268 - accuracy: 0.7383

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7052 - accuracy: 0.7396

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6934 - accuracy: 0.7437

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6928 - accuracy: 0.7500

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6999 - accuracy: 0.7422

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7140 - accuracy: 0.7308

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7077 - accuracy: 0.7321

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7134 - accuracy: 0.7271

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7193 - accuracy: 0.7246

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7107 - accuracy: 0.7335

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7114 - accuracy: 0.7292

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7041 - accuracy: 0.7270

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7092 - accuracy: 0.7250

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7051 - accuracy: 0.7292

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7113 - accuracy: 0.7259

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.7288 - accuracy: 0.7174

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7308 - accuracy: 0.7161

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7370 - accuracy: 0.7160

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7310 - accuracy: 0.7196

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7220 - accuracy: 0.7230

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7215 - accuracy: 0.7217

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7234 - accuracy: 0.7227

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7173 - accuracy: 0.7246

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7145 - accuracy: 0.7264

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7162 - accuracy: 0.7242

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7090 - accuracy: 0.7287

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7132 - accuracy: 0.7284

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7163 - accuracy: 0.7299

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7132 - accuracy: 0.7321

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7160 - accuracy: 0.7301

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7099 - accuracy: 0.7331

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7131 - accuracy: 0.7303

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7133 - accuracy: 0.7316

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7117 - accuracy: 0.7328

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7114 - accuracy: 0.7339

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7112 - accuracy: 0.7329

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7091 - accuracy: 0.7339

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7137 - accuracy: 0.7329

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7114 - accuracy: 0.7346

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7093 - accuracy: 0.7356

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7114 - accuracy: 0.7340

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7121 - accuracy: 0.7324

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7092 - accuracy: 0.7328

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7079 - accuracy: 0.7337

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7085 - accuracy: 0.7334

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7120 - accuracy: 0.7308

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7117 - accuracy: 0.7300

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7155 - accuracy: 0.7287

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7195 - accuracy: 0.7258

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7172 - accuracy: 0.7267

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7187 - accuracy: 0.7271

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7186 - accuracy: 0.7265

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7225 - accuracy: 0.7238

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7242 - accuracy: 0.7232

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7237 - accuracy: 0.7236

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7217 - accuracy: 0.7235

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7183 - accuracy: 0.7259

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7153 - accuracy: 0.7277

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7136 - accuracy: 0.7271

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7128 - accuracy: 0.7279

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7108 - accuracy: 0.7282

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7082 - accuracy: 0.7289

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7050 - accuracy: 0.7306

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7038 - accuracy: 0.7313

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7067 - accuracy: 0.7298

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7068 - accuracy: 0.7309

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7052 - accuracy: 0.7316

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7052 - accuracy: 0.7314

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7054 - accuracy: 0.7321

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7053 - accuracy: 0.7315

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7049 - accuracy: 0.7317

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7071 - accuracy: 0.7320

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7066 - accuracy: 0.7318

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7082 - accuracy: 0.7313

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7089 - accuracy: 0.7300

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7077 - accuracy: 0.7306

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7059 - accuracy: 0.7312

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7055 - accuracy: 0.7314

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7052 - accuracy: 0.7320

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7079 - accuracy: 0.7304

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7078 - accuracy: 0.7299

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7070 - accuracy: 0.7302

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7067 - accuracy: 0.7307

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7102 - accuracy: 0.7282

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7102 - accuracy: 0.7282 - val_loss: 0.7665 - val_accuracy: 0.6975


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6429 - accuracy: 0.8125

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6328 - accuracy: 0.7656

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6075 - accuracy: 0.7917

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6311 - accuracy: 0.7891

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6645 - accuracy: 0.7688

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.6880 - accuracy: 0.7604

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 0.6684 - accuracy: 0.7634

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6916 - accuracy: 0.7656

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6996 - accuracy: 0.7639

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7071 - accuracy: 0.7688

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6794 - accuracy: 0.7784

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6579 - accuracy: 0.7839

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6671 - accuracy: 0.7716

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6760 - accuracy: 0.7679

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6857 - accuracy: 0.7646

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6918 - accuracy: 0.7559

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6793 - accuracy: 0.7629

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6737 - accuracy: 0.7622

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6803 - accuracy: 0.7599

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6706 - accuracy: 0.7656

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6698 - accuracy: 0.7619

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6614 - accuracy: 0.7642

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6595 - accuracy: 0.7636

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6612 - accuracy: 0.7578

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6610 - accuracy: 0.7550

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6603 - accuracy: 0.7536

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6530 - accuracy: 0.7581

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6518 - accuracy: 0.7589

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6531 - accuracy: 0.7608

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6507 - accuracy: 0.7604

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6612 - accuracy: 0.7530

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6616 - accuracy: 0.7529

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6596 - accuracy: 0.7528

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6555 - accuracy: 0.7555

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6563 - accuracy: 0.7571

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6626 - accuracy: 0.7535

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6559 - accuracy: 0.7551

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6512 - accuracy: 0.7558

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6511 - accuracy: 0.7572

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.6505 - accuracy: 0.7570

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6500 - accuracy: 0.7576

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6540 - accuracy: 0.7560

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6520 - accuracy: 0.7573

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6500 - accuracy: 0.7570

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6470 - accuracy: 0.7582

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6432 - accuracy: 0.7594

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6406 - accuracy: 0.7605

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6400 - accuracy: 0.7590

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6419 - accuracy: 0.7588

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6457 - accuracy: 0.7574

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6440 - accuracy: 0.7591

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6410 - accuracy: 0.7601

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6392 - accuracy: 0.7605

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6432 - accuracy: 0.7591

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6399 - accuracy: 0.7601

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6451 - accuracy: 0.7577

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6517 - accuracy: 0.7560

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6576 - accuracy: 0.7527

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6590 - accuracy: 0.7516

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6579 - accuracy: 0.7526

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6588 - accuracy: 0.7505

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6574 - accuracy: 0.7510

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6607 - accuracy: 0.7495

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6598 - accuracy: 0.7510

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6594 - accuracy: 0.7510

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6580 - accuracy: 0.7514

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6580 - accuracy: 0.7528

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6599 - accuracy: 0.7518

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6596 - accuracy: 0.7509

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6596 - accuracy: 0.7504

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6632 - accuracy: 0.7478

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6645 - accuracy: 0.7474

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6663 - accuracy: 0.7462

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6681 - accuracy: 0.7454

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6695 - accuracy: 0.7459

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6687 - accuracy: 0.7463

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6707 - accuracy: 0.7456

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6712 - accuracy: 0.7456

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6756 - accuracy: 0.7437

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6806 - accuracy: 0.7426

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6810 - accuracy: 0.7431

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6793 - accuracy: 0.7440

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6795 - accuracy: 0.7429

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6818 - accuracy: 0.7419

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6827 - accuracy: 0.7413

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6849 - accuracy: 0.7403

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6824 - accuracy: 0.7411

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6821 - accuracy: 0.7408

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6821 - accuracy: 0.7416

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6827 - accuracy: 0.7421

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6812 - accuracy: 0.7418

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.6812 - accuracy: 0.7418 - val_loss: 0.7163 - val_accuracy: 0.7003


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4859 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4878 - accuracy: 0.8125

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5885 - accuracy: 0.7708

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5981 - accuracy: 0.7578

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6235 - accuracy: 0.7563

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.6443 - accuracy: 0.7396

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6278 - accuracy: 0.7500

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6351 - accuracy: 0.7500

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6267 - accuracy: 0.7604

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6297 - accuracy: 0.7531

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6419 - accuracy: 0.7415

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6213 - accuracy: 0.7526

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6274 - accuracy: 0.7500

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6199 - accuracy: 0.7522

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6102 - accuracy: 0.7604

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6067 - accuracy: 0.7617

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6044 - accuracy: 0.7629

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6070 - accuracy: 0.7587

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6002 - accuracy: 0.7615

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5924 - accuracy: 0.7656

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5875 - accuracy: 0.7708

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6032 - accuracy: 0.7628

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6057 - accuracy: 0.7649

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6131 - accuracy: 0.7630

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6136 - accuracy: 0.7600

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6136 - accuracy: 0.7608

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6083 - accuracy: 0.7616

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6082 - accuracy: 0.7612

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6114 - accuracy: 0.7586

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6075 - accuracy: 0.7604

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6088 - accuracy: 0.7581

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6130 - accuracy: 0.7578

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6101 - accuracy: 0.7595

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6047 - accuracy: 0.7619

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6082 - accuracy: 0.7607

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6041 - accuracy: 0.7604

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6027 - accuracy: 0.7593

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6009 - accuracy: 0.7590

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5975 - accuracy: 0.7604

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5991 - accuracy: 0.7602

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5939 - accuracy: 0.7637

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5956 - accuracy: 0.7626

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5914 - accuracy: 0.7660

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5930 - accuracy: 0.7649

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5971 - accuracy: 0.7632

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5989 - accuracy: 0.7609

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6007 - accuracy: 0.7593

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6066 - accuracy: 0.7578

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6104 - accuracy: 0.7570

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6138 - accuracy: 0.7556

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6108 - accuracy: 0.7567

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6091 - accuracy: 0.7572

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6072 - accuracy: 0.7571

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6053 - accuracy: 0.7581

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6053 - accuracy: 0.7597

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6105 - accuracy: 0.7584

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6197 - accuracy: 0.7555

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6204 - accuracy: 0.7554

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6280 - accuracy: 0.7516

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6272 - accuracy: 0.7521

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6289 - accuracy: 0.7510

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6289 - accuracy: 0.7500

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6304 - accuracy: 0.7495

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6308 - accuracy: 0.7495

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6309 - accuracy: 0.7490

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6330 - accuracy: 0.7500

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6325 - accuracy: 0.7514

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6330 - accuracy: 0.7514

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6307 - accuracy: 0.7532

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6321 - accuracy: 0.7527

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6323 - accuracy: 0.7531

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6326 - accuracy: 0.7539

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6342 - accuracy: 0.7530

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6354 - accuracy: 0.7521

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6330 - accuracy: 0.7538

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6330 - accuracy: 0.7533

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6333 - accuracy: 0.7524

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6342 - accuracy: 0.7524

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6316 - accuracy: 0.7540

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6328 - accuracy: 0.7527

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6338 - accuracy: 0.7523

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6320 - accuracy: 0.7542

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6315 - accuracy: 0.7545

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6300 - accuracy: 0.7552

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6309 - accuracy: 0.7555

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6281 - accuracy: 0.7565

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6295 - accuracy: 0.7557

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6297 - accuracy: 0.7567

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6309 - accuracy: 0.7559

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6344 - accuracy: 0.7538

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6325 - accuracy: 0.7548

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.6325 - accuracy: 0.7548 - val_loss: 0.7267 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4414 - accuracy: 0.8750

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6197 - accuracy: 0.7969

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5988 - accuracy: 0.7917

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5926 - accuracy: 0.7891

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5962 - accuracy: 0.7750

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.5652 - accuracy: 0.7917

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5500 - accuracy: 0.7946

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5681 - accuracy: 0.7852

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5577 - accuracy: 0.7917

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5664 - accuracy: 0.7812

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5948 - accuracy: 0.7642

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5978 - accuracy: 0.7656

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6077 - accuracy: 0.7620

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6048 - accuracy: 0.7679

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6200 - accuracy: 0.7604

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6529 - accuracy: 0.7500

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6507 - accuracy: 0.7537

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6477 - accuracy: 0.7517

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6368 - accuracy: 0.7549

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6362 - accuracy: 0.7547

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6267 - accuracy: 0.7604

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6224 - accuracy: 0.7614

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.6301 - accuracy: 0.7582

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6345 - accuracy: 0.7578

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6323 - accuracy: 0.7575

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6296 - accuracy: 0.7572

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6253 - accuracy: 0.7604

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6251 - accuracy: 0.7598

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6279 - accuracy: 0.7616

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6253 - accuracy: 0.7622

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6263 - accuracy: 0.7618

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6277 - accuracy: 0.7595

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6326 - accuracy: 0.7583

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6308 - accuracy: 0.7581

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6271 - accuracy: 0.7596

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6211 - accuracy: 0.7636

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6252 - accuracy: 0.7616

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6181 - accuracy: 0.7637

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.6147 - accuracy: 0.7649

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6146 - accuracy: 0.7653

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6089 - accuracy: 0.7672

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6097 - accuracy: 0.7675

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6140 - accuracy: 0.7679

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6121 - accuracy: 0.7675

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6149 - accuracy: 0.7650

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6149 - accuracy: 0.7647

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6115 - accuracy: 0.7657

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6096 - accuracy: 0.7673

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6077 - accuracy: 0.7670

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6121 - accuracy: 0.7642

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6125 - accuracy: 0.7651

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6168 - accuracy: 0.7630

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6179 - accuracy: 0.7628

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6214 - accuracy: 0.7603

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6205 - accuracy: 0.7607

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6223 - accuracy: 0.7605

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6253 - accuracy: 0.7597

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6266 - accuracy: 0.7580

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6234 - accuracy: 0.7594

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6209 - accuracy: 0.7598

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6209 - accuracy: 0.7611

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6186 - accuracy: 0.7625

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6197 - accuracy: 0.7627

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6222 - accuracy: 0.7606

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6238 - accuracy: 0.7600

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6261 - accuracy: 0.7584

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6238 - accuracy: 0.7597

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6225 - accuracy: 0.7605

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6208 - accuracy: 0.7603

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6187 - accuracy: 0.7619

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6160 - accuracy: 0.7631

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6164 - accuracy: 0.7620

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6231 - accuracy: 0.7593

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6237 - accuracy: 0.7592

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6214 - accuracy: 0.7599

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6225 - accuracy: 0.7590

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6230 - accuracy: 0.7584

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6209 - accuracy: 0.7591

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6191 - accuracy: 0.7598

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6199 - accuracy: 0.7593

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6191 - accuracy: 0.7592

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6200 - accuracy: 0.7591

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6207 - accuracy: 0.7582

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6228 - accuracy: 0.7574

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6220 - accuracy: 0.7584

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6236 - accuracy: 0.7586

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6255 - accuracy: 0.7582

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6274 - accuracy: 0.7570

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6267 - accuracy: 0.7577

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6293 - accuracy: 0.7565

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6294 - accuracy: 0.7572

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.6294 - accuracy: 0.7572 - val_loss: 0.7292 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.5121 - accuracy: 0.8125

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.5089 - accuracy: 0.7812

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5442 - accuracy: 0.7604

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5357 - accuracy: 0.7734

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5131 - accuracy: 0.8000

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5069 - accuracy: 0.8125

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5518 - accuracy: 0.7857

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5834 - accuracy: 0.7773

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6041 - accuracy: 0.7674

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6003 - accuracy: 0.7625

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5965 - accuracy: 0.7614

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5923 - accuracy: 0.7708

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6298 - accuracy: 0.7524

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6227 - accuracy: 0.7545

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6301 - accuracy: 0.7542

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6299 - accuracy: 0.7598

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6185 - accuracy: 0.7684

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6180 - accuracy: 0.7691

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6139 - accuracy: 0.7730

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6101 - accuracy: 0.7719

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6172 - accuracy: 0.7679

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7670

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6214 - accuracy: 0.7663

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6174 - accuracy: 0.7656

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6331 - accuracy: 0.7575

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6312 - accuracy: 0.7560

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6307 - accuracy: 0.7535

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6340 - accuracy: 0.7500

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6288 - accuracy: 0.7522

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6223 - accuracy: 0.7563

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6179 - accuracy: 0.7591

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6127 - accuracy: 0.7607

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6108 - accuracy: 0.7614

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6193 - accuracy: 0.7592

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6161 - accuracy: 0.7598

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6112 - accuracy: 0.7613

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6149 - accuracy: 0.7601

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6123 - accuracy: 0.7607

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6060 - accuracy: 0.7628

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.6065 - accuracy: 0.7633

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6145 - accuracy: 0.7622

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6129 - accuracy: 0.7624

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6177 - accuracy: 0.7607

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6185 - accuracy: 0.7598

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6221 - accuracy: 0.7568

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6203 - accuracy: 0.7560

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6221 - accuracy: 0.7546

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6206 - accuracy: 0.7551

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6168 - accuracy: 0.7569

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6145 - accuracy: 0.7574

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6111 - accuracy: 0.7585

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6131 - accuracy: 0.7571

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6106 - accuracy: 0.7587

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6113 - accuracy: 0.7574

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6113 - accuracy: 0.7578

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6100 - accuracy: 0.7577

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6084 - accuracy: 0.7587

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6090 - accuracy: 0.7580

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6075 - accuracy: 0.7599

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6055 - accuracy: 0.7608

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6009 - accuracy: 0.7621

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5991 - accuracy: 0.7634

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5977 - accuracy: 0.7642

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5978 - accuracy: 0.7650

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5960 - accuracy: 0.7652

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5935 - accuracy: 0.7654

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5920 - accuracy: 0.7652

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5888 - accuracy: 0.7664

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5900 - accuracy: 0.7657

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5891 - accuracy: 0.7659

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5911 - accuracy: 0.7657

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5886 - accuracy: 0.7668

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5889 - accuracy: 0.7669

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5892 - accuracy: 0.7667

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5918 - accuracy: 0.7653

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5888 - accuracy: 0.7667

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5879 - accuracy: 0.7665

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5914 - accuracy: 0.7663

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5941 - accuracy: 0.7645

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5929 - accuracy: 0.7659

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5914 - accuracy: 0.7664

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5927 - accuracy: 0.7670

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5940 - accuracy: 0.7668

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5929 - accuracy: 0.7673

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5953 - accuracy: 0.7671

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5958 - accuracy: 0.7669

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5939 - accuracy: 0.7682

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5923 - accuracy: 0.7690

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5928 - accuracy: 0.7685

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5922 - accuracy: 0.7686

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5923 - accuracy: 0.7681

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5923 - accuracy: 0.7681 - val_loss: 0.7234 - val_accuracy: 0.7289


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7009 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6660 - accuracy: 0.8125

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6351 - accuracy: 0.8125

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6258 - accuracy: 0.8047

.. parsed-literal::

     5/92 [>.............................] - ETA: 4s - loss: 0.5979 - accuracy: 0.8062

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.5561 - accuracy: 0.8125

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5242 - accuracy: 0.8170

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5384 - accuracy: 0.8047

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5178 - accuracy: 0.8056

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5115 - accuracy: 0.8062

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5135 - accuracy: 0.8097

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5093 - accuracy: 0.8177

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5014 - accuracy: 0.8245

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5135 - accuracy: 0.8170

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5246 - accuracy: 0.8104

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5239 - accuracy: 0.8125

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5166 - accuracy: 0.8125

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5155 - accuracy: 0.8090

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5174 - accuracy: 0.8059

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5213 - accuracy: 0.8062

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5215 - accuracy: 0.8080

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5301 - accuracy: 0.8026

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5472 - accuracy: 0.7935

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5472 - accuracy: 0.7956

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5460 - accuracy: 0.7937

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5437 - accuracy: 0.7957

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5405 - accuracy: 0.7940

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5368 - accuracy: 0.7969

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5328 - accuracy: 0.7985

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5351 - accuracy: 0.7990

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5376 - accuracy: 0.7964

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5428 - accuracy: 0.7930

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5397 - accuracy: 0.7964

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5346 - accuracy: 0.7996

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5320 - accuracy: 0.8018

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5356 - accuracy: 0.8003

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5352 - accuracy: 0.7990

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5464 - accuracy: 0.7961

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5438 - accuracy: 0.7965

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5459 - accuracy: 0.7953

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5479 - accuracy: 0.7942

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5502 - accuracy: 0.7939

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5479 - accuracy: 0.7958

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5518 - accuracy: 0.7933

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5560 - accuracy: 0.7903

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5599 - accuracy: 0.7880

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5615 - accuracy: 0.7879

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5639 - accuracy: 0.7858

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5593 - accuracy: 0.7883

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5634 - accuracy: 0.7862

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5672 - accuracy: 0.7837

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5652 - accuracy: 0.7849

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5644 - accuracy: 0.7860

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5645 - accuracy: 0.7859

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5633 - accuracy: 0.7869

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5662 - accuracy: 0.7846

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5615 - accuracy: 0.7867

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5597 - accuracy: 0.7883

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5580 - accuracy: 0.7887

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5647 - accuracy: 0.7849

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5637 - accuracy: 0.7848

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5610 - accuracy: 0.7868

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5619 - accuracy: 0.7862

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5635 - accuracy: 0.7856

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5635 - accuracy: 0.7856

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5616 - accuracy: 0.7869

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5638 - accuracy: 0.7854

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5657 - accuracy: 0.7845

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5638 - accuracy: 0.7862

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5619 - accuracy: 0.7871

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5634 - accuracy: 0.7870

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5635 - accuracy: 0.7865

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5638 - accuracy: 0.7851

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5660 - accuracy: 0.7846

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5668 - accuracy: 0.7850

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5704 - accuracy: 0.7837

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5693 - accuracy: 0.7841

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5732 - accuracy: 0.7837

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5729 - accuracy: 0.7836

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5724 - accuracy: 0.7832

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5708 - accuracy: 0.7843

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5692 - accuracy: 0.7851

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5694 - accuracy: 0.7850

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5703 - accuracy: 0.7839

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5684 - accuracy: 0.7835

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5660 - accuracy: 0.7849

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5657 - accuracy: 0.7849

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5660 - accuracy: 0.7849

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5648 - accuracy: 0.7852

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5652 - accuracy: 0.7858

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5657 - accuracy: 0.7858

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5657 - accuracy: 0.7858 - val_loss: 0.7111 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.3127 - accuracy: 0.9062

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4240 - accuracy: 0.8594

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.4431 - accuracy: 0.8438

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.4689 - accuracy: 0.8359

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.4951 - accuracy: 0.8375

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5012 - accuracy: 0.8281

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.4621 - accuracy: 0.8393

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.4918 - accuracy: 0.8281

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.4819 - accuracy: 0.8299

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.4886 - accuracy: 0.8285

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5028 - accuracy: 0.8218

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5097 - accuracy: 0.8260

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5035 - accuracy: 0.8250

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5337 - accuracy: 0.8093

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5283 - accuracy: 0.8075

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5273 - accuracy: 0.8078

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5367 - accuracy: 0.8046

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5309 - accuracy: 0.8100

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5288 - accuracy: 0.8101

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5356 - accuracy: 0.8072

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5370 - accuracy: 0.8046

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5387 - accuracy: 0.8008

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5340 - accuracy: 0.8013

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5306 - accuracy: 0.8043

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5286 - accuracy: 0.8058

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5277 - accuracy: 0.8096

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5236 - accuracy: 0.8119

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5195 - accuracy: 0.8130

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5226 - accuracy: 0.8120

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5202 - accuracy: 0.8140

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5190 - accuracy: 0.8150

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5217 - accuracy: 0.8139

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5191 - accuracy: 0.8139

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5162 - accuracy: 0.8138

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5187 - accuracy: 0.8156

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5217 - accuracy: 0.8121

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5228 - accuracy: 0.8121

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5255 - accuracy: 0.8105

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5237 - accuracy: 0.8113

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5214 - accuracy: 0.8129

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5200 - accuracy: 0.8136

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5169 - accuracy: 0.8151

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5211 - accuracy: 0.8129

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5178 - accuracy: 0.8135

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5224 - accuracy: 0.8128

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5217 - accuracy: 0.8122

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5193 - accuracy: 0.8115

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5174 - accuracy: 0.8128

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5136 - accuracy: 0.8141

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5186 - accuracy: 0.8134

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5150 - accuracy: 0.8152

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5109 - accuracy: 0.8175

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5123 - accuracy: 0.8145

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5110 - accuracy: 0.8151

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5100 - accuracy: 0.8150

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5121 - accuracy: 0.8133

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5145 - accuracy: 0.8117

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5184 - accuracy: 0.8096

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5255 - accuracy: 0.8075

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5249 - accuracy: 0.8071

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5269 - accuracy: 0.8062

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5244 - accuracy: 0.8063

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5266 - accuracy: 0.8049

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5244 - accuracy: 0.8050

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5240 - accuracy: 0.8047

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5245 - accuracy: 0.8048

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5268 - accuracy: 0.8040

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5310 - accuracy: 0.8027

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5304 - accuracy: 0.8029

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5333 - accuracy: 0.8008

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5355 - accuracy: 0.8001

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5360 - accuracy: 0.7994

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5333 - accuracy: 0.8000

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5323 - accuracy: 0.8010

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5311 - accuracy: 0.8012

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5324 - accuracy: 0.8005

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5339 - accuracy: 0.8002

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5322 - accuracy: 0.8004

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5329 - accuracy: 0.7998

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5328 - accuracy: 0.8003

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5344 - accuracy: 0.7993

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5382 - accuracy: 0.7976

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5366 - accuracy: 0.7981

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5376 - accuracy: 0.7972

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5368 - accuracy: 0.7981

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5354 - accuracy: 0.7990

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5368 - accuracy: 0.7988

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5403 - accuracy: 0.7979

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5408 - accuracy: 0.7977

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5422 - accuracy: 0.7972

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5413 - accuracy: 0.7977

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5413 - accuracy: 0.7977 - val_loss: 0.7537 - val_accuracy: 0.7289


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4472 - accuracy: 0.8750

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.5776 - accuracy: 0.7812

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5818 - accuracy: 0.7708

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5331 - accuracy: 0.7891

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5391 - accuracy: 0.8000

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5250 - accuracy: 0.8073

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 0.5363 - accuracy: 0.8125

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5330 - accuracy: 0.8047

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5161 - accuracy: 0.8160

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5484 - accuracy: 0.7969

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5352 - accuracy: 0.8040

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5256 - accuracy: 0.8125

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5419 - accuracy: 0.8101

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5403 - accuracy: 0.8080

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5356 - accuracy: 0.8062

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5163 - accuracy: 0.8164

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5220 - accuracy: 0.8162

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5239 - accuracy: 0.8142

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5285 - accuracy: 0.8125

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5161 - accuracy: 0.8172

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5171 - accuracy: 0.8155

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5114 - accuracy: 0.8139

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.5075 - accuracy: 0.8166

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5163 - accuracy: 0.8151

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5172 - accuracy: 0.8150

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5110 - accuracy: 0.8161

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5100 - accuracy: 0.8148

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5098 - accuracy: 0.8136

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5117 - accuracy: 0.8114

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5120 - accuracy: 0.8115

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5076 - accuracy: 0.8125

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.4992 - accuracy: 0.8154

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.4993 - accuracy: 0.8163

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5049 - accuracy: 0.8153

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5054 - accuracy: 0.8152

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5103 - accuracy: 0.8108

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5127 - accuracy: 0.8074

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5108 - accuracy: 0.8076

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5119 - accuracy: 0.8085

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5068 - accuracy: 0.8109

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5106 - accuracy: 0.8110

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5133 - accuracy: 0.8103

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5155 - accuracy: 0.8089

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5116 - accuracy: 0.8097

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5086 - accuracy: 0.8111

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5061 - accuracy: 0.8118

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5041 - accuracy: 0.8138

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5065 - accuracy: 0.8112

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5041 - accuracy: 0.8112

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5005 - accuracy: 0.8125

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5013 - accuracy: 0.8125

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5059 - accuracy: 0.8107

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5042 - accuracy: 0.8119

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5034 - accuracy: 0.8119

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5031 - accuracy: 0.8119

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5018 - accuracy: 0.8114

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5007 - accuracy: 0.8120

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5019 - accuracy: 0.8109

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5019 - accuracy: 0.8104

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5007 - accuracy: 0.8099

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5018 - accuracy: 0.8094

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5059 - accuracy: 0.8090

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5055 - accuracy: 0.8085

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5088 - accuracy: 0.8076

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5086 - accuracy: 0.8072

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5075 - accuracy: 0.8073

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5034 - accuracy: 0.8092

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5038 - accuracy: 0.8093

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5001 - accuracy: 0.8107

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.4985 - accuracy: 0.8112

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5001 - accuracy: 0.8110

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5026 - accuracy: 0.8101

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5036 - accuracy: 0.8102

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5020 - accuracy: 0.8106

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5033 - accuracy: 0.8102

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5058 - accuracy: 0.8099

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5050 - accuracy: 0.8103

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5072 - accuracy: 0.8087

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5050 - accuracy: 0.8103

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5077 - accuracy: 0.8088

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5061 - accuracy: 0.8096

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5055 - accuracy: 0.8093

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5071 - accuracy: 0.8082

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5069 - accuracy: 0.8079

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5074 - accuracy: 0.8076

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5071 - accuracy: 0.8080

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5067 - accuracy: 0.8080

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5066 - accuracy: 0.8077

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5042 - accuracy: 0.8092

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5036 - accuracy: 0.8092

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5030 - accuracy: 0.8093

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5030 - accuracy: 0.8093 - val_loss: 0.6997 - val_accuracy: 0.7289



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1468.png


.. parsed-literal::

    1/1 [==============================] - ETA: 0s

.. parsed-literal::

    1/1 [==============================] - 0s 77ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.96 percent confidence.


.. parsed-literal::

    2024-04-10 00:34:56.231190: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-10 00:34:56.328135: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.338506: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-10 00:34:56.350087: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.358069: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.365272: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.376675: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.417209: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-04-10 00:34:56.490294: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.512402: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-04-10 00:34:56.553862: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.579259: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:56.654416: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-10 00:34:56.977477: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:57.128242: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:57.165128: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-10 00:34:57.196447: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:34:57.246670: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets



.. parsed-literal::

    output/A_Close_Up_Photo_of_a_Dandelion.jpg:   0%|          | 0.00/21.7k [00:00<?, ?B/s]


.. parsed-literal::

    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 99.76 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1480.png


Imports
~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The Post Training Quantization API is implemented in the ``nncf``
library.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import nncf
    from openvino.runtime import Core
    from openvino.runtime import serialize
    from PIL import Image
    from sklearn.metrics import accuracy_score
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Post-training Quantization with NNCF
------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop.

Create a quantized model from the pre-trained FP32 model and the
calibration dataset. The optimization process contains the following
steps:

1. Create a Dataset for quantization.
2. Run nncf.quantize for getting an optimized model.

The validation dataset already defined in the training notebook.

.. code:: ipython3

    img_height = 180
    img_width = 180
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=1
    )
    
    for a, b in val_dataset:
        print(type(a), type(b))
        break


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    <class 'tensorflow.python.framework.ops.EagerTensor'> <class 'tensorflow.python.framework.ops.EagerTensor'>


.. parsed-literal::

    2024-04-10 00:35:00.588920: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:35:00.589163: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


The validation dataset can be reused in quantization process. But it
returns a tuple (images, labels), whereas calibration_dataset should
only return images. The transformation function helps to transform a
user validation dataset to the calibration dataset.

.. code:: ipython3

    def transform_fn(data_item):
        """
        The transformation function transforms a data item into model input data.
        This function should be passed when the data item cannot be used as model's input.
        """
        images, _ = data_item
        return images.numpy()
    
    
    calibration_dataset = nncf.Dataset(val_dataset, transform_fn)

Download Intermediate Representation (IR) model.

.. code:: ipython3

    core = Core()
    ir_model = core.read_model(model_xml)

Use `Basic Quantization
Flow <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__.
To use the most advanced quantization flow that allows to apply 8-bit
quantization to the model with accuracy control see `Quantizing with
accuracy
control <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html>`__.

.. code:: ipython3

    quantized_model = nncf.quantize(
        ir_model,
        calibration_dataset,
        subset_size=1000
    )



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Exception in thread Thread-88:
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Traceback (most recent call last):
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.run()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/ops/math_ops.py", line 1569, in _truediv_python3
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise TypeError(f"`x` and `y` must have the same dtype, "
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">TypeError: `x` and `y` must have the same dtype, got tf.int64 != tf.float32.
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Save quantized model to benchmark.

.. code:: ipython3

    compressed_model_dir = Path("model/optimized")
    compressed_model_dir.mkdir(parents=True, exist_ok=True)
    compressed_model_xml = compressed_model_dir / "flower_ir.xml"
    serialize(quantized_model, str(compressed_model_xml))

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"] if not "GPU" in core.available_devices else ["AUTO", "MULTY:CPU,GPU"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Compare Metrics
---------------

`back to top ⬆️ <#Table-of-contents:>`__

Define a metric to determine the performance of the model.

For this demo we define validate function to compute accuracy metrics.

.. code:: ipython3

    def validate(model, validation_loader):
        """
        Evaluate model and compute accuracy metrics.
    
        :param model: Model to validate
        :param validation_loader: Validation dataset
        :returns: Accuracy scores
        """
        predictions = []
        references = []
    
        output = model.outputs[0]
    
        for images, target in validation_loader:
            pred = model(images.numpy())[output]
    
            predictions.append(np.argmax(pred, axis=1))
            references.append(target)
    
        predictions = np.concatenate(predictions, axis=0)
        references = np.concatenate(references, axis=0)
    
        scores = accuracy_score(references, predictions)
    
        return scores

Calculate accuracy for the original model and the quantized model.

.. code:: ipython3

    original_compiled_model = core.compile_model(model=ir_model, device_name=device.value)
    quantized_compiled_model = core.compile_model(model=quantized_model, device_name=device.value)
    
    original_accuracy = validate(original_compiled_model, val_dataset)
    quantized_accuracy = validate(quantized_compiled_model, val_dataset)
    
    print(f"Accuracy of the original model: {original_accuracy:.3f}")
    print(f"Accuracy of the quantized model: {quantized_accuracy:.3f}")


.. parsed-literal::

    Accuracy of the original model: 0.729
    Accuracy of the quantized model: 0.741


Compare file size of the models.

.. code:: ipython3

    original_model_size = model_xml.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = compressed_model_xml.with_suffix(".bin").stat().st_size / 1024
    
    print(f"Original model size: {original_model_size:.2f} KB")
    print(f"Quantized model size: {quantized_model_size:.2f} KB")


.. parsed-literal::

    Original model size: 7791.65 KB
    Quantized model size: 3897.08 KB


So, we can see that the original and quantized models have similar
accuracy with a much smaller size of the quantized model.

Run Inference on Quantized Model
--------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Copy the preprocess function from the training notebook and run
inference on the quantized model with Inference Engine. See the
`OpenVINO API tutorial <openvino-api-with-output.html>`__ for more
information about running inference with Inference Engine Python API.

.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)
    
        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
    
        input_image = image.reshape((n, h, w, c))
    
        return input_image

.. code:: ipython3

    # Get the names of the input and output layer
    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)
    
    # Get the class names: a list of directory names in alphabetical order
    class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])
    
    # Run inference on an input image...
    inp_img_url = (
        "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    )
    directory = "output"
    inp_file_name = "A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(directory)/Path(inp_file_name)
    # Download the image if it does not exist yet
    if not Path(inp_file_name).exists():
        download_file(inp_img_url, inp_file_name, directory=directory)
    
    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(imagePath=file_path)
    print(f'input image shape: {input_image.shape}')
    print(f'input layer shape: {input_layer.shape}')
    
    res = quantized_compiled_model([input_image])[output_layer]
    
    score = tf.nn.softmax(res[0])
    
    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    input image shape: (1, 180, 180, 3)
    input layer shape: [1,180,180,3]
    This image most likely belongs to dandelion with a 99.79 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_27_1.png


Compare Inference Speed
-----------------------

`back to top ⬆️ <#Table-of-contents:>`__

Measure inference speed with the `OpenVINO Benchmark
App <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.

Benchmark App is a command line tool that measures raw inference
performance for a specified OpenVINO IR model. Run
``benchmark_app --help`` to see a list of available parameters. By
default, Benchmark App tests the performance of the model specified with
the ``-m`` parameter with asynchronous inference on CPU, for one minute.
Use the ``-d`` parameter to test performance on a different device, for
example an Intel integrated Graphics (iGPU), and ``-t`` to set the
number of seconds to run inference. See the
`documentation <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
for more information.

This tutorial uses a wrapper function from `Notebook
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/utils/notebook_utils.ipynb>`__.
It prints the ``benchmark_app`` command with the chosen parameters.

In the next cells, inference speed will be measured for the original and
quantized model on CPU. If an iGPU is available, inference speed will be
measured for CPU+GPU as well. The number of seconds is set to 15.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    # print the available devices on this system
    print("Device information:")
    
    for ov_device in core.available_devices:
        print(f'{ov_device} - {core.get_property(ov_device, "FULL_DEVICE_NAME")}')


.. parsed-literal::

    Device information:
    CPU - Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


.. code:: ipython3

    # Original model benchmarking
    ! benchmark_app -m $model_xml -d $device.value -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.31 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 125.98 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 4.11 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55992 iterations
    [ INFO ] Duration:         15002.40 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.02 ms
    [ INFO ]    Average:       3.03 ms
    [ INFO ]    Min:           1.82 ms
    [ INFO ]    Max:           13.70 ms
    [ INFO ] Throughput:   3732.20 FPS


.. code:: ipython3

    # Quantized model benchmarking
    ! benchmark_app -m $compressed_model_xml -d $device.value -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.78 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 121.61 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 2.33 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report


.. parsed-literal::

    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178212 iterations
    [ INFO ] Duration:         15001.39 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.56 ms
    [ INFO ]    Max:           6.96 ms
    [ INFO ] Throughput:   11879.70 FPS


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

-  `Preparation <#preparation>`__

   -  `Imports <#imports>`__

-  `Post-training Quantization with
   NNCF <#post-training-quantization-with-nncf>`__

   -  `Select inference device <#select-inference-device>`__

-  `Compare Metrics <#compare-metrics>`__
-  `Run Inference on Quantized
   Model <#run-inference-on-quantized-model>`__
-  `Compare Inference Speed <#compare-inference-speed>`__

Preparation
-----------

`back to top ⬆️ <#table-of-contents>`__

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

    2024-03-27 15:02:25.646956: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-27 15:02:25.682246: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-27 15:02:26.276811: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-03-27 15:02:54.129816: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-27 15:02:54.129850: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-27 15:02:54.129855: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-27 15:02:54.129986: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-27 15:02:54.130003: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-27 15:02:54.130006: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-03-27 15:02:54.442858: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:02:54.443168: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_28.png


.. parsed-literal::

    2024-03-27 15:02:55.418195: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:02:55.418426: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:02:55.557824: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:02:55.558109: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-03-27 15:02:56.251433: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:02:56.251745: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



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

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         




                                                                     


.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       




                                                                     


.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      




                                                                     


.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     




                                                                     


.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0         




                                                                     


.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0         




                                                                     


.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056   




                                                                     


.. parsed-literal::

     outputs (Dense)             (None, 5)                 645       




                                                                     


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

    2024-03-27 15:02:57.236178: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:02:57.236563: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
 1/92 [..............................] - ETA: 1:31 - loss: 1.6281 - accuracy: 0.1875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.7584 - accuracy: 0.2344  

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.7147 - accuracy: 0.2292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.6797 - accuracy: 0.2578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.6677 - accuracy: 0.2562

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.6459 - accuracy: 0.2500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.6290 - accuracy: 0.2455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.6178 - accuracy: 0.2617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.5984 - accuracy: 0.2708

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.6006 - accuracy: 0.2781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.5839 - accuracy: 0.2869

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.5713 - accuracy: 0.2917

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.5516 - accuracy: 0.3053

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.5458 - accuracy: 0.3147

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.5435 - accuracy: 0.3104

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.5367 - accuracy: 0.3086

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.5176 - accuracy: 0.3217

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.5111 - accuracy: 0.3299

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.4979 - accuracy: 0.3388

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.4923 - accuracy: 0.3469

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.4837 - accuracy: 0.3542

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.4843 - accuracy: 0.3537

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.4875 - accuracy: 0.3519

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.4799 - accuracy: 0.3568

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.4705 - accuracy: 0.3613

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.4707 - accuracy: 0.3630

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.4583 - accuracy: 0.3669

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.4506 - accuracy: 0.3728

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.4442 - accuracy: 0.3782

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.4355 - accuracy: 0.3854

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.4294 - accuracy: 0.3901

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.4294 - accuracy: 0.3896

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.4237 - accuracy: 0.3892

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.4207 - accuracy: 0.3906

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.4162 - accuracy: 0.3902

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.4051 - accuracy: 0.3967

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.3996 - accuracy: 0.3978

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.3975 - accuracy: 0.3980

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.3952 - accuracy: 0.4006

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.3889 - accuracy: 0.4039

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.3874 - accuracy: 0.4024

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.3801 - accuracy: 0.4092

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.3737 - accuracy: 0.4128

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.3714 - accuracy: 0.4148

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.3665 - accuracy: 0.4153

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.3664 - accuracy: 0.4144

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.3599 - accuracy: 0.4169

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.3556 - accuracy: 0.4199

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.3484 - accuracy: 0.4216

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.3443 - accuracy: 0.4238

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.3409 - accuracy: 0.4234

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.3329 - accuracy: 0.4297

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.3337 - accuracy: 0.4298

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.3286 - accuracy: 0.4329

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.3215 - accuracy: 0.4381

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.3182 - accuracy: 0.4392

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.3125 - accuracy: 0.4408

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.3112 - accuracy: 0.4423

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.3071 - accuracy: 0.4439

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.3031 - accuracy: 0.4464

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.2996 - accuracy: 0.4447

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.2956 - accuracy: 0.4477

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.2922 - accuracy: 0.4490

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.2913 - accuracy: 0.4488

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.2860 - accuracy: 0.4496

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.2816 - accuracy: 0.4527

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.2777 - accuracy: 0.4543

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.2751 - accuracy: 0.4559

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.2724 - accuracy: 0.4583

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.2677 - accuracy: 0.4607

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.2623 - accuracy: 0.4639

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.2589 - accuracy: 0.4656

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.2603 - accuracy: 0.4661

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.2569 - accuracy: 0.4678

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.2592 - accuracy: 0.4682

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.2615 - accuracy: 0.4658

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.2585 - accuracy: 0.4670

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.2576 - accuracy: 0.4683

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.2548 - accuracy: 0.4694

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.2567 - accuracy: 0.4679

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.2545 - accuracy: 0.4694

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.2548 - accuracy: 0.4690

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.2554 - accuracy: 0.4668

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.2531 - accuracy: 0.4690

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.2524 - accuracy: 0.4712

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.2520 - accuracy: 0.4719

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.2502 - accuracy: 0.4744

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.2495 - accuracy: 0.4754

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.2478 - accuracy: 0.4767

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.2455 - accuracy: 0.4776

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.2425 - accuracy: 0.4792

.. parsed-literal::

    2024-03-27 15:03:03.531087: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:03:03.531360: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 65ms/step - loss: 1.2425 - accuracy: 0.4792 - val_loss: 1.0357 - val_accuracy: 0.5886


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8856 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8903 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9323 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0647 - accuracy: 0.6172

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0238 - accuracy: 0.6313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0408 - accuracy: 0.6094

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0062 - accuracy: 0.6250

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0154 - accuracy: 0.6250

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9873 - accuracy: 0.6354

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0364 - accuracy: 0.6062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0300 - accuracy: 0.6136

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0274 - accuracy: 0.6068

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0281 - accuracy: 0.6106

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0288 - accuracy: 0.6027

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0405 - accuracy: 0.6000

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0330 - accuracy: 0.5996

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0308 - accuracy: 0.6011

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0193 - accuracy: 0.6094

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0267 - accuracy: 0.6020

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0246 - accuracy: 0.6016

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0291 - accuracy: 0.5952

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0209 - accuracy: 0.5980

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.0160 - accuracy: 0.6033

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0234 - accuracy: 0.6042

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0276 - accuracy: 0.6012

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0340 - accuracy: 0.6034

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0352 - accuracy: 0.6076

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0308 - accuracy: 0.6105

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0307 - accuracy: 0.6088

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0301 - accuracy: 0.6083

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0351 - accuracy: 0.6058

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0384 - accuracy: 0.6006

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0385 - accuracy: 0.5975

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0394 - accuracy: 0.5965

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0433 - accuracy: 0.5938

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0428 - accuracy: 0.5946

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0428 - accuracy: 0.5912

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0385 - accuracy: 0.5905

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0386 - accuracy: 0.5913

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 1.0356 - accuracy: 0.5953

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0302 - accuracy: 0.5998

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0289 - accuracy: 0.5997

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0307 - accuracy: 0.5988

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0355 - accuracy: 0.5959

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0325 - accuracy: 0.5972

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0373 - accuracy: 0.5931

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0330 - accuracy: 0.5951

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0300 - accuracy: 0.5977

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0297 - accuracy: 0.5982

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0260 - accuracy: 0.6000

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0279 - accuracy: 0.5993

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0244 - accuracy: 0.6004

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0214 - accuracy: 0.6026

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0203 - accuracy: 0.6024

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0179 - accuracy: 0.6017

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0151 - accuracy: 0.6027

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0139 - accuracy: 0.6014

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0118 - accuracy: 0.6013

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0105 - accuracy: 0.6028

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0087 - accuracy: 0.6026

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0023 - accuracy: 0.6050

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0026 - accuracy: 0.6043

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9982 - accuracy: 0.6062

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0006 - accuracy: 0.6057

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0000 - accuracy: 0.6055

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0026 - accuracy: 0.6039

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0067 - accuracy: 0.6029

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0094 - accuracy: 0.6018

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0102 - accuracy: 0.6017

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0103 - accuracy: 0.6007

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0118 - accuracy: 0.6006

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0118 - accuracy: 0.6001

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0111 - accuracy: 0.5996

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0112 - accuracy: 0.6003

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0115 - accuracy: 0.6011

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0142 - accuracy: 0.6010

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0143 - accuracy: 0.6009

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0140 - accuracy: 0.6004

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0147 - accuracy: 0.5999

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0133 - accuracy: 0.5998

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0132 - accuracy: 0.6005

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0129 - accuracy: 0.6012

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0129 - accuracy: 0.6011

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0104 - accuracy: 0.6021

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0084 - accuracy: 0.6031

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0093 - accuracy: 0.6030

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0091 - accuracy: 0.6040

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0096 - accuracy: 0.6032

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0093 - accuracy: 0.6038

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0100 - accuracy: 0.6030

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0091 - accuracy: 0.6035

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 1.0091 - accuracy: 0.6035 - val_loss: 0.9558 - val_accuracy: 0.6281


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9741 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8332 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8440 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8663 - accuracy: 0.6094

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8776 - accuracy: 0.6313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8749 - accuracy: 0.6406

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8638 - accuracy: 0.6518

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8876 - accuracy: 0.6523

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8924 - accuracy: 0.6354

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8999 - accuracy: 0.6281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8924 - accuracy: 0.6250

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9050 - accuracy: 0.6302

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9262 - accuracy: 0.6250

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9319 - accuracy: 0.6272

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9237 - accuracy: 0.6354

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9174 - accuracy: 0.6367

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9123 - accuracy: 0.6434

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9153 - accuracy: 0.6476

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9247 - accuracy: 0.6398

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9383 - accuracy: 0.6313

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9451 - accuracy: 0.6280

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9434 - accuracy: 0.6335

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9400 - accuracy: 0.6345

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9366 - accuracy: 0.6354

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9386 - accuracy: 0.6350

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9321 - accuracy: 0.6346

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9335 - accuracy: 0.6354

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9378 - accuracy: 0.6362

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9375 - accuracy: 0.6401

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9283 - accuracy: 0.6448

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9302 - accuracy: 0.6452

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9345 - accuracy: 0.6436

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9299 - accuracy: 0.6468

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9323 - accuracy: 0.6452

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9348 - accuracy: 0.6446

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9359 - accuracy: 0.6441

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9364 - accuracy: 0.6419

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9412 - accuracy: 0.6390

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9411 - accuracy: 0.6378

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9336 - accuracy: 0.6406

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9396 - accuracy: 0.6380

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9390 - accuracy: 0.6376

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9364 - accuracy: 0.6410

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9332 - accuracy: 0.6420

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9307 - accuracy: 0.6431

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9243 - accuracy: 0.6457

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9283 - accuracy: 0.6440

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9295 - accuracy: 0.6429

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9337 - accuracy: 0.6413

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9317 - accuracy: 0.6410

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9291 - accuracy: 0.6437

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9263 - accuracy: 0.6457

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9266 - accuracy: 0.6459

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9212 - accuracy: 0.6478

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9166 - accuracy: 0.6502

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9151 - accuracy: 0.6492

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9132 - accuracy: 0.6494

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9132 - accuracy: 0.6484

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9104 - accuracy: 0.6491

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9106 - accuracy: 0.6476

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9150 - accuracy: 0.6468

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9136 - accuracy: 0.6469

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9202 - accuracy: 0.6446

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9207 - accuracy: 0.6433

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9191 - accuracy: 0.6445

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9150 - accuracy: 0.6470

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9145 - accuracy: 0.6467

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9161 - accuracy: 0.6459

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9235 - accuracy: 0.6420

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9214 - accuracy: 0.6440

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9192 - accuracy: 0.6442

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9207 - accuracy: 0.6422

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9216 - accuracy: 0.6432

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9203 - accuracy: 0.6430

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9209 - accuracy: 0.6411

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9213 - accuracy: 0.6409

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9213 - accuracy: 0.6419

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9190 - accuracy: 0.6429

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9195 - accuracy: 0.6426

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9168 - accuracy: 0.6440

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9177 - accuracy: 0.6441

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9153 - accuracy: 0.6446

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9138 - accuracy: 0.6451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9128 - accuracy: 0.6453

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9120 - accuracy: 0.6454

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9117 - accuracy: 0.6463

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9158 - accuracy: 0.6446

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9167 - accuracy: 0.6447

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9158 - accuracy: 0.6455

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9162 - accuracy: 0.6453

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9174 - accuracy: 0.6434

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9174 - accuracy: 0.6434 - val_loss: 0.8940 - val_accuracy: 0.6417


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6430 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7140 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8808 - accuracy: 0.6354

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9407 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9232 - accuracy: 0.6438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9207 - accuracy: 0.6562

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9423 - accuracy: 0.6473

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9553 - accuracy: 0.6367

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9542 - accuracy: 0.6493

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9251 - accuracy: 0.6562

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9324 - accuracy: 0.6506

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9481 - accuracy: 0.6458

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9435 - accuracy: 0.6442

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9349 - accuracy: 0.6384

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9296 - accuracy: 0.6375

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9264 - accuracy: 0.6465

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9243 - accuracy: 0.6452

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9298 - accuracy: 0.6441

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9347 - accuracy: 0.6398

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9365 - accuracy: 0.6375

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9328 - accuracy: 0.6369

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9377 - accuracy: 0.6364

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9321 - accuracy: 0.6413

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9297 - accuracy: 0.6432

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9286 - accuracy: 0.6425

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9248 - accuracy: 0.6418

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9337 - accuracy: 0.6377

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9426 - accuracy: 0.6328

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9340 - accuracy: 0.6358

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9362 - accuracy: 0.6365

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9315 - accuracy: 0.6391

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9255 - accuracy: 0.6406

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9217 - accuracy: 0.6439

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9211 - accuracy: 0.6425

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9169 - accuracy: 0.6473

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9108 - accuracy: 0.6493

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9104 - accuracy: 0.6486

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9115 - accuracy: 0.6456

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9104 - accuracy: 0.6426

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9101 - accuracy: 0.6406

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9096 - accuracy: 0.6410

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9062 - accuracy: 0.6414

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9038 - accuracy: 0.6417

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9039 - accuracy: 0.6413

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8987 - accuracy: 0.6444

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8943 - accuracy: 0.6461

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8931 - accuracy: 0.6463

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8978 - accuracy: 0.6439

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8914 - accuracy: 0.6467

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8893 - accuracy: 0.6463

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8848 - accuracy: 0.6477

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8840 - accuracy: 0.6490

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8822 - accuracy: 0.6492

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8792 - accuracy: 0.6522

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8776 - accuracy: 0.6528

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8756 - accuracy: 0.6546

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8753 - accuracy: 0.6552

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8731 - accuracy: 0.6562

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8677 - accuracy: 0.6600

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8684 - accuracy: 0.6599

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8666 - accuracy: 0.6629

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8645 - accuracy: 0.6648

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8670 - accuracy: 0.6642

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8659 - accuracy: 0.6650

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8659 - accuracy: 0.6644

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8628 - accuracy: 0.6652

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8625 - accuracy: 0.6646

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8636 - accuracy: 0.6641

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8682 - accuracy: 0.6621

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8674 - accuracy: 0.6625

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8668 - accuracy: 0.6624

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8679 - accuracy: 0.6615

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8692 - accuracy: 0.6610

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8729 - accuracy: 0.6605

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8737 - accuracy: 0.6612

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8721 - accuracy: 0.6620

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8787 - accuracy: 0.6587

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8780 - accuracy: 0.6587

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8782 - accuracy: 0.6582

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8783 - accuracy: 0.6586

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8789 - accuracy: 0.6582

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8777 - accuracy: 0.6601

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8770 - accuracy: 0.6608

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8790 - accuracy: 0.6604

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8785 - accuracy: 0.6611

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8791 - accuracy: 0.6610

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8781 - accuracy: 0.6620

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8778 - accuracy: 0.6613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8797 - accuracy: 0.6605

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8796 - accuracy: 0.6608

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8782 - accuracy: 0.6614

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8782 - accuracy: 0.6614 - val_loss: 0.8111 - val_accuracy: 0.6730


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8712 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0088 - accuracy: 0.5625

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9116 - accuracy: 0.5938

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8865 - accuracy: 0.6172

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8920 - accuracy: 0.6125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8827 - accuracy: 0.6146

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8733 - accuracy: 0.6250

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8588 - accuracy: 0.6250

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8446 - accuracy: 0.6319

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8271 - accuracy: 0.6500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8096 - accuracy: 0.6591

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7952 - accuracy: 0.6615

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7982 - accuracy: 0.6659

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8012 - accuracy: 0.6674

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7988 - accuracy: 0.6625

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7948 - accuracy: 0.6680

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7960 - accuracy: 0.6618

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7920 - accuracy: 0.6649

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7854 - accuracy: 0.6678

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7952 - accuracy: 0.6656

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8005 - accuracy: 0.6652

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8057 - accuracy: 0.6634

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8074 - accuracy: 0.6630

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8106 - accuracy: 0.6628

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8049 - accuracy: 0.6662

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8017 - accuracy: 0.6671

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8043 - accuracy: 0.6667

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8037 - accuracy: 0.6674

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8099 - accuracy: 0.6659

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8127 - accuracy: 0.6677

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8178 - accuracy: 0.6643

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8176 - accuracy: 0.6631

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8172 - accuracy: 0.6638

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8128 - accuracy: 0.6664

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8086 - accuracy: 0.6679

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8082 - accuracy: 0.6701

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8150 - accuracy: 0.6681

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8123 - accuracy: 0.6727

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8087 - accuracy: 0.6755

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8078 - accuracy: 0.6758

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8047 - accuracy: 0.6776

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8085 - accuracy: 0.6749

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8088 - accuracy: 0.6773

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8164 - accuracy: 0.6747

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8129 - accuracy: 0.6757

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8148 - accuracy: 0.6753

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8175 - accuracy: 0.6762

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8163 - accuracy: 0.6764

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8110 - accuracy: 0.6786

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8072 - accuracy: 0.6806

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8054 - accuracy: 0.6826

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8031 - accuracy: 0.6833

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8042 - accuracy: 0.6840

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8058 - accuracy: 0.6834

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8078 - accuracy: 0.6835

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8104 - accuracy: 0.6819

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8057 - accuracy: 0.6859

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8013 - accuracy: 0.6875

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8022 - accuracy: 0.6859

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7992 - accuracy: 0.6865

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7994 - accuracy: 0.6880

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8016 - accuracy: 0.6880

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8053 - accuracy: 0.6865

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8066 - accuracy: 0.6870

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8064 - accuracy: 0.6870

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8076 - accuracy: 0.6870

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8078 - accuracy: 0.6870

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8055 - accuracy: 0.6884

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8052 - accuracy: 0.6891

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8067 - accuracy: 0.6882

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8069 - accuracy: 0.6877

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8067 - accuracy: 0.6869

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8070 - accuracy: 0.6869

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8046 - accuracy: 0.6873

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8085 - accuracy: 0.6836

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8084 - accuracy: 0.6849

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8086 - accuracy: 0.6849

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8063 - accuracy: 0.6849

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8050 - accuracy: 0.6850

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8048 - accuracy: 0.6850

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8077 - accuracy: 0.6831

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8107 - accuracy: 0.6816

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8091 - accuracy: 0.6825

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8099 - accuracy: 0.6818

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8117 - accuracy: 0.6826

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8125 - accuracy: 0.6816

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8128 - accuracy: 0.6820

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8118 - accuracy: 0.6817

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8098 - accuracy: 0.6825

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8087 - accuracy: 0.6832

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8100 - accuracy: 0.6826

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8100 - accuracy: 0.6826 - val_loss: 0.8562 - val_accuracy: 0.6771


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6740 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8300 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7399 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7548 - accuracy: 0.7266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7643 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8003 - accuracy: 0.7135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7909 - accuracy: 0.7143

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8270 - accuracy: 0.6914

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8362 - accuracy: 0.6875

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8319 - accuracy: 0.6844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8052 - accuracy: 0.6903

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8045 - accuracy: 0.6875

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8040 - accuracy: 0.6923

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8020 - accuracy: 0.6942

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8096 - accuracy: 0.6875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8079 - accuracy: 0.6855

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8086 - accuracy: 0.6838

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8030 - accuracy: 0.6858

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8104 - accuracy: 0.6826

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8117 - accuracy: 0.6844

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8160 - accuracy: 0.6815

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8043 - accuracy: 0.6827

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8047 - accuracy: 0.6855

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8239 - accuracy: 0.6806

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8275 - accuracy: 0.6820

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8284 - accuracy: 0.6846

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8312 - accuracy: 0.6791

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8393 - accuracy: 0.6750

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8310 - accuracy: 0.6754

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8286 - accuracy: 0.6799

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8274 - accuracy: 0.6801

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8205 - accuracy: 0.6851

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8227 - accuracy: 0.6861

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8234 - accuracy: 0.6871

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8236 - accuracy: 0.6853

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8232 - accuracy: 0.6854

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8229 - accuracy: 0.6838

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8210 - accuracy: 0.6847

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8211 - accuracy: 0.6840

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8222 - accuracy: 0.6817

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8215 - accuracy: 0.6826

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8227 - accuracy: 0.6820

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8207 - accuracy: 0.6821

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8169 - accuracy: 0.6830

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8118 - accuracy: 0.6844

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8098 - accuracy: 0.6865

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8078 - accuracy: 0.6859

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8048 - accuracy: 0.6853

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8006 - accuracy: 0.6878

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8087 - accuracy: 0.6823

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8121 - accuracy: 0.6824

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8103 - accuracy: 0.6836

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8065 - accuracy: 0.6849

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8029 - accuracy: 0.6878

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7978 - accuracy: 0.6911

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7963 - accuracy: 0.6916

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7935 - accuracy: 0.6921

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7941 - accuracy: 0.6926

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7913 - accuracy: 0.6930

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7909 - accuracy: 0.6939

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7912 - accuracy: 0.6943

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7891 - accuracy: 0.6947

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7921 - accuracy: 0.6936

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7904 - accuracy: 0.6945

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7876 - accuracy: 0.6949

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7869 - accuracy: 0.6952

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7887 - accuracy: 0.6942

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7869 - accuracy: 0.6941

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7848 - accuracy: 0.6940

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7824 - accuracy: 0.6952

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7859 - accuracy: 0.6921

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7862 - accuracy: 0.6920

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7840 - accuracy: 0.6928

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7846 - accuracy: 0.6936

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7846 - accuracy: 0.6922

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7841 - accuracy: 0.6922

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7860 - accuracy: 0.6913

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7838 - accuracy: 0.6929

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7817 - accuracy: 0.6932

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7813 - accuracy: 0.6935

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7798 - accuracy: 0.6934

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7803 - accuracy: 0.6930

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7775 - accuracy: 0.6944

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7803 - accuracy: 0.6947

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7793 - accuracy: 0.6950

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7798 - accuracy: 0.6945

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7822 - accuracy: 0.6944

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7810 - accuracy: 0.6951

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7787 - accuracy: 0.6964

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7770 - accuracy: 0.6970

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7786 - accuracy: 0.6955

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7786 - accuracy: 0.6955 - val_loss: 0.7756 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.4867 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5594 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6202 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7364 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7007 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7136 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7170 - accuracy: 0.7277

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7124 - accuracy: 0.7344

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6875 - accuracy: 0.7396

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6749 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6839 - accuracy: 0.7386

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6863 - accuracy: 0.7396

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6681 - accuracy: 0.7476

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6612 - accuracy: 0.7478

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6720 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6726 - accuracy: 0.7441

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6807 - accuracy: 0.7426

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6687 - accuracy: 0.7465

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6684 - accuracy: 0.7484

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6674 - accuracy: 0.7531

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6788 - accuracy: 0.7485

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6756 - accuracy: 0.7514

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6682 - accuracy: 0.7541

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6753 - accuracy: 0.7552

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6846 - accuracy: 0.7487

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6738 - accuracy: 0.7536

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6730 - accuracy: 0.7523

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6643 - accuracy: 0.7567

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6617 - accuracy: 0.7597

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6714 - accuracy: 0.7542

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6733 - accuracy: 0.7500

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6708 - accuracy: 0.7520

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6821 - accuracy: 0.7453

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6884 - accuracy: 0.7426

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6886 - accuracy: 0.7437

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6924 - accuracy: 0.7406

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6915 - accuracy: 0.7392

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6979 - accuracy: 0.7379

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6933 - accuracy: 0.7398

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6927 - accuracy: 0.7408

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6989 - accuracy: 0.7365

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7006 - accuracy: 0.7368

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7003 - accuracy: 0.7379

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7038 - accuracy: 0.7367

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7086 - accuracy: 0.7336

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7102 - accuracy: 0.7340

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7091 - accuracy: 0.7356

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7091 - accuracy: 0.7359

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7127 - accuracy: 0.7337

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7097 - accuracy: 0.7352

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7067 - accuracy: 0.7379

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7067 - accuracy: 0.7382

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7059 - accuracy: 0.7390

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7048 - accuracy: 0.7392

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7026 - accuracy: 0.7382

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7016 - accuracy: 0.7384

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6970 - accuracy: 0.7408

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6989 - accuracy: 0.7399

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7019 - accuracy: 0.7380

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7031 - accuracy: 0.7382

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7034 - accuracy: 0.7373

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7063 - accuracy: 0.7351

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7069 - accuracy: 0.7333

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7049 - accuracy: 0.7350

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7076 - accuracy: 0.7348

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7080 - accuracy: 0.7350

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7074 - accuracy: 0.7352

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7058 - accuracy: 0.7364

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7088 - accuracy: 0.7352

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7103 - accuracy: 0.7350

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7123 - accuracy: 0.7348

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7113 - accuracy: 0.7345

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7135 - accuracy: 0.7331

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7117 - accuracy: 0.7345

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7114 - accuracy: 0.7339

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7110 - accuracy: 0.7345

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7179 - accuracy: 0.7307

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7191 - accuracy: 0.7310

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7209 - accuracy: 0.7308

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7202 - accuracy: 0.7314

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7231 - accuracy: 0.7294

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7221 - accuracy: 0.7296

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7244 - accuracy: 0.7287

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7246 - accuracy: 0.7286

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7237 - accuracy: 0.7285

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7247 - accuracy: 0.7273

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7266 - accuracy: 0.7269

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7276 - accuracy: 0.7261

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7284 - accuracy: 0.7253

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7313 - accuracy: 0.7249

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7346 - accuracy: 0.7245

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7346 - accuracy: 0.7245 - val_loss: 0.7343 - val_accuracy: 0.7030


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7169 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7937 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8234 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7630 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7232 - accuracy: 0.7375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7257 - accuracy: 0.7292

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7488 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7645 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 0.7638 - accuracy: 0.6979

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 5s - loss: 0.7574 - accuracy: 0.7063

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7471 - accuracy: 0.7102

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7577 - accuracy: 0.7083

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7472 - accuracy: 0.7139

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7531 - accuracy: 0.7054

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7413 - accuracy: 0.7104

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7654 - accuracy: 0.6992

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7495 - accuracy: 0.7077

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7436 - accuracy: 0.7153

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7593 - accuracy: 0.7056

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7452 - accuracy: 0.7141

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7354 - accuracy: 0.7188

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7313 - accuracy: 0.7202

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7292 - accuracy: 0.7174

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.7217 - accuracy: 0.7214

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7123 - accuracy: 0.7237

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7054 - accuracy: 0.7260

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7054 - accuracy: 0.7269

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6962 - accuracy: 0.7310

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6956 - accuracy: 0.7306

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7053 - accuracy: 0.7271

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7046 - accuracy: 0.7278

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7048 - accuracy: 0.7266

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7095 - accuracy: 0.7254

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7060 - accuracy: 0.7270

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7076 - accuracy: 0.7277

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7002 - accuracy: 0.7292

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6964 - accuracy: 0.7289

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7044 - accuracy: 0.7278

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7023 - accuracy: 0.7292

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7088 - accuracy: 0.7258

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7118 - accuracy: 0.7264

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7144 - accuracy: 0.7254

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7148 - accuracy: 0.7246

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7158 - accuracy: 0.7244

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7160 - accuracy: 0.7257

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7149 - accuracy: 0.7269

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7117 - accuracy: 0.7287

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7076 - accuracy: 0.7311

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7073 - accuracy: 0.7299

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7036 - accuracy: 0.7321

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7047 - accuracy: 0.7313

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7010 - accuracy: 0.7340

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7029 - accuracy: 0.7320

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7051 - accuracy: 0.7312

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7027 - accuracy: 0.7326

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7012 - accuracy: 0.7329

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6977 - accuracy: 0.7343

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6990 - accuracy: 0.7330

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6950 - accuracy: 0.7343

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6916 - accuracy: 0.7356

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6946 - accuracy: 0.7353

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6923 - accuracy: 0.7371

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6913 - accuracy: 0.7377

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6923 - accuracy: 0.7370

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6933 - accuracy: 0.7362

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6939 - accuracy: 0.7346

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6944 - accuracy: 0.7348

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6955 - accuracy: 0.7341

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6960 - accuracy: 0.7330

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6936 - accuracy: 0.7345

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6945 - accuracy: 0.7343

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6958 - accuracy: 0.7341

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6983 - accuracy: 0.7322

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7004 - accuracy: 0.7308

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7027 - accuracy: 0.7302

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7044 - accuracy: 0.7305

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7032 - accuracy: 0.7303

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7051 - accuracy: 0.7298

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7036 - accuracy: 0.7308

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7024 - accuracy: 0.7310

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7011 - accuracy: 0.7309

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7002 - accuracy: 0.7311

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7021 - accuracy: 0.7299

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7022 - accuracy: 0.7301

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7001 - accuracy: 0.7318

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6993 - accuracy: 0.7320

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6966 - accuracy: 0.7336

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6966 - accuracy: 0.7335

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6986 - accuracy: 0.7333

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6980 - accuracy: 0.7338

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6968 - accuracy: 0.7343

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6968 - accuracy: 0.7343 - val_loss: 0.7260 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5965 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7222 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6660 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6516 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6707 - accuracy: 0.7750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6921 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6900 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7005 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6816 - accuracy: 0.7188

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6636 - accuracy: 0.7281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6557 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6456 - accuracy: 0.7526

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6603 - accuracy: 0.7452

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6485 - accuracy: 0.7478

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6272 - accuracy: 0.7563

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6226 - accuracy: 0.7617

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6223 - accuracy: 0.7610

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6116 - accuracy: 0.7656

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6049 - accuracy: 0.7681

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6077 - accuracy: 0.7688

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6306 - accuracy: 0.7574

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6234 - accuracy: 0.7614

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6139 - accuracy: 0.7663

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6146 - accuracy: 0.7656

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6106 - accuracy: 0.7675

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6207 - accuracy: 0.7656

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6235 - accuracy: 0.7650

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6235 - accuracy: 0.7656

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6232 - accuracy: 0.7672

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6141 - accuracy: 0.7729

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6093 - accuracy: 0.7752

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6097 - accuracy: 0.7744

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6051 - accuracy: 0.7765

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6120 - accuracy: 0.7748

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6200 - accuracy: 0.7684

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6149 - accuracy: 0.7696

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6235 - accuracy: 0.7666

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6206 - accuracy: 0.7677

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6178 - accuracy: 0.7681

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6196 - accuracy: 0.7692

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6237 - accuracy: 0.7695

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6307 - accuracy: 0.7654

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6292 - accuracy: 0.7650

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6359 - accuracy: 0.7647

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6352 - accuracy: 0.7650

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6359 - accuracy: 0.7647

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6417 - accuracy: 0.7624

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6433 - accuracy: 0.7609

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6415 - accuracy: 0.7613

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6479 - accuracy: 0.7586

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6447 - accuracy: 0.7603

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6463 - accuracy: 0.7589

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6489 - accuracy: 0.7564

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6526 - accuracy: 0.7557

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6529 - accuracy: 0.7562

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6543 - accuracy: 0.7544

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6551 - accuracy: 0.7527

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6538 - accuracy: 0.7521

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6554 - accuracy: 0.7521

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6554 - accuracy: 0.7526

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6584 - accuracy: 0.7505

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6569 - accuracy: 0.7520

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6555 - accuracy: 0.7529

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6603 - accuracy: 0.7510

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6613 - accuracy: 0.7495

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6613 - accuracy: 0.7495

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6600 - accuracy: 0.7509

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6600 - accuracy: 0.7505

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6618 - accuracy: 0.7496

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6640 - accuracy: 0.7482

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6612 - accuracy: 0.7500

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6620 - accuracy: 0.7487

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6652 - accuracy: 0.7470

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6630 - accuracy: 0.7487

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6636 - accuracy: 0.7496

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6615 - accuracy: 0.7512

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6630 - accuracy: 0.7508

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6614 - accuracy: 0.7516

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6613 - accuracy: 0.7516

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6607 - accuracy: 0.7531

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6613 - accuracy: 0.7531

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6604 - accuracy: 0.7526

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6598 - accuracy: 0.7522

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6600 - accuracy: 0.7522

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6582 - accuracy: 0.7526

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6579 - accuracy: 0.7518

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6589 - accuracy: 0.7521

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6588 - accuracy: 0.7525

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6603 - accuracy: 0.7528

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6643 - accuracy: 0.7507

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6626 - accuracy: 0.7514

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6626 - accuracy: 0.7514 - val_loss: 0.7079 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5440 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5661 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5792 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6032 - accuracy: 0.7656

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6309 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6053 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5958 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5887 - accuracy: 0.7656

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5754 - accuracy: 0.7812

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5823 - accuracy: 0.7812

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5640 - accuracy: 0.7869

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6135 - accuracy: 0.7708

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6153 - accuracy: 0.7692

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6081 - accuracy: 0.7679

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5941 - accuracy: 0.7750

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5964 - accuracy: 0.7773

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5978 - accuracy: 0.7757

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5907 - accuracy: 0.7795

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5970 - accuracy: 0.7763

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5913 - accuracy: 0.7781

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5807 - accuracy: 0.7842

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5708 - accuracy: 0.7855

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5603 - accuracy: 0.7894

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7826

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5815 - accuracy: 0.7800

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7825

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5802 - accuracy: 0.7801

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5800 - accuracy: 0.7835

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5745 - accuracy: 0.7866

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5703 - accuracy: 0.7865

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5714 - accuracy: 0.7853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5702 - accuracy: 0.7871

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5691 - accuracy: 0.7888

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5741 - accuracy: 0.7858

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5794 - accuracy: 0.7848

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5793 - accuracy: 0.7839

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5822 - accuracy: 0.7829

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5827 - accuracy: 0.7829

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5825 - accuracy: 0.7837

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5824 - accuracy: 0.7836

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5780 - accuracy: 0.7851

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5813 - accuracy: 0.7842

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5853 - accuracy: 0.7827

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5855 - accuracy: 0.7820

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5892 - accuracy: 0.7812

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5917 - accuracy: 0.7792

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5953 - accuracy: 0.7786

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5937 - accuracy: 0.7799

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5921 - accuracy: 0.7800

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5905 - accuracy: 0.7812

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5926 - accuracy: 0.7794

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5942 - accuracy: 0.7788

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5934 - accuracy: 0.7789

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5935 - accuracy: 0.7789

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5940 - accuracy: 0.7790

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5990 - accuracy: 0.7773

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5996 - accuracy: 0.7769

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6030 - accuracy: 0.7764

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6055 - accuracy: 0.7744

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6033 - accuracy: 0.7750

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5999 - accuracy: 0.7766

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5989 - accuracy: 0.7777

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6011 - accuracy: 0.7768

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6025 - accuracy: 0.7754

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6009 - accuracy: 0.7764

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6029 - accuracy: 0.7746

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6048 - accuracy: 0.7729

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6027 - accuracy: 0.7734

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6033 - accuracy: 0.7731

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6024 - accuracy: 0.7728

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6046 - accuracy: 0.7720

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6050 - accuracy: 0.7721

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6056 - accuracy: 0.7727

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6062 - accuracy: 0.7720

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6072 - accuracy: 0.7721

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6067 - accuracy: 0.7722

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6073 - accuracy: 0.7715

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6110 - accuracy: 0.7700

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6103 - accuracy: 0.7706

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6128 - accuracy: 0.7703

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6149 - accuracy: 0.7689

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6146 - accuracy: 0.7687

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6150 - accuracy: 0.7692

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6141 - accuracy: 0.7697

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6176 - accuracy: 0.7684

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6180 - accuracy: 0.7674

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6170 - accuracy: 0.7675

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6186 - accuracy: 0.7662

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6197 - accuracy: 0.7653

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6217 - accuracy: 0.7638

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6218 - accuracy: 0.7640

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6218 - accuracy: 0.7640 - val_loss: 0.7259 - val_accuracy: 0.7016


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6973 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7355 - accuracy: 0.6719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6664 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6047 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5954 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5779 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5624 - accuracy: 0.7812

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5703 - accuracy: 0.7891

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5594 - accuracy: 0.7847

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5413 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5488 - accuracy: 0.7926

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5462 - accuracy: 0.7917

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5529 - accuracy: 0.7861

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5540 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5550 - accuracy: 0.7875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5547 - accuracy: 0.7852

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5479 - accuracy: 0.7868

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5522 - accuracy: 0.7865

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5759 - accuracy: 0.7796

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5766 - accuracy: 0.7781

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5713 - accuracy: 0.7812

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5731 - accuracy: 0.7798

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5676 - accuracy: 0.7853

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5848 - accuracy: 0.7760

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5813 - accuracy: 0.7763

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5765 - accuracy: 0.7776

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5803 - accuracy: 0.7789

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5819 - accuracy: 0.7790

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5818 - accuracy: 0.7759

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5808 - accuracy: 0.7771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5859 - accuracy: 0.7782

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5913 - accuracy: 0.7754

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5957 - accuracy: 0.7718

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5923 - accuracy: 0.7748

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5905 - accuracy: 0.7750

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5893 - accuracy: 0.7743

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5978 - accuracy: 0.7711

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5979 - accuracy: 0.7706

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6004 - accuracy: 0.7676

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6060 - accuracy: 0.7672

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6151 - accuracy: 0.7652

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6231 - accuracy: 0.7619

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6223 - accuracy: 0.7616

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6215 - accuracy: 0.7635

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6171 - accuracy: 0.7653

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6209 - accuracy: 0.7622

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6257 - accuracy: 0.7593

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6326 - accuracy: 0.7559

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6314 - accuracy: 0.7545

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6288 - accuracy: 0.7569

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6291 - accuracy: 0.7574

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6266 - accuracy: 0.7572

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6301 - accuracy: 0.7564

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6285 - accuracy: 0.7574

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6281 - accuracy: 0.7562

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6228 - accuracy: 0.7583

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6211 - accuracy: 0.7592

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6205 - accuracy: 0.7596

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6185 - accuracy: 0.7610

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6161 - accuracy: 0.7613

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6153 - accuracy: 0.7621

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6190 - accuracy: 0.7600

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6186 - accuracy: 0.7598

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6193 - accuracy: 0.7597

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6149 - accuracy: 0.7609

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6131 - accuracy: 0.7612

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6105 - accuracy: 0.7615

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6093 - accuracy: 0.7623

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6065 - accuracy: 0.7634

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6100 - accuracy: 0.7637

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6081 - accuracy: 0.7652

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6092 - accuracy: 0.7642

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6101 - accuracy: 0.7640

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6087 - accuracy: 0.7651

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6068 - accuracy: 0.7665

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6082 - accuracy: 0.7655

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6061 - accuracy: 0.7661

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6080 - accuracy: 0.7651

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6072 - accuracy: 0.7657

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6046 - accuracy: 0.7674

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6060 - accuracy: 0.7664

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6042 - accuracy: 0.7670

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6045 - accuracy: 0.7664

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6030 - accuracy: 0.7673

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6032 - accuracy: 0.7675

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6001 - accuracy: 0.7698

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6006 - accuracy: 0.7692

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5982 - accuracy: 0.7701

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5971 - accuracy: 0.7709

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5942 - accuracy: 0.7724

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5953 - accuracy: 0.7725

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5953 - accuracy: 0.7725 - val_loss: 0.7457 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 6s - loss: 0.7281 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6456 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5425 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5572 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.5569 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5682 - accuracy: 0.7865

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5539 - accuracy: 0.7768

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5789 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5878 - accuracy: 0.7778

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5775 - accuracy: 0.7844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5881 - accuracy: 0.7756

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5704 - accuracy: 0.7812

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5584 - accuracy: 0.7885

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5632 - accuracy: 0.7857

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5582 - accuracy: 0.7875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5599 - accuracy: 0.7910

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5499 - accuracy: 0.7923

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5438 - accuracy: 0.7951

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5484 - accuracy: 0.7961

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5549 - accuracy: 0.7937

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5580 - accuracy: 0.7932

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5612 - accuracy: 0.7898

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5673 - accuracy: 0.7867

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5602 - accuracy: 0.7917

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5575 - accuracy: 0.7925

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5587 - accuracy: 0.7897

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5514 - accuracy: 0.7928

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5519 - accuracy: 0.7935

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5501 - accuracy: 0.7931

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5517 - accuracy: 0.7917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5554 - accuracy: 0.7944

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5519 - accuracy: 0.7959

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5541 - accuracy: 0.7945

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5540 - accuracy: 0.7941

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5505 - accuracy: 0.7946

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5463 - accuracy: 0.7960

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5491 - accuracy: 0.7948

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5500 - accuracy: 0.7928

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5489 - accuracy: 0.7925

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5522 - accuracy: 0.7914

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5513 - accuracy: 0.7927

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5608 - accuracy: 0.7872

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5573 - accuracy: 0.7892

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5574 - accuracy: 0.7891

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5576 - accuracy: 0.7889

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5600 - accuracy: 0.7894

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5589 - accuracy: 0.7912

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5561 - accuracy: 0.7917

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5549 - accuracy: 0.7927

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5565 - accuracy: 0.7931

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5568 - accuracy: 0.7917

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5522 - accuracy: 0.7927

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5494 - accuracy: 0.7936

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5473 - accuracy: 0.7940

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5486 - accuracy: 0.7926

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5463 - accuracy: 0.7935

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5444 - accuracy: 0.7939

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5485 - accuracy: 0.7909

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5482 - accuracy: 0.7908

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5483 - accuracy: 0.7917

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5477 - accuracy: 0.7920

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5484 - accuracy: 0.7908

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5496 - accuracy: 0.7902

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5492 - accuracy: 0.7905

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5522 - accuracy: 0.7904

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5537 - accuracy: 0.7888

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5510 - accuracy: 0.7906

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5511 - accuracy: 0.7904

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5555 - accuracy: 0.7889

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5545 - accuracy: 0.7888

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5555 - accuracy: 0.7887

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5556 - accuracy: 0.7882

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5566 - accuracy: 0.7881

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5535 - accuracy: 0.7897

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5594 - accuracy: 0.7879

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5601 - accuracy: 0.7874

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5617 - accuracy: 0.7857

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5602 - accuracy: 0.7865

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5635 - accuracy: 0.7856

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5640 - accuracy: 0.7852

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5694 - accuracy: 0.7820

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5706 - accuracy: 0.7820

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5700 - accuracy: 0.7828

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5705 - accuracy: 0.7827

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5726 - accuracy: 0.7820

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5749 - accuracy: 0.7805

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5750 - accuracy: 0.7792

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5745 - accuracy: 0.7789

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5737 - accuracy: 0.7786

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5719 - accuracy: 0.7796

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5731 - accuracy: 0.7786

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5731 - accuracy: 0.7786 - val_loss: 0.7397 - val_accuracy: 0.7371


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5347 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5655 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5373 - accuracy: 0.8021

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5310 - accuracy: 0.7969

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5502 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5126 - accuracy: 0.8073

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.5603 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5611 - accuracy: 0.7930

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5338 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5360 - accuracy: 0.8031

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5238 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5411 - accuracy: 0.7995

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5450 - accuracy: 0.7957

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5465 - accuracy: 0.7924

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5376 - accuracy: 0.8000

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5325 - accuracy: 0.8047

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5312 - accuracy: 0.8033

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5385 - accuracy: 0.8038

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5304 - accuracy: 0.8076

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5327 - accuracy: 0.8078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5345 - accuracy: 0.8051

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5326 - accuracy: 0.8026

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5340 - accuracy: 0.8016

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5494 - accuracy: 0.7995

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5514 - accuracy: 0.7987

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5607 - accuracy: 0.7957

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5664 - accuracy: 0.7928

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5660 - accuracy: 0.7946

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5647 - accuracy: 0.7963

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5620 - accuracy: 0.8000

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5658 - accuracy: 0.7994

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5685 - accuracy: 0.7969

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5605 - accuracy: 0.8002

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5687 - accuracy: 0.7960

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5645 - accuracy: 0.7982

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5661 - accuracy: 0.7960

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5624 - accuracy: 0.7965

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5622 - accuracy: 0.7977

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5629 - accuracy: 0.7973

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5618 - accuracy: 0.7992

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5606 - accuracy: 0.8003

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5668 - accuracy: 0.7976

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5680 - accuracy: 0.7965

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5631 - accuracy: 0.7990

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5608 - accuracy: 0.8000

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5592 - accuracy: 0.8016

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5590 - accuracy: 0.8025

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5556 - accuracy: 0.8034

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5582 - accuracy: 0.8017

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5548 - accuracy: 0.8031

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5593 - accuracy: 0.8015

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5612 - accuracy: 0.8011

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5642 - accuracy: 0.8001

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5670 - accuracy: 0.7969

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5694 - accuracy: 0.7955

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5638 - accuracy: 0.7980

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5642 - accuracy: 0.7977

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5610 - accuracy: 0.7985

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5620 - accuracy: 0.7977

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5611 - accuracy: 0.7979

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5576 - accuracy: 0.7987

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5568 - accuracy: 0.7999

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5615 - accuracy: 0.7996

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5642 - accuracy: 0.7979

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5647 - accuracy: 0.7971

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5625 - accuracy: 0.7983

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5644 - accuracy: 0.7976

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5624 - accuracy: 0.7978

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5618 - accuracy: 0.7976

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5587 - accuracy: 0.7977

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5589 - accuracy: 0.7962

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5633 - accuracy: 0.7947

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5627 - accuracy: 0.7936

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5605 - accuracy: 0.7952

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5596 - accuracy: 0.7954

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5614 - accuracy: 0.7932

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5595 - accuracy: 0.7930

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5593 - accuracy: 0.7929

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5611 - accuracy: 0.7911

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5609 - accuracy: 0.7914

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5592 - accuracy: 0.7920

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5587 - accuracy: 0.7923

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5576 - accuracy: 0.7925

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5572 - accuracy: 0.7931

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5571 - accuracy: 0.7937

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5574 - accuracy: 0.7943

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5583 - accuracy: 0.7934

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5595 - accuracy: 0.7933

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5574 - accuracy: 0.7939

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5574 - accuracy: 0.7941

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5571 - accuracy: 0.7939

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5571 - accuracy: 0.7939 - val_loss: 0.6982 - val_accuracy: 0.7507


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.4412 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4557 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4736 - accuracy: 0.8021

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4721 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4684 - accuracy: 0.8062

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4830 - accuracy: 0.8073

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4970 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4801 - accuracy: 0.8125

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5058 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5191 - accuracy: 0.8062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5255 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5227 - accuracy: 0.8047

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5193 - accuracy: 0.8077

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5050 - accuracy: 0.8147

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5090 - accuracy: 0.8146

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5005 - accuracy: 0.8164

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5030 - accuracy: 0.8180

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5127 - accuracy: 0.8108

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5088 - accuracy: 0.8141

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4953 - accuracy: 0.8219

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4894 - accuracy: 0.8229

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4922 - accuracy: 0.8182

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4894 - accuracy: 0.8207

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4847 - accuracy: 0.8216

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4822 - accuracy: 0.8213

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4931 - accuracy: 0.8161

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4913 - accuracy: 0.8171

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4947 - accuracy: 0.8125

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5014 - accuracy: 0.8093

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4990 - accuracy: 0.8125

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5032 - accuracy: 0.8095

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5005 - accuracy: 0.8105

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5012 - accuracy: 0.8097

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4996 - accuracy: 0.8107

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4970 - accuracy: 0.8116

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4958 - accuracy: 0.8125

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4958 - accuracy: 0.8133

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5026 - accuracy: 0.8117

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5010 - accuracy: 0.8125

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.4993 - accuracy: 0.8125

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5002 - accuracy: 0.8102

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5031 - accuracy: 0.8088

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5004 - accuracy: 0.8096

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5043 - accuracy: 0.8082

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5030 - accuracy: 0.8083

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5005 - accuracy: 0.8084

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4992 - accuracy: 0.8085

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5090 - accuracy: 0.8066

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5108 - accuracy: 0.8061

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5186 - accuracy: 0.8012

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5199 - accuracy: 0.8021

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5218 - accuracy: 0.8011

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5235 - accuracy: 0.8001

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5229 - accuracy: 0.7998

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5258 - accuracy: 0.8000

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5235 - accuracy: 0.8008

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5246 - accuracy: 0.8010

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5243 - accuracy: 0.8028

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5248 - accuracy: 0.8024

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5221 - accuracy: 0.8042

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5240 - accuracy: 0.8017

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5278 - accuracy: 0.7999

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5256 - accuracy: 0.8016

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5278 - accuracy: 0.8008

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5275 - accuracy: 0.8019

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5277 - accuracy: 0.8021

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5319 - accuracy: 0.7999

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5304 - accuracy: 0.8001

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5319 - accuracy: 0.7980

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5342 - accuracy: 0.7978

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5341 - accuracy: 0.7975

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5321 - accuracy: 0.7986

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5308 - accuracy: 0.7992

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5290 - accuracy: 0.7994

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5293 - accuracy: 0.7987

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5328 - accuracy: 0.7985

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5312 - accuracy: 0.7987

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5312 - accuracy: 0.7989

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5300 - accuracy: 0.7998

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5288 - accuracy: 0.8000

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5280 - accuracy: 0.8002

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5294 - accuracy: 0.7980

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5272 - accuracy: 0.7986

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5247 - accuracy: 0.8005

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5257 - accuracy: 0.7996

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5230 - accuracy: 0.8008

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5243 - accuracy: 0.8002

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5227 - accuracy: 0.8004

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5215 - accuracy: 0.8008

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5201 - accuracy: 0.8013

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5184 - accuracy: 0.8025

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5184 - accuracy: 0.8025 - val_loss: 0.7446 - val_accuracy: 0.7316


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.2660 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4481 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5043 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5654 - accuracy: 0.8125

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.5641 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5965 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5880 - accuracy: 0.7946

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5866 - accuracy: 0.7891

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5590 - accuracy: 0.7986

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5527 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5583 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5488 - accuracy: 0.7969

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5534 - accuracy: 0.7909

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5381 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5332 - accuracy: 0.7979

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5351 - accuracy: 0.7929

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5463 - accuracy: 0.7905

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5453 - accuracy: 0.7933

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5377 - accuracy: 0.7959

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5325 - accuracy: 0.7982

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5348 - accuracy: 0.7989

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5293 - accuracy: 0.8022

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5359 - accuracy: 0.7974

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5275 - accuracy: 0.8018

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5224 - accuracy: 0.8022

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5186 - accuracy: 0.8049

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5181 - accuracy: 0.8074

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5219 - accuracy: 0.8054

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5256 - accuracy: 0.8046

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5294 - accuracy: 0.8049

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5211 - accuracy: 0.8081

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5310 - accuracy: 0.8044

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5325 - accuracy: 0.8046

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5371 - accuracy: 0.8022

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5349 - accuracy: 0.8024

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5429 - accuracy: 0.7976

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5461 - accuracy: 0.7955

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5418 - accuracy: 0.7984

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5376 - accuracy: 0.7995

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5367 - accuracy: 0.7983

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5369 - accuracy: 0.7994

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5366 - accuracy: 0.7982

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5313 - accuracy: 0.8000

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5290 - accuracy: 0.8017

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5273 - accuracy: 0.8019

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5243 - accuracy: 0.8035

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5229 - accuracy: 0.8037

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5209 - accuracy: 0.8051

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5221 - accuracy: 0.8053

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5181 - accuracy: 0.8067

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5162 - accuracy: 0.8086

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5192 - accuracy: 0.8069

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5174 - accuracy: 0.8064

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5166 - accuracy: 0.8054

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5170 - accuracy: 0.8049

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5166 - accuracy: 0.8040

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5160 - accuracy: 0.8041

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5175 - accuracy: 0.8037

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5164 - accuracy: 0.8044

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5128 - accuracy: 0.8071

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5114 - accuracy: 0.8077

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5102 - accuracy: 0.8078

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5079 - accuracy: 0.8103

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5057 - accuracy: 0.8113

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5078 - accuracy: 0.8108

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5136 - accuracy: 0.8090

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5133 - accuracy: 0.8095

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5147 - accuracy: 0.8082

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5185 - accuracy: 0.8065

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5181 - accuracy: 0.8065

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5155 - accuracy: 0.8075

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5137 - accuracy: 0.8084

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5118 - accuracy: 0.8097

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5153 - accuracy: 0.8081

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5148 - accuracy: 0.8090

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5169 - accuracy: 0.8078

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5187 - accuracy: 0.8067

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5183 - accuracy: 0.8067

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5214 - accuracy: 0.8049

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5226 - accuracy: 0.8046

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5210 - accuracy: 0.8050

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5199 - accuracy: 0.8048

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5192 - accuracy: 0.8049

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5185 - accuracy: 0.8053

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5189 - accuracy: 0.8054

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5184 - accuracy: 0.8055

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5173 - accuracy: 0.8052

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5161 - accuracy: 0.8060

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5143 - accuracy: 0.8064

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5137 - accuracy: 0.8061

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5146 - accuracy: 0.8062

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5146 - accuracy: 0.8062 - val_loss: 0.6658 - val_accuracy: 0.7561



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1467.png


.. parsed-literal::

    
1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 77ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 98.82 percent confidence.


.. parsed-literal::

    2024-03-27 15:04:26.754852: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-27 15:04:26.854837: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:26.865352: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-27 15:04:26.877087: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:26.884479: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:26.891678: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:26.903170: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:26.944274: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-27 15:04:27.016961: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.039269: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-27 15:04:27.082184: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.108178: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.184170: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-27 15:04:27.503824: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.654382: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.690808: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-27 15:04:27.722051: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:04:27.772290: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 99.92 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1479.png


Imports
~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

The Post Training Quantization API is implemented in the ``nncf``
library.

.. code:: ipython3

    import sys
    
    import matplotlib.pyplot as plt
    import numpy as np
    import nncf
    from openvino.runtime import Core
    from openvino.runtime import serialize
    from PIL import Image
    from sklearn.metrics import accuracy_score
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Post-training Quantization with NNCF
------------------------------------

`back to top ⬆️ <#table-of-contents>`__

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

    2024-03-27 15:04:30.609228: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:04:30.609486: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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

`back to top ⬆️ <#table-of-contents>`__

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

`back to top ⬆️ <#table-of-contents>`__

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

    Accuracy of the original model: 0.756
    Accuracy of the quantized model: 0.760


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

`back to top ⬆️ <#table-of-contents>`__

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


.. parsed-literal::

    This image most likely belongs to dandelion with a 99.93 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_27_2.png


Compare Inference Speed
-----------------------

`back to top ⬆️ <#table-of-contents>`__

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
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/master/notebooks/utils/notebook_utils.ipynb>`__.
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
    [ INFO ] Read model took 4.24 ms
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

    [ INFO ] Compile model took 120.76 ms
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
    [ INFO ] First inference took 3.83 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55740 iterations
    [ INFO ] Duration:         15003.13 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.04 ms
    [ INFO ]    Average:       3.04 ms
    [ INFO ]    Min:           2.29 ms
    [ INFO ]    Max:           12.25 ms
    [ INFO ] Throughput:   3715.22 FPS


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


.. parsed-literal::

    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.80 ms
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

    [ INFO ] Compile model took 115.73 ms
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
    [ INFO ] First inference took 2.13 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178620 iterations
    [ INFO ] Duration:         15001.84 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.59 ms
    [ INFO ]    Max:           6.77 ms
    [ INFO ] Throughput:   11906.54 FPS


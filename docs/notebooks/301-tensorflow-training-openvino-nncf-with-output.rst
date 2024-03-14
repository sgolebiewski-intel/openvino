Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `301-tensorflow-training-openvino
notebook <301-tensorflow-training-openvino-with-output.html>`__, to improve
inference speed. Quantization is performed with `Post-training
Quantization with
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
    
    %pip install -q tensorflow Pillow numpy tqdm nncf
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 4.25.3 which is incompatible.
    tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 4.25.3 which is incompatible.
    tf2onnx 1.16.1 requires protobuf~=3.20, but you have protobuf 4.25.3 which is incompatible.
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    
    import tensorflow as tf
    
    model_xml = Path("model/flower/flower_ir.xml")
    dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    )
    data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))
    
    if not model_xml.exists():
        print("Executing training notebook. This will take a while...")
        %run 301-tensorflow-training-openvino.ipynb


.. parsed-literal::

    2024-03-14 01:02:28.966516: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-14 01:02:29.001573: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-14 01:02:29.586570: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


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

    2024-03-14 01:02:35.714124: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-14 01:02:35.714160: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-14 01:02:35.714165: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-14 01:02:35.714296: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-14 01:02:35.714313: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-14 01:02:35.714317: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-03-14 01:02:36.053140: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-14 01:02:36.053525: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_12.png


.. parsed-literal::

    2024-03-14 01:02:37.026725: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:02:37.026973: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-14 01:02:37.204867: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-14 01:02:37.205518: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-03-14 01:02:37.886390: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:02:37.886698: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_17.png


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

    2024-03-14 01:02:38.903609: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:02:38.904000: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
 1/92 [..............................] - ETA: 1:25 - loss: 1.6209 - accuracy: 0.1875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.8485 - accuracy: 0.2500  

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.0158 - accuracy: 0.2188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.9726 - accuracy: 0.1875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.9061 - accuracy: 0.1937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.8514 - accuracy: 0.2240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.8129 - accuracy: 0.2545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.7851 - accuracy: 0.2461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 1.7553 - accuracy: 0.2569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.7365 - accuracy: 0.2531

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.7248 - accuracy: 0.2500

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.7025 - accuracy: 0.2500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.6808 - accuracy: 0.2500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.6663 - accuracy: 0.2589

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.6494 - accuracy: 0.2750

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.6309 - accuracy: 0.2910

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.6224 - accuracy: 0.2960

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.6079 - accuracy: 0.3038

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.5889 - accuracy: 0.3125

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.5663 - accuracy: 0.3203

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.5541 - accuracy: 0.3274

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.5433 - accuracy: 0.3338

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.5293 - accuracy: 0.3438

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.5135 - accuracy: 0.3516

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.5037 - accuracy: 0.3537

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.5061 - accuracy: 0.3534

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.4957 - accuracy: 0.3519

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.4967 - accuracy: 0.3516

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.4941 - accuracy: 0.3491

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.4859 - accuracy: 0.3490

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.4793 - accuracy: 0.3528

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.4793 - accuracy: 0.3496

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.4795 - accuracy: 0.3456

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.4738 - accuracy: 0.3502

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.4683 - accuracy: 0.3527

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.4660 - accuracy: 0.3533

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.4596 - accuracy: 0.3547

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.4579 - accuracy: 0.3528

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.4535 - accuracy: 0.3558

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.4504 - accuracy: 0.3570

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.4479 - accuracy: 0.3651

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.4439 - accuracy: 0.3661

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.4385 - accuracy: 0.3677

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.4330 - accuracy: 0.3679

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.4269 - accuracy: 0.3701

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.4243 - accuracy: 0.3736

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.4283 - accuracy: 0.3777

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.4251 - accuracy: 0.3802

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.4195 - accuracy: 0.3833

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.4176 - accuracy: 0.3856

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.4157 - accuracy: 0.3848

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.4107 - accuracy: 0.3870

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.4023 - accuracy: 0.3927

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.4001 - accuracy: 0.3935

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.3933 - accuracy: 0.3960

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.3903 - accuracy: 0.3973

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.3865 - accuracy: 0.3953

.. parsed-literal::

    
58/92 [=================>............] - ETA: 2s - loss: 1.3884 - accuracy: 0.3944

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.3818 - accuracy: 0.3962

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.3762 - accuracy: 0.3979

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.3718 - accuracy: 0.3981

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.3699 - accuracy: 0.3987

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.3669 - accuracy: 0.4013

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.3643 - accuracy: 0.4048

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.3591 - accuracy: 0.4087

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.3594 - accuracy: 0.4100

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.3587 - accuracy: 0.4109

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.3566 - accuracy: 0.4108

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.3520 - accuracy: 0.4139

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.3478 - accuracy: 0.4147

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.3477 - accuracy: 0.4151

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.3467 - accuracy: 0.4158

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.3419 - accuracy: 0.4199

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.3381 - accuracy: 0.4210

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3371 - accuracy: 0.4217

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3349 - accuracy: 0.4231

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3323 - accuracy: 0.4255

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.3296 - accuracy: 0.4268

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.3250 - accuracy: 0.4317

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.3248 - accuracy: 0.4318

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.3248 - accuracy: 0.4315

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.3195 - accuracy: 0.4327

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.3181 - accuracy: 0.4335

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3153 - accuracy: 0.4358

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3125 - accuracy: 0.4388

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3105 - accuracy: 0.4388

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3072 - accuracy: 0.4409

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3024 - accuracy: 0.4448

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3031 - accuracy: 0.4447

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.2991 - accuracy: 0.4467

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.2962 - accuracy: 0.4483

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.2932 - accuracy: 0.4503

.. parsed-literal::

    2024-03-14 01:02:45.185812: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:02:45.186092: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.2932 - accuracy: 0.4503 - val_loss: 1.1489 - val_accuracy: 0.5531


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 1.1903 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1082 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0623 - accuracy: 0.5833

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0313 - accuracy: 0.5938

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 1.0238 - accuracy: 0.5875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0054 - accuracy: 0.5885

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0306 - accuracy: 0.5759

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0614 - accuracy: 0.5547

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0628 - accuracy: 0.5625

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0666 - accuracy: 0.5594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0514 - accuracy: 0.5568

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0524 - accuracy: 0.5599

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0402 - accuracy: 0.5721

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0482 - accuracy: 0.5670

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0458 - accuracy: 0.5708

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0449 - accuracy: 0.5645

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0550 - accuracy: 0.5662

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0573 - accuracy: 0.5608

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0473 - accuracy: 0.5658

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0434 - accuracy: 0.5734

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0409 - accuracy: 0.5759

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0389 - accuracy: 0.5781

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0506 - accuracy: 0.5747

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0557 - accuracy: 0.5677

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0501 - accuracy: 0.5700

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0558 - accuracy: 0.5697

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0517 - accuracy: 0.5718

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0559 - accuracy: 0.5692

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0615 - accuracy: 0.5679

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0568 - accuracy: 0.5688

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0589 - accuracy: 0.5675

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0562 - accuracy: 0.5674

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0529 - accuracy: 0.5691

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0538 - accuracy: 0.5680

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0605 - accuracy: 0.5670

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0570 - accuracy: 0.5677

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0588 - accuracy: 0.5693

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0556 - accuracy: 0.5715

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0525 - accuracy: 0.5753

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0468 - accuracy: 0.5781

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0545 - accuracy: 0.5739

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0570 - accuracy: 0.5744

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0633 - accuracy: 0.5683

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0592 - accuracy: 0.5682

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0601 - accuracy: 0.5667

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0585 - accuracy: 0.5673

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0592 - accuracy: 0.5678

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0554 - accuracy: 0.5710

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0531 - accuracy: 0.5727

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0519 - accuracy: 0.5725

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0568 - accuracy: 0.5735

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0570 - accuracy: 0.5739

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0569 - accuracy: 0.5719

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0555 - accuracy: 0.5723

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0573 - accuracy: 0.5716

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0533 - accuracy: 0.5737

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0527 - accuracy: 0.5735

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0480 - accuracy: 0.5760

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0459 - accuracy: 0.5768

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0440 - accuracy: 0.5776

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0413 - accuracy: 0.5789

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0499 - accuracy: 0.5726

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0506 - accuracy: 0.5719

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0522 - accuracy: 0.5724

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0532 - accuracy: 0.5732

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0531 - accuracy: 0.5735

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0503 - accuracy: 0.5743

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0500 - accuracy: 0.5764

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0490 - accuracy: 0.5762

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0461 - accuracy: 0.5777

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0423 - accuracy: 0.5788

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0400 - accuracy: 0.5786

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0372 - accuracy: 0.5805

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0349 - accuracy: 0.5811

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0330 - accuracy: 0.5813

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0335 - accuracy: 0.5831

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0353 - accuracy: 0.5824

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0321 - accuracy: 0.5845

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0296 - accuracy: 0.5870

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0290 - accuracy: 0.5867

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0291 - accuracy: 0.5868

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0290 - accuracy: 0.5872

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0266 - accuracy: 0.5884

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0261 - accuracy: 0.5878

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0279 - accuracy: 0.5878

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0248 - accuracy: 0.5901

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0263 - accuracy: 0.5901

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0278 - accuracy: 0.5894

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0293 - accuracy: 0.5884

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0289 - accuracy: 0.5888

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0280 - accuracy: 0.5886

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0280 - accuracy: 0.5886 - val_loss: 1.0141 - val_accuracy: 0.6008


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8305 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9560 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9379 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9268 - accuracy: 0.6719

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9382 - accuracy: 0.6687

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9196 - accuracy: 0.6875

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9127 - accuracy: 0.6964

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9217 - accuracy: 0.6992

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9246 - accuracy: 0.6944

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9171 - accuracy: 0.6938

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9103 - accuracy: 0.6932

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9032 - accuracy: 0.7031

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9036 - accuracy: 0.7019

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9301 - accuracy: 0.6830

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9365 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9213 - accuracy: 0.6914

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9258 - accuracy: 0.6857

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9360 - accuracy: 0.6788

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9299 - accuracy: 0.6793

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9328 - accuracy: 0.6781

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9357 - accuracy: 0.6771

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9218 - accuracy: 0.6790

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9413 - accuracy: 0.6712

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9626 - accuracy: 0.6615

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9626 - accuracy: 0.6562

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9579 - accuracy: 0.6562

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9543 - accuracy: 0.6562

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9508 - accuracy: 0.6596

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9566 - accuracy: 0.6552

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9534 - accuracy: 0.6552

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9534 - accuracy: 0.6542

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9532 - accuracy: 0.6533

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9574 - accuracy: 0.6496

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9619 - accuracy: 0.6480

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9564 - accuracy: 0.6518

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9622 - accuracy: 0.6476

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9549 - accuracy: 0.6503

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9582 - accuracy: 0.6480

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9561 - accuracy: 0.6482

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9551 - accuracy: 0.6477

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9548 - accuracy: 0.6452

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9629 - accuracy: 0.6411

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9599 - accuracy: 0.6421

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9567 - accuracy: 0.6439

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9549 - accuracy: 0.6448

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9507 - accuracy: 0.6451

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9552 - accuracy: 0.6440

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9512 - accuracy: 0.6442

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9463 - accuracy: 0.6470

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9478 - accuracy: 0.6453

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9494 - accuracy: 0.6431

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9463 - accuracy: 0.6451

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9452 - accuracy: 0.6453

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9442 - accuracy: 0.6450

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9425 - accuracy: 0.6446

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9419 - accuracy: 0.6448

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9376 - accuracy: 0.6456

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9348 - accuracy: 0.6479

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9365 - accuracy: 0.6475

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9378 - accuracy: 0.6461

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9342 - accuracy: 0.6468

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9331 - accuracy: 0.6464

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9318 - accuracy: 0.6471

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9330 - accuracy: 0.6477

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9340 - accuracy: 0.6473

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9331 - accuracy: 0.6489

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9329 - accuracy: 0.6494

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9293 - accuracy: 0.6509

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9314 - accuracy: 0.6492

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9312 - accuracy: 0.6484

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9308 - accuracy: 0.6485

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9325 - accuracy: 0.6469

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9307 - accuracy: 0.6475

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9296 - accuracy: 0.6480

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9301 - accuracy: 0.6477

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9304 - accuracy: 0.6466

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9324 - accuracy: 0.6455

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9341 - accuracy: 0.6440

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9347 - accuracy: 0.6458

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9349 - accuracy: 0.6463

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9322 - accuracy: 0.6479

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9343 - accuracy: 0.6465

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9351 - accuracy: 0.6451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9352 - accuracy: 0.6456

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9348 - accuracy: 0.6454

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9339 - accuracy: 0.6452

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9335 - accuracy: 0.6453

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9315 - accuracy: 0.6458

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9326 - accuracy: 0.6445

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9330 - accuracy: 0.6446

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9340 - accuracy: 0.6441

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9340 - accuracy: 0.6441 - val_loss: 1.0167 - val_accuracy: 0.6090


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8363 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9165 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9982 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9931 - accuracy: 0.6719

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9837 - accuracy: 0.6687

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9827 - accuracy: 0.6615

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9676 - accuracy: 0.6696

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9634 - accuracy: 0.6602

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9596 - accuracy: 0.6562

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9570 - accuracy: 0.6500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9321 - accuracy: 0.6619

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9242 - accuracy: 0.6589

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9338 - accuracy: 0.6538

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9465 - accuracy: 0.6473

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9401 - accuracy: 0.6438

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9214 - accuracy: 0.6484

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9199 - accuracy: 0.6452

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9139 - accuracy: 0.6458

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9010 - accuracy: 0.6530

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9039 - accuracy: 0.6453

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8918 - accuracy: 0.6503

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8839 - accuracy: 0.6534

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8704 - accuracy: 0.6603

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8684 - accuracy: 0.6602

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8628 - accuracy: 0.6650

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8638 - accuracy: 0.6635

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8694 - accuracy: 0.6609

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8612 - accuracy: 0.6607

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8626 - accuracy: 0.6606

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8654 - accuracy: 0.6604

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8782 - accuracy: 0.6532

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8736 - accuracy: 0.6553

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8755 - accuracy: 0.6562

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8733 - accuracy: 0.6590

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8788 - accuracy: 0.6562

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8838 - accuracy: 0.6545

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8797 - accuracy: 0.6579

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8767 - accuracy: 0.6595

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8766 - accuracy: 0.6595

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8789 - accuracy: 0.6578

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8782 - accuracy: 0.6578

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8783 - accuracy: 0.6577

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8799 - accuracy: 0.6562

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8822 - accuracy: 0.6570

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8799 - accuracy: 0.6576

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8775 - accuracy: 0.6596

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8768 - accuracy: 0.6602

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8804 - accuracy: 0.6582

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8797 - accuracy: 0.6582

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8788 - accuracy: 0.6587

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8869 - accuracy: 0.6532

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8863 - accuracy: 0.6538

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8853 - accuracy: 0.6539

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8844 - accuracy: 0.6545

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8887 - accuracy: 0.6551

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8882 - accuracy: 0.6546

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8852 - accuracy: 0.6552

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8820 - accuracy: 0.6568

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8829 - accuracy: 0.6557

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8830 - accuracy: 0.6557

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8850 - accuracy: 0.6557

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8812 - accuracy: 0.6578

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8795 - accuracy: 0.6592

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8778 - accuracy: 0.6602

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8757 - accuracy: 0.6611

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8745 - accuracy: 0.6624

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8751 - accuracy: 0.6618

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8736 - accuracy: 0.6613

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8746 - accuracy: 0.6603

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8721 - accuracy: 0.6612

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8745 - accuracy: 0.6615

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8726 - accuracy: 0.6623

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8705 - accuracy: 0.6622

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8672 - accuracy: 0.6643

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8684 - accuracy: 0.6633

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8702 - accuracy: 0.6620

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8736 - accuracy: 0.6611

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8710 - accuracy: 0.6627

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8715 - accuracy: 0.6634

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8709 - accuracy: 0.6637

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8709 - accuracy: 0.6632

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8703 - accuracy: 0.6631

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8671 - accuracy: 0.6642

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8674 - accuracy: 0.6637

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8664 - accuracy: 0.6644

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8665 - accuracy: 0.6639

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8664 - accuracy: 0.6635

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8714 - accuracy: 0.6616

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8733 - accuracy: 0.6609

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8726 - accuracy: 0.6612

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8707 - accuracy: 0.6628

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8707 - accuracy: 0.6628 - val_loss: 0.8705 - val_accuracy: 0.6580


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6741 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8270 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8538 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8606 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8789 - accuracy: 0.6812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8877 - accuracy: 0.6927

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8723 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8522 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8361 - accuracy: 0.7222

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8267 - accuracy: 0.7219

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8355 - accuracy: 0.7017

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8326 - accuracy: 0.6953

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8229 - accuracy: 0.6971

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8100 - accuracy: 0.7009

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7887 - accuracy: 0.7104

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7933 - accuracy: 0.7070

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7905 - accuracy: 0.7096

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7968 - accuracy: 0.7066

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7963 - accuracy: 0.7039

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7825 - accuracy: 0.7078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7853 - accuracy: 0.7024

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7859 - accuracy: 0.7031

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7891 - accuracy: 0.7011

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7910 - accuracy: 0.7005

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7809 - accuracy: 0.7050

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7819 - accuracy: 0.7043

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7933 - accuracy: 0.7049

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7988 - accuracy: 0.7054

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7976 - accuracy: 0.7037

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7978 - accuracy: 0.7042

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7991 - accuracy: 0.7036

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7989 - accuracy: 0.7041

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7951 - accuracy: 0.7036

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7887 - accuracy: 0.7077

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7926 - accuracy: 0.7054

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7897 - accuracy: 0.7066

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7887 - accuracy: 0.7095

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7841 - accuracy: 0.7113

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7842 - accuracy: 0.7123

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7803 - accuracy: 0.7141

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7811 - accuracy: 0.7134

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7828 - accuracy: 0.7113

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7874 - accuracy: 0.7086

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7879 - accuracy: 0.7088

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7855 - accuracy: 0.7111

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7882 - accuracy: 0.7126

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7878 - accuracy: 0.7121

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7918 - accuracy: 0.7090

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7957 - accuracy: 0.7060

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7964 - accuracy: 0.7069

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8005 - accuracy: 0.7053

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8004 - accuracy: 0.7049

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8022 - accuracy: 0.7028

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8011 - accuracy: 0.7037

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8015 - accuracy: 0.7034

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7996 - accuracy: 0.7037

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7987 - accuracy: 0.7039

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7975 - accuracy: 0.7031

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7952 - accuracy: 0.7034

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7986 - accuracy: 0.7010

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8070 - accuracy: 0.6972

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8102 - accuracy: 0.6981

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8076 - accuracy: 0.6994

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8069 - accuracy: 0.7007

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8088 - accuracy: 0.6995

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8059 - accuracy: 0.7008

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8097 - accuracy: 0.7006

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8159 - accuracy: 0.6967

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8119 - accuracy: 0.6984

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8091 - accuracy: 0.6991

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8096 - accuracy: 0.6989

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8116 - accuracy: 0.6979

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8116 - accuracy: 0.6969

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8119 - accuracy: 0.6964

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8093 - accuracy: 0.6971

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8102 - accuracy: 0.6965

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8119 - accuracy: 0.6952

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8124 - accuracy: 0.6947

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8098 - accuracy: 0.6954

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8111 - accuracy: 0.6945

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8111 - accuracy: 0.6952

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8116 - accuracy: 0.6959

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8142 - accuracy: 0.6947

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8132 - accuracy: 0.6949

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8123 - accuracy: 0.6952

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8140 - accuracy: 0.6944

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8123 - accuracy: 0.6943

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8107 - accuracy: 0.6950

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8095 - accuracy: 0.6956

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8081 - accuracy: 0.6965

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8098 - accuracy: 0.6951

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8092 - accuracy: 0.6948 - val_loss: 0.8903 - val_accuracy: 0.6621


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6048 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7650 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7359 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7559 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7392 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7327 - accuracy: 0.7188

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7080 - accuracy: 0.7232

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6999 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7135 - accuracy: 0.7326

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7248 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7132 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7367 - accuracy: 0.7422

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7653 - accuracy: 0.7356

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7713 - accuracy: 0.7299

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7611 - accuracy: 0.7292

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7623 - accuracy: 0.7246

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7621 - accuracy: 0.7261

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7717 - accuracy: 0.7222

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7866 - accuracy: 0.7138

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7912 - accuracy: 0.7109

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7917 - accuracy: 0.7098

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7904 - accuracy: 0.7102

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7837 - accuracy: 0.7133

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7811 - accuracy: 0.7135

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7770 - accuracy: 0.7113

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7824 - accuracy: 0.7115

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7846 - accuracy: 0.7118

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7880 - accuracy: 0.7109

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7905 - accuracy: 0.7091

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7907 - accuracy: 0.7083

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7875 - accuracy: 0.7077

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7920 - accuracy: 0.7051

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8038 - accuracy: 0.6970

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8070 - accuracy: 0.6985

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8008 - accuracy: 0.7018

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7996 - accuracy: 0.7005

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7951 - accuracy: 0.7010

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8025 - accuracy: 0.7007

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8020 - accuracy: 0.7019

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8055 - accuracy: 0.6969

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8086 - accuracy: 0.6951

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8065 - accuracy: 0.6949

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8090 - accuracy: 0.6926

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8070 - accuracy: 0.6932

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8057 - accuracy: 0.6944

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8006 - accuracy: 0.6957

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7984 - accuracy: 0.6988

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8040 - accuracy: 0.6974

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7983 - accuracy: 0.7010

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7924 - accuracy: 0.7044

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7966 - accuracy: 0.7029

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7964 - accuracy: 0.7032

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7944 - accuracy: 0.7047

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7917 - accuracy: 0.7055

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7926 - accuracy: 0.7057

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7900 - accuracy: 0.7087

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7880 - accuracy: 0.7105

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7851 - accuracy: 0.7122

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7819 - accuracy: 0.7139

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7826 - accuracy: 0.7150

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7794 - accuracy: 0.7161

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7789 - accuracy: 0.7166

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7802 - accuracy: 0.7147

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7835 - accuracy: 0.7128

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7817 - accuracy: 0.7120

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7799 - accuracy: 0.7135

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7805 - accuracy: 0.7136

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7807 - accuracy: 0.7123

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7794 - accuracy: 0.7128

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7804 - accuracy: 0.7125

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7822 - accuracy: 0.7117

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7835 - accuracy: 0.7113

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7802 - accuracy: 0.7123

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7834 - accuracy: 0.7103

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7825 - accuracy: 0.7104

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7823 - accuracy: 0.7097

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7810 - accuracy: 0.7102

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7804 - accuracy: 0.7103

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7803 - accuracy: 0.7104

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7788 - accuracy: 0.7117

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7779 - accuracy: 0.7122

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7781 - accuracy: 0.7122

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7805 - accuracy: 0.7101

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7799 - accuracy: 0.7098

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7798 - accuracy: 0.7099

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7801 - accuracy: 0.7097

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7801 - accuracy: 0.7083

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7800 - accuracy: 0.7092

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7811 - accuracy: 0.7089

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7804 - accuracy: 0.7094

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7804 - accuracy: 0.7091

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7804 - accuracy: 0.7091 - val_loss: 0.8211 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5864 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7039 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7027 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7723 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7507 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7378 - accuracy: 0.7135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7458 - accuracy: 0.7054

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7400 - accuracy: 0.7148

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7549 - accuracy: 0.7118

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7396 - accuracy: 0.7156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7310 - accuracy: 0.7159

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7593 - accuracy: 0.7005

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7588 - accuracy: 0.6995

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7543 - accuracy: 0.7031

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7643 - accuracy: 0.6979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7672 - accuracy: 0.7012

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7592 - accuracy: 0.7022

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7625 - accuracy: 0.7031

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7574 - accuracy: 0.7072

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7602 - accuracy: 0.7063

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7660 - accuracy: 0.7039

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7599 - accuracy: 0.7088

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7611 - accuracy: 0.7133

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7655 - accuracy: 0.7109

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7638 - accuracy: 0.7088

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7642 - accuracy: 0.7079

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7538 - accuracy: 0.7141

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7542 - accuracy: 0.7143

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7481 - accuracy: 0.7166

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7486 - accuracy: 0.7167

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7545 - accuracy: 0.7127

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7521 - accuracy: 0.7148

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7468 - accuracy: 0.7197

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7525 - accuracy: 0.7178

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7562 - accuracy: 0.7152

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7556 - accuracy: 0.7153

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7522 - accuracy: 0.7179

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7503 - accuracy: 0.7179

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7438 - accuracy: 0.7196

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7425 - accuracy: 0.7180

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7433 - accuracy: 0.7157

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7386 - accuracy: 0.7180

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7385 - accuracy: 0.7180

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7385 - accuracy: 0.7180

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7373 - accuracy: 0.7181

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7400 - accuracy: 0.7174

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7457 - accuracy: 0.7141

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7428 - accuracy: 0.7148

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7388 - accuracy: 0.7149

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7390 - accuracy: 0.7150

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7341 - accuracy: 0.7169

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7332 - accuracy: 0.7181

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7414 - accuracy: 0.7146

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7393 - accuracy: 0.7147

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7404 - accuracy: 0.7153

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7461 - accuracy: 0.7132

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7417 - accuracy: 0.7155

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7421 - accuracy: 0.7161

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7380 - accuracy: 0.7182

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7427 - accuracy: 0.7167

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7413 - accuracy: 0.7176

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7406 - accuracy: 0.7176

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7438 - accuracy: 0.7167

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7414 - accuracy: 0.7167

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7423 - accuracy: 0.7158

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7417 - accuracy: 0.7154

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7426 - accuracy: 0.7173

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7468 - accuracy: 0.7145

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7515 - accuracy: 0.7133

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7499 - accuracy: 0.7133

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7464 - accuracy: 0.7152

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7439 - accuracy: 0.7165

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7418 - accuracy: 0.7169

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7423 - accuracy: 0.7170

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7402 - accuracy: 0.7174

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7386 - accuracy: 0.7182

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7363 - accuracy: 0.7191

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7362 - accuracy: 0.7194

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7355 - accuracy: 0.7198

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7352 - accuracy: 0.7190

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7354 - accuracy: 0.7187

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7380 - accuracy: 0.7171

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7426 - accuracy: 0.7179

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7408 - accuracy: 0.7187

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7391 - accuracy: 0.7201

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7411 - accuracy: 0.7179

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7416 - accuracy: 0.7176

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7405 - accuracy: 0.7180

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7431 - accuracy: 0.7176

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7454 - accuracy: 0.7163

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7467 - accuracy: 0.7156

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7467 - accuracy: 0.7156 - val_loss: 0.7641 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6508 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6867 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7119 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 4s - loss: 0.7359 - accuracy: 0.7266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.7326 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6950 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7118 - accuracy: 0.7188

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7066 - accuracy: 0.7266

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7091 - accuracy: 0.7257

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6953 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6835 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6656 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6653 - accuracy: 0.7572

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6764 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6726 - accuracy: 0.7542

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6752 - accuracy: 0.7559

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6682 - accuracy: 0.7592

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6557 - accuracy: 0.7656

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6444 - accuracy: 0.7697

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6521 - accuracy: 0.7641

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6511 - accuracy: 0.7664

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6622 - accuracy: 0.7628

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6523 - accuracy: 0.7636

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6451 - accuracy: 0.7643

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6400 - accuracy: 0.7650

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6392 - accuracy: 0.7644

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6597 - accuracy: 0.7558

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6556 - accuracy: 0.7578

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6533 - accuracy: 0.7565

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6564 - accuracy: 0.7541

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6497 - accuracy: 0.7569

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6487 - accuracy: 0.7567

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6453 - accuracy: 0.7583

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6487 - accuracy: 0.7572

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6446 - accuracy: 0.7570

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6435 - accuracy: 0.7577

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6437 - accuracy: 0.7550

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6438 - accuracy: 0.7532

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6458 - accuracy: 0.7539

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6456 - accuracy: 0.7515

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6474 - accuracy: 0.7522

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6428 - accuracy: 0.7537

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6442 - accuracy: 0.7529

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6470 - accuracy: 0.7535

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6562 - accuracy: 0.7500

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6570 - accuracy: 0.7507

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6554 - accuracy: 0.7513

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6602 - accuracy: 0.7487

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6643 - accuracy: 0.7494

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6673 - accuracy: 0.7488

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6643 - accuracy: 0.7494

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6674 - accuracy: 0.7476

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6687 - accuracy: 0.7453

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6678 - accuracy: 0.7460

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6668 - accuracy: 0.7472

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6666 - accuracy: 0.7461

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6666 - accuracy: 0.7468

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6702 - accuracy: 0.7452

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6726 - accuracy: 0.7453

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6757 - accuracy: 0.7443

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6771 - accuracy: 0.7439

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6753 - accuracy: 0.7445

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6801 - accuracy: 0.7431

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6810 - accuracy: 0.7437

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6823 - accuracy: 0.7419

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6813 - accuracy: 0.7430

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6830 - accuracy: 0.7426

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6815 - accuracy: 0.7432

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6850 - accuracy: 0.7433

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6853 - accuracy: 0.7434

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6837 - accuracy: 0.7443

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6828 - accuracy: 0.7436

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6827 - accuracy: 0.7424

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6819 - accuracy: 0.7425

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6846 - accuracy: 0.7413

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6857 - accuracy: 0.7410

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6860 - accuracy: 0.7404

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6869 - accuracy: 0.7397

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6927 - accuracy: 0.7371

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6903 - accuracy: 0.7380

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6883 - accuracy: 0.7389

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6879 - accuracy: 0.7387

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6879 - accuracy: 0.7396

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6893 - accuracy: 0.7393

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6899 - accuracy: 0.7387

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6911 - accuracy: 0.7374

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6919 - accuracy: 0.7372

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6934 - accuracy: 0.7359

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6958 - accuracy: 0.7347

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6970 - accuracy: 0.7345

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6957 - accuracy: 0.7354

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6957 - accuracy: 0.7354 - val_loss: 0.7505 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 6s - loss: 0.7546 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8370 - accuracy: 0.6719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8342 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8279 - accuracy: 0.6562

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7963 - accuracy: 0.6793

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8026 - accuracy: 0.6759

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7866 - accuracy: 0.6815

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7387 - accuracy: 0.7071

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7436 - accuracy: 0.7019

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7342 - accuracy: 0.7151

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7144 - accuracy: 0.7207

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7117 - accuracy: 0.7157

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7080 - accuracy: 0.7205

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7039 - accuracy: 0.7246

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7006 - accuracy: 0.7262

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6964 - accuracy: 0.7295

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6970 - accuracy: 0.7271

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6961 - accuracy: 0.7317

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7129 - accuracy: 0.7215

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7005 - accuracy: 0.7289

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.7004 - accuracy: 0.7313

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6972 - accuracy: 0.7349

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6906 - accuracy: 0.7355

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6947 - accuracy: 0.7311

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6928 - accuracy: 0.7318

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6925 - accuracy: 0.7336

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6943 - accuracy: 0.7320

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6862 - accuracy: 0.7359

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6890 - accuracy: 0.7353

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6888 - accuracy: 0.7337

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6848 - accuracy: 0.7352

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6909 - accuracy: 0.7357

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6884 - accuracy: 0.7370

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6860 - accuracy: 0.7356

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6857 - accuracy: 0.7351

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6853 - accuracy: 0.7338

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6834 - accuracy: 0.7343

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6836 - accuracy: 0.7347

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6787 - accuracy: 0.7358

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6782 - accuracy: 0.7362

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6754 - accuracy: 0.7373

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6786 - accuracy: 0.7361

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6741 - accuracy: 0.7371

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6711 - accuracy: 0.7395

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6708 - accuracy: 0.7391

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6685 - accuracy: 0.7413

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6659 - accuracy: 0.7435

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6680 - accuracy: 0.7442

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6656 - accuracy: 0.7469

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6694 - accuracy: 0.7475

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6645 - accuracy: 0.7512

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6712 - accuracy: 0.7488

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6652 - accuracy: 0.7517

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6647 - accuracy: 0.7511

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6652 - accuracy: 0.7500

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6638 - accuracy: 0.7500

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6623 - accuracy: 0.7516

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6596 - accuracy: 0.7532

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6587 - accuracy: 0.7537

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6593 - accuracy: 0.7526

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6590 - accuracy: 0.7525

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6594 - accuracy: 0.7510

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6649 - accuracy: 0.7480

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6651 - accuracy: 0.7481

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6669 - accuracy: 0.7462

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6673 - accuracy: 0.7458

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6654 - accuracy: 0.7458

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6693 - accuracy: 0.7441

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6671 - accuracy: 0.7460

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6662 - accuracy: 0.7465

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6674 - accuracy: 0.7456

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6704 - accuracy: 0.7444

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6705 - accuracy: 0.7441

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6753 - accuracy: 0.7425

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6736 - accuracy: 0.7438

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6752 - accuracy: 0.7427

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6730 - accuracy: 0.7428

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6712 - accuracy: 0.7433

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6715 - accuracy: 0.7429

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6697 - accuracy: 0.7438

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6716 - accuracy: 0.7439

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6730 - accuracy: 0.7432

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6761 - accuracy: 0.7429

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6755 - accuracy: 0.7434

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6740 - accuracy: 0.7438

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6750 - accuracy: 0.7432

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6751 - accuracy: 0.7447

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6743 - accuracy: 0.7454

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6736 - accuracy: 0.7455

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6732 - accuracy: 0.7466

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6741 - accuracy: 0.7452

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6741 - accuracy: 0.7452 - val_loss: 0.7935 - val_accuracy: 0.7044


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.3921 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5044 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5442 - accuracy: 0.8021

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5108 - accuracy: 0.8203

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5488 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5497 - accuracy: 0.8177

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5828 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5639 - accuracy: 0.7969

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5383 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5516 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5671 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5639 - accuracy: 0.7969

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5522 - accuracy: 0.8005

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5456 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5563 - accuracy: 0.7958

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5779 - accuracy: 0.7891

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5689 - accuracy: 0.7923

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5782 - accuracy: 0.7882

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5837 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5865 - accuracy: 0.7828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5900 - accuracy: 0.7783

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6162 - accuracy: 0.7628

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6209 - accuracy: 0.7609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6174 - accuracy: 0.7617

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6076 - accuracy: 0.7650

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6031 - accuracy: 0.7680

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6074 - accuracy: 0.7685

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6108 - accuracy: 0.7667

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6132 - accuracy: 0.7651

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6188 - accuracy: 0.7615

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6164 - accuracy: 0.7601

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6227 - accuracy: 0.7559

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6189 - accuracy: 0.7566

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6174 - accuracy: 0.7574

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6222 - accuracy: 0.7580

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6275 - accuracy: 0.7561

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6323 - accuracy: 0.7525

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6289 - accuracy: 0.7525

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6338 - accuracy: 0.7516

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6304 - accuracy: 0.7531

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6296 - accuracy: 0.7538

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6279 - accuracy: 0.7545

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6293 - accuracy: 0.7536

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6270 - accuracy: 0.7543

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6332 - accuracy: 0.7521

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6324 - accuracy: 0.7520

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6307 - accuracy: 0.7547

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6299 - accuracy: 0.7539

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6312 - accuracy: 0.7545

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6317 - accuracy: 0.7544

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6303 - accuracy: 0.7549

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6279 - accuracy: 0.7560

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6241 - accuracy: 0.7577

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6259 - accuracy: 0.7558

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6274 - accuracy: 0.7557

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6273 - accuracy: 0.7556

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6275 - accuracy: 0.7560

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6244 - accuracy: 0.7570

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6211 - accuracy: 0.7585

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6219 - accuracy: 0.7578

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6187 - accuracy: 0.7592

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6172 - accuracy: 0.7606

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6171 - accuracy: 0.7589

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6184 - accuracy: 0.7578

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6160 - accuracy: 0.7591

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6153 - accuracy: 0.7595

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6163 - accuracy: 0.7589

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6196 - accuracy: 0.7587

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6193 - accuracy: 0.7591

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6201 - accuracy: 0.7580

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6231 - accuracy: 0.7566

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6322 - accuracy: 0.7535

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6332 - accuracy: 0.7539

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6296 - accuracy: 0.7555

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6278 - accuracy: 0.7571

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6275 - accuracy: 0.7570

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6272 - accuracy: 0.7565

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6290 - accuracy: 0.7568

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6279 - accuracy: 0.7575

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6291 - accuracy: 0.7570

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6300 - accuracy: 0.7562

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6307 - accuracy: 0.7550

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6308 - accuracy: 0.7553

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6309 - accuracy: 0.7556

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6326 - accuracy: 0.7548

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6310 - accuracy: 0.7562

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6322 - accuracy: 0.7554

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6335 - accuracy: 0.7556

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6322 - accuracy: 0.7563

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6296 - accuracy: 0.7569

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6308 - accuracy: 0.7561

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6308 - accuracy: 0.7561 - val_loss: 0.7393 - val_accuracy: 0.7221


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7374 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5940 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5806 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5891 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5532 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5924 - accuracy: 0.7708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5735 - accuracy: 0.7812

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6109 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6237 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6301 - accuracy: 0.7688

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6402 - accuracy: 0.7614

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6603 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6484 - accuracy: 0.7548

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6488 - accuracy: 0.7522

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6512 - accuracy: 0.7521

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6404 - accuracy: 0.7559

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6379 - accuracy: 0.7574

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6497 - accuracy: 0.7500

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6467 - accuracy: 0.7533

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6539 - accuracy: 0.7516

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6518 - accuracy: 0.7530

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6471 - accuracy: 0.7557

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6479 - accuracy: 0.7541

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6451 - accuracy: 0.7552

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6359 - accuracy: 0.7613

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6388 - accuracy: 0.7608

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6383 - accuracy: 0.7616

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6323 - accuracy: 0.7634

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6352 - accuracy: 0.7629

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6346 - accuracy: 0.7625

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6315 - accuracy: 0.7651

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6301 - accuracy: 0.7656

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6387 - accuracy: 0.7614

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6383 - accuracy: 0.7610

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6382 - accuracy: 0.7607

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6420 - accuracy: 0.7578

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6389 - accuracy: 0.7584

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6351 - accuracy: 0.7590

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6352 - accuracy: 0.7588

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6320 - accuracy: 0.7594

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6321 - accuracy: 0.7576

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6309 - accuracy: 0.7582

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6272 - accuracy: 0.7594

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6287 - accuracy: 0.7578

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6312 - accuracy: 0.7569

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6302 - accuracy: 0.7582

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6305 - accuracy: 0.7580

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6363 - accuracy: 0.7565

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6389 - accuracy: 0.7551

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6377 - accuracy: 0.7550

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6383 - accuracy: 0.7549

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6396 - accuracy: 0.7524

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6377 - accuracy: 0.7553

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6378 - accuracy: 0.7564

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6412 - accuracy: 0.7551

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6432 - accuracy: 0.7539

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6399 - accuracy: 0.7549

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6409 - accuracy: 0.7554

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6402 - accuracy: 0.7569

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6408 - accuracy: 0.7573

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6422 - accuracy: 0.7567

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6405 - accuracy: 0.7591

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6371 - accuracy: 0.7609

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6367 - accuracy: 0.7612

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6395 - accuracy: 0.7611

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6385 - accuracy: 0.7595

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6365 - accuracy: 0.7603

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6328 - accuracy: 0.7619

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6307 - accuracy: 0.7627

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6310 - accuracy: 0.7615

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6300 - accuracy: 0.7618

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6297 - accuracy: 0.7612

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6266 - accuracy: 0.7619

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6274 - accuracy: 0.7613

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6298 - accuracy: 0.7599

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6342 - accuracy: 0.7581

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6385 - accuracy: 0.7580

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6351 - accuracy: 0.7595

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6340 - accuracy: 0.7610

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6327 - accuracy: 0.7620

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6299 - accuracy: 0.7626

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6321 - accuracy: 0.7628

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6347 - accuracy: 0.7616

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6362 - accuracy: 0.7607

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6351 - accuracy: 0.7609

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6338 - accuracy: 0.7612

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6319 - accuracy: 0.7614

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6319 - accuracy: 0.7616

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6328 - accuracy: 0.7615

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6358 - accuracy: 0.7596

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6349 - accuracy: 0.7599

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6349 - accuracy: 0.7599 - val_loss: 0.6843 - val_accuracy: 0.7452


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5008 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5083 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5794 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5236 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5030 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5277 - accuracy: 0.7865

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5342 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5459 - accuracy: 0.7891

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5298 - accuracy: 0.7986

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5305 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5452 - accuracy: 0.7841

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5694 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5580 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5695 - accuracy: 0.7812

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5588 - accuracy: 0.7854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5573 - accuracy: 0.7871

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5477 - accuracy: 0.7868

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5455 - accuracy: 0.7865

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5433 - accuracy: 0.7895

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5480 - accuracy: 0.7844

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5460 - accuracy: 0.7857

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5508 - accuracy: 0.7841

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5609 - accuracy: 0.7785

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5736 - accuracy: 0.7747

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5693 - accuracy: 0.7775

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5614 - accuracy: 0.7812

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5653 - accuracy: 0.7824

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5672 - accuracy: 0.7801

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5609 - accuracy: 0.7834

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5651 - accuracy: 0.7792

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5622 - accuracy: 0.7802

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5549 - accuracy: 0.7852

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5630 - accuracy: 0.7822

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5581 - accuracy: 0.7831

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5547 - accuracy: 0.7857

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5544 - accuracy: 0.7847

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5479 - accuracy: 0.7863

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5455 - accuracy: 0.7854

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5483 - accuracy: 0.7837

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5475 - accuracy: 0.7836

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5435 - accuracy: 0.7843

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5408 - accuracy: 0.7857

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5388 - accuracy: 0.7856

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5399 - accuracy: 0.7862

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5432 - accuracy: 0.7854

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5479 - accuracy: 0.7846

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5512 - accuracy: 0.7852

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5498 - accuracy: 0.7852

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5510 - accuracy: 0.7838

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5494 - accuracy: 0.7837

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5571 - accuracy: 0.7812

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5541 - accuracy: 0.7825

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5517 - accuracy: 0.7842

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5519 - accuracy: 0.7847

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5534 - accuracy: 0.7847

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5532 - accuracy: 0.7852

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5551 - accuracy: 0.7841

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5515 - accuracy: 0.7856

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5506 - accuracy: 0.7866

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5510 - accuracy: 0.7865

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5552 - accuracy: 0.7849

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5548 - accuracy: 0.7854

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5563 - accuracy: 0.7853

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5557 - accuracy: 0.7852

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5553 - accuracy: 0.7856

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5559 - accuracy: 0.7860

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5574 - accuracy: 0.7864

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5585 - accuracy: 0.7855

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5677 - accuracy: 0.7823

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5668 - accuracy: 0.7827

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5697 - accuracy: 0.7801

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5693 - accuracy: 0.7809

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5700 - accuracy: 0.7805

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5720 - accuracy: 0.7805

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5746 - accuracy: 0.7797

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5816 - accuracy: 0.7765

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5821 - accuracy: 0.7765

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5797 - accuracy: 0.7778

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5827 - accuracy: 0.7766

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5842 - accuracy: 0.7763

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5869 - accuracy: 0.7764

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5837 - accuracy: 0.7779

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5852 - accuracy: 0.7776

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5848 - accuracy: 0.7773

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5834 - accuracy: 0.7777

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5833 - accuracy: 0.7777

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5828 - accuracy: 0.7774

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5827 - accuracy: 0.7775

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5829 - accuracy: 0.7782

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5851 - accuracy: 0.7775

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5843 - accuracy: 0.7783

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5843 - accuracy: 0.7783 - val_loss: 0.6886 - val_accuracy: 0.7302


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5936 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5854 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6270 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5810 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5644 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5366 - accuracy: 0.8125

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5273 - accuracy: 0.8125

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5348 - accuracy: 0.8047

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5247 - accuracy: 0.8090

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5446 - accuracy: 0.8031

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5462 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5626 - accuracy: 0.7995

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5472 - accuracy: 0.8077

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5356 - accuracy: 0.8080

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5179 - accuracy: 0.8167

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5306 - accuracy: 0.8105

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5438 - accuracy: 0.8033

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5360 - accuracy: 0.8056

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5505 - accuracy: 0.7993

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5467 - accuracy: 0.8000

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5567 - accuracy: 0.7991

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5628 - accuracy: 0.7983

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5690 - accuracy: 0.7962

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5586 - accuracy: 0.8021

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5696 - accuracy: 0.7950

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5740 - accuracy: 0.7909

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5748 - accuracy: 0.7894

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5754 - accuracy: 0.7902

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5756 - accuracy: 0.7920

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5693 - accuracy: 0.7958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5658 - accuracy: 0.7974

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5734 - accuracy: 0.7959

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5688 - accuracy: 0.7973

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5661 - accuracy: 0.7978

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5587 - accuracy: 0.8016

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5532 - accuracy: 0.8036

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5555 - accuracy: 0.8030

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5515 - accuracy: 0.8024

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5515 - accuracy: 0.8011

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5458 - accuracy: 0.8014

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5519 - accuracy: 0.7994

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5587 - accuracy: 0.7990

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5598 - accuracy: 0.7993

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5626 - accuracy: 0.7968

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5618 - accuracy: 0.7978

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5625 - accuracy: 0.7975

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5628 - accuracy: 0.7965

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5640 - accuracy: 0.7962

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5651 - accuracy: 0.7946

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5647 - accuracy: 0.7950

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5665 - accuracy: 0.7935

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5639 - accuracy: 0.7950

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5616 - accuracy: 0.7971

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5628 - accuracy: 0.7962

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5654 - accuracy: 0.7948

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5626 - accuracy: 0.7957

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5612 - accuracy: 0.7960

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5600 - accuracy: 0.7957

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5580 - accuracy: 0.7965

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5601 - accuracy: 0.7953

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5603 - accuracy: 0.7935

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5609 - accuracy: 0.7923

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5572 - accuracy: 0.7936

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5587 - accuracy: 0.7934

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5591 - accuracy: 0.7928

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5586 - accuracy: 0.7926

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5590 - accuracy: 0.7929

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5575 - accuracy: 0.7932

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5532 - accuracy: 0.7953

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5540 - accuracy: 0.7951

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5528 - accuracy: 0.7949

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5536 - accuracy: 0.7951

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5529 - accuracy: 0.7949

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5608 - accuracy: 0.7931

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5582 - accuracy: 0.7941

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5585 - accuracy: 0.7923

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5599 - accuracy: 0.7910

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5617 - accuracy: 0.7905

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5596 - accuracy: 0.7911

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5580 - accuracy: 0.7906

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5585 - accuracy: 0.7909

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5581 - accuracy: 0.7908

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5566 - accuracy: 0.7910

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5565 - accuracy: 0.7909

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5566 - accuracy: 0.7905

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5557 - accuracy: 0.7900

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5600 - accuracy: 0.7881

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5626 - accuracy: 0.7877

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5625 - accuracy: 0.7880

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5648 - accuracy: 0.7868

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5659 - accuracy: 0.7868

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5659 - accuracy: 0.7868 - val_loss: 0.6845 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.3633 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5144 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5932 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5963 - accuracy: 0.7656

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5554 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5253 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5132 - accuracy: 0.8080

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5200 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4983 - accuracy: 0.8160

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5056 - accuracy: 0.8062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4949 - accuracy: 0.8125

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4922 - accuracy: 0.8099

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5006 - accuracy: 0.8125

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5237 - accuracy: 0.8058

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5311 - accuracy: 0.8042

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5431 - accuracy: 0.8008

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5631 - accuracy: 0.7941

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5663 - accuracy: 0.7934

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5619 - accuracy: 0.7944

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5589 - accuracy: 0.7969

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5521 - accuracy: 0.8006

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5421 - accuracy: 0.8040

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5454 - accuracy: 0.8043

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5477 - accuracy: 0.8021

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5558 - accuracy: 0.8012

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5561 - accuracy: 0.7993

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5545 - accuracy: 0.8021

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5617 - accuracy: 0.8002

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5633 - accuracy: 0.8006

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5721 - accuracy: 0.7958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5684 - accuracy: 0.7974

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7939

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5780 - accuracy: 0.7898

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5722 - accuracy: 0.7923

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5681 - accuracy: 0.7937

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5648 - accuracy: 0.7960

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5636 - accuracy: 0.7981

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5624 - accuracy: 0.7985

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5611 - accuracy: 0.7989

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5606 - accuracy: 0.7984

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5590 - accuracy: 0.7995

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5574 - accuracy: 0.7999

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5538 - accuracy: 0.8001

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5490 - accuracy: 0.8018

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5502 - accuracy: 0.8021

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5601 - accuracy: 0.7976

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5608 - accuracy: 0.7965

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5640 - accuracy: 0.7962

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5660 - accuracy: 0.7940

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5655 - accuracy: 0.7944

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5677 - accuracy: 0.7941

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5625 - accuracy: 0.7963

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5615 - accuracy: 0.7948

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5590 - accuracy: 0.7951

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5588 - accuracy: 0.7932

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5609 - accuracy: 0.7913

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5606 - accuracy: 0.7917

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5578 - accuracy: 0.7926

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5541 - accuracy: 0.7940

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5537 - accuracy: 0.7948

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5510 - accuracy: 0.7961

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5504 - accuracy: 0.7969

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5466 - accuracy: 0.7986

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5458 - accuracy: 0.7983

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5461 - accuracy: 0.7981

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5461 - accuracy: 0.7983

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5454 - accuracy: 0.7990

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5437 - accuracy: 0.7987

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5423 - accuracy: 0.7994

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5384 - accuracy: 0.8012

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5389 - accuracy: 0.8010

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5383 - accuracy: 0.8011

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5394 - accuracy: 0.8004

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5388 - accuracy: 0.8010

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5371 - accuracy: 0.8016

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5354 - accuracy: 0.8025

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5353 - accuracy: 0.8027

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5375 - accuracy: 0.8024

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5380 - accuracy: 0.8025

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5396 - accuracy: 0.8019

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5413 - accuracy: 0.8016

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5407 - accuracy: 0.8021

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5421 - accuracy: 0.8007

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5435 - accuracy: 0.7998

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5414 - accuracy: 0.8003

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5399 - accuracy: 0.8004

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5412 - accuracy: 0.7999

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5420 - accuracy: 0.7982

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5413 - accuracy: 0.7987

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5412 - accuracy: 0.7992

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5400 - accuracy: 0.8001

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5400 - accuracy: 0.8001 - val_loss: 0.6695 - val_accuracy: 0.7480


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.3171 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4687 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4490 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4317 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4367 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4802 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4849 - accuracy: 0.7991

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4706 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4855 - accuracy: 0.7951

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4929 - accuracy: 0.7875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4951 - accuracy: 0.7898

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4927 - accuracy: 0.7917

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4840 - accuracy: 0.7957

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4906 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4818 - accuracy: 0.8021

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4905 - accuracy: 0.8027

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4971 - accuracy: 0.8015

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4972 - accuracy: 0.8003

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4890 - accuracy: 0.8059

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4925 - accuracy: 0.8016

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4877 - accuracy: 0.8051

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4934 - accuracy: 0.8026

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5068 - accuracy: 0.7935

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5068 - accuracy: 0.7917

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5119 - accuracy: 0.7900

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5062 - accuracy: 0.7933

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4993 - accuracy: 0.7986

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4950 - accuracy: 0.8013

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4890 - accuracy: 0.8060

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4862 - accuracy: 0.8083

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4962 - accuracy: 0.8054

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4989 - accuracy: 0.8047

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4942 - accuracy: 0.8068

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4952 - accuracy: 0.8079

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5047 - accuracy: 0.8062

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5072 - accuracy: 0.8056

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5077 - accuracy: 0.8066

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5035 - accuracy: 0.8076

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5033 - accuracy: 0.8069

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5141 - accuracy: 0.8031

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5181 - accuracy: 0.8011

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5241 - accuracy: 0.7984

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5278 - accuracy: 0.7972

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5245 - accuracy: 0.7997

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5251 - accuracy: 0.8000

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5257 - accuracy: 0.7989

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5283 - accuracy: 0.7972

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5295 - accuracy: 0.7975

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5316 - accuracy: 0.7972

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5433 - accuracy: 0.7931

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5421 - accuracy: 0.7935

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5409 - accuracy: 0.7945

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5461 - accuracy: 0.7913

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5444 - accuracy: 0.7922

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5449 - accuracy: 0.7920

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5412 - accuracy: 0.7946

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5366 - accuracy: 0.7976

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5339 - accuracy: 0.7995

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5340 - accuracy: 0.8013

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5345 - accuracy: 0.8014

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5357 - accuracy: 0.8011

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5334 - accuracy: 0.8023

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5309 - accuracy: 0.8025

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5316 - accuracy: 0.8012

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5304 - accuracy: 0.8013

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5299 - accuracy: 0.8010

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5273 - accuracy: 0.8026

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5283 - accuracy: 0.8023

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5308 - accuracy: 0.8020

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5333 - accuracy: 0.8012

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5322 - accuracy: 0.8018

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5337 - accuracy: 0.8003

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5357 - accuracy: 0.7996

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5351 - accuracy: 0.8002

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5345 - accuracy: 0.8007

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5349 - accuracy: 0.7997

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5327 - accuracy: 0.8010

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5311 - accuracy: 0.8028

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5322 - accuracy: 0.8033

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5310 - accuracy: 0.8042

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5279 - accuracy: 0.8058

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5271 - accuracy: 0.8063

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5273 - accuracy: 0.8052

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5288 - accuracy: 0.8038

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5283 - accuracy: 0.8039

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5271 - accuracy: 0.8037

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5254 - accuracy: 0.8045

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5254 - accuracy: 0.8049

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5250 - accuracy: 0.8047

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5241 - accuracy: 0.8054

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5250 - accuracy: 0.8055

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5250 - accuracy: 0.8055 - val_loss: 0.6989 - val_accuracy: 0.7316



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1452.png


.. parsed-literal::

    
1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 82ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.40 percent confidence.


.. parsed-literal::

    2024-03-14 01:04:08.593555: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-14 01:04:08.679252: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.689364: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-14 01:04:08.700383: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.707382: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.714152: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.725040: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.764387: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-14 01:04:08.831354: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:08.851679: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-14 01:04:08.891033: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-14 01:04:09.080039: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:09.157882: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-14 01:04:09.300555: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:09.439169: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:09.473030: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:04:09.500780: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-14 01:04:09.547473: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 99.27 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1465.png


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

    2024-03-14 01:04:12.132954: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:04:12.133304: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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

    Accuracy of the original model: 0.732
    Accuracy of the quantized model: 0.728


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
`OpenVINO API tutorial <002-openvino-api-with-output.html>`__
for more information about running inference with Inference Engine
Python API.

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
    This image most likely belongs to dandelion with a 99.27 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_1.png


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
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/utils/notebook_utils.ipynb>`__.
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
    [ INFO ] Read model took 4.37 ms
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

    [ INFO ] Compile model took 120.34 ms
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
    [ INFO ] First inference took 4.07 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55716 iterations
    [ INFO ] Duration:         15002.78 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.04 ms
    [ INFO ]    Average:       3.05 ms
    [ INFO ]    Min:           1.38 ms
    [ INFO ]    Max:           12.62 ms
    [ INFO ] Throughput:   3713.71 FPS


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
    [ INFO ] Read model took 4.79 ms
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

    [ INFO ] Compile model took 114.15 ms
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
    [ INFO ] First inference took 2.35 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178836 iterations
    [ INFO ] Duration:         15001.24 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.57 ms
    [ INFO ]    Max:           6.78 ms
    [ INFO ] Throughput:   11921.42 FPS


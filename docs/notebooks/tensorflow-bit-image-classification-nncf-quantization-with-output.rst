Big Transfer Image Classification Model Quantization pipeline with NNCF
=======================================================================

This tutorial demonstrates the Quantization of the Big Transfer Image
Classification model, which is fine-tuned on the sub-set of ImageNet
dataset with 10 class labels with
`NNCF <https://github.com/openvinotoolkit/nncf>`__. It uses
`BiT-M-R50x1/1 <https://www.kaggle.com/models/google/bit/frameworks/tensorFlow2/variations/m-r50x1/versions/1?tfhub-redirect=true>`__
model, which is trained on ImageNet-21k. Big Transfer is a recipe for
pre-training image classification models on large supervised datasets
and efficiently fine-tuning them on any given target task. The recipe
achieves excellent performance on a wide variety of tasks, even when
using very few labeled examples from the target dataset. This tutorial
uses OpenVINO backend for performing model quantization in NNCF.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prepare Dataset <#prepare-dataset>`__
-  `Plotting data samples <#plotting-data-samples>`__
-  `Model Fine-tuning <#model-fine-tuning>`__
-  `Perform model optimization (IR)
   step <#perform-model-optimization-ir-step>`__
-  `Compute accuracy of the TF
   model <#compute-accuracy-of-the-tf-model>`__
-  `Compute accuracy of the OpenVINO
   model <#compute-accuracy-of-the-openvino-model>`__
-  `Quantize OpenVINO model using
   NNCF <#quantize-openvino-model-using-nncf>`__
-  `Compute accuracy of the quantized
   model <#compute-accuracy-of-the-quantized-model>`__
-  `Compare FP32 and INT8 accuracy <#compare-fp32-and-int8-accuracy>`__
-  `Compare inference results on one
   picture <#compare-inference-results-on-one-picture>`__

.. code:: ipython3

    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"
    
    %pip install -q "openvino>=2024.0.0" "nncf>=2.7.0" "tensorflow-hub>=0.15.0" "tensorflow_datasets" tf_keras
    %pip install -q "scikit-learn>=1.3.2"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import os
    import sys
    import numpy as np
    from pathlib import Path 
    
    from openvino.runtime import Core
    import openvino as ov
    import nncf
    import logging
    
    sys.path.append("../utils")
    from nncf.common.logging.logger import set_log_level
    set_log_level(logging.ERROR)
    
    from sklearn.metrics import accuracy_score
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_hub as hub
    
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
    os.environ['NO_GCE_CHECK'] = 'true'


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. code:: ipython3

    core = Core()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    
    # For top 5 labels.
    MAX_PREDS = 1
    TRAINING_BATCH_SIZE = 128
    BATCH_SIZE = 1
    IMG_SIZE = (256, 256)  # Default Imagenet image size
    NUM_CLASSES = 10  # For Imagenette dataset
    FINE_TUNING_STEPS = 1
    LR = 1e-5
    
    MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)  # From Imagenet dataset
    STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)  # From Imagenet dataset


Prepare Dataset
~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    datasets, datasets_info = tfds.load('imagenette/160px', shuffle_files=True, as_supervised=True, with_info=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=0))
    train_ds, validation_ds = datasets['train'], datasets['validation']



.. parsed-literal::

    2024-03-27 11:42:48.701097: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-27 11:42:48.701318: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    def preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label
    
    train_dataset = (train_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))
    validation_dataset = (validation_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          .batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))

.. code:: ipython3

    # Class labels dictionary with imagenette sample names and classes
    lbl_dict = dict(
        n01440764='tench',
        n02102040='English springer',
        n02979186='cassette player',
        n03000684='chain saw',
        n03028079='church',
        n03394916='French horn',
        n03417042='garbage truck',
        n03425413='gas pump',
        n03445777='golf ball',
        n03888257='parachute'
    )
    
    # Imagenette samples name index
    class_idx_dict = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 
                      'n03028079', 'n03394916', 'n03417042', 'n03425413', 
                      'n03445777', 'n03888257']
    
    def label_func(key):
        return lbl_dict[key]

Plotting data samples
~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    # Get the class labels from the dataset info
    class_labels = datasets_info.features['label'].names
    
    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))
    
    for i, (image, label_index) in enumerate(train_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]
    
        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis('off')
        plt.tight_layout()
    plt.show()



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_9_0.png


.. code:: ipython3

    # Get the class labels from the dataset info
    class_labels = datasets_info.features['label'].names
    
    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))
    
    for i, (image, label_index) in enumerate(validation_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]
    
        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis('off')
        plt.tight_layout()
    plt.show()



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_10_0.png


Model Fine-tuning
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    # Load the Big Transfer model
    bit_model_url = "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
    bit_m = hub.KerasLayer(bit_model_url, trainable=True)
    
    # Customize the model for the new task
    model = tf.keras.Sequential([
        bit_m,
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fine-tune the model
    model.fit(train_dataset.take(3000),
              epochs=FINE_TUNING_STEPS,
              validation_data=validation_dataset.take(1000))
    model.save("./bit_tf_model/", save_format='tf')


.. parsed-literal::

    
  1/101 [..............................] - ETA: 46:26 - loss: 5.4761 - accuracy: 0.1016

.. parsed-literal::

    
  2/101 [..............................] - ETA: 15:22 - loss: 5.3169 - accuracy: 0.1055

.. parsed-literal::

    
  3/101 [..............................] - ETA: 15:14 - loss: 4.9516 - accuracy: 0.1406

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 15:04 - loss: 4.6974 - accuracy: 0.1484

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 14:55 - loss: 4.4972 - accuracy: 0.1625

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 14:45 - loss: 4.2184 - accuracy: 0.1862

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 14:34 - loss: 3.9324 - accuracy: 0.2221

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 14:24 - loss: 3.7196 - accuracy: 0.2480

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 14:15 - loss: 3.4670 - accuracy: 0.2847

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 14:06 - loss: 3.2443 - accuracy: 0.3195

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 13:57 - loss: 3.0633 - accuracy: 0.3480

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 13:49 - loss: 2.8770 - accuracy: 0.3789

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 13:40 - loss: 2.7271 - accuracy: 0.4044

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 13:31 - loss: 2.5891 - accuracy: 0.4302

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 13:21 - loss: 2.4632 - accuracy: 0.4516

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 13:12 - loss: 2.3348 - accuracy: 0.4780

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 13:03 - loss: 2.2315 - accuracy: 0.4982

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 12:53 - loss: 2.1380 - accuracy: 0.5200

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 12:44 - loss: 2.0508 - accuracy: 0.5378

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 12:34 - loss: 1.9710 - accuracy: 0.5547

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 12:25 - loss: 1.8944 - accuracy: 0.5699

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 12:15 - loss: 1.8224 - accuracy: 0.5849

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 12:06 - loss: 1.7658 - accuracy: 0.5965

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 11:57 - loss: 1.6996 - accuracy: 0.6104

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 11:47 - loss: 1.6494 - accuracy: 0.6216

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 11:38 - loss: 1.6008 - accuracy: 0.6310

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 11:29 - loss: 1.5512 - accuracy: 0.6412

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 11:19 - loss: 1.5022 - accuracy: 0.6521

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 11:10 - loss: 1.4630 - accuracy: 0.6600

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 11:01 - loss: 1.4235 - accuracy: 0.6687

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 10:51 - loss: 1.3814 - accuracy: 0.6782

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 10:42 - loss: 1.3495 - accuracy: 0.6858

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 10:33 - loss: 1.3177 - accuracy: 0.6925

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 10:24 - loss: 1.2886 - accuracy: 0.6994

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 10:14 - loss: 1.2618 - accuracy: 0.7054

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 10:05 - loss: 1.2306 - accuracy: 0.7122

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 9:56 - loss: 1.2005 - accuracy: 0.7188 

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 9:46 - loss: 1.1780 - accuracy: 0.7237

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 9:37 - loss: 1.1487 - accuracy: 0.7306

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 9:28 - loss: 1.1251 - accuracy: 0.7361

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 9:18 - loss: 1.1013 - accuracy: 0.7414

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 9:09 - loss: 1.0777 - accuracy: 0.7463

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 9:00 - loss: 1.0557 - accuracy: 0.7511

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 8:50 - loss: 1.0377 - accuracy: 0.7548

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 8:41 - loss: 1.0203 - accuracy: 0.7585

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 8:32 - loss: 1.0012 - accuracy: 0.7632

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 8:22 - loss: 0.9842 - accuracy: 0.7663

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 8:13 - loss: 0.9650 - accuracy: 0.7707

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 8:04 - loss: 0.9492 - accuracy: 0.7746

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 7:54 - loss: 0.9340 - accuracy: 0.7778

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 7:45 - loss: 0.9188 - accuracy: 0.7814

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 7:36 - loss: 0.9038 - accuracy: 0.7852

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 7:26 - loss: 0.8908 - accuracy: 0.7880

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 7:17 - loss: 0.8762 - accuracy: 0.7912

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 7:08 - loss: 0.8617 - accuracy: 0.7946

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 6:58 - loss: 0.8471 - accuracy: 0.7981

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 6:49 - loss: 0.8377 - accuracy: 0.8002

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 6:40 - loss: 0.8250 - accuracy: 0.8032

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 6:30 - loss: 0.8134 - accuracy: 0.8057

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 6:21 - loss: 0.8016 - accuracy: 0.8085

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 6:12 - loss: 0.7903 - accuracy: 0.8110

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 6:03 - loss: 0.7788 - accuracy: 0.8138

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 5:53 - loss: 0.7695 - accuracy: 0.8158

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 5:44 - loss: 0.7618 - accuracy: 0.8177

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 5:35 - loss: 0.7543 - accuracy: 0.8192

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 5:25 - loss: 0.7452 - accuracy: 0.8210

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 5:16 - loss: 0.7364 - accuracy: 0.8228

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 5:07 - loss: 0.7293 - accuracy: 0.8246

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 4:57 - loss: 0.7202 - accuracy: 0.8268

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 4:48 - loss: 0.7118 - accuracy: 0.8283

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 4:39 - loss: 0.7039 - accuracy: 0.8301

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 4:29 - loss: 0.6969 - accuracy: 0.8319

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 4:20 - loss: 0.6875 - accuracy: 0.8342

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 4:11 - loss: 0.6791 - accuracy: 0.8360

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 4:01 - loss: 0.6723 - accuracy: 0.8375

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 3:52 - loss: 0.6656 - accuracy: 0.8390

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 3:43 - loss: 0.6616 - accuracy: 0.8400

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 3:34 - loss: 0.6543 - accuracy: 0.8414

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 3:24 - loss: 0.6466 - accuracy: 0.8433

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 3:15 - loss: 0.6396 - accuracy: 0.8449

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 3:06 - loss: 0.6318 - accuracy: 0.8468

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 2:56 - loss: 0.6250 - accuracy: 0.8485

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 2:47 - loss: 0.6187 - accuracy: 0.8500

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 2:38 - loss: 0.6119 - accuracy: 0.8515

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 2:28 - loss: 0.6053 - accuracy: 0.8528

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 2:19 - loss: 0.6007 - accuracy: 0.8539

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 2:10 - loss: 0.5945 - accuracy: 0.8552

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 2:00 - loss: 0.5881 - accuracy: 0.8567

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 1:51 - loss: 0.5822 - accuracy: 0.8580

.. parsed-literal::

    
 90/101 [=========================>....] - ETA: 1:42 - loss: 0.5771 - accuracy: 0.8592

.. parsed-literal::

    
 91/101 [==========================>...] - ETA: 1:33 - loss: 0.5711 - accuracy: 0.8607

.. parsed-literal::

    
 92/101 [==========================>...] - ETA: 1:23 - loss: 0.5654 - accuracy: 0.8619

.. parsed-literal::

    
 93/101 [==========================>...] - ETA: 1:14 - loss: 0.5596 - accuracy: 0.8632

.. parsed-literal::

    
 94/101 [==========================>...] - ETA: 1:05 - loss: 0.5554 - accuracy: 0.8640

.. parsed-literal::

    
 95/101 [===========================>..] - ETA: 55s - loss: 0.5497 - accuracy: 0.8654 

.. parsed-literal::

    
 96/101 [===========================>..] - ETA: 46s - loss: 0.5451 - accuracy: 0.8666

.. parsed-literal::

    
 97/101 [===========================>..] - ETA: 37s - loss: 0.5403 - accuracy: 0.8675

.. parsed-literal::

    
 98/101 [============================>.] - ETA: 27s - loss: 0.5354 - accuracy: 0.8687

.. parsed-literal::

    
 99/101 [============================>.] - ETA: 18s - loss: 0.5321 - accuracy: 0.8695

.. parsed-literal::

    
100/101 [============================>.] - ETA: 9s - loss: 0.5278 - accuracy: 0.8705 

.. parsed-literal::

    
101/101 [==============================] - ETA: 0s - loss: 0.5249 - accuracy: 0.8711

.. parsed-literal::

    
101/101 [==============================] - 972s 9s/step - loss: 0.5249 - accuracy: 0.8711 - val_loss: 0.0656 - val_accuracy: 0.9820


.. parsed-literal::

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


Perform model optimization (IR) step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    ir_path = Path("./bit_ov_model/bit_m_r50x1_1.xml")
    if not ir_path.exists():
        print("Initiating model optimization..!!!")
        ov_model = ov.convert_model("./bit_tf_model")
        ov.save_model(ov_model, ir_path)
    else:
        print(f"IR model {ir_path} already exists.")


.. parsed-literal::

    Initiating model optimization..!!!


Compute accuracy of the TF model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    tf_model = tf.keras.models.load_model("./bit_tf_model/")
       
    tf_predictions = []
    gt_label = []
    
    for _, label in validation_dataset:
        for cls_label in label:
            l_list = cls_label.numpy().tolist()
            gt_label.append(l_list.index(1))
            
    for img_batch, label_batch in validation_dataset:
        tf_result_batch = tf_model.predict(img_batch, verbose=0)
        for i in range(len(img_batch)):
            tf_result = tf_result_batch[i]
            tf_result = tf.reshape(tf_result, [-1])
            top5_label_idx = np.argsort(tf_result)[-MAX_PREDS::][::-1]
            tf_predictions.append(top5_label_idx)
    
    # Convert the lists to NumPy arrays for accuracy calculation
    tf_predictions = np.array(tf_predictions)
    gt_label = np.array(gt_label)
    
    tf_acc_score = accuracy_score(tf_predictions, gt_label)


Compute accuracy of the OpenVINO model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Select device for inference:

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
    ov_fp32_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # Target device set to CPU (Other options Ex: AUTO/GPU/dGPU/)
    compiled_model = ov.compile_model(ov_fp32_model, device.value)
    output = compiled_model.outputs[0]
    
    ov_predictions = []
    for img_batch, _ in validation_dataset:
        for image in img_batch:
            image = tf.expand_dims(image, axis=0)
            pred = compiled_model(image)[output]
            ov_result = tf.reshape(pred, [-1])
            top_label_idx = np.argsort(ov_result)[-MAX_PREDS::][::-1]
            ov_predictions.append(top_label_idx)
    
    fp32_acc_score = accuracy_score(ov_predictions, gt_label)


Quantize OpenVINO model using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Model Quantization using NNCF

1. Preprocessing and preparing validation samples for NNCF calibration
2. Perform NNCF Quantization on OpenVINO FP32 model
3. Serialize Quantized OpenVINO INT8 model

.. code:: ipython3

    def nncf_preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = image - MEAN_RGB
        image = image / STDDEV_RGB
        return image
    
    val_ds = (validation_ds.map(nncf_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
              .batch(1)
              .prefetch(tf.data.experimental.AUTOTUNE))
    
    calibration_dataset = nncf.Dataset(val_ds)
        
    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
    
    ov_int8_model = nncf.quantize(ov_fp32_model, calibration_dataset, fast_bias_correction=False)
    
    ov.save_model(ov_int8_model, "./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")



.. parsed-literal::

    Output()



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



Compute accuracy of the quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    nncf_quantized_model = core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
    nncf_quantized_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # Target device set to CPU by default
    compiled_model = ov.compile_model(nncf_quantized_model, device.value)
    output = compiled_model.outputs[0]
    
    ov_predictions = []
    inp_tensor = nncf_quantized_model.inputs[0]
    out_tensor = nncf_quantized_model.outputs[0]
            
    for img_batch, _ in validation_dataset:
        for image in img_batch:
            image = tf.expand_dims(image, axis=0)
            pred = compiled_model(image)[output]
            ov_result = tf.reshape(pred, [-1])
            top_label_idx = np.argsort(ov_result)[-MAX_PREDS::][::-1]
            ov_predictions.append(top_label_idx)
            
    int8_acc_score = accuracy_score(ov_predictions, gt_label)


Compare FP32 and INT8 accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    print(f"Accuracy of the tensorflow model (fp32): {tf_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO optimized model (fp32): {fp32_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO quantized model (int8): {int8_acc_score * 100: .2f}%")
    accuracy_drop = fp32_acc_score - int8_acc_score
    print(f"Accuracy drop between OV FP32 and INT8 model: {accuracy_drop * 100:.1f}% ")


.. parsed-literal::

    Accuracy of the tensorflow model (fp32):  98.20%
    Accuracy of the OpenVINO optimized model (fp32):  98.20%
    Accuracy of the OpenVINO quantized model (int8):  97.40%
    Accuracy drop between OV FP32 and INT8 model: 0.8% 


Compare inference results on one picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    
    # Accessing validation sample
    sample_idx = 50
    vds = datasets['validation']
    
    if len(vds) > sample_idx:
        sample = vds.take(sample_idx + 1).skip(sample_idx).as_numpy_iterator().next()
    else:
        print("Dataset does not have enough samples...!!!")
    
    # Image data
    sample_data = sample[0]
    
    # Label info
    sample_label = sample[1]
    
    # Image data pre-processing
    image = tf.image.resize(sample_data, IMG_SIZE)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0
    
    # OpenVINO inference 
    def ov_inference(model: ov.Model, image) -> str:
        compiled_model = ov.compile_model(model, device.value)
        output = compiled_model.outputs[0]
        pred = compiled_model(image)[output]
        ov_result = tf.reshape(pred, [-1])
        pred_label = np.argsort(ov_result)[-MAX_PREDS::][::-1]
        return pred_label
    
    # OpenVINO FP32 model
    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
    ov_fp32_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # OpenVINO INT8 model
    ov_int8_model = core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
    ov_int8_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # OpenVINO FP32 model inference
    ov_fp32_pred_label = ov_inference(ov_fp32_model, image)
    
    print(f"Predicted label for the sample picture by float (fp32) model: {label_func(class_idx_dict[int(ov_fp32_pred_label)])}\n")
    
    # OpenVINO FP32 model inference
    ov_int8_pred_label = ov_inference(ov_int8_model, image)
    print(f"Predicted label for the sample picture by qunatized (int8) model: {label_func(class_idx_dict[int(ov_int8_pred_label)])}\n")
    
    # Plotting the image sample with ground truth
    plt.figure()
    plt.imshow(sample_data)
    plt.title(f"Ground truth: {label_func(class_idx_dict[sample_label])}")
    plt.axis('off')
    plt.show()



.. parsed-literal::

    Predicted label for the sample picture by float (fp32) model: gas pump
    


.. parsed-literal::

    Predicted label for the sample picture by qunatized (int8) model: gas pump
    



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_27_2.png


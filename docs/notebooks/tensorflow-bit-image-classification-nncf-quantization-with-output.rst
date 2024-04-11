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
    import numpy as np
    from pathlib import Path 
    
    from openvino.runtime import Core
    import openvino as ov
    import nncf
    import logging
    
    from nncf.common.logging.logger import set_log_level
    set_log_level(logging.ERROR)
    
    from sklearn.metrics import accuracy_score
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["TFHUB_CACHE_DIR"] = str(Path("./tfhub_modules").resolve())
    
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



.. code:: ipython3

    datasets, datasets_info = tfds.load('imagenette/160px', shuffle_files=True, as_supervised=True, with_info=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=0))
    train_ds, validation_ds = datasets['train'], datasets['validation']



.. parsed-literal::

    2024-04-09 22:30:27.030835: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-09 22:30:27.031063: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

    
  1/101 [..............................] - ETA: 45:32 - loss: 7.0695 - accuracy: 0.0938

.. parsed-literal::

    
  2/101 [..............................] - ETA: 15:04 - loss: 6.3424 - accuracy: 0.1211

.. parsed-literal::

    
  3/101 [..............................] - ETA: 14:58 - loss: 6.1591 - accuracy: 0.1042

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 14:50 - loss: 5.6085 - accuracy: 0.1523

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 14:40 - loss: 5.3571 - accuracy: 0.1656

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 14:31 - loss: 4.9866 - accuracy: 0.1966

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 14:22 - loss: 4.6566 - accuracy: 0.2143

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 14:12 - loss: 4.3641 - accuracy: 0.2344

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 14:02 - loss: 4.1022 - accuracy: 0.2648

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 13:53 - loss: 3.8635 - accuracy: 0.2906

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 13:44 - loss: 3.6445 - accuracy: 0.3210

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 13:34 - loss: 3.4491 - accuracy: 0.3490

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 13:25 - loss: 3.2624 - accuracy: 0.3774

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 13:16 - loss: 3.1082 - accuracy: 0.3968

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 13:07 - loss: 2.9578 - accuracy: 0.4182

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 12:58 - loss: 2.8174 - accuracy: 0.4404

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 12:48 - loss: 2.6809 - accuracy: 0.4660

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 12:39 - loss: 2.5545 - accuracy: 0.4883

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 12:30 - loss: 2.4575 - accuracy: 0.5070

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 12:21 - loss: 2.3525 - accuracy: 0.5262

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 12:12 - loss: 2.2660 - accuracy: 0.5417

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 12:03 - loss: 2.1716 - accuracy: 0.5604

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 11:54 - loss: 2.1015 - accuracy: 0.5727

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 11:44 - loss: 2.0252 - accuracy: 0.5866

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 11:35 - loss: 1.9607 - accuracy: 0.5984

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 11:26 - loss: 1.8966 - accuracy: 0.6103

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 11:17 - loss: 1.8331 - accuracy: 0.6215

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 11:08 - loss: 1.7761 - accuracy: 0.6320

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 10:58 - loss: 1.7297 - accuracy: 0.6409

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 10:49 - loss: 1.6821 - accuracy: 0.6508

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 10:40 - loss: 1.6321 - accuracy: 0.6605

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 10:31 - loss: 1.5889 - accuracy: 0.6687

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 10:22 - loss: 1.5473 - accuracy: 0.6764

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 10:13 - loss: 1.5124 - accuracy: 0.6834

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 10:03 - loss: 1.4755 - accuracy: 0.6902

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 9:54 - loss: 1.4387 - accuracy: 0.6973 

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 9:45 - loss: 1.4028 - accuracy: 0.7044

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 9:36 - loss: 1.3740 - accuracy: 0.7095

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 9:27 - loss: 1.3393 - accuracy: 0.7167

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 9:18 - loss: 1.3125 - accuracy: 0.7221

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 9:09 - loss: 1.2849 - accuracy: 0.7271

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 8:59 - loss: 1.2569 - accuracy: 0.7329

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 8:50 - loss: 1.2318 - accuracy: 0.7375

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 8:41 - loss: 1.2079 - accuracy: 0.7418

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 8:32 - loss: 1.1865 - accuracy: 0.7460

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 8:23 - loss: 1.1637 - accuracy: 0.7512

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 8:14 - loss: 1.1433 - accuracy: 0.7548

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 8:05 - loss: 1.1226 - accuracy: 0.7593

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 7:55 - loss: 1.1035 - accuracy: 0.7628

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 7:46 - loss: 1.0842 - accuracy: 0.7667

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 7:37 - loss: 1.0661 - accuracy: 0.7701

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 7:28 - loss: 1.0490 - accuracy: 0.7733

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 7:19 - loss: 1.0322 - accuracy: 0.7770

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 7:10 - loss: 1.0143 - accuracy: 0.7807

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 7:01 - loss: 0.9969 - accuracy: 0.7845

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 6:51 - loss: 0.9803 - accuracy: 0.7877

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 6:42 - loss: 0.9665 - accuracy: 0.7907

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 6:33 - loss: 0.9510 - accuracy: 0.7939

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 6:24 - loss: 0.9370 - accuracy: 0.7969

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 6:15 - loss: 0.9243 - accuracy: 0.7996

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 6:06 - loss: 0.9104 - accuracy: 0.8026

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 5:56 - loss: 0.8975 - accuracy: 0.8052

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 5:47 - loss: 0.8849 - accuracy: 0.8079

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 5:38 - loss: 0.8747 - accuracy: 0.8099

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 5:29 - loss: 0.8655 - accuracy: 0.8118

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 5:20 - loss: 0.8551 - accuracy: 0.8142

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 5:11 - loss: 0.8446 - accuracy: 0.8162

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 5:01 - loss: 0.8350 - accuracy: 0.8182

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 4:52 - loss: 0.8244 - accuracy: 0.8203

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 4:43 - loss: 0.8148 - accuracy: 0.8222

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 4:34 - loss: 0.8050 - accuracy: 0.8241

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 4:25 - loss: 0.7958 - accuracy: 0.8260

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 4:16 - loss: 0.7862 - accuracy: 0.8279

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 4:07 - loss: 0.7774 - accuracy: 0.8297

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 3:57 - loss: 0.7698 - accuracy: 0.8311

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 3:48 - loss: 0.7614 - accuracy: 0.8330

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 3:39 - loss: 0.7550 - accuracy: 0.8346

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 3:30 - loss: 0.7464 - accuracy: 0.8363

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 3:21 - loss: 0.7376 - accuracy: 0.8381

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 3:12 - loss: 0.7299 - accuracy: 0.8397

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 3:03 - loss: 0.7214 - accuracy: 0.8415

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 2:53 - loss: 0.7131 - accuracy: 0.8434

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 2:44 - loss: 0.7052 - accuracy: 0.8449

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 2:35 - loss: 0.6979 - accuracy: 0.8464

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 2:26 - loss: 0.6906 - accuracy: 0.8479

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 2:17 - loss: 0.6839 - accuracy: 0.8493

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 2:08 - loss: 0.6769 - accuracy: 0.8507

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 1:59 - loss: 0.6702 - accuracy: 0.8521

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 1:49 - loss: 0.6631 - accuracy: 0.8536

.. parsed-literal::

    
 90/101 [=========================>....] - ETA: 1:40 - loss: 0.6572 - accuracy: 0.8548

.. parsed-literal::

    
 91/101 [==========================>...] - ETA: 1:31 - loss: 0.6503 - accuracy: 0.8563

.. parsed-literal::

    
 92/101 [==========================>...] - ETA: 1:22 - loss: 0.6433 - accuracy: 0.8578

.. parsed-literal::

    
 93/101 [==========================>...] - ETA: 1:13 - loss: 0.6366 - accuracy: 0.8594

.. parsed-literal::

    
 94/101 [==========================>...] - ETA: 1:04 - loss: 0.6316 - accuracy: 0.8605

.. parsed-literal::

    
 95/101 [===========================>..] - ETA: 54s - loss: 0.6253 - accuracy: 0.8618 

.. parsed-literal::

    
 96/101 [===========================>..] - ETA: 45s - loss: 0.6194 - accuracy: 0.8630

.. parsed-literal::

    
 97/101 [===========================>..] - ETA: 36s - loss: 0.6137 - accuracy: 0.8641

.. parsed-literal::

    
 98/101 [============================>.] - ETA: 27s - loss: 0.6079 - accuracy: 0.8654

.. parsed-literal::

    
 99/101 [============================>.] - ETA: 18s - loss: 0.6039 - accuracy: 0.8664

.. parsed-literal::

    
100/101 [============================>.] - ETA: 9s - loss: 0.5991 - accuracy: 0.8673 

.. parsed-literal::

    
101/101 [==============================] - ETA: 0s - loss: 0.5959 - accuracy: 0.8678

.. parsed-literal::

    
101/101 [==============================] - 956s 9s/step - loss: 0.5959 - accuracy: 0.8678 - val_loss: 0.0800 - val_accuracy: 0.9760


.. parsed-literal::

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


Perform model optimization (IR) step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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



.. code:: ipython3

    print(f"Accuracy of the tensorflow model (fp32): {tf_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO optimized model (fp32): {fp32_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO quantized model (int8): {int8_acc_score * 100: .2f}%")
    accuracy_drop = fp32_acc_score - int8_acc_score
    print(f"Accuracy drop between OV FP32 and INT8 model: {accuracy_drop * 100:.1f}% ")


.. parsed-literal::

    Accuracy of the tensorflow model (fp32):  97.60%
    Accuracy of the OpenVINO optimized model (fp32):  97.60%
    Accuracy of the OpenVINO quantized model (int8):  96.80%
    Accuracy drop between OV FP32 and INT8 model: 0.8% 


Compare inference results on one picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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


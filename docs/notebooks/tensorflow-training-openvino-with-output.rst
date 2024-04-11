From Training to Deployment with TensorFlow and OpenVINO™
=========================================================

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `TensorFlow Image Classification
   Training <#tensorflow-image-classification-training>`__
-  `Import TensorFlow and Other
   Libraries <#import-tensorflow-and-other-libraries>`__
-  `Download and Explore the
   Dataset <#download-and-explore-the-dataset>`__
-  `Load Using keras.preprocessing <#load-using-keras-preprocessing>`__
-  `Create a Dataset <#create-a-dataset>`__
-  `Visualize the Data <#visualize-the-data>`__
-  `Configure the Dataset for
   Performance <#configure-the-dataset-for-performance>`__
-  `Standardize the Data <#standardize-the-data>`__
-  `Create the Model <#create-the-model>`__
-  `Compile the Model <#compile-the-model>`__
-  `Model Summary <#model-summary>`__
-  `Train the Model <#train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Overfitting <#overfitting>`__
-  `Data Augmentation <#data-augmentation>`__
-  `Dropout <#dropout>`__
-  `Compile and Train the Model <#compile-and-train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Predict on New Data <#predict-on-new-data>`__
-  `Save the TensorFlow Model <#save-the-tensorflow-model>`__
-  `Convert the TensorFlow model with OpenVINO Model Conversion
   API <#convert-the-tensorflow-model-with-openvino-model-conversion-api>`__
-  `Preprocessing Image Function <#preprocessing-image-function>`__
-  `OpenVINO Runtime Setup <#openvino-runtime-setup>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run the Inference Step <#run-the-inference-step>`__
-  `The Next Steps <#the-next-steps>`__

.. code:: ipython3

    # @title Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # Copyright 2018 The TensorFlow Authors
    #
    # Modified for OpenVINO Notebooks

This tutorial demonstrates how to train, convert, and deploy an image
classification model with TensorFlow and OpenVINO. This particular
notebook shows the process where we perform the inference step on the
freshly trained model that is converted to OpenVINO IR with model
conversion API. For faster inference speed on the model created in this
notebook, check out the `Post-Training Quantization with TensorFlow
Classification Model <./tensorflow-training-openvino-nncf.ipynb>`__
notebook.

This training code comprises the official `TensorFlow Image
Classification
Tutorial <https://www.tensorflow.org/tutorials/images/classification>`__
in its entirety.

The ``flower_ir.bin`` and ``flower_ir.xml`` (pre-trained models) can be
obtained by executing the code with ‘Runtime->Run All’ or the
``Ctrl+F9`` command.

.. code:: ipython3

    import platform
    
    %pip install -q "openvino>=2023.1.0" "pillow"
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"
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


TensorFlow Image Classification Training
----------------------------------------



The first part of the tutorial shows how to classify images of flowers
(based on the TensorFlow’s official tutorial). It creates an image
classifier using a ``keras.Sequential`` model, and loads data using
``preprocessing.image_dataset_from_directory``. You will gain practical
experience with the following concepts:

-  Efficiently loading a dataset off disk.
-  Identifying overfitting and applying techniques to mitigate it,
   including data augmentation and Dropout.

This tutorial follows a basic machine learning workflow:

1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model

Import TensorFlow and Other Libraries
-------------------------------------



.. code:: ipython3

    import os
    from pathlib import Path
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file


.. parsed-literal::

    2024-04-10 00:36:10.563849: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-10 00:36:10.598515: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-10 00:36:11.114096: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download and Explore the Dataset
--------------------------------



This tutorial uses a dataset of about 3,700 photos of flowers. The
dataset contains 5 sub-directories, one per class:

::

   flower_photo/
     daisy/
     dandelion/
     roses/
     sunflowers/
     tulips/

.. code:: ipython3

    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

After downloading, you should now have a copy of the dataset available.
There are 3,670 total images:

.. code:: ipython3

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)


.. parsed-literal::

    3670


Here are some roses:

.. code:: ipython3

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_14_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_15_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_17_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_18_0.png



Load Using keras.preprocessing
------------------------------



Let’s load these images off disk using the helpful
`image_dataset_from_directory <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`__
utility. This will take you from a directory of images on disk to a
``tf.data.Dataset`` in just a couple lines of code. If you like, you can
also write your own data loading code from scratch by visiting the `load
images <https://www.tensorflow.org/tutorials/load_data/images>`__
tutorial.

Create a Dataset
----------------



Define some parameters for the loader:

.. code:: ipython3

    batch_size = 32
    img_height = 180
    img_width = 180

It’s good practice to use a validation split when developing your model.
Let’s use 80% of the images for training, and 20% for validation.

.. code:: ipython3

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-04-10 00:36:14.403818: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-10 00:36:14.403850: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-04-10 00:36:14.403854: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-04-10 00:36:14.403983: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-04-10 00:36:14.403998: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-04-10 00:36:14.404001: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.


You can find the class names in the ``class_names`` attribute on these
datasets. These correspond to the directory names in alphabetical order.

.. code:: ipython3

    class_names = train_ds.class_names
    print(class_names)


.. parsed-literal::

    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


Visualize the Data
------------------



Here are the first 9 images from the training dataset.

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


.. parsed-literal::

    2024-04-10 00:36:14.711678: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:36:14.712092: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_29_1.png


You will train a model using these datasets by passing them to
``model.fit`` in a moment. If you like, you can also manually iterate
over the dataset and retrieve batches of images:

.. code:: ipython3

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2024-04-10 00:36:15.553168: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:36:15.553658: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


The ``image_batch`` is a tensor of the shape ``(32, 180, 180, 3)``. This
is a batch of 32 images of shape ``180x180x3`` (the last dimension
refers to color channels RGB). The ``label_batch`` is a tensor of the
shape ``(32,)``, these are corresponding labels to the 32 images.

You can call ``.numpy()`` on the ``image_batch`` and ``labels_batch``
tensors to convert them to a ``numpy.ndarray``.

Configure the Dataset for Performance
-------------------------------------



Let’s make sure to use buffered prefetching so you can yield data from
disk without having I/O become blocking. These are two important methods
you should use when loading data.

``Dataset.cache()`` keeps the images in memory after they’re loaded off
disk during the first epoch. This will ensure the dataset does not
become a bottleneck while training your model. If your dataset is too
large to fit into memory, you can also use this method to create a
performant on-disk cache.

``Dataset.prefetch()`` overlaps data preprocessing and model execution
while training.

Interested readers can learn more about both methods, as well as how to
cache data to disk in the `data performance
guide <https://www.tensorflow.org/guide/data_performance#prefetching>`__.

.. code:: ipython3

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

Standardize the Data
--------------------



The RGB channel values are in the ``[0, 255]`` range. This is not ideal
for a neural network; in general you should seek to make your input
values small. Here, you will standardize values to be in the ``[0, 1]``
range by using a Rescaling layer.

.. code:: ipython3

    normalization_layer = tf.keras.layers.Rescaling(1./255)

Note: The Keras Preprocessing utilities and layers introduced in this
section are currently experimental and may change.

There are two ways to use this layer. You can apply it to the dataset by
calling map:

.. code:: ipython3

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image)) 


.. parsed-literal::

    2024-04-10 00:36:15.755605: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-10 00:36:15.755999: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    0.0 1.0


Or, you can include the layer inside your model definition, which can
simplify deployment. Let’s use the second approach here.

Note: you previously resized images using the ``image_size`` argument of
``image_dataset_from_directory``. If you want to include the resizing
logic in your model as well, you can use the
`Resizing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing>`__
layer.

Create the Model
----------------



The model consists of three convolution blocks with a max pool layer in
each of them. There’s a fully connected layer with 128 units on top of
it that is activated by a ``relu`` activation function. This model has
not been tuned for high accuracy, the goal of this tutorial is to show a
standard approach.

.. code:: ipython3

    num_classes = 5
    
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])

Compile the Model
-----------------



For this tutorial, choose the ``optimizers.Adam`` optimizer and
``losses.SparseCategoricalCrossentropy`` loss function. To view training
and validation accuracy for each training epoch, pass the ``metrics``
argument.

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

Model Summary
-------------



View all the layers of the network using the model’s ``summary`` method.

   **NOTE:** This section is commented out for performance reasons.
   Please feel free to uncomment these to compare the results.

.. code:: ipython3

    # model.summary()

Train the Model
---------------



.. code:: ipython3

    # epochs=10
    # history = model.fit(
    #   train_ds,
    #   validation_data=val_ds,
    #   epochs=epochs
    # )

Visualize Training Results
--------------------------



Create plots of loss and accuracy on the training and validation sets.

.. code:: ipython3

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    
    # epochs_range = range(epochs)
    
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

As you can see from the plots, training accuracy and validation accuracy
are off by large margin and the model has achieved only around 60%
accuracy on the validation set.

Let’s look at what went wrong and try to increase the overall
performance of the model.

Overfitting
-----------



In the plots above, the training accuracy is increasing linearly over
time, whereas validation accuracy stalls around 60% in the training
process. Also, the difference in accuracy between training and
validation accuracy is noticeable — a sign of
`overfitting <https://www.tensorflow.org/tutorials/keras/overfit_and_underfit>`__.

When there are a small number of training examples, the model sometimes
learns from noises or unwanted details from training examples—to an
extent that it negatively impacts the performance of the model on new
examples. This phenomenon is known as overfitting. It means that the
model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In
this tutorial, you’ll use *data augmentation* and add *Dropout* to your
model.

Data Augmentation
-----------------



Overfitting generally occurs when there are a small number of training
examples. `Data
augmentation <https://www.tensorflow.org/tutorials/images/data_augmentation>`__
takes the approach of generating additional training data from your
existing examples by augmenting them using random transformations that
yield believable-looking images. This helps expose the model to more
aspects of the data and generalize better.

You will implement data augmentation using the layers from
``tf.keras.layers.experimental.preprocessing``. These can be included
inside your model like other layers, and run on the GPU.

.. code:: ipython3

    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
      ]
    )

Let’s visualize what a few augmented examples look like by applying data
augmentation to the same image several times:

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


.. parsed-literal::

    2024-04-10 00:36:16.512377: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:36:16.512681: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_57_1.png


You will use data augmentation to train a model in a moment.

Dropout
-------



Another technique to reduce overfitting is to introduce
`Dropout <https://developers.google.com/machine-learning/glossary#dropout_regularization>`__
to the network, a form of *regularization*.

When you apply Dropout to a layer it randomly drops out (by setting the
activation to zero) a number of output units from the layer during the
training process. Dropout takes a fractional number as its input value,
in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20%
or 40% of the output units randomly from the applied layer.

Let’s create a new neural network using ``layers.Dropout``, then train
it using augmented images.

.. code:: ipython3

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")
    ])

Compile and Train the Model
---------------------------



.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

.. code:: ipython3

    model.summary()


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


.. code:: ipython3

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-04-10 00:36:17.658374: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:36:17.658933: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
 1/92 [..............................] - ETA: 1:30 - loss: 1.6254 - accuracy: 0.2500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.8802 - accuracy: 0.2188  

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.8494 - accuracy: 0.2083

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.7768 - accuracy: 0.2344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.7178 - accuracy: 0.2750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.6917 - accuracy: 0.2708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.6766 - accuracy: 0.2679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.6552 - accuracy: 0.2773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.6476 - accuracy: 0.2674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.6338 - accuracy: 0.2656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.6258 - accuracy: 0.2557

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.6134 - accuracy: 0.2604

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.5975 - accuracy: 0.2861

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.5853 - accuracy: 0.2991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.5741 - accuracy: 0.3104

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.5578 - accuracy: 0.3164

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.5412 - accuracy: 0.3272

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.5250 - accuracy: 0.3385

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.5063 - accuracy: 0.3438

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.4973 - accuracy: 0.3500

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.4830 - accuracy: 0.3527

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.4775 - accuracy: 0.3523

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.4640 - accuracy: 0.3546

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.4472 - accuracy: 0.3620

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.4358 - accuracy: 0.3613

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.4341 - accuracy: 0.3582

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.4176 - accuracy: 0.3669

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.4124 - accuracy: 0.3728

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.4113 - accuracy: 0.3761

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.4084 - accuracy: 0.3771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.4027 - accuracy: 0.3770

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.3942 - accuracy: 0.3818

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.3917 - accuracy: 0.3807

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.3890 - accuracy: 0.3842

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.3812 - accuracy: 0.3893

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.3763 - accuracy: 0.3880

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.3768 - accuracy: 0.3910

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.3718 - accuracy: 0.3914

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.3679 - accuracy: 0.3966

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.3643 - accuracy: 0.4008

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.3595 - accuracy: 0.4024

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.3558 - accuracy: 0.4048

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.3592 - accuracy: 0.4012

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.3587 - accuracy: 0.4034

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.3577 - accuracy: 0.4014

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.3530 - accuracy: 0.4044

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.3490 - accuracy: 0.4057

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.3463 - accuracy: 0.4077

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.3401 - accuracy: 0.4122

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.3347 - accuracy: 0.4146

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.3310 - accuracy: 0.4163

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.3341 - accuracy: 0.4149

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.3330 - accuracy: 0.4165

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.3280 - accuracy: 0.4192

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.3256 - accuracy: 0.4218

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.3183 - accuracy: 0.4260

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.3201 - accuracy: 0.4279

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.3184 - accuracy: 0.4291

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.3144 - accuracy: 0.4324

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.3115 - accuracy: 0.4351

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.3071 - accuracy: 0.4383

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.3007 - accuracy: 0.4408

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.2987 - accuracy: 0.4427

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.2959 - accuracy: 0.4431

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.2951 - accuracy: 0.4435

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.2936 - accuracy: 0.4444

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.2929 - accuracy: 0.4452

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.2920 - accuracy: 0.4451

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.2921 - accuracy: 0.4436

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.2909 - accuracy: 0.4431

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.2896 - accuracy: 0.4430

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.2849 - accuracy: 0.4451

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.2814 - accuracy: 0.4476

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.2803 - accuracy: 0.4475

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.2784 - accuracy: 0.4498

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.2785 - accuracy: 0.4517

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.2747 - accuracy: 0.4540

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.2725 - accuracy: 0.4562

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.2737 - accuracy: 0.4563

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.2718 - accuracy: 0.4573

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.2692 - accuracy: 0.4601

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.2706 - accuracy: 0.4602

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.2688 - accuracy: 0.4607

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.2655 - accuracy: 0.4616

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.2600 - accuracy: 0.4642

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.2564 - accuracy: 0.4661

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.2530 - accuracy: 0.4683

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.2518 - accuracy: 0.4687

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.2480 - accuracy: 0.4708

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.2475 - accuracy: 0.4711

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.2469 - accuracy: 0.4721

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.2436 - accuracy: 0.4741

.. parsed-literal::

    2024-04-10 00:36:23.967428: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-04-10 00:36:23.967697: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.2436 - accuracy: 0.4741 - val_loss: 1.0610 - val_accuracy: 0.5668


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9926 - accuracy: 0.4688

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9464 - accuracy: 0.5625

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9667 - accuracy: 0.5625

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9745 - accuracy: 0.5703

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0252 - accuracy: 0.5688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.0164 - accuracy: 0.5833

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0330 - accuracy: 0.5804

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0636 - accuracy: 0.5625

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0733 - accuracy: 0.5556

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0681 - accuracy: 0.5656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0437 - accuracy: 0.5739

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0481 - accuracy: 0.5677

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0520 - accuracy: 0.5673

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0580 - accuracy: 0.5670

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0621 - accuracy: 0.5667

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0553 - accuracy: 0.5684

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0413 - accuracy: 0.5790

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0394 - accuracy: 0.5868

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0310 - accuracy: 0.5888

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0411 - accuracy: 0.5875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0440 - accuracy: 0.5923

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0432 - accuracy: 0.5881

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0472 - accuracy: 0.5883

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0372 - accuracy: 0.5964

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0427 - accuracy: 0.5938

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0379 - accuracy: 0.5950

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0399 - accuracy: 0.5949

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0429 - accuracy: 0.5926

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0483 - accuracy: 0.5927

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0407 - accuracy: 0.5948

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0384 - accuracy: 0.5958

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0407 - accuracy: 0.5947

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0350 - accuracy: 0.5956

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0305 - accuracy: 0.5974

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0283 - accuracy: 0.5982

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0214 - accuracy: 0.6016

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0179 - accuracy: 0.6039

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0244 - accuracy: 0.5987

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0233 - accuracy: 0.5994

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0187 - accuracy: 0.6008

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0227 - accuracy: 0.5998

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0249 - accuracy: 0.5997

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0297 - accuracy: 0.5996

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0265 - accuracy: 0.5994

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0286 - accuracy: 0.5965

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0298 - accuracy: 0.5944

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0290 - accuracy: 0.5944

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0316 - accuracy: 0.5924

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0315 - accuracy: 0.5931

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0305 - accuracy: 0.5944

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0312 - accuracy: 0.5925

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0307 - accuracy: 0.5931

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0293 - accuracy: 0.5949

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0261 - accuracy: 0.5961

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0246 - accuracy: 0.5966

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0249 - accuracy: 0.5971

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0248 - accuracy: 0.5959

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0259 - accuracy: 0.5975

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0185 - accuracy: 0.6012

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0163 - accuracy: 0.6016

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0146 - accuracy: 0.6019

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0149 - accuracy: 0.6013

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0140 - accuracy: 0.6017

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0102 - accuracy: 0.6021

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0070 - accuracy: 0.6029

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0057 - accuracy: 0.6037

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0025 - accuracy: 0.6054

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0081 - accuracy: 0.6048

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0080 - accuracy: 0.6046

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0045 - accuracy: 0.6067

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0022 - accuracy: 0.6061

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9999 - accuracy: 0.6072

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0013 - accuracy: 0.6074

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9992 - accuracy: 0.6085

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0003 - accuracy: 0.6083

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9987 - accuracy: 0.6094

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9990 - accuracy: 0.6092

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9975 - accuracy: 0.6079

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9996 - accuracy: 0.6066

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9980 - accuracy: 0.6076

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9979 - accuracy: 0.6086

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9977 - accuracy: 0.6099

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9967 - accuracy: 0.6104

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9955 - accuracy: 0.6106

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9995 - accuracy: 0.6104

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0005 - accuracy: 0.6099

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0063 - accuracy: 0.6079

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0066 - accuracy: 0.6070

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0065 - accuracy: 0.6072

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0063 - accuracy: 0.6074

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0093 - accuracy: 0.6073

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0093 - accuracy: 0.6073 - val_loss: 1.1116 - val_accuracy: 0.5341


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9978 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0612 - accuracy: 0.5781

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 4s - loss: 1.0151 - accuracy: 0.5833

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.9507 - accuracy: 0.6184

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9301 - accuracy: 0.6304

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9061 - accuracy: 0.6481

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8961 - accuracy: 0.6573

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9187 - accuracy: 0.6536

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9102 - accuracy: 0.6635

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9060 - accuracy: 0.6686

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9207 - accuracy: 0.6649

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9224 - accuracy: 0.6618

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9218 - accuracy: 0.6591

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9255 - accuracy: 0.6547

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9191 - accuracy: 0.6627

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9139 - accuracy: 0.6623

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9208 - accuracy: 0.6549

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9245 - accuracy: 0.6533

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9257 - accuracy: 0.6551

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9187 - accuracy: 0.6581

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9227 - accuracy: 0.6638

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9152 - accuracy: 0.6676

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9191 - accuracy: 0.6645

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9160 - accuracy: 0.6641

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9209 - accuracy: 0.6590

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9193 - accuracy: 0.6577

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9148 - accuracy: 0.6599

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9160 - accuracy: 0.6587

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9114 - accuracy: 0.6597

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9214 - accuracy: 0.6545

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9185 - accuracy: 0.6555

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9151 - accuracy: 0.6555

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9131 - accuracy: 0.6556

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9155 - accuracy: 0.6574

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9216 - accuracy: 0.6547

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9181 - accuracy: 0.6565

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9247 - accuracy: 0.6531

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9216 - accuracy: 0.6548

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9224 - accuracy: 0.6557

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9240 - accuracy: 0.6526

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9289 - accuracy: 0.6512

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9259 - accuracy: 0.6528

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9231 - accuracy: 0.6529

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9244 - accuracy: 0.6508

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9200 - accuracy: 0.6516

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9193 - accuracy: 0.6524

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9147 - accuracy: 0.6531

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9191 - accuracy: 0.6506

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9164 - accuracy: 0.6514

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9151 - accuracy: 0.6515

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9137 - accuracy: 0.6528

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9123 - accuracy: 0.6523

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9113 - accuracy: 0.6535

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9111 - accuracy: 0.6547

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9091 - accuracy: 0.6564

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9065 - accuracy: 0.6575

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9113 - accuracy: 0.6564

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9133 - accuracy: 0.6559

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9106 - accuracy: 0.6569

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9103 - accuracy: 0.6569

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9105 - accuracy: 0.6564

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9110 - accuracy: 0.6554

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9201 - accuracy: 0.6529

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9225 - accuracy: 0.6525

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9206 - accuracy: 0.6530

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9166 - accuracy: 0.6545

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9158 - accuracy: 0.6545

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9154 - accuracy: 0.6550

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9146 - accuracy: 0.6555

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9139 - accuracy: 0.6559

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9158 - accuracy: 0.6551

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9158 - accuracy: 0.6542

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9123 - accuracy: 0.6555

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9084 - accuracy: 0.6568

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9108 - accuracy: 0.6555

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9087 - accuracy: 0.6564

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9082 - accuracy: 0.6564

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9080 - accuracy: 0.6560

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9059 - accuracy: 0.6563

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9053 - accuracy: 0.6563

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9040 - accuracy: 0.6571

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9024 - accuracy: 0.6575

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9047 - accuracy: 0.6556

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9031 - accuracy: 0.6556

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9046 - accuracy: 0.6542

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9043 - accuracy: 0.6538

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9061 - accuracy: 0.6531

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9080 - accuracy: 0.6518

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9105 - accuracy: 0.6497

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9101 - accuracy: 0.6501

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9092 - accuracy: 0.6522

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9092 - accuracy: 0.6522 - val_loss: 0.9226 - val_accuracy: 0.6349


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9628 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9099 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9602 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8620 - accuracy: 0.6797

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8884 - accuracy: 0.6625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8812 - accuracy: 0.6615

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.8419 - accuracy: 0.6696

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8568 - accuracy: 0.6758

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8463 - accuracy: 0.6736

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8360 - accuracy: 0.6844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8493 - accuracy: 0.6733

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8491 - accuracy: 0.6745

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8531 - accuracy: 0.6707

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8614 - accuracy: 0.6674

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8602 - accuracy: 0.6646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8544 - accuracy: 0.6738

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8451 - accuracy: 0.6801

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8434 - accuracy: 0.6806

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8549 - accuracy: 0.6743

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8483 - accuracy: 0.6734

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8415 - accuracy: 0.6786

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8404 - accuracy: 0.6776

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8589 - accuracy: 0.6780

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8696 - accuracy: 0.6732

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8621 - accuracy: 0.6762

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8740 - accuracy: 0.6683

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8806 - accuracy: 0.6655

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8829 - accuracy: 0.6652

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8772 - accuracy: 0.6681

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8851 - accuracy: 0.6667

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9013 - accuracy: 0.6603

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8933 - accuracy: 0.6641

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8937 - accuracy: 0.6629

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8927 - accuracy: 0.6618

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8863 - accuracy: 0.6652

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8830 - accuracy: 0.6675

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8817 - accuracy: 0.6681

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8786 - accuracy: 0.6669

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8833 - accuracy: 0.6651

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8849 - accuracy: 0.6625

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8943 - accuracy: 0.6593

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8909 - accuracy: 0.6607

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8916 - accuracy: 0.6606

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8890 - accuracy: 0.6634

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8873 - accuracy: 0.6625

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8885 - accuracy: 0.6630

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8854 - accuracy: 0.6656

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8885 - accuracy: 0.6628

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8878 - accuracy: 0.6633

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8860 - accuracy: 0.6650

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8792 - accuracy: 0.6673

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8757 - accuracy: 0.6700

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8815 - accuracy: 0.6686

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8803 - accuracy: 0.6684

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8800 - accuracy: 0.6676

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8757 - accuracy: 0.6691

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8740 - accuracy: 0.6694

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8716 - accuracy: 0.6697

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8688 - accuracy: 0.6710

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8689 - accuracy: 0.6703

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8680 - accuracy: 0.6700

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8667 - accuracy: 0.6708

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8649 - accuracy: 0.6716

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8680 - accuracy: 0.6708

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8644 - accuracy: 0.6721

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8676 - accuracy: 0.6704

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8663 - accuracy: 0.6711

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8619 - accuracy: 0.6732

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8635 - accuracy: 0.6725

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8595 - accuracy: 0.6740

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8569 - accuracy: 0.6751

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8525 - accuracy: 0.6765

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8528 - accuracy: 0.6771

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8530 - accuracy: 0.6760

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8525 - accuracy: 0.6770

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8499 - accuracy: 0.6787

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8529 - accuracy: 0.6777

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8526 - accuracy: 0.6762

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8508 - accuracy: 0.6775

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8494 - accuracy: 0.6788

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8467 - accuracy: 0.6797

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8442 - accuracy: 0.6809

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8442 - accuracy: 0.6806

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8419 - accuracy: 0.6810

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8420 - accuracy: 0.6811

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8423 - accuracy: 0.6801

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8458 - accuracy: 0.6788

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8447 - accuracy: 0.6785

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8420 - accuracy: 0.6804

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8398 - accuracy: 0.6808

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8403 - accuracy: 0.6802

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8403 - accuracy: 0.6802 - val_loss: 0.8762 - val_accuracy: 0.6594


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8061 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7852 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8109 - accuracy: 0.6354

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8089 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8564 - accuracy: 0.6375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8358 - accuracy: 0.6458

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8498 - accuracy: 0.6384

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8212 - accuracy: 0.6562

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8133 - accuracy: 0.6562

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8092 - accuracy: 0.6562

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8179 - accuracy: 0.6506

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8066 - accuracy: 0.6589

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7964 - accuracy: 0.6635

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7924 - accuracy: 0.6763

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7909 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7857 - accuracy: 0.6914

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7814 - accuracy: 0.6967

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7756 - accuracy: 0.7031

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7749 - accuracy: 0.7039

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7638 - accuracy: 0.7094

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7787 - accuracy: 0.7009

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7934 - accuracy: 0.6974

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7863 - accuracy: 0.7011

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7759 - accuracy: 0.7018

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7742 - accuracy: 0.7038

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7693 - accuracy: 0.7043

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7648 - accuracy: 0.7095

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7691 - accuracy: 0.7054

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7742 - accuracy: 0.7015

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7667 - accuracy: 0.7031

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7733 - accuracy: 0.6996

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7726 - accuracy: 0.6992

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7711 - accuracy: 0.7008

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7738 - accuracy: 0.6994

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7718 - accuracy: 0.7018

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7673 - accuracy: 0.7049

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7727 - accuracy: 0.7035

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7726 - accuracy: 0.7039

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7686 - accuracy: 0.7051

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7637 - accuracy: 0.7078

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7621 - accuracy: 0.7088

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7598 - accuracy: 0.7098

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7565 - accuracy: 0.7108

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7562 - accuracy: 0.7102

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7509 - accuracy: 0.7104

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7564 - accuracy: 0.7072

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7580 - accuracy: 0.7055

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7547 - accuracy: 0.7077

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7539 - accuracy: 0.7066

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7488 - accuracy: 0.7088

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7442 - accuracy: 0.7114

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7488 - accuracy: 0.7097

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7507 - accuracy: 0.7075

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7483 - accuracy: 0.7083

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7453 - accuracy: 0.7097

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7441 - accuracy: 0.7104

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7405 - accuracy: 0.7122

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7442 - accuracy: 0.7117

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7461 - accuracy: 0.7108

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7475 - accuracy: 0.7109

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7467 - accuracy: 0.7116

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7452 - accuracy: 0.7117

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7473 - accuracy: 0.7098

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7465 - accuracy: 0.7109

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7498 - accuracy: 0.7087

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7505 - accuracy: 0.7074

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7527 - accuracy: 0.7076

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7516 - accuracy: 0.7077

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7562 - accuracy: 0.7052

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7593 - accuracy: 0.7045

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7590 - accuracy: 0.7051

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7613 - accuracy: 0.7032

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7624 - accuracy: 0.7025

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7692 - accuracy: 0.6994

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7701 - accuracy: 0.6988

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7676 - accuracy: 0.7007

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7684 - accuracy: 0.7006

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7674 - accuracy: 0.7016

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7662 - accuracy: 0.7018

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7654 - accuracy: 0.7012

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7656 - accuracy: 0.7015

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7651 - accuracy: 0.7005

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7652 - accuracy: 0.6996

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7656 - accuracy: 0.6995

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7641 - accuracy: 0.7004

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7669 - accuracy: 0.6992

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7660 - accuracy: 0.6991

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7641 - accuracy: 0.7007

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7660 - accuracy: 0.6999

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7650 - accuracy: 0.6997

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7640 - accuracy: 0.6999

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7640 - accuracy: 0.6999 - val_loss: 0.8612 - val_accuracy: 0.6662


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9668 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9202 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8439 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8501 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7736 - accuracy: 0.6750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7435 - accuracy: 0.6979

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7173 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7068 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6919 - accuracy: 0.7257

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6877 - accuracy: 0.7281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6728 - accuracy: 0.7386

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6814 - accuracy: 0.7318

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6736 - accuracy: 0.7380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6763 - accuracy: 0.7344

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6788 - accuracy: 0.7292

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6629 - accuracy: 0.7363

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6734 - accuracy: 0.7298

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6877 - accuracy: 0.7257

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6886 - accuracy: 0.7270

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6861 - accuracy: 0.7250

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6835 - accuracy: 0.7247

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6882 - accuracy: 0.7259

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6803 - accuracy: 0.7296

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6728 - accuracy: 0.7318

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6708 - accuracy: 0.7312

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6645 - accuracy: 0.7320

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6628 - accuracy: 0.7326

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6705 - accuracy: 0.7310

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6644 - accuracy: 0.7349

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6804 - accuracy: 0.7302

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6797 - accuracy: 0.7298

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6797 - accuracy: 0.7285

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6776 - accuracy: 0.7311

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6765 - accuracy: 0.7307

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6751 - accuracy: 0.7321

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6851 - accuracy: 0.7248

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6839 - accuracy: 0.7255

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6885 - accuracy: 0.7212

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6883 - accuracy: 0.7220

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6922 - accuracy: 0.7203

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6967 - accuracy: 0.7180

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6993 - accuracy: 0.7165

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7015 - accuracy: 0.7158

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7032 - accuracy: 0.7159

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7045 - accuracy: 0.7160

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7027 - accuracy: 0.7167

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6980 - accuracy: 0.7188

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6970 - accuracy: 0.7207

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6979 - accuracy: 0.7213

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6955 - accuracy: 0.7219

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6938 - accuracy: 0.7218

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6925 - accuracy: 0.7236

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6887 - accuracy: 0.7264

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6881 - accuracy: 0.7280

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6863 - accuracy: 0.7290

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6899 - accuracy: 0.7277

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6909 - accuracy: 0.7281

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6890 - accuracy: 0.7290

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6874 - accuracy: 0.7304

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6883 - accuracy: 0.7297

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6896 - accuracy: 0.7285

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6937 - accuracy: 0.7263

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6911 - accuracy: 0.7272

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6948 - accuracy: 0.7261

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6942 - accuracy: 0.7250

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6977 - accuracy: 0.7249

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6995 - accuracy: 0.7239

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7000 - accuracy: 0.7233

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7040 - accuracy: 0.7215

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7078 - accuracy: 0.7205

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7083 - accuracy: 0.7201

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7095 - accuracy: 0.7192

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7064 - accuracy: 0.7213

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7051 - accuracy: 0.7226

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7045 - accuracy: 0.7233

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7096 - accuracy: 0.7216

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7098 - accuracy: 0.7220

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7113 - accuracy: 0.7208

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7124 - accuracy: 0.7215

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7109 - accuracy: 0.7219

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7110 - accuracy: 0.7222

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7114 - accuracy: 0.7214

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7115 - accuracy: 0.7214

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7108 - accuracy: 0.7210

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7131 - accuracy: 0.7191

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7124 - accuracy: 0.7188

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7142 - accuracy: 0.7180

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7136 - accuracy: 0.7180

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7134 - accuracy: 0.7184

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7121 - accuracy: 0.7184

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7095 - accuracy: 0.7190

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7095 - accuracy: 0.7190 - val_loss: 0.7873 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7619 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7227 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7356 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7238 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7504 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7324 - accuracy: 0.7292

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7237 - accuracy: 0.7188

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6999 - accuracy: 0.7266

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7017 - accuracy: 0.7222

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6875 - accuracy: 0.7312

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6986 - accuracy: 0.7330

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7072 - accuracy: 0.7292

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6977 - accuracy: 0.7332

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7091 - accuracy: 0.7277

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7012 - accuracy: 0.7312

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6881 - accuracy: 0.7344

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6789 - accuracy: 0.7371

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6681 - accuracy: 0.7431

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6670 - accuracy: 0.7484

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6606 - accuracy: 0.7500

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6655 - accuracy: 0.7470

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6679 - accuracy: 0.7472

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6726 - accuracy: 0.7500

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6847 - accuracy: 0.7461

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6880 - accuracy: 0.7450

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6829 - accuracy: 0.7488

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6934 - accuracy: 0.7431

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6930 - accuracy: 0.7433

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6943 - accuracy: 0.7446

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6947 - accuracy: 0.7417

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6940 - accuracy: 0.7369

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7071 - accuracy: 0.7305

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7036 - accuracy: 0.7311

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7062 - accuracy: 0.7307

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7002 - accuracy: 0.7321

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6960 - accuracy: 0.7344

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6913 - accuracy: 0.7373

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6982 - accuracy: 0.7352

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6926 - accuracy: 0.7372

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6963 - accuracy: 0.7336

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6964 - accuracy: 0.7332

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6940 - accuracy: 0.7366

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6953 - accuracy: 0.7369

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7053 - accuracy: 0.7337

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7056 - accuracy: 0.7333

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7028 - accuracy: 0.7330

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7003 - accuracy: 0.7354

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7033 - accuracy: 0.7350

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7022 - accuracy: 0.7353

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7006 - accuracy: 0.7369

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7026 - accuracy: 0.7365

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7029 - accuracy: 0.7362

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7035 - accuracy: 0.7353

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7029 - accuracy: 0.7344

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7015 - accuracy: 0.7347

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6992 - accuracy: 0.7366

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6964 - accuracy: 0.7368

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6947 - accuracy: 0.7371

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6950 - accuracy: 0.7373

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6898 - accuracy: 0.7385

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6898 - accuracy: 0.7387

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6882 - accuracy: 0.7389

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6894 - accuracy: 0.7391

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6905 - accuracy: 0.7383

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6935 - accuracy: 0.7380

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6956 - accuracy: 0.7367

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6964 - accuracy: 0.7360

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6953 - accuracy: 0.7353

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7010 - accuracy: 0.7337

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6997 - accuracy: 0.7339

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6987 - accuracy: 0.7346

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6966 - accuracy: 0.7357

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6961 - accuracy: 0.7359

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6935 - accuracy: 0.7365

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6911 - accuracy: 0.7375

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6927 - accuracy: 0.7364

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6929 - accuracy: 0.7362

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6907 - accuracy: 0.7368

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6903 - accuracy: 0.7377

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6916 - accuracy: 0.7375

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6927 - accuracy: 0.7373

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6911 - accuracy: 0.7378

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6902 - accuracy: 0.7380

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6926 - accuracy: 0.7374

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6912 - accuracy: 0.7379

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6923 - accuracy: 0.7373

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6907 - accuracy: 0.7385

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6891 - accuracy: 0.7384

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6877 - accuracy: 0.7392

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6903 - accuracy: 0.7376

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6908 - accuracy: 0.7377

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6908 - accuracy: 0.7377 - val_loss: 0.7734 - val_accuracy: 0.6975


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6553 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6930 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6298 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6024 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6189 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6363 - accuracy: 0.7708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7136 - accuracy: 0.7277

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7383 - accuracy: 0.7305

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7386 - accuracy: 0.7292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7261 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7043 - accuracy: 0.7472

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7033 - accuracy: 0.7448

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6981 - accuracy: 0.7476

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6911 - accuracy: 0.7478

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6973 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7257 - accuracy: 0.7285

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7163 - accuracy: 0.7316

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7062 - accuracy: 0.7361

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7008 - accuracy: 0.7319

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6931 - accuracy: 0.7359

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6888 - accuracy: 0.7351

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6827 - accuracy: 0.7344

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6873 - accuracy: 0.7323

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6864 - accuracy: 0.7298

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6866 - accuracy: 0.7294

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6803 - accuracy: 0.7336

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6783 - accuracy: 0.7365

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6736 - accuracy: 0.7359

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6828 - accuracy: 0.7321

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6909 - accuracy: 0.7287

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6973 - accuracy: 0.7244

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6989 - accuracy: 0.7233

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6927 - accuracy: 0.7259

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6901 - accuracy: 0.7266

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6874 - accuracy: 0.7290

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6845 - accuracy: 0.7304

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6859 - accuracy: 0.7310

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6845 - accuracy: 0.7323

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6788 - accuracy: 0.7358

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6765 - accuracy: 0.7354

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6721 - accuracy: 0.7388

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6734 - accuracy: 0.7368

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6679 - accuracy: 0.7393

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6666 - accuracy: 0.7402

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6647 - accuracy: 0.7404

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6646 - accuracy: 0.7386

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6609 - accuracy: 0.7395

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6672 - accuracy: 0.7365

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6683 - accuracy: 0.7356

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6656 - accuracy: 0.7365

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6601 - accuracy: 0.7385

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6565 - accuracy: 0.7405

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6549 - accuracy: 0.7395

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6558 - accuracy: 0.7392

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6530 - accuracy: 0.7393

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6505 - accuracy: 0.7401

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6521 - accuracy: 0.7397

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6536 - accuracy: 0.7388

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6540 - accuracy: 0.7380

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6514 - accuracy: 0.7397

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6485 - accuracy: 0.7404

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6510 - accuracy: 0.7415

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6514 - accuracy: 0.7417

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6516 - accuracy: 0.7418

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6541 - accuracy: 0.7419

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6563 - accuracy: 0.7425

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6596 - accuracy: 0.7422

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6564 - accuracy: 0.7427

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6565 - accuracy: 0.7424

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6581 - accuracy: 0.7407

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6576 - accuracy: 0.7413

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6559 - accuracy: 0.7418

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6576 - accuracy: 0.7415

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6603 - accuracy: 0.7408

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6593 - accuracy: 0.7422

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6579 - accuracy: 0.7435

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6572 - accuracy: 0.7440

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6548 - accuracy: 0.7448

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6558 - accuracy: 0.7433

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6549 - accuracy: 0.7442

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6570 - accuracy: 0.7427

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6579 - accuracy: 0.7428

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6570 - accuracy: 0.7429

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6564 - accuracy: 0.7441

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6540 - accuracy: 0.7453

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6577 - accuracy: 0.7435

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6578 - accuracy: 0.7429

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6583 - accuracy: 0.7433

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6572 - accuracy: 0.7437

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6568 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6556 - accuracy: 0.7439

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6556 - accuracy: 0.7439 - val_loss: 0.8480 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7106 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6278 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6246 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5777 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6274 - accuracy: 0.7750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6061 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6002 - accuracy: 0.7857

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6605 - accuracy: 0.7607

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6775 - accuracy: 0.7596

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6786 - accuracy: 0.7616

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6878 - accuracy: 0.7633

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6842 - accuracy: 0.7623

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6684 - accuracy: 0.7682

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6657 - accuracy: 0.7585

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6649 - accuracy: 0.7579

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6571 - accuracy: 0.7612

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6664 - accuracy: 0.7553

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6702 - accuracy: 0.7500

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6679 - accuracy: 0.7532

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6730 - accuracy: 0.7500

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.6621 - accuracy: 0.7557

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6661 - accuracy: 0.7527

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6696 - accuracy: 0.7461

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6663 - accuracy: 0.7462

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6777 - accuracy: 0.7500

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6741 - accuracy: 0.7512

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6665 - accuracy: 0.7523

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6654 - accuracy: 0.7511

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6639 - accuracy: 0.7542

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6558 - accuracy: 0.7581

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6499 - accuracy: 0.7598

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6529 - accuracy: 0.7567

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6515 - accuracy: 0.7565

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6487 - accuracy: 0.7572

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6431 - accuracy: 0.7596

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6507 - accuracy: 0.7585

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6477 - accuracy: 0.7608

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6524 - accuracy: 0.7581

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6481 - accuracy: 0.7586

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6462 - accuracy: 0.7577

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6484 - accuracy: 0.7567

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6509 - accuracy: 0.7558

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6608 - accuracy: 0.7507

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6568 - accuracy: 0.7528

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6549 - accuracy: 0.7541

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6597 - accuracy: 0.7513

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6602 - accuracy: 0.7513

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6602 - accuracy: 0.7519

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6572 - accuracy: 0.7531

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6587 - accuracy: 0.7531

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6566 - accuracy: 0.7542

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6574 - accuracy: 0.7536

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6594 - accuracy: 0.7523

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6572 - accuracy: 0.7534

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6545 - accuracy: 0.7550

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6531 - accuracy: 0.7555

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6522 - accuracy: 0.7570

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6486 - accuracy: 0.7580

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6487 - accuracy: 0.7563

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6475 - accuracy: 0.7557

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6472 - accuracy: 0.7556

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6492 - accuracy: 0.7555

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6466 - accuracy: 0.7574

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6436 - accuracy: 0.7587

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6403 - accuracy: 0.7605

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6389 - accuracy: 0.7603

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6366 - accuracy: 0.7606

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6332 - accuracy: 0.7618

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6381 - accuracy: 0.7603

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6382 - accuracy: 0.7584

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6373 - accuracy: 0.7574

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6353 - accuracy: 0.7586

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6372 - accuracy: 0.7585

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6374 - accuracy: 0.7579

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6362 - accuracy: 0.7587

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6345 - accuracy: 0.7598

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6335 - accuracy: 0.7600

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6322 - accuracy: 0.7615

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6304 - accuracy: 0.7621

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6297 - accuracy: 0.7624

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6289 - accuracy: 0.7630

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6291 - accuracy: 0.7632

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6273 - accuracy: 0.7646

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6254 - accuracy: 0.7659

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6271 - accuracy: 0.7653

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6241 - accuracy: 0.7669

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6218 - accuracy: 0.7685

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6204 - accuracy: 0.7694

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6257 - accuracy: 0.7671

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6248 - accuracy: 0.7672

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6243 - accuracy: 0.7681

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6243 - accuracy: 0.7681 - val_loss: 0.8191 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8403 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7135 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6704 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6850 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7091 - accuracy: 0.7375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6714 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6865 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6866 - accuracy: 0.7383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6926 - accuracy: 0.7431

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6884 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6822 - accuracy: 0.7557

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6843 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6627 - accuracy: 0.7596

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6499 - accuracy: 0.7634

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6314 - accuracy: 0.7729

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6323 - accuracy: 0.7754

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6352 - accuracy: 0.7721

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6184 - accuracy: 0.7778

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6277 - accuracy: 0.7763

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6343 - accuracy: 0.7703

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6324 - accuracy: 0.7693

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6423 - accuracy: 0.7628

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6481 - accuracy: 0.7554

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6488 - accuracy: 0.7539

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6422 - accuracy: 0.7588

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6425 - accuracy: 0.7584

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6358 - accuracy: 0.7639

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6324 - accuracy: 0.7634

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6252 - accuracy: 0.7672

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6221 - accuracy: 0.7677

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6208 - accuracy: 0.7661

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6192 - accuracy: 0.7666

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6114 - accuracy: 0.7708

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6065 - accuracy: 0.7721

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6104 - accuracy: 0.7696

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6110 - accuracy: 0.7674

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6069 - accuracy: 0.7703

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6074 - accuracy: 0.7706

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6086 - accuracy: 0.7700

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6052 - accuracy: 0.7703

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6041 - accuracy: 0.7721

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6044 - accuracy: 0.7716

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6134 - accuracy: 0.7689

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6151 - accuracy: 0.7649

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6141 - accuracy: 0.7653

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6091 - accuracy: 0.7677

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6099 - accuracy: 0.7673

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6129 - accuracy: 0.7656

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6109 - accuracy: 0.7672

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6108 - accuracy: 0.7650

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6083 - accuracy: 0.7672

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6100 - accuracy: 0.7668

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6138 - accuracy: 0.7653

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6169 - accuracy: 0.7650

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6175 - accuracy: 0.7665

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6194 - accuracy: 0.7651

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6168 - accuracy: 0.7664

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6182 - accuracy: 0.7645

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6181 - accuracy: 0.7648

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6167 - accuracy: 0.7661

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6123 - accuracy: 0.7674

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6083 - accuracy: 0.7707

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6047 - accuracy: 0.7713

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6035 - accuracy: 0.7734

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6010 - accuracy: 0.7745

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6059 - accuracy: 0.7708

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6039 - accuracy: 0.7719

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6032 - accuracy: 0.7725

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6040 - accuracy: 0.7722

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6091 - accuracy: 0.7705

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6109 - accuracy: 0.7689

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6136 - accuracy: 0.7682

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6135 - accuracy: 0.7678

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6121 - accuracy: 0.7684

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6099 - accuracy: 0.7702

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6167 - accuracy: 0.7671

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6160 - accuracy: 0.7669

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6145 - accuracy: 0.7671

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6198 - accuracy: 0.7645

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6219 - accuracy: 0.7635

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6218 - accuracy: 0.7634

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6223 - accuracy: 0.7621

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6231 - accuracy: 0.7623

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6224 - accuracy: 0.7618

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6206 - accuracy: 0.7628

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6206 - accuracy: 0.7622

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6229 - accuracy: 0.7625

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6232 - accuracy: 0.7616

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6219 - accuracy: 0.7625

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6210 - accuracy: 0.7624

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6232 - accuracy: 0.7609

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6232 - accuracy: 0.7609 - val_loss: 0.7610 - val_accuracy: 0.7289


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6637 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5223 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5329 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4862 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4967 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5279 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5257 - accuracy: 0.7812

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5301 - accuracy: 0.7812

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5320 - accuracy: 0.7847

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5286 - accuracy: 0.7820

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5265 - accuracy: 0.7819

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5450 - accuracy: 0.7696

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5446 - accuracy: 0.7705

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5457 - accuracy: 0.7712

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5386 - accuracy: 0.7778

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5292 - accuracy: 0.7817

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5353 - accuracy: 0.7817

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5649 - accuracy: 0.7717

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5697 - accuracy: 0.7690

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5804 - accuracy: 0.7636

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5730 - accuracy: 0.7687

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5707 - accuracy: 0.7706

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7671

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5730 - accuracy: 0.7702

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5683 - accuracy: 0.7718

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5617 - accuracy: 0.7745

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5628 - accuracy: 0.7725

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5656 - accuracy: 0.7696

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5631 - accuracy: 0.7700

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5562 - accuracy: 0.7724

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5484 - accuracy: 0.7776

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5429 - accuracy: 0.7805

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5418 - accuracy: 0.7824

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5418 - accuracy: 0.7824

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5445 - accuracy: 0.7806

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5423 - accuracy: 0.7823

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5437 - accuracy: 0.7823

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5459 - accuracy: 0.7823

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5443 - accuracy: 0.7838

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5413 - accuracy: 0.7853

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5485 - accuracy: 0.7829

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5487 - accuracy: 0.7829

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5489 - accuracy: 0.7829

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5465 - accuracy: 0.7828

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5441 - accuracy: 0.7842

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5431 - accuracy: 0.7841

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5442 - accuracy: 0.7840

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5412 - accuracy: 0.7846

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5455 - accuracy: 0.7833

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5438 - accuracy: 0.7839

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5451 - accuracy: 0.7838

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5463 - accuracy: 0.7826

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5503 - accuracy: 0.7802

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5515 - accuracy: 0.7803

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5510 - accuracy: 0.7803

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5555 - accuracy: 0.7775

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5544 - accuracy: 0.7781

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5557 - accuracy: 0.7782

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5567 - accuracy: 0.7782

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5540 - accuracy: 0.7798

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5538 - accuracy: 0.7799

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5537 - accuracy: 0.7804

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5531 - accuracy: 0.7809

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5513 - accuracy: 0.7814

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5487 - accuracy: 0.7823

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5479 - accuracy: 0.7823

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5494 - accuracy: 0.7809

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5495 - accuracy: 0.7814

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5508 - accuracy: 0.7809

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5502 - accuracy: 0.7814

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5568 - accuracy: 0.7801

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5539 - accuracy: 0.7814

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5558 - accuracy: 0.7809

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5533 - accuracy: 0.7818

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5542 - accuracy: 0.7814

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5523 - accuracy: 0.7822

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5547 - accuracy: 0.7805

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5583 - accuracy: 0.7802

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5597 - accuracy: 0.7790

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5621 - accuracy: 0.7783

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5618 - accuracy: 0.7787

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5613 - accuracy: 0.7798

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5623 - accuracy: 0.7791

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5653 - accuracy: 0.7777

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5642 - accuracy: 0.7784

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5638 - accuracy: 0.7795

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5633 - accuracy: 0.7799

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5633 - accuracy: 0.7789

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5639 - accuracy: 0.7786

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5642 - accuracy: 0.7782

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5624 - accuracy: 0.7793

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5624 - accuracy: 0.7793 - val_loss: 0.7041 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5774 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6539 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6227 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5784 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5343 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5418 - accuracy: 0.7865

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5335 - accuracy: 0.7857

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5468 - accuracy: 0.7857

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5308 - accuracy: 0.7917

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5244 - accuracy: 0.7907

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5128 - accuracy: 0.7952

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5203 - accuracy: 0.7941

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5227 - accuracy: 0.7955

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5139 - accuracy: 0.8030

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5174 - accuracy: 0.8036

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5157 - accuracy: 0.8041

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5425 - accuracy: 0.7923

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5456 - accuracy: 0.7933

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5425 - accuracy: 0.7943

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5363 - accuracy: 0.7967

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5314 - accuracy: 0.7960

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5374 - accuracy: 0.7953

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5331 - accuracy: 0.7961

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5385 - accuracy: 0.7942

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5381 - accuracy: 0.7913

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5386 - accuracy: 0.7909

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5329 - accuracy: 0.7939

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5394 - accuracy: 0.7913

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5455 - accuracy: 0.7899

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5422 - accuracy: 0.7917

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5422 - accuracy: 0.7923

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5413 - accuracy: 0.7910

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5371 - accuracy: 0.7926

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5363 - accuracy: 0.7914

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5380 - accuracy: 0.7911

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5392 - accuracy: 0.7908

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5361 - accuracy: 0.7939

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5345 - accuracy: 0.7952

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5373 - accuracy: 0.7956

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5388 - accuracy: 0.7983

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5436 - accuracy: 0.7949

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5495 - accuracy: 0.7939

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5483 - accuracy: 0.7950

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5474 - accuracy: 0.7961

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5487 - accuracy: 0.7958

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5433 - accuracy: 0.7988

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5422 - accuracy: 0.7991

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5363 - accuracy: 0.8013

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5374 - accuracy: 0.8028

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5379 - accuracy: 0.8023

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5355 - accuracy: 0.8037

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5373 - accuracy: 0.8021

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5344 - accuracy: 0.8023

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5374 - accuracy: 0.8014

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5387 - accuracy: 0.8004

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5356 - accuracy: 0.8018

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5343 - accuracy: 0.8019

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5362 - accuracy: 0.8016

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5395 - accuracy: 0.7997

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5402 - accuracy: 0.7994

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5374 - accuracy: 0.8011

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5382 - accuracy: 0.8013

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5364 - accuracy: 0.8025

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5350 - accuracy: 0.8031

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5313 - accuracy: 0.8042

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5321 - accuracy: 0.8038

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5362 - accuracy: 0.8026

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5385 - accuracy: 0.8018

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5389 - accuracy: 0.8015

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5395 - accuracy: 0.8012

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5410 - accuracy: 0.7997

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5388 - accuracy: 0.8007

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5388 - accuracy: 0.8008

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5386 - accuracy: 0.8006

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5383 - accuracy: 0.8003

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5392 - accuracy: 0.7980

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5393 - accuracy: 0.7970

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5422 - accuracy: 0.7968

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5417 - accuracy: 0.7966

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5423 - accuracy: 0.7957

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5471 - accuracy: 0.7936

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5477 - accuracy: 0.7915

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5460 - accuracy: 0.7918

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5436 - accuracy: 0.7928

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5450 - accuracy: 0.7919

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5438 - accuracy: 0.7925

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5446 - accuracy: 0.7924

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5441 - accuracy: 0.7930

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5453 - accuracy: 0.7932

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5451 - accuracy: 0.7934

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5440 - accuracy: 0.7933

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5440 - accuracy: 0.7933 - val_loss: 0.7243 - val_accuracy: 0.7316


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.4342 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4928 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4438 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4853 - accuracy: 0.7969

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4719 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4650 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4772 - accuracy: 0.7946

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4753 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4611 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4624 - accuracy: 0.8062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4830 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4733 - accuracy: 0.8047

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4786 - accuracy: 0.8005

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4822 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4848 - accuracy: 0.7979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4883 - accuracy: 0.7949

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4868 - accuracy: 0.7941

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4879 - accuracy: 0.7917

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4954 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4943 - accuracy: 0.7891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4930 - accuracy: 0.7932

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4897 - accuracy: 0.7955

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.4905 - accuracy: 0.7948

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4926 - accuracy: 0.7943

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4850 - accuracy: 0.7975

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4841 - accuracy: 0.7969

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4887 - accuracy: 0.7951

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4849 - accuracy: 0.7980

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4906 - accuracy: 0.7953

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5043 - accuracy: 0.7896

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5066 - accuracy: 0.7893

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5212 - accuracy: 0.7861

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5221 - accuracy: 0.7879

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5257 - accuracy: 0.7858

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5269 - accuracy: 0.7839

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5321 - accuracy: 0.7812

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5273 - accuracy: 0.7838

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5305 - accuracy: 0.7829

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5410 - accuracy: 0.7788

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5356 - accuracy: 0.7828

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5439 - accuracy: 0.7805

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5488 - accuracy: 0.7798

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5458 - accuracy: 0.7812

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5466 - accuracy: 0.7812

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5471 - accuracy: 0.7792

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5437 - accuracy: 0.7819

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5444 - accuracy: 0.7819

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5397 - accuracy: 0.7852

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5393 - accuracy: 0.7864

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5367 - accuracy: 0.7862

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5342 - accuracy: 0.7886

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5400 - accuracy: 0.7873

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5390 - accuracy: 0.7877

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5343 - accuracy: 0.7899

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5371 - accuracy: 0.7886

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5405 - accuracy: 0.7868

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5410 - accuracy: 0.7878

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5435 - accuracy: 0.7866

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5445 - accuracy: 0.7860

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5425 - accuracy: 0.7859

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5416 - accuracy: 0.7864

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5428 - accuracy: 0.7853

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5406 - accuracy: 0.7877

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5394 - accuracy: 0.7881

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5406 - accuracy: 0.7870

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5400 - accuracy: 0.7874

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5402 - accuracy: 0.7878

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5390 - accuracy: 0.7868

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5365 - accuracy: 0.7872

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5359 - accuracy: 0.7875

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5342 - accuracy: 0.7883

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5364 - accuracy: 0.7882

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5388 - accuracy: 0.7869

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5383 - accuracy: 0.7880

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5382 - accuracy: 0.7884

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5356 - accuracy: 0.7891

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5379 - accuracy: 0.7882

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5379 - accuracy: 0.7873

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5376 - accuracy: 0.7876

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5354 - accuracy: 0.7883

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5329 - accuracy: 0.7890

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5306 - accuracy: 0.7904

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5288 - accuracy: 0.7918

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5305 - accuracy: 0.7913

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5305 - accuracy: 0.7915

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5306 - accuracy: 0.7918

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5318 - accuracy: 0.7917

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5294 - accuracy: 0.7930

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5299 - accuracy: 0.7925

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5290 - accuracy: 0.7927

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5285 - accuracy: 0.7933

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5285 - accuracy: 0.7933 - val_loss: 0.8408 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 5s - loss: 0.4326 - accuracy: 0.7917

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4314 - accuracy: 0.8214

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4084 - accuracy: 0.8068

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5428 - accuracy: 0.7833

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5849 - accuracy: 0.7632

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5580 - accuracy: 0.7772

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5636 - accuracy: 0.7731

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5732 - accuracy: 0.7702

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5639 - accuracy: 0.7714

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5618 - accuracy: 0.7724

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5609 - accuracy: 0.7733

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5667 - accuracy: 0.7660

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5505 - accuracy: 0.7745

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5466 - accuracy: 0.7773

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5523 - accuracy: 0.7754

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5692 - accuracy: 0.7639

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5666 - accuracy: 0.7649

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5596 - accuracy: 0.7676

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5462 - accuracy: 0.7750

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5497 - accuracy: 0.7690

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5488 - accuracy: 0.7711

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5589 - accuracy: 0.7672

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5508 - accuracy: 0.7720

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5575 - accuracy: 0.7697

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5481 - accuracy: 0.7727

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5493 - accuracy: 0.7694

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5529 - accuracy: 0.7675

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5565 - accuracy: 0.7658

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5594 - accuracy: 0.7663

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5588 - accuracy: 0.7647

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5601 - accuracy: 0.7652

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5587 - accuracy: 0.7648

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5540 - accuracy: 0.7681

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5489 - accuracy: 0.7713

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5400 - accuracy: 0.7770

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5368 - accuracy: 0.7788

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5348 - accuracy: 0.7798

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5309 - accuracy: 0.7823

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5280 - accuracy: 0.7831

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5259 - accuracy: 0.7846

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5245 - accuracy: 0.7860

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5274 - accuracy: 0.7837

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5231 - accuracy: 0.7865

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5282 - accuracy: 0.7843

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5298 - accuracy: 0.7828

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5310 - accuracy: 0.7821

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5326 - accuracy: 0.7821

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5322 - accuracy: 0.7814

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5271 - accuracy: 0.7827

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5246 - accuracy: 0.7833

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5268 - accuracy: 0.7851

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5225 - accuracy: 0.7874

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5286 - accuracy: 0.7855

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5314 - accuracy: 0.7837

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5265 - accuracy: 0.7860

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5304 - accuracy: 0.7836

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5295 - accuracy: 0.7841

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5244 - accuracy: 0.7863

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5277 - accuracy: 0.7851

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5243 - accuracy: 0.7871

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5250 - accuracy: 0.7876

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5214 - accuracy: 0.7895

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5220 - accuracy: 0.7888

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5221 - accuracy: 0.7887

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5203 - accuracy: 0.7901

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5215 - accuracy: 0.7894

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5217 - accuracy: 0.7898

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5249 - accuracy: 0.7874

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5277 - accuracy: 0.7877

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5278 - accuracy: 0.7876

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5243 - accuracy: 0.7893

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5243 - accuracy: 0.7896

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5245 - accuracy: 0.7899

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5242 - accuracy: 0.7907

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5217 - accuracy: 0.7918

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5238 - accuracy: 0.7896

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5237 - accuracy: 0.7899

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5241 - accuracy: 0.7898

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5232 - accuracy: 0.7893

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5237 - accuracy: 0.7888

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5265 - accuracy: 0.7875

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5275 - accuracy: 0.7867

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5264 - accuracy: 0.7878

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5263 - accuracy: 0.7881

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5258 - accuracy: 0.7880

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5264 - accuracy: 0.7875

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5262 - accuracy: 0.7875

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5251 - accuracy: 0.7881

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5249 - accuracy: 0.7884

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5247 - accuracy: 0.7886

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5245 - accuracy: 0.7886

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5261 - accuracy: 0.7885

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5261 - accuracy: 0.7885 - val_loss: 0.7809 - val_accuracy: 0.7193


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5846 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4486 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4537 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4100 - accuracy: 0.8438

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4282 - accuracy: 0.8438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4492 - accuracy: 0.8333

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4354 - accuracy: 0.8438

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4212 - accuracy: 0.8516

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4339 - accuracy: 0.8368

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4501 - accuracy: 0.8372

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4389 - accuracy: 0.8404

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4464 - accuracy: 0.8382

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4573 - accuracy: 0.8341

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4534 - accuracy: 0.8390

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4516 - accuracy: 0.8393

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4463 - accuracy: 0.8414

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4470 - accuracy: 0.8380

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4381 - accuracy: 0.8383

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4492 - accuracy: 0.8370

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4542 - accuracy: 0.8328

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4567 - accuracy: 0.8333

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4542 - accuracy: 0.8338

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4540 - accuracy: 0.8355

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4461 - accuracy: 0.8371

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4510 - accuracy: 0.8313

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4438 - accuracy: 0.8353

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4462 - accuracy: 0.8333

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4445 - accuracy: 0.8348

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4534 - accuracy: 0.8309

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4491 - accuracy: 0.8293

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4506 - accuracy: 0.8287

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4495 - accuracy: 0.8273

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4474 - accuracy: 0.8287

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4455 - accuracy: 0.8300

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4412 - accuracy: 0.8330

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4376 - accuracy: 0.8367

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4423 - accuracy: 0.8369

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4444 - accuracy: 0.8363

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.4451 - accuracy: 0.8349

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4487 - accuracy: 0.8336

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4451 - accuracy: 0.8346

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4468 - accuracy: 0.8341

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4450 - accuracy: 0.8357

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.4441 - accuracy: 0.8359

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4512 - accuracy: 0.8320

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4642 - accuracy: 0.8275

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4606 - accuracy: 0.8292

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4613 - accuracy: 0.8295

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4634 - accuracy: 0.8273

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4627 - accuracy: 0.8282

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4619 - accuracy: 0.8267

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4627 - accuracy: 0.8264

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4615 - accuracy: 0.8267

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.4631 - accuracy: 0.8265

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.4655 - accuracy: 0.8251

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.4664 - accuracy: 0.8254

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4652 - accuracy: 0.8263

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4661 - accuracy: 0.8261

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.4669 - accuracy: 0.8253

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.4677 - accuracy: 0.8246

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.4690 - accuracy: 0.8239

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.4667 - accuracy: 0.8262

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.4668 - accuracy: 0.8260

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.4645 - accuracy: 0.8272

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.4631 - accuracy: 0.8275

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.4610 - accuracy: 0.8291

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.4633 - accuracy: 0.8280

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.4649 - accuracy: 0.8264

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.4650 - accuracy: 0.8262

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.4662 - accuracy: 0.8255

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.4704 - accuracy: 0.8240

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.4714 - accuracy: 0.8239

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.4704 - accuracy: 0.8237

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.4701 - accuracy: 0.8236

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.4719 - accuracy: 0.8230

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.4744 - accuracy: 0.8225

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.4753 - accuracy: 0.8232

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.4735 - accuracy: 0.8242

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.4733 - accuracy: 0.8241

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.4731 - accuracy: 0.8243

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.4742 - accuracy: 0.8238

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.4759 - accuracy: 0.8240

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.4766 - accuracy: 0.8239

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.4771 - accuracy: 0.8237

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.4786 - accuracy: 0.8225

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.4784 - accuracy: 0.8220

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.4804 - accuracy: 0.8219

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.4826 - accuracy: 0.8201

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.4815 - accuracy: 0.8207

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.4814 - accuracy: 0.8206

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.4850 - accuracy: 0.8188

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.4850 - accuracy: 0.8188 - val_loss: 0.7300 - val_accuracy: 0.7330


Visualize Training Results
--------------------------



After applying data augmentation and Dropout, there is less overfitting
than before, and training and validation accuracy are closer aligned.

.. code:: ipython3

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_66_0.png


Predict on New Data
-------------------



Finally, let us use the model to classify an image that was not included
in the training or validation sets.

   **Note**: Data augmentation and Dropout layers are inactive at
   inference time.

.. code:: ipython3

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    
    img = tf.keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    
1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 77ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.86 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.h5" is the pre-trained model
    model_dir = "model"
    saved_model_path = f"{model_dir}/flower/saved_model"
    model.save(saved_model_path)


.. parsed-literal::

    2024-04-10 00:37:47.272355: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-10 00:37:47.368801: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.379158: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-10 00:37:47.390758: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.398073: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.405276: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.416624: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.457774: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-04-10 00:37:47.530734: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.552824: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-04-10 00:37:47.594764: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.620560: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:47.694944: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-10 00:37:47.852219: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:48.031043: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-10 00:37:48.068866: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:48.099900: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-10 00:37:48.150433: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


Convert the TensorFlow model with OpenVINO Model Conversion API
---------------------------------------------------------------

To convert the model to
OpenVINO IR with ``FP16`` precision, use model conversion Python API.

.. code:: ipython3

    # Convert the model to ir model format and save it.
    ir_model_path = Path("model/flower")
    ir_model_path.mkdir(parents=True, exist_ok=True)
    ir_model = ov.convert_model(saved_model_path, input=[1,180,180,3])
    ov.save_model(ir_model, ir_model_path / "flower_ir.xml")

Preprocessing Image Function
----------------------------



.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, h, w, c = [1, img_height, img_height, 3]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)
    
        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
        input_image = image.reshape((n, h, w, c))
    
        return input_image

OpenVINO Runtime Setup
----------------------



Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    # Initialize OpenVINO runtime
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

    class_names=["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    
    compiled_model = core.compile_model(model=ir_model, device_name=device.value)
    
    del ir_model
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

Run the Inference Step
----------------------



.. code:: ipython3

    # Run inference on the input image...
    inp_img_url = "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    OUTPUT_DIR = "output"
    inp_file_name = f"A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(OUTPUT_DIR)/Path(inp_file_name)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download the image
    download_file(inp_img_url, inp_file_name, directory=OUTPUT_DIR)
    
    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(file_path)
    
    print(input_image.shape)
    print(input_layer.shape)
    res = compiled_model([input_image])[output_layer]
    
    score = tf.nn.softmax(res[0])
    
    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 99.70 percent confidence.



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------



This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <./tensorflow-training-openvino-nncf.ipynb>`__ notebook.

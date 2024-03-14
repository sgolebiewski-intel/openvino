Working with Open Model Zoo Models
==================================

This tutorial shows how to download a model from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__, convert it
to OpenVINO™ IR format, show information about the model, and benchmark
the model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `OpenVINO and Open Model Zoo
   Tools <#openvino-and-open-model-zoo-tools>`__
-  `Preparation <#preparation>`__

   -  `Model Name <#model-name>`__
   -  `Imports <#imports>`__
   -  `Settings and Configuration <#settings-and-configuration>`__

-  `Download a Model from Open Model
   Zoo <#download-a-model-from-open-model-zoo>`__
-  `Convert a Model to OpenVINO IR
   format <#convert-a-model-to-openvino-ir-format>`__
-  `Get Model Information <#get-model-information>`__
-  `Run Benchmark Tool <#run-benchmark-tool>`__

   -  `Benchmark with Different
      Settings <#benchmark-with-different-settings>`__

OpenVINO and Open Model Zoo Tools
---------------------------------

`back to top ⬆️ <#table-of-contents>`__

OpenVINO and Open Model Zoo tools are listed in the table below.

+------------+--------------+-----------------------------------------+
| Tool       | Command      | Description                             |
+============+==============+=========================================+
| Model      | ``omz_downlo | Download models from Open Model Zoo.    |
| Downloader | ader``       |                                         |
+------------+--------------+-----------------------------------------+
| Model      | ``omz_conver | Convert Open Model Zoo models to        |
| Converter  | ter``        | OpenVINO’s IR format.                   |
+------------+--------------+-----------------------------------------+
| Info       | ``omz_info_d | Print information about Open Model Zoo  |
| Dumper     | umper``      | models.                                 |
+------------+--------------+-----------------------------------------+
| Benchmark  | ``benchmark_ | Benchmark model performance by          |
| Tool       | app``        | computing inference time.               |
+------------+--------------+-----------------------------------------+

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino-dev>=2024.0.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Preparation
-----------

`back to top ⬆️ <#table-of-contents>`__

Model Name
~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Set ``model_name`` to the name of the Open Model Zoo model to use in
this notebook. Refer to the list of
`public <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
and
`Intel <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md>`__
pre-trained models for a full list of models that can be used. Set
``model_name`` to the model you want to use.

.. code:: ipython3

    # model_name = "resnet-50-pytorch"
    model_name = "mobilenet-v2-pytorch"

Imports
~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import json
    from pathlib import Path
    
    import openvino as ov
    from IPython.display import Markdown, display
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import DeviceNotFoundAlert, NotebookAlert

Settings and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Set the file and directory paths. By default, this notebook downloads
models from Open Model Zoo to the ``open_model_zoo_models`` directory in
your ``$HOME`` directory. On Windows, the $HOME directory is usually
``c:\users\username``, on Linux ``/home/username``. To change the
folder, change ``base_model_dir`` in the cell below.

The following settings can be changed:

-  ``base_model_dir``: Models will be downloaded into the ``intel`` and
   ``public`` folders in this directory.
-  ``omz_cache_dir``: Cache folder for Open Model Zoo. Specifying a
   cache directory is not required for Model Downloader and Model
   Converter, but it speeds up subsequent downloads.
-  ``precision``: If specified, only models with this precision will be
   downloaded and converted.

.. code:: ipython3

    base_model_dir = Path("model")
    omz_cache_dir = Path("cache")
    precision = "FP16"
    
    # Check if an GPU is available on this system to use with Benchmark App.
    core = ov.Core()
    gpu_available = "GPU" in core.available_devices
    
    print(
        f"base_model_dir: {base_model_dir}, omz_cache_dir: {omz_cache_dir}, gpu_availble: {gpu_available}"
    )


.. parsed-literal::

    base_model_dir: model, omz_cache_dir: cache, gpu_availble: False


Download a Model from Open Model Zoo
------------------------------------

`back to top ⬆️ <#table-of-contents>`__

Specify, display and run the Model Downloader command to download the
model.

.. code:: ipython3

    ## Uncomment the next line to show help in omz_downloader which explains the command-line options.
    
    # !omz_downloader --help

.. code:: ipython3

    download_command = (
        f"omz_downloader --name {model_name} --output_dir {base_model_dir} --cache_dir {omz_cache_dir}"
    )
    display(Markdown(f"Download command: `{download_command}`"))
    display(Markdown(f"Downloading {model_name}..."))
    ! $download_command



Download command:
``omz_downloader --name mobilenet-v2-pytorch --output_dir model --cache_dir cache``



Downloading mobilenet-v2-pytorch…


.. parsed-literal::

    ################|| Downloading mobilenet-v2-pytorch ||################
    
    ========== Downloading model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth


.. parsed-literal::

    ... 0%, 32 KB, 879 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 891 KB/s, 0 seconds passed
... 0%, 96 KB, 1321 KB/s, 0 seconds passed
... 0%, 128 KB, 1213 KB/s, 0 seconds passed
... 1%, 160 KB, 1491 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 192 KB, 1758 KB/s, 0 seconds passed
... 1%, 224 KB, 2017 KB/s, 0 seconds passed
... 1%, 256 KB, 2275 KB/s, 0 seconds passed
... 2%, 288 KB, 2036 KB/s, 0 seconds passed
... 2%, 320 KB, 2253 KB/s, 0 seconds passed
... 2%, 352 KB, 2464 KB/s, 0 seconds passed
... 2%, 384 KB, 2680 KB/s, 0 seconds passed
... 2%, 416 KB, 2896 KB/s, 0 seconds passed
... 3%, 448 KB, 3108 KB/s, 0 seconds passed
... 3%, 480 KB, 3313 KB/s, 0 seconds passed
... 3%, 512 KB, 3524 KB/s, 0 seconds passed
... 3%, 544 KB, 3673 KB/s, 0 seconds passed
... 4%, 576 KB, 3876 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 608 KB, 3450 KB/s, 0 seconds passed
... 4%, 640 KB, 3615 KB/s, 0 seconds passed
... 4%, 672 KB, 3787 KB/s, 0 seconds passed
... 5%, 704 KB, 3958 KB/s, 0 seconds passed
... 5%, 736 KB, 4129 KB/s, 0 seconds passed
... 5%, 768 KB, 4299 KB/s, 0 seconds passed
... 5%, 800 KB, 4468 KB/s, 0 seconds passed
... 5%, 832 KB, 4638 KB/s, 0 seconds passed
... 6%, 864 KB, 4735 KB/s, 0 seconds passed
... 6%, 896 KB, 4899 KB/s, 0 seconds passed
... 6%, 928 KB, 5033 KB/s, 0 seconds passed
... 6%, 960 KB, 5195 KB/s, 0 seconds passed
... 7%, 992 KB, 5357 KB/s, 0 seconds passed
... 7%, 1024 KB, 5520 KB/s, 0 seconds passed
... 7%, 1056 KB, 5681 KB/s, 0 seconds passed
... 7%, 1088 KB, 5841 KB/s, 0 seconds passed
... 8%, 1120 KB, 6001 KB/s, 0 seconds passed
... 8%, 1152 KB, 6159 KB/s, 0 seconds passed
... 8%, 1184 KB, 6319 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1216 KB, 5739 KB/s, 0 seconds passed
... 8%, 1248 KB, 5844 KB/s, 0 seconds passed
... 9%, 1280 KB, 5977 KB/s, 0 seconds passed
... 9%, 1312 KB, 6109 KB/s, 0 seconds passed
... 9%, 1344 KB, 6241 KB/s, 0 seconds passed
... 9%, 1376 KB, 6372 KB/s, 0 seconds passed
... 10%, 1408 KB, 6510 KB/s, 0 seconds passed
... 10%, 1440 KB, 6614 KB/s, 0 seconds passed
... 10%, 1472 KB, 6744 KB/s, 0 seconds passed
... 10%, 1504 KB, 6879 KB/s, 0 seconds passed
... 11%, 1536 KB, 7012 KB/s, 0 seconds passed
... 11%, 1568 KB, 7146 KB/s, 0 seconds passed
... 11%, 1600 KB, 7278 KB/s, 0 seconds passed
... 11%, 1632 KB, 7410 KB/s, 0 seconds passed
... 11%, 1664 KB, 7542 KB/s, 0 seconds passed
... 12%, 1696 KB, 7675 KB/s, 0 seconds passed
... 12%, 1728 KB, 7806 KB/s, 0 seconds passed
... 12%, 1760 KB, 7937 KB/s, 0 seconds passed
... 12%, 1792 KB, 8067 KB/s, 0 seconds passed
... 13%, 1824 KB, 8198 KB/s, 0 seconds passed
... 13%, 1856 KB, 8327 KB/s, 0 seconds passed
... 13%, 1888 KB, 8456 KB/s, 0 seconds passed
... 13%, 1920 KB, 8585 KB/s, 0 seconds passed
... 14%, 1952 KB, 8712 KB/s, 0 seconds passed
... 14%, 1984 KB, 8840 KB/s, 0 seconds passed
... 14%, 2016 KB, 8967 KB/s, 0 seconds passed
... 14%, 2048 KB, 9094 KB/s, 0 seconds passed
... 14%, 2080 KB, 9220 KB/s, 0 seconds passed
... 15%, 2112 KB, 9346 KB/s, 0 seconds passed
... 15%, 2144 KB, 9472 KB/s, 0 seconds passed
... 15%, 2176 KB, 9603 KB/s, 0 seconds passed
... 15%, 2208 KB, 9735 KB/s, 0 seconds passed
... 16%, 2240 KB, 9866 KB/s, 0 seconds passed
... 16%, 2272 KB, 9997 KB/s, 0 seconds passed
... 16%, 2304 KB, 10128 KB/s, 0 seconds passed
... 16%, 2336 KB, 10258 KB/s, 0 seconds passed
... 17%, 2368 KB, 10387 KB/s, 0 seconds passed
... 17%, 2400 KB, 10516 KB/s, 0 seconds passed
... 17%, 2432 KB, 10646 KB/s, 0 seconds passed
... 17%, 2464 KB, 9846 KB/s, 0 seconds passed
... 17%, 2496 KB, 9953 KB/s, 0 seconds passed
... 18%, 2528 KB, 10063 KB/s, 0 seconds passed
... 18%, 2560 KB, 10174 KB/s, 0 seconds passed
... 18%, 2592 KB, 10284 KB/s, 0 seconds passed
... 18%, 2624 KB, 10395 KB/s, 0 seconds passed
... 19%, 2656 KB, 10506 KB/s, 0 seconds passed
... 19%, 2688 KB, 10617 KB/s, 0 seconds passed
... 19%, 2720 KB, 10728 KB/s, 0 seconds passed
... 19%, 2752 KB, 10838 KB/s, 0 seconds passed
... 20%, 2784 KB, 10948 KB/s, 0 seconds passed
... 20%, 2816 KB, 11057 KB/s, 0 seconds passed
... 20%, 2848 KB, 11167 KB/s, 0 seconds passed
... 20%, 2880 KB, 11275 KB/s, 0 seconds passed
... 20%, 2912 KB, 11382 KB/s, 0 seconds passed
... 21%, 2944 KB, 11489 KB/s, 0 seconds passed
... 21%, 2976 KB, 11597 KB/s, 0 seconds passed
... 21%, 3008 KB, 11705 KB/s, 0 seconds passed
... 21%, 3040 KB, 11811 KB/s, 0 seconds passed
... 22%, 3072 KB, 11918 KB/s, 0 seconds passed
... 22%, 3104 KB, 12025 KB/s, 0 seconds passed
... 22%, 3136 KB, 12131 KB/s, 0 seconds passed
... 22%, 3168 KB, 12237 KB/s, 0 seconds passed
... 23%, 3200 KB, 12343 KB/s, 0 seconds passed
... 23%, 3232 KB, 12448 KB/s, 0 seconds passed
... 23%, 3264 KB, 12552 KB/s, 0 seconds passed
... 23%, 3296 KB, 12657 KB/s, 0 seconds passed
... 23%, 3328 KB, 12761 KB/s, 0 seconds passed
... 24%, 3360 KB, 12865 KB/s, 0 seconds passed
... 24%, 3392 KB, 12969 KB/s, 0 seconds passed
... 24%, 3424 KB, 13071 KB/s, 0 seconds passed
... 24%, 3456 KB, 13175 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 3488 KB, 13277 KB/s, 0 seconds passed
... 25%, 3520 KB, 13380 KB/s, 0 seconds passed
... 25%, 3552 KB, 13482 KB/s, 0 seconds passed
... 25%, 3584 KB, 13582 KB/s, 0 seconds passed
... 26%, 3616 KB, 13684 KB/s, 0 seconds passed
... 26%, 3648 KB, 13784 KB/s, 0 seconds passed
... 26%, 3680 KB, 13886 KB/s, 0 seconds passed
... 26%, 3712 KB, 13987 KB/s, 0 seconds passed
... 26%, 3744 KB, 14088 KB/s, 0 seconds passed
... 27%, 3776 KB, 14188 KB/s, 0 seconds passed
... 27%, 3808 KB, 14289 KB/s, 0 seconds passed
... 27%, 3840 KB, 14396 KB/s, 0 seconds passed
... 27%, 3872 KB, 14503 KB/s, 0 seconds passed
... 28%, 3904 KB, 14609 KB/s, 0 seconds passed
... 28%, 3936 KB, 14715 KB/s, 0 seconds passed
... 28%, 3968 KB, 14821 KB/s, 0 seconds passed
... 28%, 4000 KB, 14926 KB/s, 0 seconds passed
... 29%, 4032 KB, 15032 KB/s, 0 seconds passed
... 29%, 4064 KB, 15137 KB/s, 0 seconds passed
... 29%, 4096 KB, 15242 KB/s, 0 seconds passed
... 29%, 4128 KB, 15347 KB/s, 0 seconds passed
... 29%, 4160 KB, 15451 KB/s, 0 seconds passed
... 30%, 4192 KB, 15556 KB/s, 0 seconds passed
... 30%, 4224 KB, 15661 KB/s, 0 seconds passed
... 30%, 4256 KB, 15766 KB/s, 0 seconds passed
... 30%, 4288 KB, 15870 KB/s, 0 seconds passed
... 31%, 4320 KB, 15974 KB/s, 0 seconds passed
... 31%, 4352 KB, 16078 KB/s, 0 seconds passed
... 31%, 4384 KB, 16182 KB/s, 0 seconds passed
... 31%, 4416 KB, 16285 KB/s, 0 seconds passed
... 32%, 4448 KB, 16388 KB/s, 0 seconds passed
... 32%, 4480 KB, 16491 KB/s, 0 seconds passed
... 32%, 4512 KB, 16595 KB/s, 0 seconds passed
... 32%, 4544 KB, 16701 KB/s, 0 seconds passed
... 32%, 4576 KB, 16807 KB/s, 0 seconds passed
... 33%, 4608 KB, 16912 KB/s, 0 seconds passed
... 33%, 4640 KB, 17018 KB/s, 0 seconds passed
... 33%, 4672 KB, 17122 KB/s, 0 seconds passed
... 33%, 4704 KB, 17228 KB/s, 0 seconds passed
... 34%, 4736 KB, 17333 KB/s, 0 seconds passed
... 34%, 4768 KB, 17438 KB/s, 0 seconds passed
... 34%, 4800 KB, 17543 KB/s, 0 seconds passed
... 34%, 4832 KB, 17647 KB/s, 0 seconds passed
... 35%, 4864 KB, 17752 KB/s, 0 seconds passed
... 35%, 4896 KB, 17856 KB/s, 0 seconds passed
... 35%, 4928 KB, 17960 KB/s, 0 seconds passed
... 35%, 4960 KB, 17454 KB/s, 0 seconds passed
... 35%, 4992 KB, 17389 KB/s, 0 seconds passed
... 36%, 5024 KB, 17471 KB/s, 0 seconds passed
... 36%, 5056 KB, 17460 KB/s, 0 seconds passed
... 36%, 5088 KB, 17539 KB/s, 0 seconds passed
... 36%, 5120 KB, 17624 KB/s, 0 seconds passed
... 37%, 5152 KB, 17710 KB/s, 0 seconds passed
... 37%, 5184 KB, 17798 KB/s, 0 seconds passed
... 37%, 5216 KB, 17889 KB/s, 0 seconds passed
... 37%, 5248 KB, 17972 KB/s, 0 seconds passed
... 38%, 5280 KB, 18058 KB/s, 0 seconds passed
... 38%, 5312 KB, 18144 KB/s, 0 seconds passed
... 38%, 5344 KB, 18230 KB/s, 0 seconds passed
... 38%, 5376 KB, 18315 KB/s, 0 seconds passed
... 38%, 5408 KB, 18401 KB/s, 0 seconds passed
... 39%, 5440 KB, 18485 KB/s, 0 seconds passed
... 39%, 5472 KB, 18570 KB/s, 0 seconds passed
... 39%, 5504 KB, 18654 KB/s, 0 seconds passed
... 39%, 5536 KB, 18738 KB/s, 0 seconds passed
... 40%, 5568 KB, 18821 KB/s, 0 seconds passed
... 40%, 5600 KB, 18905 KB/s, 0 seconds passed
... 40%, 5632 KB, 18990 KB/s, 0 seconds passed
... 40%, 5664 KB, 19076 KB/s, 0 seconds passed
... 41%, 5696 KB, 19162 KB/s, 0 seconds passed
... 41%, 5728 KB, 19248 KB/s, 0 seconds passed
... 41%, 5760 KB, 19335 KB/s, 0 seconds passed
... 41%, 5792 KB, 19422 KB/s, 0 seconds passed
... 41%, 5824 KB, 19508 KB/s, 0 seconds passed
... 42%, 5856 KB, 19594 KB/s, 0 seconds passed
... 42%, 5888 KB, 19679 KB/s, 0 seconds passed
... 42%, 5920 KB, 19765 KB/s, 0 seconds passed
... 42%, 5952 KB, 19849 KB/s, 0 seconds passed
... 43%, 5984 KB, 19933 KB/s, 0 seconds passed
... 43%, 6016 KB, 20018 KB/s, 0 seconds passed
... 43%, 6048 KB, 20103 KB/s, 0 seconds passed
... 43%, 6080 KB, 20188 KB/s, 0 seconds passed
... 44%, 6112 KB, 20272 KB/s, 0 seconds passed
... 44%, 6144 KB, 20357 KB/s, 0 seconds passed
... 44%, 6176 KB, 20440 KB/s, 0 seconds passed
... 44%, 6208 KB, 20524 KB/s, 0 seconds passed
... 44%, 6240 KB, 20608 KB/s, 0 seconds passed
... 45%, 6272 KB, 20691 KB/s, 0 seconds passed
... 45%, 6304 KB, 20775 KB/s, 0 seconds passed
... 45%, 6336 KB, 20857 KB/s, 0 seconds passed
... 45%, 6368 KB, 20940 KB/s, 0 seconds passed
... 46%, 6400 KB, 21023 KB/s, 0 seconds passed
... 46%, 6432 KB, 21105 KB/s, 0 seconds passed
... 46%, 6464 KB, 21187 KB/s, 0 seconds passed
... 46%, 6496 KB, 21268 KB/s, 0 seconds passed
... 47%, 6528 KB, 21351 KB/s, 0 seconds passed
... 47%, 6560 KB, 21434 KB/s, 0 seconds passed
... 47%, 6592 KB, 21516 KB/s, 0 seconds passed
... 47%, 6624 KB, 21598 KB/s, 0 seconds passed
... 47%, 6656 KB, 21680 KB/s, 0 seconds passed
... 48%, 6688 KB, 21768 KB/s, 0 seconds passed
... 48%, 6720 KB, 21857 KB/s, 0 seconds passed
... 48%, 6752 KB, 21946 KB/s, 0 seconds passed
... 48%, 6784 KB, 22034 KB/s, 0 seconds passed
... 49%, 6816 KB, 22123 KB/s, 0 seconds passed
... 49%, 6848 KB, 22212 KB/s, 0 seconds passed
... 49%, 6880 KB, 22300 KB/s, 0 seconds passed
... 49%, 6912 KB, 22389 KB/s, 0 seconds passed
... 50%, 6944 KB, 22476 KB/s, 0 seconds passed
... 50%, 6976 KB, 22565 KB/s, 0 seconds passed
... 50%, 7008 KB, 22653 KB/s, 0 seconds passed
... 50%, 7040 KB, 22741 KB/s, 0 seconds passed
... 50%, 7072 KB, 22829 KB/s, 0 seconds passed
... 51%, 7104 KB, 22916 KB/s, 0 seconds passed
... 51%, 7136 KB, 23004 KB/s, 0 seconds passed
... 51%, 7168 KB, 23090 KB/s, 0 seconds passed
... 51%, 7200 KB, 23178 KB/s, 0 seconds passed
... 52%, 7232 KB, 23264 KB/s, 0 seconds passed
... 52%, 7264 KB, 23351 KB/s, 0 seconds passed
... 52%, 7296 KB, 23438 KB/s, 0 seconds passed
... 52%, 7328 KB, 23525 KB/s, 0 seconds passed
... 53%, 7360 KB, 23612 KB/s, 0 seconds passed
... 53%, 7392 KB, 23698 KB/s, 0 seconds passed
... 53%, 7424 KB, 23784 KB/s, 0 seconds passed
... 53%, 7456 KB, 23871 KB/s, 0 seconds passed
... 53%, 7488 KB, 23957 KB/s, 0 seconds passed
... 54%, 7520 KB, 24044 KB/s, 0 seconds passed
... 54%, 7552 KB, 24130 KB/s, 0 seconds passed
... 54%, 7584 KB, 24216 KB/s, 0 seconds passed
... 54%, 7616 KB, 24302 KB/s, 0 seconds passed
... 55%, 7648 KB, 24388 KB/s, 0 seconds passed
... 55%, 7680 KB, 24474 KB/s, 0 seconds passed

.. parsed-literal::

    ... 55%, 7712 KB, 24557 KB/s, 0 seconds passed
... 55%, 7744 KB, 24643 KB/s, 0 seconds passed
... 56%, 7776 KB, 24728 KB/s, 0 seconds passed
... 56%, 7808 KB, 24814 KB/s, 0 seconds passed
... 56%, 7840 KB, 24900 KB/s, 0 seconds passed
... 56%, 7872 KB, 24985 KB/s, 0 seconds passed
... 56%, 7904 KB, 25069 KB/s, 0 seconds passed
... 57%, 7936 KB, 25154 KB/s, 0 seconds passed
... 57%, 7968 KB, 25239 KB/s, 0 seconds passed
... 57%, 8000 KB, 25319 KB/s, 0 seconds passed
... 57%, 8032 KB, 25399 KB/s, 0 seconds passed
... 58%, 8064 KB, 25480 KB/s, 0 seconds passed
... 58%, 8096 KB, 25555 KB/s, 0 seconds passed
... 58%, 8128 KB, 25634 KB/s, 0 seconds passed
... 58%, 8160 KB, 25715 KB/s, 0 seconds passed
... 59%, 8192 KB, 25794 KB/s, 0 seconds passed
... 59%, 8224 KB, 25874 KB/s, 0 seconds passed
... 59%, 8256 KB, 25953 KB/s, 0 seconds passed
... 59%, 8288 KB, 26032 KB/s, 0 seconds passed
... 59%, 8320 KB, 26107 KB/s, 0 seconds passed
... 60%, 8352 KB, 26185 KB/s, 0 seconds passed
... 60%, 8384 KB, 26264 KB/s, 0 seconds passed
... 60%, 8416 KB, 26343 KB/s, 0 seconds passed
... 60%, 8448 KB, 26416 KB/s, 0 seconds passed
... 61%, 8480 KB, 26486 KB/s, 0 seconds passed
... 61%, 8512 KB, 26564 KB/s, 0 seconds passed
... 61%, 8544 KB, 26642 KB/s, 0 seconds passed
... 61%, 8576 KB, 26715 KB/s, 0 seconds passed
... 62%, 8608 KB, 26793 KB/s, 0 seconds passed
... 62%, 8640 KB, 26871 KB/s, 0 seconds passed
... 62%, 8672 KB, 26944 KB/s, 0 seconds passed
... 62%, 8704 KB, 27020 KB/s, 0 seconds passed
... 62%, 8736 KB, 26808 KB/s, 0 seconds passed
... 63%, 8768 KB, 26874 KB/s, 0 seconds passed
... 63%, 8800 KB, 26942 KB/s, 0 seconds passed
... 63%, 8832 KB, 27004 KB/s, 0 seconds passed
... 63%, 8864 KB, 27068 KB/s, 0 seconds passed
... 64%, 8896 KB, 27133 KB/s, 0 seconds passed
... 64%, 8928 KB, 27202 KB/s, 0 seconds passed
... 64%, 8960 KB, 27274 KB/s, 0 seconds passed
... 64%, 8992 KB, 27225 KB/s, 0 seconds passed
... 65%, 9024 KB, 27291 KB/s, 0 seconds passed
... 65%, 9056 KB, 27357 KB/s, 0 seconds passed
... 65%, 9088 KB, 27423 KB/s, 0 seconds passed
... 65%, 9120 KB, 27488 KB/s, 0 seconds passed
... 65%, 9152 KB, 27554 KB/s, 0 seconds passed
... 66%, 9184 KB, 27622 KB/s, 0 seconds passed
... 66%, 9216 KB, 27694 KB/s, 0 seconds passed
... 66%, 9248 KB, 27486 KB/s, 0 seconds passed
... 66%, 9280 KB, 27549 KB/s, 0 seconds passed
... 67%, 9312 KB, 27611 KB/s, 0 seconds passed
... 67%, 9344 KB, 27675 KB/s, 0 seconds passed
... 67%, 9376 KB, 27740 KB/s, 0 seconds passed
... 67%, 9408 KB, 27803 KB/s, 0 seconds passed
... 68%, 9440 KB, 27871 KB/s, 0 seconds passed
... 68%, 9472 KB, 27942 KB/s, 0 seconds passed
... 68%, 9504 KB, 27191 KB/s, 0 seconds passed
... 68%, 9536 KB, 27241 KB/s, 0 seconds passed
... 68%, 9568 KB, 27300 KB/s, 0 seconds passed
... 69%, 9600 KB, 27361 KB/s, 0 seconds passed
... 69%, 9632 KB, 27422 KB/s, 0 seconds passed
... 69%, 9664 KB, 27484 KB/s, 0 seconds passed
... 69%, 9696 KB, 27543 KB/s, 0 seconds passed
... 70%, 9728 KB, 27603 KB/s, 0 seconds passed
... 70%, 9760 KB, 27664 KB/s, 0 seconds passed
... 70%, 9792 KB, 27724 KB/s, 0 seconds passed
... 70%, 9824 KB, 27785 KB/s, 0 seconds passed
... 71%, 9856 KB, 27846 KB/s, 0 seconds passed
... 71%, 9888 KB, 27905 KB/s, 0 seconds passed
... 71%, 9920 KB, 27966 KB/s, 0 seconds passed
... 71%, 9952 KB, 28026 KB/s, 0 seconds passed
... 71%, 9984 KB, 28087 KB/s, 0 seconds passed
... 72%, 10016 KB, 28145 KB/s, 0 seconds passed
... 72%, 10048 KB, 28204 KB/s, 0 seconds passed
... 72%, 10080 KB, 28264 KB/s, 0 seconds passed
... 72%, 10112 KB, 28324 KB/s, 0 seconds passed
... 73%, 10144 KB, 28385 KB/s, 0 seconds passed
... 73%, 10176 KB, 28450 KB/s, 0 seconds passed
... 73%, 10208 KB, 28518 KB/s, 0 seconds passed
... 73%, 10240 KB, 28588 KB/s, 0 seconds passed
... 74%, 10272 KB, 28657 KB/s, 0 seconds passed
... 74%, 10304 KB, 28726 KB/s, 0 seconds passed
... 74%, 10336 KB, 28796 KB/s, 0 seconds passed
... 74%, 10368 KB, 28865 KB/s, 0 seconds passed
... 74%, 10400 KB, 28934 KB/s, 0 seconds passed
... 75%, 10432 KB, 28960 KB/s, 0 seconds passed
... 75%, 10464 KB, 29021 KB/s, 0 seconds passed
... 75%, 10496 KB, 29082 KB/s, 0 seconds passed
... 75%, 10528 KB, 29144 KB/s, 0 seconds passed
... 76%, 10560 KB, 29205 KB/s, 0 seconds passed
... 76%, 10592 KB, 29267 KB/s, 0 seconds passed
... 76%, 10624 KB, 29330 KB/s, 0 seconds passed
... 76%, 10656 KB, 29397 KB/s, 0 seconds passed

.. parsed-literal::

    ... 77%, 10688 KB, 29110 KB/s, 0 seconds passed
... 77%, 10720 KB, 29168 KB/s, 0 seconds passed
... 77%, 10752 KB, 29224 KB/s, 0 seconds passed
... 77%, 10784 KB, 29281 KB/s, 0 seconds passed
... 77%, 10816 KB, 29338 KB/s, 0 seconds passed
... 78%, 10848 KB, 29395 KB/s, 0 seconds passed
... 78%, 10880 KB, 29451 KB/s, 0 seconds passed
... 78%, 10912 KB, 29508 KB/s, 0 seconds passed
... 78%, 10944 KB, 29564 KB/s, 0 seconds passed
... 79%, 10976 KB, 29619 KB/s, 0 seconds passed
... 79%, 11008 KB, 29674 KB/s, 0 seconds passed
... 79%, 11040 KB, 29731 KB/s, 0 seconds passed
... 79%, 11072 KB, 29787 KB/s, 0 seconds passed
... 80%, 11104 KB, 29840 KB/s, 0 seconds passed
... 80%, 11136 KB, 29895 KB/s, 0 seconds passed
... 80%, 11168 KB, 29949 KB/s, 0 seconds passed
... 80%, 11200 KB, 30004 KB/s, 0 seconds passed
... 80%, 11232 KB, 30063 KB/s, 0 seconds passed
... 81%, 11264 KB, 30124 KB/s, 0 seconds passed
... 81%, 11296 KB, 30186 KB/s, 0 seconds passed
... 81%, 11328 KB, 30248 KB/s, 0 seconds passed
... 81%, 11360 KB, 30308 KB/s, 0 seconds passed
... 82%, 11392 KB, 30363 KB/s, 0 seconds passed
... 82%, 11424 KB, 30419 KB/s, 0 seconds passed
... 82%, 11456 KB, 30470 KB/s, 0 seconds passed
... 82%, 11488 KB, 30525 KB/s, 0 seconds passed
... 82%, 11520 KB, 30579 KB/s, 0 seconds passed
... 83%, 11552 KB, 30635 KB/s, 0 seconds passed
... 83%, 11584 KB, 30696 KB/s, 0 seconds passed
... 83%, 11616 KB, 30757 KB/s, 0 seconds passed
... 83%, 11648 KB, 30818 KB/s, 0 seconds passed
... 84%, 11680 KB, 30878 KB/s, 0 seconds passed
... 84%, 11712 KB, 30941 KB/s, 0 seconds passed
... 84%, 11744 KB, 31003 KB/s, 0 seconds passed
... 84%, 11776 KB, 31066 KB/s, 0 seconds passed
... 85%, 11808 KB, 31128 KB/s, 0 seconds passed
... 85%, 11840 KB, 31186 KB/s, 0 seconds passed
... 85%, 11872 KB, 31247 KB/s, 0 seconds passed
... 85%, 11904 KB, 31311 KB/s, 0 seconds passed
... 85%, 11936 KB, 31365 KB/s, 0 seconds passed
... 86%, 11968 KB, 31426 KB/s, 0 seconds passed
... 86%, 12000 KB, 31488 KB/s, 0 seconds passed
... 86%, 12032 KB, 31550 KB/s, 0 seconds passed
... 86%, 12064 KB, 31607 KB/s, 0 seconds passed
... 87%, 12096 KB, 31670 KB/s, 0 seconds passed
... 87%, 12128 KB, 31731 KB/s, 0 seconds passed
... 87%, 12160 KB, 31794 KB/s, 0 seconds passed
... 87%, 12192 KB, 31854 KB/s, 0 seconds passed
... 88%, 12224 KB, 31916 KB/s, 0 seconds passed
... 88%, 12256 KB, 31977 KB/s, 0 seconds passed
... 88%, 12288 KB, 32034 KB/s, 0 seconds passed
... 88%, 12320 KB, 32095 KB/s, 0 seconds passed
... 88%, 12352 KB, 32156 KB/s, 0 seconds passed
... 89%, 12384 KB, 32218 KB/s, 0 seconds passed
... 89%, 12416 KB, 32269 KB/s, 0 seconds passed
... 89%, 12448 KB, 32266 KB/s, 0 seconds passed
... 89%, 12480 KB, 32327 KB/s, 0 seconds passed
... 90%, 12512 KB, 32351 KB/s, 0 seconds passed
... 90%, 12544 KB, 32412 KB/s, 0 seconds passed
... 90%, 12576 KB, 32475 KB/s, 0 seconds passed
... 90%, 12608 KB, 32530 KB/s, 0 seconds passed
... 91%, 12640 KB, 32590 KB/s, 0 seconds passed
... 91%, 12672 KB, 32655 KB/s, 0 seconds passed
... 91%, 12704 KB, 32715 KB/s, 0 seconds passed
... 91%, 12736 KB, 32776 KB/s, 0 seconds passed
... 91%, 12768 KB, 32832 KB/s, 0 seconds passed
... 92%, 12800 KB, 32891 KB/s, 0 seconds passed
... 92%, 12832 KB, 32951 KB/s, 0 seconds passed
... 92%, 12864 KB, 33007 KB/s, 0 seconds passed
... 92%, 12896 KB, 33066 KB/s, 0 seconds passed
... 93%, 12928 KB, 33131 KB/s, 0 seconds passed
... 93%, 12960 KB, 33190 KB/s, 0 seconds passed
... 93%, 12992 KB, 33245 KB/s, 0 seconds passed
... 93%, 13024 KB, 33305 KB/s, 0 seconds passed
... 94%, 13056 KB, 33364 KB/s, 0 seconds passed
... 94%, 13088 KB, 33423 KB/s, 0 seconds passed
... 94%, 13120 KB, 33478 KB/s, 0 seconds passed
... 94%, 13152 KB, 33537 KB/s, 0 seconds passed
... 94%, 13184 KB, 33587 KB/s, 0 seconds passed
... 95%, 13216 KB, 33645 KB/s, 0 seconds passed
... 95%, 13248 KB, 33700 KB/s, 0 seconds passed
... 95%, 13280 KB, 33759 KB/s, 0 seconds passed
... 95%, 13312 KB, 33817 KB/s, 0 seconds passed
... 96%, 13344 KB, 33876 KB/s, 0 seconds passed
... 96%, 13376 KB, 33930 KB/s, 0 seconds passed
... 96%, 13408 KB, 33993 KB/s, 0 seconds passed
... 96%, 13440 KB, 34051 KB/s, 0 seconds passed
... 97%, 13472 KB, 34104 KB/s, 0 seconds passed
... 97%, 13504 KB, 34166 KB/s, 0 seconds passed
... 97%, 13536 KB, 34220 KB/s, 0 seconds passed
... 97%, 13568 KB, 34278 KB/s, 0 seconds passed
... 97%, 13600 KB, 34336 KB/s, 0 seconds passed
... 98%, 13632 KB, 34389 KB/s, 0 seconds passed
... 98%, 13664 KB, 34438 KB/s, 0 seconds passed
... 98%, 13696 KB, 34496 KB/s, 0 seconds passed
... 98%, 13728 KB, 34553 KB/s, 0 seconds passed
... 99%, 13760 KB, 34610 KB/s, 0 seconds passed
... 99%, 13792 KB, 34664 KB/s, 0 seconds passed
... 99%, 13824 KB, 34722 KB/s, 0 seconds passed
... 99%, 13856 KB, 34779 KB/s, 0 seconds passed
... 100%, 13879 KB, 34821 KB/s, 0 seconds passed



.. parsed-literal::

    


Convert a Model to OpenVINO IR format
-------------------------------------

`back to top ⬆️ <#table-of-contents>`__

Specify, display and run the Model Converter command to convert the
model to OpenVINO IR format. Model conversion may take a while. The
output of the Model Converter command will be displayed. When the
conversion is successful, the last lines of the output will include:
``[ SUCCESS ] Generated IR version 11 model.`` For downloaded models
that are already in OpenVINO IR format, conversion will be skipped.

.. code:: ipython3

    ## Uncomment the next line to show Help in omz_converter which explains the command-line options.
    
    # !omz_converter --help

.. code:: ipython3

    convert_command = f"omz_converter --name {model_name} --precisions {precision} --download_dir {base_model_dir} --output_dir {base_model_dir}"
    display(Markdown(f"Convert command: `{convert_command}`"))
    display(Markdown(f"Converting {model_name}..."))
    
    ! $convert_command



Convert command:
``omz_converter --name mobilenet-v2-pytorch --precisions FP16 --download_dir model --output_dir model``



Converting mobilenet-v2-pytorch…


.. parsed-literal::

    ========== Converting mobilenet-v2-pytorch to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-name=mobilenet_v2 --weights=model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth --import-module=torchvision.models --input-shape=1,3,224,224 --output-file=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx --input-names=data --output-names=prob
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting mobilenet-v2-pytorch to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/mobilenet-v2-pytorch/FP16 --model_name=mobilenet-v2-pytorch --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.624,57.12,57.375]' --reverse_input_channels --output=prob --input_model=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 224, 224]' --compress_to_fp16=True
    


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.bin


.. parsed-literal::

    


Get Model Information
---------------------

`back to top ⬆️ <#table-of-contents>`__

The Info Dumper prints the following information for Open Model Zoo
models:

-  Model name
-  Description
-  Framework that was used to train the model
-  License URL
-  Precisions supported by the model
-  Subdirectory: the location of the downloaded model
-  Task type

This information can be shown by running
``omz_info_dumper --name model_name`` in a terminal. The information can
also be parsed and used in scripts.

In the next cell, run Info Dumper and use ``json`` to load the
information in a dictionary.

.. code:: ipython3

    model_info_output = %sx omz_info_dumper --name $model_name
    model_info = json.loads(model_info_output.get_nlstr())
    
    if len(model_info) > 1:
        NotebookAlert(
            f"There are multiple IR files for the {model_name} model. The first model in the "
            "omz_info_dumper output will be used for benchmarking. Change "
            "`selected_model_info` in the cell below to select a different model from the list.",
            "warning",
        )
    
    model_info




.. parsed-literal::

    [{'name': 'mobilenet-v2-pytorch',
      'composite_model_name': None,
      'description': 'MobileNet V2 is image classification model pre-trained on ImageNet dataset. This is a PyTorch* implementation of MobileNetV2 architecture as described in the paper "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation" <https://arxiv.org/abs/1801.04381>.\nThe model input is a blob that consists of a single image of "1, 3, 224, 224" in "RGB" order.\nThe model output is typical object classifier for the 1000 different classifications matching with those in the ImageNet database.',
      'framework': 'pytorch',
      'license_url': 'https://raw.githubusercontent.com/pytorch/vision/master/LICENSE',
      'accuracy_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/accuracy-check.yml',
      'model_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/model.yml',
      'precisions': ['FP16', 'FP32'],
      'subdirectory': 'public/mobilenet-v2-pytorch',
      'task_type': 'classification',
      'input_info': [{'name': 'data',
        'shape': [1, 3, 224, 224],
        'layout': 'NCHW'}],
      'model_stages': []}]



Having information of the model in a JSON file enables extraction of the
path to the model directory, and building the path to the OpenVINO IR
file.

.. code:: ipython3

    selected_model_info = model_info[0]
    model_path = (
        base_model_dir
        / Path(selected_model_info["subdirectory"])
        / Path(f"{precision}/{selected_model_info['name']}.xml")
    )
    print(model_path, "exists:", model_path.exists())


.. parsed-literal::

    model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml exists: True


Run Benchmark Tool
------------------

`back to top ⬆️ <#table-of-contents>`__

By default, Benchmark Tool runs inference for 60 seconds in asynchronous
mode on CPU. It returns inference speed as latency (milliseconds per
image) and throughput values (frames per second).

.. code:: ipython3

    ## Uncomment the next line to show Help in benchmark_app which explains the command-line options.
    # !benchmark_app --help

.. code:: ipython3

    benchmark_command = f"benchmark_app -m {model_path} -t 15"
    display(Markdown(f"Benchmark command: `{benchmark_command}`"))
    display(Markdown(f"Benchmarking {model_name} on CPU with async inference for 15 seconds..."))
    
    ! $benchmark_command



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -t 15``



Benchmarking mobilenet-v2-pytorch on CPU with async inference for 15
seconds…


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 31.02 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     data (node: data) : f32 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     prob (node: prob) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     data (node: data) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     prob (node: prob) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 138.48 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   LOG_LEVEL: Level.NO
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]   KV_CACHE_PRECISION: <Type: 'float16'>
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'data'!. This input will be filled with random values!
    [ INFO ] Fill input 'data' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 6.34 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            20262 iterations
    [ INFO ] Duration:         15007.23 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.32 ms
    [ INFO ]    Average:       4.32 ms
    [ INFO ]    Min:           2.62 ms
    [ INFO ]    Max:           12.26 ms
    [ INFO ] Throughput:   1350.15 FPS


Benchmark with Different Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

The ``benchmark_app`` tool displays logging information that is not
always necessary. A more compact result is achieved when the output is
parsed with ``json``.

The following cells show some examples of ``benchmark_app`` with
different parameters. Below are some useful parameters:

-  ``-d`` A device to use for inference. For example: CPU, GPU, MULTI.
   Default: CPU.
-  ``-t`` Time expressed in number of seconds to run inference. Default:
   60.
-  ``-api`` Use asynchronous (async) or synchronous (sync) inference.
   Default: async.
-  ``-b`` Batch size. Default: 1.

Run ``! benchmark_app --help`` to get an overview of all possible
command-line parameters.

In the next cell, define the ``benchmark_model()`` function that calls
``benchmark_app``. This makes it easy to try different combinations. In
the cell below that, you display available devices on the system.

   **Note**: In this notebook, ``benchmark_app`` runs for 15 seconds to
   give a quick indication of performance. For more accurate
   performance, it is recommended to run inference for at least one
   minute by setting the ``t`` parameter to 60 or higher, and run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Copy the **benchmark command** and paste it in a
   command prompt where you have activated the ``openvino_env``
   environment.

.. code:: ipython3

    def benchmark_model(model_xml, device="CPU", seconds=60, api="async", batch=1):
        core = ov.Core()
        model_path = Path(model_xml)
        if ("GPU" in device) and ("GPU" not in core.available_devices):
            DeviceNotFoundAlert("GPU")
        else:
            benchmark_command = f"benchmark_app -m {model_path} -d {device} -t {seconds} -api {api} -b {batch}"
            display(Markdown(f"**Benchmark {model_path.name} with {device} for {seconds} seconds with {api} inference**"))
            display(Markdown(f"Benchmark command: `{benchmark_command}`"))
    
            benchmark_output = %sx $benchmark_command
            print("command ended")
            benchmark_result = [line for line in benchmark_output
                                if not (line.startswith(r"[") or line.startswith("      ") or line == "")]
            print("\n".join(benchmark_result))

.. code:: ipython3

    core = ov.Core()
    
    # Show devices available for OpenVINO Runtime
    for device in core.available_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


You can select inference device using device widget

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='CPU',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    benchmark_model(model_path, device=device.value, seconds=15, api="async")



**Benchmark mobilenet-v2-pytorch.xml with CPU for 15 seconds with async
inference**



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -d CPU -t 15 -api async -b 1``


.. parsed-literal::

    command ended
    


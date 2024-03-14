Live 3D Human Pose Estimation with OpenVINO
===========================================

This notebook demonstrates live 3D Human Pose Estimation with OpenVINO
via a webcam. We utilize the model
`human-pose-estimation-3d-0001 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__. At the end
of this notebook, you will see live inference results from your webcam
(if available). Alternatively, you can also upload a video file to test
out the algorithms. **Make sure you have properly installed
the**\ `Jupyter
extension <https://github.com/jupyter-widgets/pythreejs#jupyterlab>`__\ **and
been using JupyterLab to run the demo as suggested in the
``README.md``**

   **NOTE**: *To use a webcam, you must run this Jupyter notebook on a
   computer with a webcam. If you run on a remote server, the webcam
   will not work. However, you can still do inference on a video file in
   the final step. This demo utilizes the Python interface in
   ``Three.js`` integrated with WebGL to process data from the model
   inference. These results are processed and displayed in the
   notebook.*

*To ensure that the results are displayed correctly, run the code in a
recommended browser on one of the following operating systems:* *Ubuntu,
Windows: Chrome* *macOS: Safari*

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Imports <#imports>`__
-  `The model <#the-model>`__

   -  `Download the model <#download-the-model>`__
   -  `Convert Model to OpenVINO IR
      format <#convert-model-to-openvino-ir-format>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Load the model <#load-the-model>`__

-  `Processing <#processing>`__

   -  `Model Inference <#model-inference>`__
   -  `Draw 2D Pose Overlays <#draw-2d-pose-overlays>`__
   -  `Main Processing Function <#main-processing-function>`__

-  `Run <#run>`__

Prerequisites
-------------

`back to top ⬆️ <#table-of-contents>`__

**The ``pythreejs`` extension may not display properly when using the
latest Jupyter Notebook release (2.4.1). Therefore, it is recommended to
use Jupyter Lab instead.**

.. code:: ipython3

    %pip install pythreejs "openvino-dev>=2024.0.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)
    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.2)


.. parsed-literal::

    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (1.23.5)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.14.2)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.8.8)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl.metadata (1.0 kB)


.. parsed-literal::

    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.10)
    Requirement already satisfied: jupyterlab-widgets~=3.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.10)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


.. parsed-literal::

    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.17.2)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.10.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)


.. parsed-literal::

    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.13)
    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)


.. parsed-literal::

    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)
    Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)
    Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Installing collected packages: traittypes, ipydatawidgets, pythreejs


.. parsed-literal::

    Successfully installed ipydatawidgets-4.3.5 pythreejs-2.4.2 traittypes-0.2.1


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import collections
    import sys
    import time
    from pathlib import Path
    
    import cv2
    import ipywidgets as widgets
    import numpy as np
    from IPython.display import clear_output, display
    import openvino as ov
    
    sys.path.append("../utils")
    import notebook_utils as utils
    
    sys.path.append("./engine")
    import engine.engine3js as engine
    from engine.parse_poses import parse_poses

The model
---------

`back to top ⬆️ <#table-of-contents>`__

Download the model
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

We use ``omz_downloader``, which is a command line tool from the
``openvino-dev`` package. ``omz_downloader`` automatically creates a
directory structure and downloads the selected model.

.. code:: ipython3

    # directory where model will be downloaded
    base_model_dir = "model"
    
    # model name as named in Open Model Zoo
    model_name = "human-pose-estimation-3d-0001"
    # selected precision (FP32, FP16)
    precision = "FP32"
    
    BASE_MODEL_NAME = f"{base_model_dir}/public/{model_name}/{model_name}"
    model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
    onnx_path = Path(BASE_MODEL_NAME).with_suffix(".onnx")
    
    ir_model_path = f"model/public/{model_name}/{precision}/{model_name}.xml"
    model_weights_path = f"model/public/{model_name}/{precision}/{model_name}.bin"
    
    if not model_path.exists():
        download_command = (
            f"omz_downloader " f"--name {model_name} " f"--output_dir {base_model_dir}"
        )
        ! $download_command


.. parsed-literal::

    ################|| Downloading human-pose-estimation-3d-0001 ||################
    
    ========== Downloading model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    ... 0%, 32 KB, 920 KB/s, 0 seconds passed
... 0%, 64 KB, 949 KB/s, 0 seconds passed
... 0%, 96 KB, 1395 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1266 KB/s, 0 seconds passed
... 0%, 160 KB, 1567 KB/s, 0 seconds passed
... 1%, 192 KB, 1848 KB/s, 0 seconds passed
... 1%, 224 KB, 2120 KB/s, 0 seconds passed
... 1%, 256 KB, 2387 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 288 KB, 2133 KB/s, 0 seconds passed
... 1%, 320 KB, 2362 KB/s, 0 seconds passed
... 1%, 352 KB, 2591 KB/s, 0 seconds passed
... 2%, 384 KB, 2808 KB/s, 0 seconds passed
... 2%, 416 KB, 3028 KB/s, 0 seconds passed
... 2%, 448 KB, 3251 KB/s, 0 seconds passed
... 2%, 480 KB, 3472 KB/s, 0 seconds passed
... 2%, 512 KB, 3686 KB/s, 0 seconds passed
... 3%, 544 KB, 3906 KB/s, 0 seconds passed
... 3%, 576 KB, 3397 KB/s, 0 seconds passed
... 3%, 608 KB, 3575 KB/s, 0 seconds passed
... 3%, 640 KB, 3753 KB/s, 0 seconds passed
... 3%, 672 KB, 3931 KB/s, 0 seconds passed
... 3%, 704 KB, 4109 KB/s, 0 seconds passed
... 4%, 736 KB, 4286 KB/s, 0 seconds passed
... 4%, 768 KB, 4462 KB/s, 0 seconds passed
... 4%, 800 KB, 4637 KB/s, 0 seconds passed
... 4%, 832 KB, 4812 KB/s, 0 seconds passed
... 4%, 864 KB, 4985 KB/s, 0 seconds passed
... 4%, 896 KB, 5157 KB/s, 0 seconds passed
... 5%, 928 KB, 5328 KB/s, 0 seconds passed
... 5%, 960 KB, 5497 KB/s, 0 seconds passed
... 5%, 992 KB, 5667 KB/s, 0 seconds passed
... 5%, 1024 KB, 5838 KB/s, 0 seconds passed
... 5%, 1056 KB, 6010 KB/s, 0 seconds passed
... 6%, 1088 KB, 6181 KB/s, 0 seconds passed
... 6%, 1120 KB, 6351 KB/s, 0 seconds passed
... 6%, 1152 KB, 6520 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 1184 KB, 5831 KB/s, 0 seconds passed
... 6%, 1216 KB, 5972 KB/s, 0 seconds passed
... 6%, 1248 KB, 6117 KB/s, 0 seconds passed
... 7%, 1280 KB, 6230 KB/s, 0 seconds passed
... 7%, 1312 KB, 6371 KB/s, 0 seconds passed
... 7%, 1344 KB, 6514 KB/s, 0 seconds passed
... 7%, 1376 KB, 6621 KB/s, 0 seconds passed
... 7%, 1408 KB, 6762 KB/s, 0 seconds passed
... 8%, 1440 KB, 6903 KB/s, 0 seconds passed
... 8%, 1472 KB, 7044 KB/s, 0 seconds passed
... 8%, 1504 KB, 7184 KB/s, 0 seconds passed
... 8%, 1536 KB, 7323 KB/s, 0 seconds passed
... 8%, 1568 KB, 7462 KB/s, 0 seconds passed
... 8%, 1600 KB, 7600 KB/s, 0 seconds passed
... 9%, 1632 KB, 7737 KB/s, 0 seconds passed
... 9%, 1664 KB, 7874 KB/s, 0 seconds passed
... 9%, 1696 KB, 8011 KB/s, 0 seconds passed
... 9%, 1728 KB, 8148 KB/s, 0 seconds passed
... 9%, 1760 KB, 8284 KB/s, 0 seconds passed
... 9%, 1792 KB, 8419 KB/s, 0 seconds passed
... 10%, 1824 KB, 8554 KB/s, 0 seconds passed
... 10%, 1856 KB, 8689 KB/s, 0 seconds passed
... 10%, 1888 KB, 8823 KB/s, 0 seconds passed
... 10%, 1920 KB, 8957 KB/s, 0 seconds passed
... 10%, 1952 KB, 9089 KB/s, 0 seconds passed
... 11%, 1984 KB, 9221 KB/s, 0 seconds passed
... 11%, 2016 KB, 9353 KB/s, 0 seconds passed
... 11%, 2048 KB, 9485 KB/s, 0 seconds passed
... 11%, 2080 KB, 9617 KB/s, 0 seconds passed
... 11%, 2112 KB, 9749 KB/s, 0 seconds passed
... 11%, 2144 KB, 9883 KB/s, 0 seconds passed
... 12%, 2176 KB, 10017 KB/s, 0 seconds passed
... 12%, 2208 KB, 10150 KB/s, 0 seconds passed
... 12%, 2240 KB, 10284 KB/s, 0 seconds passed
... 12%, 2272 KB, 10417 KB/s, 0 seconds passed
... 12%, 2304 KB, 10550 KB/s, 0 seconds passed
... 12%, 2336 KB, 10682 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 2368 KB, 9949 KB/s, 0 seconds passed
... 13%, 2400 KB, 10062 KB/s, 0 seconds passed
... 13%, 2432 KB, 10178 KB/s, 0 seconds passed
... 13%, 2464 KB, 10295 KB/s, 0 seconds passed
... 13%, 2496 KB, 10411 KB/s, 0 seconds passed
... 14%, 2528 KB, 10527 KB/s, 0 seconds passed
... 14%, 2560 KB, 10644 KB/s, 0 seconds passed
... 14%, 2592 KB, 10759 KB/s, 0 seconds passed
... 14%, 2624 KB, 10874 KB/s, 0 seconds passed
... 14%, 2656 KB, 10989 KB/s, 0 seconds passed
... 14%, 2688 KB, 11104 KB/s, 0 seconds passed
... 15%, 2720 KB, 11218 KB/s, 0 seconds passed
... 15%, 2752 KB, 11332 KB/s, 0 seconds passed
... 15%, 2784 KB, 11446 KB/s, 0 seconds passed
... 15%, 2816 KB, 11560 KB/s, 0 seconds passed
... 15%, 2848 KB, 11674 KB/s, 0 seconds passed
... 16%, 2880 KB, 11787 KB/s, 0 seconds passed
... 16%, 2912 KB, 11899 KB/s, 0 seconds passed
... 16%, 2944 KB, 12010 KB/s, 0 seconds passed
... 16%, 2976 KB, 12121 KB/s, 0 seconds passed
... 16%, 3008 KB, 12233 KB/s, 0 seconds passed
... 16%, 3040 KB, 12344 KB/s, 0 seconds passed
... 17%, 3072 KB, 12453 KB/s, 0 seconds passed
... 17%, 3104 KB, 12563 KB/s, 0 seconds passed
... 17%, 3136 KB, 12673 KB/s, 0 seconds passed
... 17%, 3168 KB, 12783 KB/s, 0 seconds passed
... 17%, 3200 KB, 12893 KB/s, 0 seconds passed
... 17%, 3232 KB, 13001 KB/s, 0 seconds passed
... 18%, 3264 KB, 13113 KB/s, 0 seconds passed
... 18%, 3296 KB, 13225 KB/s, 0 seconds passed
... 18%, 3328 KB, 13338 KB/s, 0 seconds passed
... 18%, 3360 KB, 13450 KB/s, 0 seconds passed
... 18%, 3392 KB, 13563 KB/s, 0 seconds passed
... 19%, 3424 KB, 13673 KB/s, 0 seconds passed
... 19%, 3456 KB, 13779 KB/s, 0 seconds passed
... 19%, 3488 KB, 13885 KB/s, 0 seconds passed
... 19%, 3520 KB, 13991 KB/s, 0 seconds passed
... 19%, 3552 KB, 14098 KB/s, 0 seconds passed
... 19%, 3584 KB, 14210 KB/s, 0 seconds passed
... 20%, 3616 KB, 14323 KB/s, 0 seconds passed
... 20%, 3648 KB, 14436 KB/s, 0 seconds passed
... 20%, 3680 KB, 14549 KB/s, 0 seconds passed
... 20%, 3712 KB, 14660 KB/s, 0 seconds passed
... 20%, 3744 KB, 14773 KB/s, 0 seconds passed
... 20%, 3776 KB, 14885 KB/s, 0 seconds passed
... 21%, 3808 KB, 14999 KB/s, 0 seconds passed
... 21%, 3840 KB, 15113 KB/s, 0 seconds passed
... 21%, 3872 KB, 15228 KB/s, 0 seconds passed
... 21%, 3904 KB, 15341 KB/s, 0 seconds passed
... 21%, 3936 KB, 15016 KB/s, 0 seconds passed
... 22%, 3968 KB, 14757 KB/s, 0 seconds passed
... 22%, 4000 KB, 14498 KB/s, 0 seconds passed
... 22%, 4032 KB, 14564 KB/s, 0 seconds passed
... 22%, 4064 KB, 14657 KB/s, 0 seconds passed
... 22%, 4096 KB, 14752 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 4128 KB, 14627 KB/s, 0 seconds passed
... 23%, 4160 KB, 14641 KB/s, 0 seconds passed
... 23%, 4192 KB, 14728 KB/s, 0 seconds passed
... 23%, 4224 KB, 14819 KB/s, 0 seconds passed
... 23%, 4256 KB, 14910 KB/s, 0 seconds passed
... 23%, 4288 KB, 15001 KB/s, 0 seconds passed
... 24%, 4320 KB, 15097 KB/s, 0 seconds passed
... 24%, 4352 KB, 15192 KB/s, 0 seconds passed
... 24%, 4384 KB, 15149 KB/s, 0 seconds passed
... 24%, 4416 KB, 15234 KB/s, 0 seconds passed
... 24%, 4448 KB, 15324 KB/s, 0 seconds passed
... 24%, 4480 KB, 15415 KB/s, 0 seconds passed
... 25%, 4512 KB, 15476 KB/s, 0 seconds passed
... 25%, 4544 KB, 15565 KB/s, 0 seconds passed
... 25%, 4576 KB, 15654 KB/s, 0 seconds passed
... 25%, 4608 KB, 15741 KB/s, 0 seconds passed
... 25%, 4640 KB, 15829 KB/s, 0 seconds passed
... 25%, 4672 KB, 15918 KB/s, 0 seconds passed
... 26%, 4704 KB, 16005 KB/s, 0 seconds passed
... 26%, 4736 KB, 16093 KB/s, 0 seconds passed
... 26%, 4768 KB, 16180 KB/s, 0 seconds passed
... 26%, 4800 KB, 16268 KB/s, 0 seconds passed
... 26%, 4832 KB, 16358 KB/s, 0 seconds passed
... 27%, 4864 KB, 16447 KB/s, 0 seconds passed
... 27%, 4896 KB, 16538 KB/s, 0 seconds passed
... 27%, 4928 KB, 16630 KB/s, 0 seconds passed
... 27%, 4960 KB, 16712 KB/s, 0 seconds passed
... 27%, 4992 KB, 16798 KB/s, 0 seconds passed
... 27%, 5024 KB, 16803 KB/s, 0 seconds passed
... 28%, 5056 KB, 16886 KB/s, 0 seconds passed
... 28%, 5088 KB, 16969 KB/s, 0 seconds passed
... 28%, 5120 KB, 17054 KB/s, 0 seconds passed
... 28%, 5152 KB, 17139 KB/s, 0 seconds passed
... 28%, 5184 KB, 17224 KB/s, 0 seconds passed
... 28%, 5216 KB, 17308 KB/s, 0 seconds passed
... 29%, 5248 KB, 17390 KB/s, 0 seconds passed
... 29%, 5280 KB, 17476 KB/s, 0 seconds passed
... 29%, 5312 KB, 17565 KB/s, 0 seconds passed
... 29%, 5344 KB, 17653 KB/s, 0 seconds passed
... 29%, 5376 KB, 17741 KB/s, 0 seconds passed
... 30%, 5408 KB, 17830 KB/s, 0 seconds passed
... 30%, 5440 KB, 17919 KB/s, 0 seconds passed
... 30%, 5472 KB, 18007 KB/s, 0 seconds passed
... 30%, 5504 KB, 18095 KB/s, 0 seconds passed
... 30%, 5536 KB, 18183 KB/s, 0 seconds passed
... 30%, 5568 KB, 18271 KB/s, 0 seconds passed
... 31%, 5600 KB, 18359 KB/s, 0 seconds passed
... 31%, 5632 KB, 18447 KB/s, 0 seconds passed
... 31%, 5664 KB, 18534 KB/s, 0 seconds passed
... 31%, 5696 KB, 18621 KB/s, 0 seconds passed
... 31%, 5728 KB, 18708 KB/s, 0 seconds passed
... 32%, 5760 KB, 18794 KB/s, 0 seconds passed
... 32%, 5792 KB, 18880 KB/s, 0 seconds passed
... 32%, 5824 KB, 18966 KB/s, 0 seconds passed
... 32%, 5856 KB, 19053 KB/s, 0 seconds passed
... 32%, 5888 KB, 19140 KB/s, 0 seconds passed
... 32%, 5920 KB, 19224 KB/s, 0 seconds passed
... 33%, 5952 KB, 19310 KB/s, 0 seconds passed
... 33%, 5984 KB, 19395 KB/s, 0 seconds passed
... 33%, 6016 KB, 19480 KB/s, 0 seconds passed
... 33%, 6048 KB, 19565 KB/s, 0 seconds passed
... 33%, 6080 KB, 19651 KB/s, 0 seconds passed
... 33%, 6112 KB, 19736 KB/s, 0 seconds passed
... 34%, 6144 KB, 19821 KB/s, 0 seconds passed
... 34%, 6176 KB, 19905 KB/s, 0 seconds passed
... 34%, 6208 KB, 19988 KB/s, 0 seconds passed
... 34%, 6240 KB, 20072 KB/s, 0 seconds passed
... 34%, 6272 KB, 20155 KB/s, 0 seconds passed
... 35%, 6304 KB, 20240 KB/s, 0 seconds passed
... 35%, 6336 KB, 20324 KB/s, 0 seconds passed
... 35%, 6368 KB, 20406 KB/s, 0 seconds passed
... 35%, 6400 KB, 20495 KB/s, 0 seconds passed
... 35%, 6432 KB, 20584 KB/s, 0 seconds passed
... 35%, 6464 KB, 20674 KB/s, 0 seconds passed
... 36%, 6496 KB, 20763 KB/s, 0 seconds passed
... 36%, 6528 KB, 20852 KB/s, 0 seconds passed
... 36%, 6560 KB, 20941 KB/s, 0 seconds passed
... 36%, 6592 KB, 21029 KB/s, 0 seconds passed
... 36%, 6624 KB, 21118 KB/s, 0 seconds passed
... 36%, 6656 KB, 21207 KB/s, 0 seconds passed
... 37%, 6688 KB, 21296 KB/s, 0 seconds passed
... 37%, 6720 KB, 21385 KB/s, 0 seconds passed
... 37%, 6752 KB, 21473 KB/s, 0 seconds passed
... 37%, 6784 KB, 21561 KB/s, 0 seconds passed
... 37%, 6816 KB, 21650 KB/s, 0 seconds passed
... 38%, 6848 KB, 21738 KB/s, 0 seconds passed
... 38%, 6880 KB, 21827 KB/s, 0 seconds passed
... 38%, 6912 KB, 21916 KB/s, 0 seconds passed
... 38%, 6944 KB, 21998 KB/s, 0 seconds passed
... 38%, 6976 KB, 22081 KB/s, 0 seconds passed
... 38%, 7008 KB, 22164 KB/s, 0 seconds passed
... 39%, 7040 KB, 22247 KB/s, 0 seconds passed
... 39%, 7072 KB, 22326 KB/s, 0 seconds passed
... 39%, 7104 KB, 22408 KB/s, 0 seconds passed
... 39%, 7136 KB, 22490 KB/s, 0 seconds passed
... 39%, 7168 KB, 22572 KB/s, 0 seconds passed
... 40%, 7200 KB, 22654 KB/s, 0 seconds passed
... 40%, 7232 KB, 22736 KB/s, 0 seconds passed
... 40%, 7264 KB, 22814 KB/s, 0 seconds passed
... 40%, 7296 KB, 22895 KB/s, 0 seconds passed
... 40%, 7328 KB, 22977 KB/s, 0 seconds passed
... 40%, 7360 KB, 23058 KB/s, 0 seconds passed
... 41%, 7392 KB, 23139 KB/s, 0 seconds passed
... 41%, 7424 KB, 23208 KB/s, 0 seconds passed
... 41%, 7456 KB, 23280 KB/s, 0 seconds passed
... 41%, 7488 KB, 23361 KB/s, 0 seconds passed
... 41%, 7520 KB, 23438 KB/s, 0 seconds passed
... 41%, 7552 KB, 23518 KB/s, 0 seconds passed
... 42%, 7584 KB, 23602 KB/s, 0 seconds passed
... 42%, 7616 KB, 23678 KB/s, 0 seconds passed
... 42%, 7648 KB, 23759 KB/s, 0 seconds passed
... 42%, 7680 KB, 23839 KB/s, 0 seconds passed
... 42%, 7712 KB, 23918 KB/s, 0 seconds passed
... 43%, 7744 KB, 23994 KB/s, 0 seconds passed
... 43%, 7776 KB, 24069 KB/s, 0 seconds passed
... 43%, 7808 KB, 24150 KB/s, 0 seconds passed
... 43%, 7840 KB, 24229 KB/s, 0 seconds passed
... 43%, 7872 KB, 24308 KB/s, 0 seconds passed
... 43%, 7904 KB, 24383 KB/s, 0 seconds passed
... 44%, 7936 KB, 24462 KB/s, 0 seconds passed
... 44%, 7968 KB, 24541 KB/s, 0 seconds passed
... 44%, 8000 KB, 24616 KB/s, 0 seconds passed
... 44%, 8032 KB, 24683 KB/s, 0 seconds passed
... 44%, 8064 KB, 24751 KB/s, 0 seconds passed
... 45%, 8096 KB, 24829 KB/s, 0 seconds passed
... 45%, 8128 KB, 24917 KB/s, 0 seconds passed
... 45%, 8160 KB, 25004 KB/s, 0 seconds passed
... 45%, 8192 KB, 25079 KB/s, 0 seconds passed
... 45%, 8224 KB, 25157 KB/s, 0 seconds passed
... 45%, 8256 KB, 25235 KB/s, 0 seconds passed
... 46%, 8288 KB, 25308 KB/s, 0 seconds passed
... 46%, 8320 KB, 25389 KB/s, 0 seconds passed
... 46%, 8352 KB, 25463 KB/s, 0 seconds passed
... 46%, 8384 KB, 25540 KB/s, 0 seconds passed
... 46%, 8416 KB, 25617 KB/s, 0 seconds passed
... 46%, 8448 KB, 25690 KB/s, 0 seconds passed
... 47%, 8480 KB, 25766 KB/s, 0 seconds passed
... 47%, 8512 KB, 25843 KB/s, 0 seconds passed
... 47%, 8544 KB, 25915 KB/s, 0 seconds passed
... 47%, 8576 KB, 25989 KB/s, 0 seconds passed
... 47%, 8608 KB, 26053 KB/s, 0 seconds passed
... 48%, 8640 KB, 26116 KB/s, 0 seconds passed
... 48%, 8672 KB, 26192 KB/s, 0 seconds passed
... 48%, 8704 KB, 26276 KB/s, 0 seconds passed

.. parsed-literal::

    ... 48%, 8736 KB, 26361 KB/s, 0 seconds passed
... 48%, 8768 KB, 26436 KB/s, 0 seconds passed
... 48%, 8800 KB, 26512 KB/s, 0 seconds passed
... 49%, 8832 KB, 26588 KB/s, 0 seconds passed
... 49%, 8864 KB, 26659 KB/s, 0 seconds passed
... 49%, 8896 KB, 26734 KB/s, 0 seconds passed
... 49%, 8928 KB, 26805 KB/s, 0 seconds passed
... 49%, 8960 KB, 26880 KB/s, 0 seconds passed
... 49%, 8992 KB, 26955 KB/s, 0 seconds passed
... 50%, 9024 KB, 27025 KB/s, 0 seconds passed
... 50%, 9056 KB, 27100 KB/s, 0 seconds passed
... 50%, 9088 KB, 27174 KB/s, 0 seconds passed
... 50%, 9120 KB, 27245 KB/s, 0 seconds passed
... 50%, 9152 KB, 27319 KB/s, 0 seconds passed
... 51%, 9184 KB, 27393 KB/s, 0 seconds passed
... 51%, 9216 KB, 27463 KB/s, 0 seconds passed
... 51%, 9248 KB, 27537 KB/s, 0 seconds passed
... 51%, 9280 KB, 27611 KB/s, 0 seconds passed
... 51%, 9312 KB, 27684 KB/s, 0 seconds passed
... 51%, 9344 KB, 27754 KB/s, 0 seconds passed
... 52%, 9376 KB, 27827 KB/s, 0 seconds passed
... 52%, 9408 KB, 27696 KB/s, 0 seconds passed
... 52%, 9440 KB, 27750 KB/s, 0 seconds passed
... 52%, 9472 KB, 27822 KB/s, 0 seconds passed
... 52%, 9504 KB, 27891 KB/s, 0 seconds passed
... 53%, 9536 KB, 27959 KB/s, 0 seconds passed
... 53%, 9568 KB, 28017 KB/s, 0 seconds passed
... 53%, 9600 KB, 28078 KB/s, 0 seconds passed
... 53%, 9632 KB, 28155 KB/s, 0 seconds passed
... 53%, 9664 KB, 28237 KB/s, 0 seconds passed
... 53%, 9696 KB, 28313 KB/s, 0 seconds passed
... 54%, 9728 KB, 28380 KB/s, 0 seconds passed
... 54%, 9760 KB, 28442 KB/s, 0 seconds passed
... 54%, 9792 KB, 28504 KB/s, 0 seconds passed
... 54%, 9824 KB, 28565 KB/s, 0 seconds passed
... 54%, 9856 KB, 28628 KB/s, 0 seconds passed
... 54%, 9888 KB, 28690 KB/s, 0 seconds passed
... 55%, 9920 KB, 28753 KB/s, 0 seconds passed
... 55%, 9952 KB, 28816 KB/s, 0 seconds passed
... 55%, 9984 KB, 28878 KB/s, 0 seconds passed
... 55%, 10016 KB, 28940 KB/s, 0 seconds passed
... 55%, 10048 KB, 28998 KB/s, 0 seconds passed
... 56%, 10080 KB, 29059 KB/s, 0 seconds passed
... 56%, 10112 KB, 29122 KB/s, 0 seconds passed
... 56%, 10144 KB, 29188 KB/s, 0 seconds passed
... 56%, 10176 KB, 29256 KB/s, 0 seconds passed
... 56%, 10208 KB, 29324 KB/s, 0 seconds passed

.. parsed-literal::

    ... 56%, 10240 KB, 22676 KB/s, 0 seconds passed
... 57%, 10272 KB, 22720 KB/s, 0 seconds passed
... 57%, 10304 KB, 22771 KB/s, 0 seconds passed
... 57%, 10336 KB, 22826 KB/s, 0 seconds passed
... 57%, 10368 KB, 22436 KB/s, 0 seconds passed
... 57%, 10400 KB, 22478 KB/s, 0 seconds passed
... 57%, 10432 KB, 22525 KB/s, 0 seconds passed
... 58%, 10464 KB, 22574 KB/s, 0 seconds passed
... 58%, 10496 KB, 22624 KB/s, 0 seconds passed
... 58%, 10528 KB, 22675 KB/s, 0 seconds passed
... 58%, 10560 KB, 22723 KB/s, 0 seconds passed
... 58%, 10592 KB, 22773 KB/s, 0 seconds passed
... 59%, 10624 KB, 22824 KB/s, 0 seconds passed
... 59%, 10656 KB, 22874 KB/s, 0 seconds passed
... 59%, 10688 KB, 22923 KB/s, 0 seconds passed
... 59%, 10720 KB, 22971 KB/s, 0 seconds passed
... 59%, 10752 KB, 23022 KB/s, 0 seconds passed
... 59%, 10784 KB, 23072 KB/s, 0 seconds passed
... 60%, 10816 KB, 23121 KB/s, 0 seconds passed
... 60%, 10848 KB, 23171 KB/s, 0 seconds passed
... 60%, 10880 KB, 23220 KB/s, 0 seconds passed
... 60%, 10912 KB, 23270 KB/s, 0 seconds passed
... 60%, 10944 KB, 23320 KB/s, 0 seconds passed
... 61%, 10976 KB, 23369 KB/s, 0 seconds passed
... 61%, 11008 KB, 23418 KB/s, 0 seconds passed
... 61%, 11040 KB, 23466 KB/s, 0 seconds passed
... 61%, 11072 KB, 23515 KB/s, 0 seconds passed
... 61%, 11104 KB, 23564 KB/s, 0 seconds passed
... 61%, 11136 KB, 23613 KB/s, 0 seconds passed
... 62%, 11168 KB, 23662 KB/s, 0 seconds passed
... 62%, 11200 KB, 23710 KB/s, 0 seconds passed
... 62%, 11232 KB, 23759 KB/s, 0 seconds passed
... 62%, 11264 KB, 23807 KB/s, 0 seconds passed
... 62%, 11296 KB, 23857 KB/s, 0 seconds passed
... 62%, 11328 KB, 23911 KB/s, 0 seconds passed
... 63%, 11360 KB, 23967 KB/s, 0 seconds passed
... 63%, 11392 KB, 24021 KB/s, 0 seconds passed
... 63%, 11424 KB, 24076 KB/s, 0 seconds passed
... 63%, 11456 KB, 24126 KB/s, 0 seconds passed
... 63%, 11488 KB, 24171 KB/s, 0 seconds passed
... 64%, 11520 KB, 24222 KB/s, 0 seconds passed
... 64%, 11552 KB, 24267 KB/s, 0 seconds passed
... 64%, 11584 KB, 24313 KB/s, 0 seconds passed
... 64%, 11616 KB, 24358 KB/s, 0 seconds passed
... 64%, 11648 KB, 24406 KB/s, 0 seconds passed
... 64%, 11680 KB, 24446 KB/s, 0 seconds passed
... 65%, 11712 KB, 24499 KB/s, 0 seconds passed
... 65%, 11744 KB, 24545 KB/s, 0 seconds passed
... 65%, 11776 KB, 24592 KB/s, 0 seconds passed
... 65%, 11808 KB, 24632 KB/s, 0 seconds passed
... 65%, 11840 KB, 24677 KB/s, 0 seconds passed
... 65%, 11872 KB, 24726 KB/s, 0 seconds passed
... 66%, 11904 KB, 24775 KB/s, 0 seconds passed
... 66%, 11936 KB, 24825 KB/s, 0 seconds passed
... 66%, 11968 KB, 24866 KB/s, 0 seconds passed
... 66%, 12000 KB, 24913 KB/s, 0 seconds passed
... 66%, 12032 KB, 24958 KB/s, 0 seconds passed
... 67%, 12064 KB, 25002 KB/s, 0 seconds passed
... 67%, 12096 KB, 25049 KB/s, 0 seconds passed
... 67%, 12128 KB, 25092 KB/s, 0 seconds passed
... 67%, 12160 KB, 25139 KB/s, 0 seconds passed
... 67%, 12192 KB, 25184 KB/s, 0 seconds passed
... 67%, 12224 KB, 25226 KB/s, 0 seconds passed
... 68%, 12256 KB, 25273 KB/s, 0 seconds passed

.. parsed-literal::

    ... 68%, 12288 KB, 25317 KB/s, 0 seconds passed
... 68%, 12320 KB, 25360 KB/s, 0 seconds passed
... 68%, 12352 KB, 25404 KB/s, 0 seconds passed
... 68%, 12384 KB, 25453 KB/s, 0 seconds passed
... 69%, 12416 KB, 25499 KB/s, 0 seconds passed
... 69%, 12448 KB, 25546 KB/s, 0 seconds passed
... 69%, 12480 KB, 25587 KB/s, 0 seconds passed
... 69%, 12512 KB, 25633 KB/s, 0 seconds passed
... 69%, 12544 KB, 25676 KB/s, 0 seconds passed
... 69%, 12576 KB, 25716 KB/s, 0 seconds passed
... 70%, 12608 KB, 25768 KB/s, 0 seconds passed
... 70%, 12640 KB, 25808 KB/s, 0 seconds passed
... 70%, 12672 KB, 25851 KB/s, 0 seconds passed
... 70%, 12704 KB, 25897 KB/s, 0 seconds passed
... 70%, 12736 KB, 25942 KB/s, 0 seconds passed
... 70%, 12768 KB, 25977 KB/s, 0 seconds passed
... 71%, 12800 KB, 26020 KB/s, 0 seconds passed
... 71%, 12832 KB, 26069 KB/s, 0 seconds passed
... 71%, 12864 KB, 26120 KB/s, 0 seconds passed
... 71%, 12896 KB, 26168 KB/s, 0 seconds passed
... 71%, 12928 KB, 26213 KB/s, 0 seconds passed
... 72%, 12960 KB, 26253 KB/s, 0 seconds passed
... 72%, 12992 KB, 26296 KB/s, 0 seconds passed
... 72%, 13024 KB, 26348 KB/s, 0 seconds passed
... 72%, 13056 KB, 26385 KB/s, 0 seconds passed
... 72%, 13088 KB, 26427 KB/s, 0 seconds passed
... 72%, 13120 KB, 26471 KB/s, 0 seconds passed
... 73%, 13152 KB, 26514 KB/s, 0 seconds passed
... 73%, 13184 KB, 26567 KB/s, 0 seconds passed
... 73%, 13216 KB, 26615 KB/s, 0 seconds passed
... 73%, 13248 KB, 26654 KB/s, 0 seconds passed
... 73%, 13280 KB, 26704 KB/s, 0 seconds passed
... 73%, 13312 KB, 26754 KB/s, 0 seconds passed
... 74%, 13344 KB, 26793 KB/s, 0 seconds passed
... 74%, 13376 KB, 26827 KB/s, 0 seconds passed
... 74%, 13408 KB, 26879 KB/s, 0 seconds passed
... 74%, 13440 KB, 26919 KB/s, 0 seconds passed
... 74%, 13472 KB, 26958 KB/s, 0 seconds passed
... 75%, 13504 KB, 27009 KB/s, 0 seconds passed
... 75%, 13536 KB, 27057 KB/s, 0 seconds passed
... 75%, 13568 KB, 27098 KB/s, 0 seconds passed
... 75%, 13600 KB, 27147 KB/s, 0 seconds passed
... 75%, 13632 KB, 26722 KB/s, 0 seconds passed
... 75%, 13664 KB, 26693 KB/s, 0 seconds passed
... 76%, 13696 KB, 26733 KB/s, 0 seconds passed
... 76%, 13728 KB, 26775 KB/s, 0 seconds passed
... 76%, 13760 KB, 26819 KB/s, 0 seconds passed
... 76%, 13792 KB, 26862 KB/s, 0 seconds passed
... 76%, 13824 KB, 26904 KB/s, 0 seconds passed
... 77%, 13856 KB, 26947 KB/s, 0 seconds passed
... 77%, 13888 KB, 26988 KB/s, 0 seconds passed
... 77%, 13920 KB, 27030 KB/s, 0 seconds passed
... 77%, 13952 KB, 27071 KB/s, 0 seconds passed
... 77%, 13984 KB, 27114 KB/s, 0 seconds passed
... 77%, 14016 KB, 27155 KB/s, 0 seconds passed
... 78%, 14048 KB, 27198 KB/s, 0 seconds passed
... 78%, 14080 KB, 27239 KB/s, 0 seconds passed
... 78%, 14112 KB, 27281 KB/s, 0 seconds passed
... 78%, 14144 KB, 27323 KB/s, 0 seconds passed
... 78%, 14176 KB, 27365 KB/s, 0 seconds passed
... 78%, 14208 KB, 27407 KB/s, 0 seconds passed
... 79%, 14240 KB, 27447 KB/s, 0 seconds passed
... 79%, 14272 KB, 27489 KB/s, 0 seconds passed
... 79%, 14304 KB, 27534 KB/s, 0 seconds passed
... 79%, 14336 KB, 27580 KB/s, 0 seconds passed
... 79%, 14368 KB, 27626 KB/s, 0 seconds passed
... 80%, 14400 KB, 27672 KB/s, 0 seconds passed
... 80%, 14432 KB, 27718 KB/s, 0 seconds passed
... 80%, 14464 KB, 27764 KB/s, 0 seconds passed
... 80%, 14496 KB, 27811 KB/s, 0 seconds passed

.. parsed-literal::

    ... 80%, 14528 KB, 26120 KB/s, 0 seconds passed
... 80%, 14560 KB, 26154 KB/s, 0 seconds passed
... 81%, 14592 KB, 26193 KB/s, 0 seconds passed
... 81%, 14624 KB, 26232 KB/s, 0 seconds passed
... 81%, 14656 KB, 26271 KB/s, 0 seconds passed
... 81%, 14688 KB, 26311 KB/s, 0 seconds passed
... 81%, 14720 KB, 26349 KB/s, 0 seconds passed
... 82%, 14752 KB, 26389 KB/s, 0 seconds passed
... 82%, 14784 KB, 26427 KB/s, 0 seconds passed
... 82%, 14816 KB, 26466 KB/s, 0 seconds passed
... 82%, 14848 KB, 26505 KB/s, 0 seconds passed
... 82%, 14880 KB, 26544 KB/s, 0 seconds passed
... 82%, 14912 KB, 26582 KB/s, 0 seconds passed
... 83%, 14944 KB, 26621 KB/s, 0 seconds passed
... 83%, 14976 KB, 26661 KB/s, 0 seconds passed
... 83%, 15008 KB, 26700 KB/s, 0 seconds passed
... 83%, 15040 KB, 26736 KB/s, 0 seconds passed
... 83%, 15072 KB, 26774 KB/s, 0 seconds passed
... 83%, 15104 KB, 26814 KB/s, 0 seconds passed
... 84%, 15136 KB, 26856 KB/s, 0 seconds passed
... 84%, 15168 KB, 26899 KB/s, 0 seconds passed
... 84%, 15200 KB, 26942 KB/s, 0 seconds passed
... 84%, 15232 KB, 26984 KB/s, 0 seconds passed
... 84%, 15264 KB, 27027 KB/s, 0 seconds passed
... 85%, 15296 KB, 27070 KB/s, 0 seconds passed
... 85%, 15328 KB, 27114 KB/s, 0 seconds passed

.. parsed-literal::

    ... 85%, 15360 KB, 25777 KB/s, 0 seconds passed
... 85%, 15392 KB, 25807 KB/s, 0 seconds passed
... 85%, 15424 KB, 25842 KB/s, 0 seconds passed
... 85%, 15456 KB, 25880 KB/s, 0 seconds passed
... 86%, 15488 KB, 25801 KB/s, 0 seconds passed
... 86%, 15520 KB, 25831 KB/s, 0 seconds passed
... 86%, 15552 KB, 25864 KB/s, 0 seconds passed
... 86%, 15584 KB, 25900 KB/s, 0 seconds passed
... 86%, 15616 KB, 25937 KB/s, 0 seconds passed
... 86%, 15648 KB, 25973 KB/s, 0 seconds passed
... 87%, 15680 KB, 26009 KB/s, 0 seconds passed
... 87%, 15712 KB, 26045 KB/s, 0 seconds passed
... 87%, 15744 KB, 26082 KB/s, 0 seconds passed
... 87%, 15776 KB, 26121 KB/s, 0 seconds passed
... 87%, 15808 KB, 26161 KB/s, 0 seconds passed
... 88%, 15840 KB, 26201 KB/s, 0 seconds passed
... 88%, 15872 KB, 26241 KB/s, 0 seconds passed
... 88%, 15904 KB, 26276 KB/s, 0 seconds passed
... 88%, 15936 KB, 26312 KB/s, 0 seconds passed
... 88%, 15968 KB, 26347 KB/s, 0 seconds passed
... 88%, 16000 KB, 26382 KB/s, 0 seconds passed
... 89%, 16032 KB, 26418 KB/s, 0 seconds passed
... 89%, 16064 KB, 26453 KB/s, 0 seconds passed
... 89%, 16096 KB, 26490 KB/s, 0 seconds passed
... 89%, 16128 KB, 26525 KB/s, 0 seconds passed
... 89%, 16160 KB, 26562 KB/s, 0 seconds passed
... 90%, 16192 KB, 26597 KB/s, 0 seconds passed
... 90%, 16224 KB, 26633 KB/s, 0 seconds passed
... 90%, 16256 KB, 26668 KB/s, 0 seconds passed
... 90%, 16288 KB, 26704 KB/s, 0 seconds passed
... 90%, 16320 KB, 26740 KB/s, 0 seconds passed
... 90%, 16352 KB, 26775 KB/s, 0 seconds passed
... 91%, 16384 KB, 26811 KB/s, 0 seconds passed
... 91%, 16416 KB, 26845 KB/s, 0 seconds passed
... 91%, 16448 KB, 26881 KB/s, 0 seconds passed
... 91%, 16480 KB, 26917 KB/s, 0 seconds passed
... 91%, 16512 KB, 26953 KB/s, 0 seconds passed
... 91%, 16544 KB, 26988 KB/s, 0 seconds passed
... 92%, 16576 KB, 27023 KB/s, 0 seconds passed
... 92%, 16608 KB, 27059 KB/s, 0 seconds passed
... 92%, 16640 KB, 27094 KB/s, 0 seconds passed
... 92%, 16672 KB, 27130 KB/s, 0 seconds passed
... 92%, 16704 KB, 27165 KB/s, 0 seconds passed
... 93%, 16736 KB, 27205 KB/s, 0 seconds passed
... 93%, 16768 KB, 27247 KB/s, 0 seconds passed
... 93%, 16800 KB, 27288 KB/s, 0 seconds passed
... 93%, 16832 KB, 27330 KB/s, 0 seconds passed
... 93%, 16864 KB, 27372 KB/s, 0 seconds passed
... 93%, 16896 KB, 27414 KB/s, 0 seconds passed
... 94%, 16928 KB, 27456 KB/s, 0 seconds passed
... 94%, 16960 KB, 27497 KB/s, 0 seconds passed
... 94%, 16992 KB, 27538 KB/s, 0 seconds passed
... 94%, 17024 KB, 27580 KB/s, 0 seconds passed
... 94%, 17056 KB, 27621 KB/s, 0 seconds passed
... 94%, 17088 KB, 27663 KB/s, 0 seconds passed
... 95%, 17120 KB, 27705 KB/s, 0 seconds passed
... 95%, 17152 KB, 27746 KB/s, 0 seconds passed
... 95%, 17184 KB, 27788 KB/s, 0 seconds passed
... 95%, 17216 KB, 27828 KB/s, 0 seconds passed
... 95%, 17248 KB, 27870 KB/s, 0 seconds passed
... 96%, 17280 KB, 27911 KB/s, 0 seconds passed
... 96%, 17312 KB, 27953 KB/s, 0 seconds passed
... 96%, 17344 KB, 27996 KB/s, 0 seconds passed
... 96%, 17376 KB, 28039 KB/s, 0 seconds passed
... 96%, 17408 KB, 28082 KB/s, 0 seconds passed
... 96%, 17440 KB, 28125 KB/s, 0 seconds passed

.. parsed-literal::

    ... 97%, 17472 KB, 25432 KB/s, 0 seconds passed
... 97%, 17504 KB, 25462 KB/s, 0 seconds passed
... 97%, 17536 KB, 25467 KB/s, 0 seconds passed
... 97%, 17568 KB, 25496 KB/s, 0 seconds passed
... 97%, 17600 KB, 25528 KB/s, 0 seconds passed
... 98%, 17632 KB, 25562 KB/s, 0 seconds passed
... 98%, 17664 KB, 25594 KB/s, 0 seconds passed

.. parsed-literal::

    ... 98%, 17696 KB, 25624 KB/s, 0 seconds passed
... 98%, 17728 KB, 25655 KB/s, 0 seconds passed
... 98%, 17760 KB, 25687 KB/s, 0 seconds passed
... 98%, 17792 KB, 25719 KB/s, 0 seconds passed
... 99%, 17824 KB, 25751 KB/s, 0 seconds passed
... 99%, 17856 KB, 25783 KB/s, 0 seconds passed
... 99%, 17888 KB, 25818 KB/s, 0 seconds passed
... 99%, 17920 KB, 25853 KB/s, 0 seconds passed
... 99%, 17952 KB, 25888 KB/s, 0 seconds passed
... 99%, 17984 KB, 25923 KB/s, 0 seconds passed
... 100%, 17990 KB, 25919 KB/s, 0 seconds passed

    
    ========== Unpacking model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    


Convert Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

The selected model comes from the public directory, which means it must
be converted into OpenVINO Intermediate Representation (OpenVINO IR). We
use ``omz_converter`` to convert the ONNX format model to the OpenVINO
IR format.

.. code:: ipython3

    if not onnx_path.exists():
        convert_command = (
            f"omz_converter "
            f"--name {model_name} "
            f"--precisions {precision} "
            f"--download_dir {base_model_dir} "
            f"--output_dir {base_model_dir}"
        )
        ! $convert_command


.. parsed-literal::

    ========== Converting human-pose-estimation-3d-0001 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False
    


.. parsed-literal::

    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin


.. parsed-literal::

    


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

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



Load the model
~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Converted models are located in a fixed structure, which indicates
vendor, model name and precision.

First, initialize the inference engine, OpenVINO Runtime. Then, read the
network architecture and model weights from the ``.bin`` and ``.xml``
files to compile for the desired device. An inference request is then
created to infer the compiled model.

.. code:: ipython3

    # initialize inference engine
    core = ov.Core()
    # read the network and corresponding weights from file
    model = core.read_model(model=ir_model_path, weights=model_weights_path)
    # load the model on the specified device
    compiled_model = core.compile_model(model=model, device_name=device.value)
    infer_request = compiled_model.create_infer_request()
    input_tensor_name = model.inputs[0].get_any_name()
    
    # get input and output names of nodes
    input_layer = compiled_model.input(0)
    output_layers = list(compiled_model.outputs)

The input for the model is data from the input image and the outputs are
heat maps, PAF (part affinity fields) and features.

.. code:: ipython3

    input_layer.any_name, [o.any_name for o in output_layers]




.. parsed-literal::

    ('data', ['features', 'heatmaps', 'pafs'])



Processing
----------

`back to top ⬆️ <#table-of-contents>`__

Model Inference
~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Frames captured from video files or the live webcam are used as the
input for the 3D model. This is how you obtain the output heat maps, PAF
(part affinity fields) and features.

.. code:: ipython3

    def model_infer(scaled_img, stride):
        """
        Run model inference on the input image
    
        Parameters:
            scaled_img: resized image according to the input size of the model
            stride: int, the stride of the window
        """
    
        # Remove excess space from the picture
        img = scaled_img[
            0 : scaled_img.shape[0] - (scaled_img.shape[0] % stride),
            0 : scaled_img.shape[1] - (scaled_img.shape[1] % stride),
        ]
    
        img = np.transpose(img, (2, 0, 1))[
            None,
        ]
        infer_request.infer({input_tensor_name: img})
        # A set of three inference results is obtained
        results = {
            name: infer_request.get_tensor(name).data[:]
            for name in {"features", "heatmaps", "pafs"}
        }
        # Get the results
        results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])
    
        return results

Draw 2D Pose Overlays
~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

We need to define some connections between the joints in advance, so
that we can draw the structure of the human body in the resulting image
after obtaining the inference results. Joints are drawn as circles and
limbs are drawn as lines. The code is based on the `3D Human Pose
Estimation
Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/human_pose_estimation_3d_demo/python>`__
from Open Model Zoo.

.. code:: ipython3

    # 3D edge index array
    body_edges = np.array(
        [
            [0, 1], 
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [1, 15], [15, 16],            # nose - l_eye - l_ear
            [1, 17], [17, 18],            # nose - r_eye - r_ear
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )
    
    
    body_edges_2d = np.array(
        [
            [0, 1],                       # neck - nose
            [1, 16], [16, 18],            # nose - l_eye - l_ear
            [1, 15], [15, 17],            # nose - r_eye - r_ear
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )  
    
    
    def draw_poses(frame, poses_2d, scaled_img, use_popup):
        """
        Draw 2D pose overlays on the image to visualize estimated poses.
        Joints are drawn as circles and limbs are drawn as lines.
    
        :param frame: the input image
        :param poses_2d: array of human joint pairs
        """
        for pose in poses_2d:
            pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
            was_found = pose[2] > 0
    
            pose[0], pose[1] = (
                pose[0] * frame.shape[1] / scaled_img.shape[1],
                pose[1] * frame.shape[0] / scaled_img.shape[0],
            )
    
            # Draw joints.
            for edge in body_edges_2d:
                if was_found[edge[0]] and was_found[edge[1]]:
                    cv2.line(
                        frame,
                        tuple(pose[0:2, edge[0]].astype(np.int32)),
                        tuple(pose[0:2, edge[1]].astype(np.int32)),
                        (255, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
            # Draw limbs.
            for kpt_id in range(pose.shape[1]):
                if pose[2, kpt_id] != -1:
                    cv2.circle(
                        frame,
                        tuple(pose[0:2, kpt_id].astype(np.int32)),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
    
        return frame

Main Processing Function
~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Run 3D pose estimation on the specified source. It could be either a
webcam feed or a video file.

.. code:: ipython3

    def run_pose_estimation(source=0, flip=False, use_popup=False, skip_frames=0):
        """
        2D image as input, using OpenVINO as inference backend,
        get joints 3D coordinates, and draw 3D human skeleton in the scene
    
        :param source:      The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        :param flip:        To be used by VideoPlayer function for flipping capture image.
        :param use_popup:   False for showing encoded frames over this notebook, True for creating a popup window.
        :param skip_frames: Number of frames to skip at the beginning of the video.
        """
    
        focal_length = -1  # default
        stride = 8
        player = None
        skeleton_set = None
    
        try:
            # create video player to play with target fps  video_path
            # get the frame from camera
            # You can skip first N frames to fast forward video. change 'skip_first_frames'
            player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_frames)
            # start capturing
            player.start()
    
            input_image = player.next()
            # set the window size
            resize_scale = 450 / input_image.shape[1]
            windows_width = int(input_image.shape[1] * resize_scale)
            windows_height = int(input_image.shape[0] * resize_scale)
    
            # use visualization library
            engine3D = engine.Engine3js(grid=True, axis=True, view_width=windows_width, view_height=windows_height)
    
            if use_popup:
                # display the 3D human pose in this notebook, and origin frame in popup window
                display(engine3D.renderer)
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
            else:
                # set the 2D image box, show both human pose and image in the notebook
                imgbox = widgets.Image(
                    format="jpg", height=windows_height, width=windows_width
                )
                display(widgets.HBox([engine3D.renderer, imgbox]))
    
            skeleton = engine.Skeleton(body_edges=body_edges)
    
            processing_times = collections.deque()
    
            while True:
                # grab the frame
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
    
                # resize image and change dims to fit neural network input
                # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001)
                scaled_img = cv2.resize(frame, dsize=(model.inputs[0].shape[3], model.inputs[0].shape[2]))
    
                if focal_length < 0:  # Focal length is unknown
                    focal_length = np.float32(0.8 * scaled_img.shape[1])
    
                # inference start
                start_time = time.time()
                # get results
                inference_result = model_infer(scaled_img, stride)
    
                # inference stop
                stop_time = time.time()
                processing_times.append(stop_time - start_time)
                # Process the point to point coordinates of the data
                poses_3d, poses_2d = parse_poses(inference_result, 1, stride, focal_length, True)
    
                # use processing times from last 200 frames
                if len(processing_times) > 200:
                    processing_times.popleft()
    
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
    
                if len(poses_3d) > 0:
                    # From here, you can rotate the 3D point positions using the function "draw_poses",
                    # or you can directly make the correct mapping below to properly display the object image on the screen
                    poses_3d_copy = poses_3d.copy()
                    x = poses_3d_copy[:, 0::4]
                    y = poses_3d_copy[:, 1::4]
                    z = poses_3d_copy[:, 2::4]
                    poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
                        -z + np.ones(poses_3d[:, 2::4].shape) * 200,
                        -y + np.ones(poses_3d[:, 2::4].shape) * 100,
                        -x,
                    )
    
                    poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                    people = skeleton(poses_3d=poses_3d)
    
                    try:
                        engine3D.scene_remove(skeleton_set)
                    except Exception:
                        pass
    
                    engine3D.scene_add(people)
                    skeleton_set = people
    
                    # draw 2D
                    frame = draw_poses(frame, poses_2d, scaled_img, use_popup)
    
                else:
                    try:
                        engine3D.scene_remove(skeleton_set)
                        skeleton_set = None
                    except Exception:
                        pass
    
                cv2.putText(
                    frame,
                    f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
    
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27, use ESC to exit
                    if key == 27:
                        break
                else:
                    # encode numpy array to jpg
                    imgbox.value = cv2.imencode(
                        ".jpg",
                        frame,
                        params=[cv2.IMWRITE_JPEG_QUALITY, 90],
                    )[1].tobytes()
    
                engine3D.renderer.render(engine3D.scene, engine3D.cam)
    
        except KeyboardInterrupt:
            print("Interrupted")
        except RuntimeError as e:
            print(e)
        finally:
            clear_output()
            if player is not None:
                # stop capturing
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()
            if skeleton_set:
                engine3D.scene_remove(skeleton_set)

Run
---

`back to top ⬆️ <#table-of-contents>`__

Run, using a webcam as the video input. By default, the primary webcam
is set with ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering, set
``use_popup=True``.

   **NOTE**:

   *1. To use this notebook with a webcam, you need to run the notebook
   on a computer with a webcam. If you run the notebook on a server
   (e.g. Binder), the webcam will not work.*

   *2. Popup mode may not work if you run this notebook on a remote
   computer (e.g. Binder).*

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
will work.

Using the following method, you can click and move your mouse over the
picture on the left to interact.

.. code:: ipython3

    USE_WEBCAM = False
    
    cam_id = 0
    video_path = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
    
    source = cam_id if USE_WEBCAM else video_path
    
    run_pose_estimation(source=source, flip=isinstance(source, int), use_popup=False)

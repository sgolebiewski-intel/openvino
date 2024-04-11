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



**The ``pythreejs`` extension may not display properly when using the
latest Jupyter Notebook release (2.4.1). Therefore, it is recommended to
use Jupyter Lab instead.**

.. code:: ipython3

    %pip install pythreejs "openvino-dev>=2024.0.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl.metadata (5.4 kB)


.. parsed-literal::

    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)
    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.2)


.. parsed-literal::

    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (1.23.5)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.14.2)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (3.1)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.1.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl.metadata (1.0 kB)
    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.10)


.. parsed-literal::

    Requirement already satisfied: jupyterlab-widgets~=3.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.10)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


.. parsed-literal::

    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.17.2)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.11.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)


.. parsed-literal::

    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.4)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.13)
    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)


.. parsed-literal::

    Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)
    Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)


.. parsed-literal::

    Installing collected packages: traittypes, ipydatawidgets, pythreejs


.. parsed-literal::

    Successfully installed ipydatawidgets-4.3.5 pythreejs-2.4.2 traittypes-0.2.1


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



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
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    import notebook_utils as utils
    
    sys.path.append("./engine")
    import engine.engine3js as engine
    from engine.parse_poses import parse_poses

The model
---------



Download the model
~~~~~~~~~~~~~~~~~~



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

    ... 0%, 32 KB, 996 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 981 KB/s, 0 seconds passed
... 0%, 96 KB, 1406 KB/s, 0 seconds passed
... 0%, 128 KB, 1299 KB/s, 0 seconds passed
... 0%, 160 KB, 1592 KB/s, 0 seconds passed
... 1%, 192 KB, 1873 KB/s, 0 seconds passed
... 1%, 224 KB, 2130 KB/s, 0 seconds passed
... 1%, 256 KB, 2384 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 288 KB, 2181 KB/s, 0 seconds passed
... 1%, 320 KB, 2415 KB/s, 0 seconds passed
... 1%, 352 KB, 2647 KB/s, 0 seconds passed
... 2%, 384 KB, 2864 KB/s, 0 seconds passed
... 2%, 416 KB, 3083 KB/s, 0 seconds passed
... 2%, 448 KB, 3297 KB/s, 0 seconds passed
... 2%, 480 KB, 3524 KB/s, 0 seconds passed
... 2%, 512 KB, 3664 KB/s, 0 seconds passed
... 3%, 544 KB, 3872 KB/s, 0 seconds passed
... 3%, 576 KB, 3478 KB/s, 0 seconds passed
... 3%, 608 KB, 3661 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 640 KB, 3846 KB/s, 0 seconds passed
... 3%, 672 KB, 4023 KB/s, 0 seconds passed
... 3%, 704 KB, 4202 KB/s, 0 seconds passed
... 4%, 736 KB, 4381 KB/s, 0 seconds passed
... 4%, 768 KB, 4562 KB/s, 0 seconds passed
... 4%, 800 KB, 4732 KB/s, 0 seconds passed
... 4%, 832 KB, 4909 KB/s, 0 seconds passed
... 4%, 864 KB, 5086 KB/s, 0 seconds passed
... 4%, 896 KB, 5262 KB/s, 0 seconds passed
... 5%, 928 KB, 5435 KB/s, 0 seconds passed
... 5%, 960 KB, 5608 KB/s, 0 seconds passed
... 5%, 992 KB, 5781 KB/s, 0 seconds passed
... 5%, 1024 KB, 5956 KB/s, 0 seconds passed
... 5%, 1056 KB, 6131 KB/s, 0 seconds passed
... 6%, 1088 KB, 6306 KB/s, 0 seconds passed
... 6%, 1120 KB, 6438 KB/s, 0 seconds passed
... 6%, 1152 KB, 5806 KB/s, 0 seconds passed
... 6%, 1184 KB, 5938 KB/s, 0 seconds passed
... 6%, 1216 KB, 6084 KB/s, 0 seconds passed
... 6%, 1248 KB, 6228 KB/s, 0 seconds passed
... 7%, 1280 KB, 6374 KB/s, 0 seconds passed
... 7%, 1312 KB, 6521 KB/s, 0 seconds passed
... 7%, 1344 KB, 6666 KB/s, 0 seconds passed
... 7%, 1376 KB, 6812 KB/s, 0 seconds passed
... 7%, 1408 KB, 6960 KB/s, 0 seconds passed
... 8%, 1440 KB, 7108 KB/s, 0 seconds passed
... 8%, 1472 KB, 7255 KB/s, 0 seconds passed
... 8%, 1504 KB, 7394 KB/s, 0 seconds passed
... 8%, 1536 KB, 7537 KB/s, 0 seconds passed
... 8%, 1568 KB, 7680 KB/s, 0 seconds passed
... 8%, 1600 KB, 7822 KB/s, 0 seconds passed
... 9%, 1632 KB, 7963 KB/s, 0 seconds passed
... 9%, 1664 KB, 8105 KB/s, 0 seconds passed
... 9%, 1696 KB, 8246 KB/s, 0 seconds passed
... 9%, 1728 KB, 8386 KB/s, 0 seconds passed
... 9%, 1760 KB, 8525 KB/s, 0 seconds passed
... 9%, 1792 KB, 8663 KB/s, 0 seconds passed
... 10%, 1824 KB, 8801 KB/s, 0 seconds passed
... 10%, 1856 KB, 8941 KB/s, 0 seconds passed
... 10%, 1888 KB, 9080 KB/s, 0 seconds passed
... 10%, 1920 KB, 9220 KB/s, 0 seconds passed
... 10%, 1952 KB, 9358 KB/s, 0 seconds passed
... 11%, 1984 KB, 9497 KB/s, 0 seconds passed
... 11%, 2016 KB, 9636 KB/s, 0 seconds passed
... 11%, 2048 KB, 9773 KB/s, 0 seconds passed
... 11%, 2080 KB, 9910 KB/s, 0 seconds passed
... 11%, 2112 KB, 10046 KB/s, 0 seconds passed
... 11%, 2144 KB, 10183 KB/s, 0 seconds passed
... 12%, 2176 KB, 10322 KB/s, 0 seconds passed
... 12%, 2208 KB, 10461 KB/s, 0 seconds passed
... 12%, 2240 KB, 10600 KB/s, 0 seconds passed
... 12%, 2272 KB, 10739 KB/s, 0 seconds passed
... 12%, 2304 KB, 10877 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 2336 KB, 10040 KB/s, 0 seconds passed
... 13%, 2368 KB, 10032 KB/s, 0 seconds passed
... 13%, 2400 KB, 10146 KB/s, 0 seconds passed
... 13%, 2432 KB, 10264 KB/s, 0 seconds passed
... 13%, 2464 KB, 10382 KB/s, 0 seconds passed
... 13%, 2496 KB, 10499 KB/s, 0 seconds passed
... 14%, 2528 KB, 10616 KB/s, 0 seconds passed
... 14%, 2560 KB, 10734 KB/s, 0 seconds passed
... 14%, 2592 KB, 10849 KB/s, 0 seconds passed
... 14%, 2624 KB, 10965 KB/s, 0 seconds passed
... 14%, 2656 KB, 11082 KB/s, 0 seconds passed
... 14%, 2688 KB, 11197 KB/s, 0 seconds passed
... 15%, 2720 KB, 11313 KB/s, 0 seconds passed
... 15%, 2752 KB, 11429 KB/s, 0 seconds passed
... 15%, 2784 KB, 11544 KB/s, 0 seconds passed
... 15%, 2816 KB, 11658 KB/s, 0 seconds passed
... 15%, 2848 KB, 11772 KB/s, 0 seconds passed
... 16%, 2880 KB, 11887 KB/s, 0 seconds passed
... 16%, 2912 KB, 11999 KB/s, 0 seconds passed
... 16%, 2944 KB, 12111 KB/s, 0 seconds passed
... 16%, 2976 KB, 12224 KB/s, 0 seconds passed
... 16%, 3008 KB, 12336 KB/s, 0 seconds passed
... 16%, 3040 KB, 12449 KB/s, 0 seconds passed
... 17%, 3072 KB, 12559 KB/s, 0 seconds passed
... 17%, 3104 KB, 12671 KB/s, 0 seconds passed
... 17%, 3136 KB, 12782 KB/s, 0 seconds passed
... 17%, 3168 KB, 12893 KB/s, 0 seconds passed
... 17%, 3200 KB, 13003 KB/s, 0 seconds passed
... 17%, 3232 KB, 13113 KB/s, 0 seconds passed
... 18%, 3264 KB, 13221 KB/s, 0 seconds passed
... 18%, 3296 KB, 13331 KB/s, 0 seconds passed
... 18%, 3328 KB, 13440 KB/s, 0 seconds passed
... 18%, 3360 KB, 13548 KB/s, 0 seconds passed
... 18%, 3392 KB, 13657 KB/s, 0 seconds passed
... 19%, 3424 KB, 13773 KB/s, 0 seconds passed
... 19%, 3456 KB, 13888 KB/s, 0 seconds passed
... 19%, 3488 KB, 14002 KB/s, 0 seconds passed
... 19%, 3520 KB, 14116 KB/s, 0 seconds passed
... 19%, 3552 KB, 14230 KB/s, 0 seconds passed
... 19%, 3584 KB, 14344 KB/s, 0 seconds passed
... 20%, 3616 KB, 14458 KB/s, 0 seconds passed
... 20%, 3648 KB, 14572 KB/s, 0 seconds passed
... 20%, 3680 KB, 14685 KB/s, 0 seconds passed
... 20%, 3712 KB, 14799 KB/s, 0 seconds passed
... 20%, 3744 KB, 14912 KB/s, 0 seconds passed
... 20%, 3776 KB, 15026 KB/s, 0 seconds passed
... 21%, 3808 KB, 15139 KB/s, 0 seconds passed
... 21%, 3840 KB, 15251 KB/s, 0 seconds passed
... 21%, 3872 KB, 15363 KB/s, 0 seconds passed
... 21%, 3904 KB, 15476 KB/s, 0 seconds passed
... 21%, 3936 KB, 15588 KB/s, 0 seconds passed
... 22%, 3968 KB, 15700 KB/s, 0 seconds passed
... 22%, 4000 KB, 15812 KB/s, 0 seconds passed
... 22%, 4032 KB, 15923 KB/s, 0 seconds passed
... 22%, 4064 KB, 16035 KB/s, 0 seconds passed
... 22%, 4096 KB, 16145 KB/s, 0 seconds passed
... 22%, 4128 KB, 16256 KB/s, 0 seconds passed
... 23%, 4160 KB, 16367 KB/s, 0 seconds passed
... 23%, 4192 KB, 16478 KB/s, 0 seconds passed
... 23%, 4224 KB, 16587 KB/s, 0 seconds passed
... 23%, 4256 KB, 16695 KB/s, 0 seconds passed
... 23%, 4288 KB, 16804 KB/s, 0 seconds passed
... 24%, 4320 KB, 16914 KB/s, 0 seconds passed
... 24%, 4352 KB, 17022 KB/s, 0 seconds passed
... 24%, 4384 KB, 17131 KB/s, 0 seconds passed
... 24%, 4416 KB, 17241 KB/s, 0 seconds passed
... 24%, 4448 KB, 17352 KB/s, 0 seconds passed
... 24%, 4480 KB, 17464 KB/s, 0 seconds passed
... 25%, 4512 KB, 17576 KB/s, 0 seconds passed
... 25%, 4544 KB, 17687 KB/s, 0 seconds passed
... 25%, 4576 KB, 17799 KB/s, 0 seconds passed
... 25%, 4608 KB, 17910 KB/s, 0 seconds passed
... 25%, 4640 KB, 18021 KB/s, 0 seconds passed
... 25%, 4672 KB, 18132 KB/s, 0 seconds passed
... 26%, 4704 KB, 17667 KB/s, 0 seconds passed
... 26%, 4736 KB, 17769 KB/s, 0 seconds passed
... 26%, 4768 KB, 17876 KB/s, 0 seconds passed
... 26%, 4800 KB, 17976 KB/s, 0 seconds passed
... 26%, 4832 KB, 18078 KB/s, 0 seconds passed
... 27%, 4864 KB, 18176 KB/s, 0 seconds passed
... 27%, 4896 KB, 18274 KB/s, 0 seconds passed
... 27%, 4928 KB, 18364 KB/s, 0 seconds passed

.. parsed-literal::

    ... 27%, 4960 KB, 18453 KB/s, 0 seconds passed
... 27%, 4992 KB, 18544 KB/s, 0 seconds passed
... 27%, 5024 KB, 18641 KB/s, 0 seconds passed
... 28%, 5056 KB, 18744 KB/s, 0 seconds passed
... 28%, 5088 KB, 18837 KB/s, 0 seconds passed
... 28%, 5120 KB, 18933 KB/s, 0 seconds passed
... 28%, 5152 KB, 19021 KB/s, 0 seconds passed
... 28%, 5184 KB, 19112 KB/s, 0 seconds passed
... 28%, 5216 KB, 19204 KB/s, 0 seconds passed
... 29%, 5248 KB, 19299 KB/s, 0 seconds passed
... 29%, 5280 KB, 19386 KB/s, 0 seconds passed
... 29%, 5312 KB, 19414 KB/s, 0 seconds passed
... 29%, 5344 KB, 19512 KB/s, 0 seconds passed
... 29%, 5376 KB, 19610 KB/s, 0 seconds passed
... 30%, 5408 KB, 19707 KB/s, 0 seconds passed
... 30%, 5440 KB, 19804 KB/s, 0 seconds passed
... 30%, 5472 KB, 19902 KB/s, 0 seconds passed
... 30%, 5504 KB, 19995 KB/s, 0 seconds passed
... 30%, 5536 KB, 20092 KB/s, 0 seconds passed
... 30%, 5568 KB, 19746 KB/s, 0 seconds passed
... 31%, 5600 KB, 19840 KB/s, 0 seconds passed
... 31%, 5632 KB, 19935 KB/s, 0 seconds passed
... 31%, 5664 KB, 20028 KB/s, 0 seconds passed
... 31%, 5696 KB, 20120 KB/s, 0 seconds passed
... 31%, 5728 KB, 20214 KB/s, 0 seconds passed
... 32%, 5760 KB, 20307 KB/s, 0 seconds passed
... 32%, 5792 KB, 20400 KB/s, 0 seconds passed
... 32%, 5824 KB, 20491 KB/s, 0 seconds passed
... 32%, 5856 KB, 20584 KB/s, 0 seconds passed
... 32%, 5888 KB, 20677 KB/s, 0 seconds passed
... 32%, 5920 KB, 20771 KB/s, 0 seconds passed
... 33%, 5952 KB, 20859 KB/s, 0 seconds passed
... 33%, 5984 KB, 20952 KB/s, 0 seconds passed
... 33%, 6016 KB, 21045 KB/s, 0 seconds passed
... 33%, 6048 KB, 21137 KB/s, 0 seconds passed
... 33%, 6080 KB, 21225 KB/s, 0 seconds passed
... 33%, 6112 KB, 21316 KB/s, 0 seconds passed
... 34%, 6144 KB, 21409 KB/s, 0 seconds passed
... 34%, 6176 KB, 21500 KB/s, 0 seconds passed
... 34%, 6208 KB, 21588 KB/s, 0 seconds passed
... 34%, 6240 KB, 21679 KB/s, 0 seconds passed
... 34%, 6272 KB, 21770 KB/s, 0 seconds passed
... 35%, 6304 KB, 21862 KB/s, 0 seconds passed
... 35%, 6336 KB, 21950 KB/s, 0 seconds passed
... 35%, 6368 KB, 22037 KB/s, 0 seconds passed
... 35%, 6400 KB, 22125 KB/s, 0 seconds passed
... 35%, 6432 KB, 22215 KB/s, 0 seconds passed
... 35%, 6464 KB, 22304 KB/s, 0 seconds passed
... 36%, 6496 KB, 22392 KB/s, 0 seconds passed
... 36%, 6528 KB, 22480 KB/s, 0 seconds passed
... 36%, 6560 KB, 22567 KB/s, 0 seconds passed
... 36%, 6592 KB, 22655 KB/s, 0 seconds passed
... 36%, 6624 KB, 22742 KB/s, 0 seconds passed
... 36%, 6656 KB, 22830 KB/s, 0 seconds passed
... 37%, 6688 KB, 22791 KB/s, 0 seconds passed
... 37%, 6720 KB, 22858 KB/s, 0 seconds passed
... 37%, 6752 KB, 22926 KB/s, 0 seconds passed
... 37%, 6784 KB, 22999 KB/s, 0 seconds passed
... 37%, 6816 KB, 23078 KB/s, 0 seconds passed
... 38%, 6848 KB, 23164 KB/s, 0 seconds passed
... 38%, 6880 KB, 23249 KB/s, 0 seconds passed
... 38%, 6912 KB, 23335 KB/s, 0 seconds passed
... 38%, 6944 KB, 23421 KB/s, 0 seconds passed
... 38%, 6976 KB, 23506 KB/s, 0 seconds passed
... 38%, 7008 KB, 23591 KB/s, 0 seconds passed
... 39%, 7040 KB, 23676 KB/s, 0 seconds passed
... 39%, 7072 KB, 23762 KB/s, 0 seconds passed
... 39%, 7104 KB, 23846 KB/s, 0 seconds passed
... 39%, 7136 KB, 23930 KB/s, 0 seconds passed
... 39%, 7168 KB, 24017 KB/s, 0 seconds passed
... 40%, 7200 KB, 24105 KB/s, 0 seconds passed
... 40%, 7232 KB, 24192 KB/s, 0 seconds passed
... 40%, 7264 KB, 24281 KB/s, 0 seconds passed
... 40%, 7296 KB, 24369 KB/s, 0 seconds passed
... 40%, 7328 KB, 24457 KB/s, 0 seconds passed
... 40%, 7360 KB, 24545 KB/s, 0 seconds passed
... 41%, 7392 KB, 24632 KB/s, 0 seconds passed
... 41%, 7424 KB, 24718 KB/s, 0 seconds passed
... 41%, 7456 KB, 24802 KB/s, 0 seconds passed
... 41%, 7488 KB, 24887 KB/s, 0 seconds passed
... 41%, 7520 KB, 24967 KB/s, 0 seconds passed
... 41%, 7552 KB, 25051 KB/s, 0 seconds passed
... 42%, 7584 KB, 25135 KB/s, 0 seconds passed
... 42%, 7616 KB, 25219 KB/s, 0 seconds passed
... 42%, 7648 KB, 25299 KB/s, 0 seconds passed
... 42%, 7680 KB, 25381 KB/s, 0 seconds passed
... 42%, 7712 KB, 25466 KB/s, 0 seconds passed
... 43%, 7744 KB, 25548 KB/s, 0 seconds passed
... 43%, 7776 KB, 25628 KB/s, 0 seconds passed
... 43%, 7808 KB, 25711 KB/s, 0 seconds passed
... 43%, 7840 KB, 25793 KB/s, 0 seconds passed
... 43%, 7872 KB, 25872 KB/s, 0 seconds passed
... 43%, 7904 KB, 25954 KB/s, 0 seconds passed
... 44%, 7936 KB, 26036 KB/s, 0 seconds passed
... 44%, 7968 KB, 26119 KB/s, 0 seconds passed
... 44%, 8000 KB, 26201 KB/s, 0 seconds passed
... 44%, 8032 KB, 26278 KB/s, 0 seconds passed
... 44%, 8064 KB, 26360 KB/s, 0 seconds passed
... 45%, 8096 KB, 26442 KB/s, 0 seconds passed
... 45%, 8128 KB, 26523 KB/s, 0 seconds passed
... 45%, 8160 KB, 26604 KB/s, 0 seconds passed
... 45%, 8192 KB, 26681 KB/s, 0 seconds passed
... 45%, 8224 KB, 26762 KB/s, 0 seconds passed
... 45%, 8256 KB, 26843 KB/s, 0 seconds passed
... 46%, 8288 KB, 26919 KB/s, 0 seconds passed
... 46%, 8320 KB, 27000 KB/s, 0 seconds passed
... 46%, 8352 KB, 27080 KB/s, 0 seconds passed
... 46%, 8384 KB, 27158 KB/s, 0 seconds passed
... 46%, 8416 KB, 27238 KB/s, 0 seconds passed
... 46%, 8448 KB, 27315 KB/s, 0 seconds passed
... 47%, 8480 KB, 27381 KB/s, 0 seconds passed
... 47%, 8512 KB, 27449 KB/s, 0 seconds passed
... 47%, 8544 KB, 27531 KB/s, 0 seconds passed
... 47%, 8576 KB, 27621 KB/s, 0 seconds passed
... 47%, 8608 KB, 27707 KB/s, 0 seconds passed
... 48%, 8640 KB, 27786 KB/s, 0 seconds passed
... 48%, 8672 KB, 27866 KB/s, 0 seconds passed
... 48%, 8704 KB, 27945 KB/s, 0 seconds passed
... 48%, 8736 KB, 28019 KB/s, 0 seconds passed
... 48%, 8768 KB, 28098 KB/s, 0 seconds passed
... 48%, 8800 KB, 28177 KB/s, 0 seconds passed
... 49%, 8832 KB, 28251 KB/s, 0 seconds passed
... 49%, 8864 KB, 28330 KB/s, 0 seconds passed
... 49%, 8896 KB, 28408 KB/s, 0 seconds passed
... 49%, 8928 KB, 28481 KB/s, 0 seconds passed
... 49%, 8960 KB, 28559 KB/s, 0 seconds passed
... 49%, 8992 KB, 28638 KB/s, 0 seconds passed
... 50%, 9024 KB, 28715 KB/s, 0 seconds passed
... 50%, 9056 KB, 28787 KB/s, 0 seconds passed
... 50%, 9088 KB, 28865 KB/s, 0 seconds passed
... 50%, 9120 KB, 28943 KB/s, 0 seconds passed
... 50%, 9152 KB, 29015 KB/s, 0 seconds passed
... 51%, 9184 KB, 29093 KB/s, 0 seconds passed
... 51%, 9216 KB, 29170 KB/s, 0 seconds passed
... 51%, 9248 KB, 29247 KB/s, 0 seconds passed
... 51%, 9280 KB, 29319 KB/s, 0 seconds passed
... 51%, 9312 KB, 29395 KB/s, 0 seconds passed
... 51%, 9344 KB, 29472 KB/s, 0 seconds passed
... 52%, 9376 KB, 29543 KB/s, 0 seconds passed
... 52%, 9408 KB, 29620 KB/s, 0 seconds passed
... 52%, 9440 KB, 29696 KB/s, 0 seconds passed
... 52%, 9472 KB, 29767 KB/s, 0 seconds passed
... 52%, 9504 KB, 29843 KB/s, 0 seconds passed
... 53%, 9536 KB, 29919 KB/s, 0 seconds passed
... 53%, 9568 KB, 29989 KB/s, 0 seconds passed
... 53%, 9600 KB, 30065 KB/s, 0 seconds passed
... 53%, 9632 KB, 30141 KB/s, 0 seconds passed
... 53%, 9664 KB, 30215 KB/s, 0 seconds passed

.. parsed-literal::

    ... 53%, 9696 KB, 30286 KB/s, 0 seconds passed
... 54%, 9728 KB, 30361 KB/s, 0 seconds passed
... 54%, 9760 KB, 30436 KB/s, 0 seconds passed
... 54%, 9792 KB, 30511 KB/s, 0 seconds passed
... 54%, 9824 KB, 30580 KB/s, 0 seconds passed
... 54%, 9856 KB, 30654 KB/s, 0 seconds passed
... 54%, 9888 KB, 30729 KB/s, 0 seconds passed
... 55%, 9920 KB, 30799 KB/s, 0 seconds passed
... 55%, 9952 KB, 30873 KB/s, 0 seconds passed
... 55%, 9984 KB, 30946 KB/s, 0 seconds passed
... 55%, 10016 KB, 31016 KB/s, 0 seconds passed
... 55%, 10048 KB, 30988 KB/s, 0 seconds passed
... 56%, 10080 KB, 31059 KB/s, 0 seconds passed
... 56%, 10112 KB, 31134 KB/s, 0 seconds passed
... 56%, 10144 KB, 31208 KB/s, 0 seconds passed
... 56%, 10176 KB, 31281 KB/s, 0 seconds passed
... 56%, 10208 KB, 31349 KB/s, 0 seconds passed
... 56%, 10240 KB, 31422 KB/s, 0 seconds passed
... 57%, 10272 KB, 31495 KB/s, 0 seconds passed
... 57%, 10304 KB, 31562 KB/s, 0 seconds passed
... 57%, 10336 KB, 31635 KB/s, 0 seconds passed
... 57%, 10368 KB, 31707 KB/s, 0 seconds passed
... 57%, 10400 KB, 31779 KB/s, 0 seconds passed
... 57%, 10432 KB, 31852 KB/s, 0 seconds passed
... 58%, 10464 KB, 31919 KB/s, 0 seconds passed
... 58%, 10496 KB, 31986 KB/s, 0 seconds passed
... 58%, 10528 KB, 32044 KB/s, 0 seconds passed
... 58%, 10560 KB, 32103 KB/s, 0 seconds passed
... 58%, 10592 KB, 32164 KB/s, 0 seconds passed
... 59%, 10624 KB, 32226 KB/s, 0 seconds passed
... 59%, 10656 KB, 32287 KB/s, 0 seconds passed
... 59%, 10688 KB, 32348 KB/s, 0 seconds passed
... 59%, 10720 KB, 32407 KB/s, 0 seconds passed
... 59%, 10752 KB, 32468 KB/s, 0 seconds passed
... 59%, 10784 KB, 32530 KB/s, 0 seconds passed
... 60%, 10816 KB, 32590 KB/s, 0 seconds passed
... 60%, 10848 KB, 32652 KB/s, 0 seconds passed
... 60%, 10880 KB, 32712 KB/s, 0 seconds passed
... 60%, 10912 KB, 32772 KB/s, 0 seconds passed
... 60%, 10944 KB, 32832 KB/s, 0 seconds passed
... 61%, 10976 KB, 32893 KB/s, 0 seconds passed
... 61%, 11008 KB, 32950 KB/s, 0 seconds passed
... 61%, 11040 KB, 33010 KB/s, 0 seconds passed
... 61%, 11072 KB, 33067 KB/s, 0 seconds passed
... 61%, 11104 KB, 33127 KB/s, 0 seconds passed
... 61%, 11136 KB, 33187 KB/s, 0 seconds passed
... 62%, 11168 KB, 33246 KB/s, 0 seconds passed
... 62%, 11200 KB, 33306 KB/s, 0 seconds passed
... 62%, 11232 KB, 33371 KB/s, 0 seconds passed
... 62%, 11264 KB, 33439 KB/s, 0 seconds passed
... 62%, 11296 KB, 33506 KB/s, 0 seconds passed
... 62%, 11328 KB, 33573 KB/s, 0 seconds passed
... 63%, 11360 KB, 33640 KB/s, 0 seconds passed
... 63%, 11392 KB, 33708 KB/s, 0 seconds passed
... 63%, 11424 KB, 33774 KB/s, 0 seconds passed
... 63%, 11456 KB, 32654 KB/s, 0 seconds passed
... 63%, 11488 KB, 32706 KB/s, 0 seconds passed
... 64%, 11520 KB, 32760 KB/s, 0 seconds passed
... 64%, 11552 KB, 32815 KB/s, 0 seconds passed
... 64%, 11584 KB, 32872 KB/s, 0 seconds passed
... 64%, 11616 KB, 32927 KB/s, 0 seconds passed
... 64%, 11648 KB, 32981 KB/s, 0 seconds passed
... 64%, 11680 KB, 33034 KB/s, 0 seconds passed
... 65%, 11712 KB, 33090 KB/s, 0 seconds passed
... 65%, 11744 KB, 33146 KB/s, 0 seconds passed
... 65%, 11776 KB, 33199 KB/s, 0 seconds passed
... 65%, 11808 KB, 33254 KB/s, 0 seconds passed
... 65%, 11840 KB, 33306 KB/s, 0 seconds passed
... 65%, 11872 KB, 33360 KB/s, 0 seconds passed
... 66%, 11904 KB, 33413 KB/s, 0 seconds passed
... 66%, 11936 KB, 33467 KB/s, 0 seconds passed
... 66%, 11968 KB, 33521 KB/s, 0 seconds passed
... 66%, 12000 KB, 33574 KB/s, 0 seconds passed
... 66%, 12032 KB, 33628 KB/s, 0 seconds passed
... 67%, 12064 KB, 33687 KB/s, 0 seconds passed
... 67%, 12096 KB, 33748 KB/s, 0 seconds passed
... 67%, 12128 KB, 33808 KB/s, 0 seconds passed
... 67%, 12160 KB, 33872 KB/s, 0 seconds passed
... 67%, 12192 KB, 33936 KB/s, 0 seconds passed
... 67%, 12224 KB, 34000 KB/s, 0 seconds passed
... 68%, 12256 KB, 34063 KB/s, 0 seconds passed
... 68%, 12288 KB, 33910 KB/s, 0 seconds passed
... 68%, 12320 KB, 33960 KB/s, 0 seconds passed
... 68%, 12352 KB, 34015 KB/s, 0 seconds passed
... 68%, 12384 KB, 34070 KB/s, 0 seconds passed
... 69%, 12416 KB, 34124 KB/s, 0 seconds passed
... 69%, 12448 KB, 34180 KB/s, 0 seconds passed
... 69%, 12480 KB, 34235 KB/s, 0 seconds passed
... 69%, 12512 KB, 34291 KB/s, 0 seconds passed
... 69%, 12544 KB, 34344 KB/s, 0 seconds passed
... 69%, 12576 KB, 34399 KB/s, 0 seconds passed
... 70%, 12608 KB, 34453 KB/s, 0 seconds passed
... 70%, 12640 KB, 34507 KB/s, 0 seconds passed
... 70%, 12672 KB, 34559 KB/s, 0 seconds passed
... 70%, 12704 KB, 34612 KB/s, 0 seconds passed
... 70%, 12736 KB, 34666 KB/s, 0 seconds passed
... 70%, 12768 KB, 34721 KB/s, 0 seconds passed
... 71%, 12800 KB, 34775 KB/s, 0 seconds passed
... 71%, 12832 KB, 34830 KB/s, 0 seconds passed
... 71%, 12864 KB, 34883 KB/s, 0 seconds passed
... 71%, 12896 KB, 34940 KB/s, 0 seconds passed
... 71%, 12928 KB, 34999 KB/s, 0 seconds passed
... 72%, 12960 KB, 35059 KB/s, 0 seconds passed
... 72%, 12992 KB, 35118 KB/s, 0 seconds passed
... 72%, 13024 KB, 35178 KB/s, 0 seconds passed
... 72%, 13056 KB, 35236 KB/s, 0 seconds passed
... 72%, 13088 KB, 35293 KB/s, 0 seconds passed
... 72%, 13120 KB, 35352 KB/s, 0 seconds passed

.. parsed-literal::

    ... 73%, 13152 KB, 35411 KB/s, 0 seconds passed
... 73%, 13184 KB, 35470 KB/s, 0 seconds passed
... 73%, 13216 KB, 35529 KB/s, 0 seconds passed
... 73%, 13248 KB, 35588 KB/s, 0 seconds passed
... 73%, 13280 KB, 35647 KB/s, 0 seconds passed
... 73%, 13312 KB, 35703 KB/s, 0 seconds passed
... 74%, 13344 KB, 35762 KB/s, 0 seconds passed
... 74%, 13376 KB, 35821 KB/s, 0 seconds passed
... 74%, 13408 KB, 35880 KB/s, 0 seconds passed
... 74%, 13440 KB, 35937 KB/s, 0 seconds passed
... 74%, 13472 KB, 35993 KB/s, 0 seconds passed
... 75%, 13504 KB, 36051 KB/s, 0 seconds passed
... 75%, 13536 KB, 36108 KB/s, 0 seconds passed
... 75%, 13568 KB, 36166 KB/s, 0 seconds passed
... 75%, 13600 KB, 36224 KB/s, 0 seconds passed
... 75%, 13632 KB, 36282 KB/s, 0 seconds passed
... 75%, 13664 KB, 36338 KB/s, 0 seconds passed
... 76%, 13696 KB, 36396 KB/s, 0 seconds passed
... 76%, 13728 KB, 36454 KB/s, 0 seconds passed
... 76%, 13760 KB, 36511 KB/s, 0 seconds passed
... 76%, 13792 KB, 36568 KB/s, 0 seconds passed
... 76%, 13824 KB, 36625 KB/s, 0 seconds passed
... 77%, 13856 KB, 36682 KB/s, 0 seconds passed
... 77%, 13888 KB, 36740 KB/s, 0 seconds passed
... 77%, 13920 KB, 36796 KB/s, 0 seconds passed
... 77%, 13952 KB, 36853 KB/s, 0 seconds passed
... 77%, 13984 KB, 36908 KB/s, 0 seconds passed
... 77%, 14016 KB, 36975 KB/s, 0 seconds passed
... 78%, 14048 KB, 37041 KB/s, 0 seconds passed
... 78%, 14080 KB, 37107 KB/s, 0 seconds passed
... 78%, 14112 KB, 37173 KB/s, 0 seconds passed
... 78%, 14144 KB, 37238 KB/s, 0 seconds passed
... 78%, 14176 KB, 37303 KB/s, 0 seconds passed
... 78%, 14208 KB, 37368 KB/s, 0 seconds passed
... 79%, 14240 KB, 37434 KB/s, 0 seconds passed
... 79%, 14272 KB, 37500 KB/s, 0 seconds passed
... 79%, 14304 KB, 37566 KB/s, 0 seconds passed
... 79%, 14336 KB, 37631 KB/s, 0 seconds passed
... 79%, 14368 KB, 37697 KB/s, 0 seconds passed
... 80%, 14400 KB, 37762 KB/s, 0 seconds passed
... 80%, 14432 KB, 37827 KB/s, 0 seconds passed
... 80%, 14464 KB, 37893 KB/s, 0 seconds passed
... 80%, 14496 KB, 37958 KB/s, 0 seconds passed
... 80%, 14528 KB, 38023 KB/s, 0 seconds passed
... 80%, 14560 KB, 38088 KB/s, 0 seconds passed
... 81%, 14592 KB, 38152 KB/s, 0 seconds passed
... 81%, 14624 KB, 38216 KB/s, 0 seconds passed
... 81%, 14656 KB, 38276 KB/s, 0 seconds passed
... 81%, 14688 KB, 38334 KB/s, 0 seconds passed
... 81%, 14720 KB, 38385 KB/s, 0 seconds passed
... 82%, 14752 KB, 38443 KB/s, 0 seconds passed
... 82%, 14784 KB, 38499 KB/s, 0 seconds passed
... 82%, 14816 KB, 38551 KB/s, 0 seconds passed
... 82%, 14848 KB, 38608 KB/s, 0 seconds passed
... 82%, 14880 KB, 38664 KB/s, 0 seconds passed
... 82%, 14912 KB, 38710 KB/s, 0 seconds passed
... 83%, 14944 KB, 38767 KB/s, 0 seconds passed
... 83%, 14976 KB, 38818 KB/s, 0 seconds passed
... 83%, 15008 KB, 38874 KB/s, 0 seconds passed
... 83%, 15040 KB, 38931 KB/s, 0 seconds passed
... 83%, 15072 KB, 38987 KB/s, 0 seconds passed
... 83%, 15104 KB, 39038 KB/s, 0 seconds passed
... 84%, 15136 KB, 39099 KB/s, 0 seconds passed
... 84%, 15168 KB, 39155 KB/s, 0 seconds passed
... 84%, 15200 KB, 39205 KB/s, 0 seconds passed
... 84%, 15232 KB, 39261 KB/s, 0 seconds passed
... 84%, 15264 KB, 39317 KB/s, 0 seconds passed
... 85%, 15296 KB, 39373 KB/s, 0 seconds passed
... 85%, 15328 KB, 39423 KB/s, 0 seconds passed
... 85%, 15360 KB, 39479 KB/s, 0 seconds passed
... 85%, 15392 KB, 39523 KB/s, 0 seconds passed
... 85%, 15424 KB, 39579 KB/s, 0 seconds passed
... 85%, 15456 KB, 39629 KB/s, 0 seconds passed
... 86%, 15488 KB, 39685 KB/s, 0 seconds passed
... 86%, 15520 KB, 39740 KB/s, 0 seconds passed
... 86%, 15552 KB, 39789 KB/s, 0 seconds passed
... 86%, 15584 KB, 39845 KB/s, 0 seconds passed
... 86%, 15616 KB, 39899 KB/s, 0 seconds passed
... 86%, 15648 KB, 39960 KB/s, 0 seconds passed
... 87%, 15680 KB, 40009 KB/s, 0 seconds passed
... 87%, 15712 KB, 40064 KB/s, 0 seconds passed
... 87%, 15744 KB, 40119 KB/s, 0 seconds passed
... 87%, 15776 KB, 40173 KB/s, 0 seconds passed
... 87%, 15808 KB, 40228 KB/s, 0 seconds passed
... 88%, 15840 KB, 40277 KB/s, 0 seconds passed
... 88%, 15872 KB, 40332 KB/s, 0 seconds passed
... 88%, 15904 KB, 40386 KB/s, 0 seconds passed
... 88%, 15936 KB, 40435 KB/s, 0 seconds passed
... 88%, 15968 KB, 40489 KB/s, 0 seconds passed
... 88%, 16000 KB, 40543 KB/s, 0 seconds passed
... 89%, 16032 KB, 40592 KB/s, 0 seconds passed
... 89%, 16064 KB, 40641 KB/s, 0 seconds passed
... 89%, 16096 KB, 40680 KB/s, 0 seconds passed
... 89%, 16128 KB, 40733 KB/s, 0 seconds passed
... 89%, 16160 KB, 40799 KB/s, 0 seconds passed
... 90%, 16192 KB, 40850 KB/s, 0 seconds passed
... 90%, 16224 KB, 40903 KB/s, 0 seconds passed
... 90%, 16256 KB, 40951 KB/s, 0 seconds passed
... 90%, 16288 KB, 41005 KB/s, 0 seconds passed
... 90%, 16320 KB, 41058 KB/s, 0 seconds passed
... 90%, 16352 KB, 41111 KB/s, 0 seconds passed
... 91%, 16384 KB, 41159 KB/s, 0 seconds passed
... 91%, 16416 KB, 41213 KB/s, 0 seconds passed
... 91%, 16448 KB, 41260 KB/s, 0 seconds passed
... 91%, 16480 KB, 41313 KB/s, 0 seconds passed
... 91%, 16512 KB, 41366 KB/s, 0 seconds passed
... 91%, 16544 KB, 41419 KB/s, 0 seconds passed
... 92%, 16576 KB, 41472 KB/s, 0 seconds passed
... 92%, 16608 KB, 41525 KB/s, 0 seconds passed
... 92%, 16640 KB, 41577 KB/s, 0 seconds passed
... 92%, 16672 KB, 41625 KB/s, 0 seconds passed
... 92%, 16704 KB, 41677 KB/s, 0 seconds passed
... 93%, 16736 KB, 41730 KB/s, 0 seconds passed
... 93%, 16768 KB, 41777 KB/s, 0 seconds passed
... 93%, 16800 KB, 41824 KB/s, 0 seconds passed
... 93%, 16832 KB, 41861 KB/s, 0 seconds passed
... 93%, 16864 KB, 41907 KB/s, 0 seconds passed
... 93%, 16896 KB, 41973 KB/s, 0 seconds passed
... 94%, 16928 KB, 42029 KB/s, 0 seconds passed
... 94%, 16960 KB, 42082 KB/s, 0 seconds passed
... 94%, 16992 KB, 42134 KB/s, 0 seconds passed
... 94%, 17024 KB, 42180 KB/s, 0 seconds passed
... 94%, 17056 KB, 42232 KB/s, 0 seconds passed
... 94%, 17088 KB, 42270 KB/s, 0 seconds passed
... 95%, 17120 KB, 42306 KB/s, 0 seconds passed
... 95%, 17152 KB, 42344 KB/s, 0 seconds passed
... 95%, 17184 KB, 42384 KB/s, 0 seconds passed
... 95%, 17216 KB, 42450 KB/s, 0 seconds passed
... 95%, 17248 KB, 42515 KB/s, 0 seconds passed
... 96%, 17280 KB, 42579 KB/s, 0 seconds passed
... 96%, 17312 KB, 42631 KB/s, 0 seconds passed
... 96%, 17344 KB, 42683 KB/s, 0 seconds passed
... 96%, 17376 KB, 42733 KB/s, 0 seconds passed
... 96%, 17408 KB, 42779 KB/s, 0 seconds passed
... 96%, 17440 KB, 42830 KB/s, 0 seconds passed
... 97%, 17472 KB, 42875 KB/s, 0 seconds passed
... 97%, 17504 KB, 42927 KB/s, 0 seconds passed
... 97%, 17536 KB, 42972 KB/s, 0 seconds passed
... 97%, 17568 KB, 43028 KB/s, 0 seconds passed
... 97%, 17600 KB, 43079 KB/s, 0 seconds passed
... 98%, 17632 KB, 43124 KB/s, 0 seconds passed
... 98%, 17664 KB, 43175 KB/s, 0 seconds passed
... 98%, 17696 KB, 43225 KB/s, 0 seconds passed
... 98%, 17728 KB, 43270 KB/s, 0 seconds passed
... 98%, 17760 KB, 43315 KB/s, 0 seconds passed
... 98%, 17792 KB, 43365 KB/s, 0 seconds passed
... 99%, 17824 KB, 43410 KB/s, 0 seconds passed
... 99%, 17856 KB, 43455 KB/s, 0 seconds passed
... 99%, 17888 KB, 43505 KB/s, 0 seconds passed
... 99%, 17920 KB, 43555 KB/s, 0 seconds passed
... 99%, 17952 KB, 43605 KB/s, 0 seconds passed
... 99%, 17984 KB, 43655 KB/s, 0 seconds passed
... 100%, 17990 KB, 43654 KB/s, 0 seconds passed



.. parsed-literal::

    
    ========== Unpacking model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    


Convert Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False
    


.. parsed-literal::

    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-655/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin


.. parsed-literal::

    


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



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



Model Inference
~~~~~~~~~~~~~~~



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

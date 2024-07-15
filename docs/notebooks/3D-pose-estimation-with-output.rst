Live3DHumanPoseEstimationwithOpenVINO
===========================================

Thisnotebookdemonstrateslive3DHumanPoseEstimationwithOpenVINO
viaawebcam.Weutilizethemodel
`human-pose-estimation-3d-0001<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001>`__
from`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__.Attheend
ofthisnotebook,youwillseeliveinferenceresultsfromyourwebcam
(ifavailable).Alternatively,youcanalsouploadavideofiletotest
outthealgorithms.**Makesureyouhaveproperlyinstalled
the**\`Jupyter
extension<https://github.com/jupyter-widgets/pythreejs#jupyterlab>`__\**and
beenusingJupyterLabtorunthedemoassuggestedinthe
``README.md``**

**NOTE**:*Touseawebcam,youmustrunthisJupyternotebookona
computerwithawebcam.Ifyourunonaremoteserver,thewebcam
willnotwork.However,youcanstilldoinferenceonavideofilein
thefinalstep.ThisdemoutilizesthePythoninterfacein
``Three.js``integratedwithWebGLtoprocessdatafromthemodel
inference.Theseresultsareprocessedanddisplayedinthe
notebook.*

*Toensurethattheresultsaredisplayedcorrectly,runthecodeina
recommendedbrowserononeofthefollowingoperatingsystems:**Ubuntu,
Windows:Chrome**macOS:Safari*

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Imports<#imports>`__
-`Themodel<#the-model>`__

-`Downloadthemodel<#download-the-model>`__
-`ConvertModeltoOpenVINOIR
format<#convert-model-to-openvino-ir-format>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Loadthemodel<#load-the-model>`__

-`Processing<#processing>`__

-`ModelInference<#model-inference>`__
-`Draw2DPoseOverlays<#draw-2d-pose-overlays>`__
-`MainProcessingFunction<#main-processing-function>`__

-`Run<#run>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

**The``pythreejs``extensionmaynotdisplayproperlywhenusinga
JupyterNotebookrelease.Therefore,itisrecommendedtouseJupyter
Labinstead.**

..code::ipython3

%pipinstallpythreejs"openvino-dev>=2024.0.0""opencv-python""torch""onnx"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Lookinginindexes:https://pypi.org/simple,https://download.pytorch.org/whl/cpu
Collectingpythreejs
Usingcachedpythreejs-2.4.2-py3-none-any.whl.metadata(5.4kB)
Collectingopenvino-dev>=2024.0.0
Usingcachedopenvino_dev-2024.2.0-15519-py3-none-any.whl.metadata(16kB)
Collectingopencv-python
Usingcachedopencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata(20kB)
Collectingtorch
Usingcachedhttps://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp38-cp38-linux_x86_64.whl(190.4MB)
Collectingonnx
Usingcachedonnx-1.16.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata(16kB)
Requirementalreadysatisfied:ipywidgets>=7.2.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(frompythreejs)(8.1.3)
Collectingipydatawidgets>=1.1.1(frompythreejs)
Usingcachedipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata(1.4kB)
Collectingnumpy(frompythreejs)
Usingcachednumpy-1.24.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata(5.6kB)
Requirementalreadysatisfied:traitletsin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(frompythreejs)(5.14.3)
Requirementalreadysatisfied:defusedxml>=0.7.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino-dev>=2024.0.0)(0.7.1)
Collectingnetworkx<=3.1.0(fromopenvino-dev>=2024.0.0)
Usingcachednetworkx-3.1-py3-none-any.whl.metadata(5.3kB)
Collectingopenvino-telemetry>=2023.2.1(fromopenvino-dev>=2024.0.0)
Usingcachedopenvino_telemetry-2024.1.0-py3-none-any.whl.metadata(2.3kB)
Requirementalreadysatisfied:packagingin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino-dev>=2024.0.0)(24.1)
Requirementalreadysatisfied:pyyaml>=5.4.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino-dev>=2024.0.0)(6.0.1)
Requirementalreadysatisfied:requests>=2.25.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino-dev>=2024.0.0)(2.32.0)
Collectingopenvino==2024.2.0(fromopenvino-dev>=2024.0.0)
Usingcachedopenvino-2024.2.0-15519-cp38-cp38-manylinux2014_x86_64.whl.metadata(8.9kB)
Collectingfilelock(fromtorch)
Usingcachedfilelock-3.15.4-py3-none-any.whl.metadata(2.9kB)
Requirementalreadysatisfied:typing-extensions>=4.8.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch)(4.12.2)
Collectingsympy(fromtorch)
Usingcachedsympy-1.13.0-py3-none-any.whl.metadata(12kB)
Requirementalreadysatisfied:jinja2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch)(3.1.4)
Collectingfsspec(fromtorch)
Usingcachedfsspec-2024.6.1-py3-none-any.whl.metadata(11kB)
Collectingprotobuf>=3.20.2(fromonnx)
Usingcachedprotobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl.metadata(592bytes)
Collectingtraittypes>=0.2.0(fromipydatawidgets>=1.1.1->pythreejs)
Usingcachedtraittypes-0.2.1-py2.py3-none-any.whl.metadata(1.0kB)
Requirementalreadysatisfied:comm>=0.1.3in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipywidgets>=7.2.1->pythreejs)(0.2.2)
Requirementalreadysatisfied:ipython>=6.1.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipywidgets>=7.2.1->pythreejs)(8.12.3)
Requirementalreadysatisfied:widgetsnbextension~=4.0.11in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipywidgets>=7.2.1->pythreejs)(4.0.11)
Requirementalreadysatisfied:jupyterlab-widgets~=3.0.11in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipywidgets>=7.2.1->pythreejs)(3.0.11)
Requirementalreadysatisfied:charset-normalizer<4,>=2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests>=2.25.1->openvino-dev>=2024.0.0)(3.3.2)
Requirementalreadysatisfied:idna<4,>=2.5in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests>=2.25.1->openvino-dev>=2024.0.0)(3.7)
Requirementalreadysatisfied:urllib3<3,>=1.21.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests>=2.25.1->openvino-dev>=2024.0.0)(2.2.2)
Requirementalreadysatisfied:certifi>=2017.4.17in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests>=2.25.1->openvino-dev>=2024.0.0)(2024.7.4)
Requirementalreadysatisfied:MarkupSafe>=2.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromjinja2->torch)(2.1.5)
Collectingmpmath<1.4,>=1.1.0(fromsympy->torch)
Usingcachedhttps://download.pytorch.org/whl/mpmath-1.3.0-py3-none-any.whl(536kB)
Requirementalreadysatisfied:backcallin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.2.0)
Requirementalreadysatisfied:decoratorin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(5.1.1)
Requirementalreadysatisfied:jedi>=0.16in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.19.1)
Requirementalreadysatisfied:matplotlib-inlinein/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.1.7)
Requirementalreadysatisfied:picklesharein/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.7.5)
Requirementalreadysatisfied:prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(3.0.47)
Requirementalreadysatisfied:pygments>=2.4.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(2.18.0)
Requirementalreadysatisfied:stack-datain/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.6.3)
Requirementalreadysatisfied:pexpect>4.3in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(4.9.0)
Requirementalreadysatisfied:parso<0.9.0,>=0.8.3in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromjedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.8.4)
Requirementalreadysatisfied:ptyprocess>=0.5in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(frompexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.7.0)
Requirementalreadysatisfied:wcwidthin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromprompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.2.13)
Requirementalreadysatisfied:executing>=1.2.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromstack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(2.0.1)
Requirementalreadysatisfied:asttokens>=2.1.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromstack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(2.4.1)
Requirementalreadysatisfied:pure-evalin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromstack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(0.2.2)
Requirementalreadysatisfied:six>=1.12.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromasttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs)(1.16.0)
Usingcachedpythreejs-2.4.2-py3-none-any.whl(3.4MB)
Usingcachedopenvino_dev-2024.2.0-15519-py3-none-any.whl(4.7MB)
Usingcachedopenvino-2024.2.0-15519-cp38-cp38-manylinux2014_x86_64.whl(38.7MB)
Usingcachedopencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl(62.5MB)
Usingcachedonnx-1.16.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl(15.9MB)
Usingcachedipydatawidgets-4.3.5-py2.py3-none-any.whl(271kB)
Usingcachednetworkx-3.1-py3-none-any.whl(2.1MB)
Usingcachednumpy-1.24.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl(17.3MB)
Usingcachedopenvino_telemetry-2024.1.0-py3-none-any.whl(23kB)
Usingcachedprotobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl(309kB)
Usingcachedfilelock-3.15.4-py3-none-any.whl(16kB)
Usingcachedfsspec-2024.6.1-py3-none-any.whl(177kB)
Usingcachedsympy-1.13.0-py3-none-any.whl(6.2MB)
Usingcachedtraittypes-0.2.1-py2.py3-none-any.whl(8.6kB)
Installingcollectedpackages:openvino-telemetry,mpmath,traittypes,sympy,protobuf,numpy,networkx,fsspec,filelock,torch,openvino,opencv-python,onnx,openvino-dev,ipydatawidgets,pythreejs
Successfullyinstalledfilelock-3.15.4fsspec-2024.6.1ipydatawidgets-4.3.5mpmath-1.3.0networkx-3.1numpy-1.24.4onnx-1.16.1opencv-python-4.10.0.84openvino-2024.2.0openvino-dev-2024.2.0openvino-telemetry-2024.1.0protobuf-5.27.2pythreejs-2.4.2sympy-1.13.0torch-2.3.1+cputraittypes-0.2.1
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importtime
frompathlibimportPath

importcv2
importipywidgetsaswidgets
importnumpyasnp
fromIPython.displayimportclear_output,display
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
withopen("notebook_utils.py","w")asf:
f.write(r.text)

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/engine3js.py",
)
withopen("engine3js.py","w")asf:
f.write(r.text)

importnotebook_utilsasutils
importengine3jsasengine

Themodel
---------

`backtotop⬆️<#table-of-contents>`__

Downloadthemodel
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuse``omz_downloader``,whichisacommandlinetoolfromthe
``openvino-dev``package.``omz_downloader``automaticallycreatesa
directorystructureanddownloadstheselectedmodel.

..code::ipython3

#directorywheremodelwillbedownloaded
base_model_dir="model"

#modelnameasnamedinOpenModelZoo
model_name="human-pose-estimation-3d-0001"
#selectedprecision(FP32,FP16)
precision="FP32"

BASE_MODEL_NAME=f"{base_model_dir}/public/{model_name}/{model_name}"
model_path=Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path=Path(BASE_MODEL_NAME).with_suffix(".onnx")

ir_model_path=f"model/public/{model_name}/{precision}/{model_name}.xml"
model_weights_path=f"model/public/{model_name}/{precision}/{model_name}.bin"

ifnotmodel_path.exists():
download_command=f"omz_downloader"f"--name{model_name}"f"--output_dir{base_model_dir}"
!$download_command


..parsed-literal::

################||Downloadinghuman-pose-estimation-3d-0001||################

==========Downloadingmodel/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


==========Unpackingmodel/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz



ConvertModeltoOpenVINOIRformat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Theselectedmodelcomesfromthepublicdirectory,whichmeansitmust
beconvertedintoOpenVINOIntermediateRepresentation(OpenVINOIR).We
use``omz_converter``toconverttheONNXformatmodeltotheOpenVINO
IRformat.

..code::ipython3

ifnotonnx_path.exists():
convert_command=(
f"omz_converter"f"--name{model_name}"f"--precisions{precision}"f"--download_dir{base_model_dir}"f"--output_dir{base_model_dir}"
)
!$convert_command


..parsed-literal::

==========Convertinghuman-pose-estimation-3d-0001toONNX
ConversiontoONNXcommand:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/python--/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py--model-path=model/public/human-pose-estimation-3d-0001--model-name=PoseEstimationWithMobileNet--model-param=is_convertible_by_mo=True--import-module=model--weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth--input-shape=1,3,256,448--input-names=data--output-names=features,heatmaps,pafs--output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx

ONNXcheckpassedsuccessfully.

==========Convertinghuman-pose-estimation-3d-0001toIR(FP32)
Conversioncommand:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/python--/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/mo--framework=onnx--output_dir=model/public/human-pose-estimation-3d-0001/FP32--model_name=human-pose-estimation-3d-0001--input=data'--mean_values=data[128.0,128.0,128.0]''--scale_values=data[255.0,255.0,255.0]'--output=features,heatmaps,pafs--input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx'--layout=data(NCHW)''--input_shape=[1,3,256,448]'--compress_to_fp16=False

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
[SUCCESS]GeneratedIRversion11model.
[SUCCESS]XMLfile:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
[SUCCESS]BINfile:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin



Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



Loadthemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Convertedmodelsarelocatedinafixedstructure,whichindicates
vendor,modelnameandprecision.

First,initializetheinferenceengine,OpenVINORuntime.Then,readthe
networkarchitectureandmodelweightsfromthe``.bin``and``.xml``
filestocompileforthedesireddevice.Aninferencerequestisthen
createdtoinferthecompiledmodel.

..code::ipython3

#initializeinferenceengine
core=ov.Core()
#readthenetworkandcorrespondingweightsfromfile
model=core.read_model(model=ir_model_path,weights=model_weights_path)
#loadthemodelonthespecifieddevice
compiled_model=core.compile_model(model=model,device_name=device.value)
infer_request=compiled_model.create_infer_request()
input_tensor_name=model.inputs[0].get_any_name()

#getinputandoutputnamesofnodes
input_layer=compiled_model.input(0)
output_layers=list(compiled_model.outputs)

Theinputforthemodelisdatafromtheinputimageandtheoutputsare
heatmaps,PAF(partaffinityfields)andfeatures.

..code::ipython3

input_layer.any_name,[o.any_nameforoinoutput_layers]




..parsed-literal::

('data',['features','heatmaps','pafs'])



Processing
----------

`backtotop⬆️<#table-of-contents>`__

ModelInference
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Framescapturedfromvideofilesorthelivewebcamareusedasthe
inputforthe3Dmodel.Thisishowyouobtaintheoutputheatmaps,PAF
(partaffinityfields)andfeatures.

..code::ipython3

defmodel_infer(scaled_img,stride):
"""
Runmodelinferenceontheinputimage

Parameters:
scaled_img:resizedimageaccordingtotheinputsizeofthemodel
stride:int,thestrideofthewindow
"""

#Removeexcessspacefromthepicture
img=scaled_img[
0:scaled_img.shape[0]-(scaled_img.shape[0]%stride),
0:scaled_img.shape[1]-(scaled_img.shape[1]%stride),
]

img=np.transpose(img,(2,0,1))[None,]
infer_request.infer({input_tensor_name:img})
#Asetofthreeinferenceresultsisobtained
results={name:infer_request.get_tensor(name).data[:]fornamein{"features","heatmaps","pafs"}}
#Gettheresults
results=(results["features"][0],results["heatmaps"][0],results["pafs"][0])

returnresults

Draw2DPoseOverlays
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weneedtodefinesomeconnectionsbetweenthejointsinadvance,so
thatwecandrawthestructureofthehumanbodyintheresultingimage
afterobtainingtheinferenceresults.Jointsaredrawnascirclesand
limbsaredrawnaslines.Thecodeisbasedonthe`3DHumanPose
Estimation
Demo<https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/human_pose_estimation_3d_demo/python>`__
fromOpenModelZoo.

..code::ipython3

#3Dedgeindexarray
body_edges=np.array(
[
[0,1],
[0,9],
[9,10],
[10,11],#neck-r_shoulder-r_elbow-r_wrist
[0,3],
[3,4],
[4,5],#neck-l_shoulder-l_elbow-l_wrist
[1,15],
[15,16],#nose-l_eye-l_ear
[1,17],
[17,18],#nose-r_eye-r_ear
[0,6],
[6,7],
[7,8],#neck-l_hip-l_knee-l_ankle
[0,12],
[12,13],
[13,14],#neck-r_hip-r_knee-r_ankle
]
)


body_edges_2d=np.array(
[
[0,1],#neck-nose
[1,16],
[16,18],#nose-l_eye-l_ear
[1,15],
[15,17],#nose-r_eye-r_ear
[0,3],
[3,4],
[4,5],#neck-l_shoulder-l_elbow-l_wrist
[0,9],
[9,10],
[10,11],#neck-r_shoulder-r_elbow-r_wrist
[0,6],
[6,7],
[7,8],#neck-l_hip-l_knee-l_ankle
[0,12],
[12,13],
[13,14],#neck-r_hip-r_knee-r_ankle
]
)


defdraw_poses(frame,poses_2d,scaled_img,use_popup):
"""
Draw2Dposeoverlaysontheimagetovisualizeestimatedposes.
Jointsaredrawnascirclesandlimbsaredrawnaslines.

:paramframe:theinputimage
:paramposes_2d:arrayofhumanjointpairs
"""
forposeinposes_2d:
pose=np.array(pose[0:-1]).reshape((-1,3)).transpose()
was_found=pose[2]>0

pose[0],pose[1]=(
pose[0]*frame.shape[1]/scaled_img.shape[1],
pose[1]*frame.shape[0]/scaled_img.shape[0],
)

#Drawjoints.
foredgeinbody_edges_2d:
ifwas_found[edge[0]]andwas_found[edge[1]]:
cv2.line(
frame,
tuple(pose[0:2,edge[0]].astype(np.int32)),
tuple(pose[0:2,edge[1]].astype(np.int32)),
(255,255,0),
4,
cv2.LINE_AA,
)
#Drawlimbs.
forkpt_idinrange(pose.shape[1]):
ifpose[2,kpt_id]!=-1:
cv2.circle(
frame,
tuple(pose[0:2,kpt_id].astype(np.int32)),
3,
(0,255,255),
-1,
cv2.LINE_AA,
)

returnframe

MainProcessingFunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Run3Dposeestimationonthespecifiedsource.Itcouldbeeithera
webcamfeedoravideofile.

..code::ipython3

defrun_pose_estimation(source=0,flip=False,use_popup=False,skip_frames=0):
"""
2Dimageasinput,usingOpenVINOasinferencebackend,
getjoints3Dcoordinates,anddraw3Dhumanskeletoninthescene

:paramsource:Thewebcamnumbertofeedthevideostreamwithprimarywebcamsetto"0",orthevideopath.
:paramflip:TobeusedbyVideoPlayerfunctionforflippingcaptureimage.
:paramuse_popup:Falseforshowingencodedframesoverthisnotebook,Trueforcreatingapopupwindow.
:paramskip_frames:Numberofframestoskipatthebeginningofthevideo.
"""

focal_length=-1#default
stride=8
player=None
skeleton_set=None

try:
#createvideoplayertoplaywithtargetfpsvideo_path
#gettheframefromcamera
#YoucanskipfirstNframestofastforwardvideo.change'skip_first_frames'
player=utils.VideoPlayer(source,flip=flip,fps=30,skip_first_frames=skip_frames)
#startcapturing
player.start()

input_image=player.next()
#setthewindowsize
resize_scale=450/input_image.shape[1]
windows_width=int(input_image.shape[1]*resize_scale)
windows_height=int(input_image.shape[0]*resize_scale)

#usevisualizationlibrary
engine3D=engine.Engine3js(grid=True,axis=True,view_width=windows_width,view_height=windows_height)

ifuse_popup:
#displaythe3Dhumanposeinthisnotebook,andoriginframeinpopupwindow
display(engine3D.renderer)
title="PressESCtoExit"
cv2.namedWindow(title,cv2.WINDOW_KEEPRATIO|cv2.WINDOW_AUTOSIZE)
else:
#setthe2Dimagebox,showbothhumanposeandimageinthenotebook
imgbox=widgets.Image(format="jpg",height=windows_height,width=windows_width)
display(widgets.HBox([engine3D.renderer,imgbox]))

skeleton=engine.Skeleton(body_edges=body_edges)

processing_times=collections.deque()

whileTrue:
#grabtheframe
frame=player.next()
ifframeisNone:
print("Sourceended")
break

#resizeimageandchangedimstofitneuralnetworkinput
#(seehttps://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001)
scaled_img=cv2.resize(frame,dsize=(model.inputs[0].shape[3],model.inputs[0].shape[2]))

iffocal_length<0:#Focallengthisunknown
focal_length=np.float32(0.8*scaled_img.shape[1])

#inferencestart
start_time=time.time()
#getresults
inference_result=model_infer(scaled_img,stride)

#inferencestop
stop_time=time.time()
processing_times.append(stop_time-start_time)
#Processthepointtopointcoordinatesofthedata
poses_3d,poses_2d=engine.parse_poses(inference_result,1,stride,focal_length,True)

#useprocessingtimesfromlast200frames
iflen(processing_times)>200:
processing_times.popleft()

processing_time=np.mean(processing_times)*1000
fps=1000/processing_time

iflen(poses_3d)>0:
#Fromhere,youcanrotatethe3Dpointpositionsusingthefunction"draw_poses",
#oryoucandirectlymakethecorrectmappingbelowtoproperlydisplaytheobjectimageonthescreen
poses_3d_copy=poses_3d.copy()
x=poses_3d_copy[:,0::4]
y=poses_3d_copy[:,1::4]
z=poses_3d_copy[:,2::4]
poses_3d[:,0::4],poses_3d[:,1::4],poses_3d[:,2::4]=(
-z+np.ones(poses_3d[:,2::4].shape)*200,
-y+np.ones(poses_3d[:,2::4].shape)*100,
-x,
)

poses_3d=poses_3d.reshape(poses_3d.shape[0],19,-1)[:,:,0:3]
people=skeleton(poses_3d=poses_3d)

try:
engine3D.scene_remove(skeleton_set)
exceptException:
pass

engine3D.scene_add(people)
skeleton_set=people

#draw2D
frame=draw_poses(frame,poses_2d,scaled_img,use_popup)

else:
try:
engine3D.scene_remove(skeleton_set)
skeleton_set=None
exceptException:
pass

cv2.putText(
frame,
f"Inferencetime:{processing_time:.1f}ms({fps:.1f}FPS)",
(10,30),
cv2.FONT_HERSHEY_COMPLEX,
0.7,
(0,0,255),
1,
cv2.LINE_AA,
)

ifuse_popup:
cv2.imshow(title,frame)
key=cv2.waitKey(1)
#escape=27,useESCtoexit
ifkey==27:
break
else:
#encodenumpyarraytojpg
imgbox.value=cv2.imencode(
".jpg",
frame,
params=[cv2.IMWRITE_JPEG_QUALITY,90],
)[1].tobytes()

engine3D.renderer.render(engine3D.scene,engine3D.cam)

exceptKeyboardInterrupt:
print("Interrupted")
exceptRuntimeErrorase:
print(e)
finally:
clear_output()
ifplayerisnotNone:
#stopcapturing
player.stop()
ifuse_popup:
cv2.destroyAllWindows()
ifskeleton_set:
engine3D.scene_remove(skeleton_set)

Run
---

`backtotop⬆️<#table-of-contents>`__

Run,usingawebcamasthevideoinput.Bydefault,theprimarywebcam
issetwith``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set``flip=True``when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,set
``use_popup=True``.

**NOTE**:

*1.Tousethisnotebookwithawebcam,youneedtorunthenotebook
onacomputerwithawebcam.Ifyourunthenotebookonaserver
(e.g. Binder),thewebcamwillnotwork.*

*2.Popupmodemaynotworkifyourunthisnotebookonaremote
computer(e.g. Binder).*

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
willwork.

Usingthefollowingmethod,youcanclickandmoveyourmouseoverthe
pictureonthelefttointeract.

..code::ipython3

USE_WEBCAM=False

cam_id=0
video_path="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"

source=cam_idifUSE_WEBCAMelsevideo_path

run_pose_estimation(source=source,flip=isinstance(source,int),use_popup=False)

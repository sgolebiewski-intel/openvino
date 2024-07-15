ConvertaPyTorchModeltoONNXandOpenVINO™IR
================================================

Thistutorialdemonstratesstep-by-stepinstructionsonhowtodo
inferenceonaPyTorchsemanticsegmentationmodel,usingOpenVINO
Runtime.

First,thePyTorchmodelisexportedin`ONNX<https://onnx.ai/>`__
formatandthenconvertedtoOpenVINOIR.ThentherespectiveONNXand
OpenVINOIRmodelsareloadedintoOpenVINORuntimetoshowmodel
predictions.Inthistutorial,wewilluseLR-ASPPmodelwith
MobileNetV3backbone.

Accordingtothepaper,`Searchingfor
MobileNetV3<https://arxiv.org/pdf/1905.02244.pdf>`__,LR-ASPPorLite
ReducedAtrousSpatialPyramidPoolinghasalightweightandefficient
segmentationdecoderarchitecture.Thediagrambelowillustratesthe
modelarchitecture:

..figure::https://user-images.githubusercontent.com/29454499/207099169-48dca3dc-a8eb-4e11-be92-40cebeec7a88.png
:alt:image

image

Themodelispre-trainedonthe`MS
COCO<https://cocodataset.org/#home>`__dataset.Insteadoftrainingon
all80classes,thesegmentationmodelhasbeentrainedon20classes
fromthe`PASCALVOC<http://host.robots.ox.ac.uk/pascal/VOC/>`__
dataset:**background,aeroplane,bicycle,bird,boat,bottle,bus,car,
cat,chair,cow,diningtable,dog,horse,motorbike,person,potted
plant,sheep,sofa,train,tvmonitor**

Moreinformationaboutthemodelisavailableinthe`torchvision
documentation<https://pytorch.org/vision/main/models/lraspp.html>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Imports<#imports>`__
-`Settings<#settings>`__
-`LoadModel<#load-model>`__

-`ONNXModelConversion<#onnx-model-conversion>`__

-`ConvertPyTorchmodeltoONNX<#convert-pytorch-model-to-onnx>`__
-`ConvertONNXModeltoOpenVINOIR
Format<#convert-onnx-model-to-openvino-ir-format>`__

-`ShowResults<#show-results>`__

-`LoadandPreprocessanInput
Image<#load-and-preprocess-an-input-image>`__
-`LoadtheOpenVINOIRNetworkandRunInferenceontheONNX
model<#load-the-openvino-ir-network-and-run-inference-on-the-onnx-model>`__

-`1.ONNXModelinOpenVINO
Runtime<#1--onnx-model-in-openvino-runtime>`__
-`Selectinferencedevice<#select-inference-device>`__
-`2.OpenVINOIRModelinOpenVINO
Runtime<#2--openvino-ir-model-in-openvino-runtime>`__
-`Selectinferencedevice<#select-inference-device>`__

-`PyTorchComparison<#pytorch-comparison>`__
-`PerformanceComparison<#performance-comparison>`__
-`References<#references>`__

..code::ipython3

#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0"onnxtorchtorchvisionopencv-pythontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtime
importwarnings
frompathlibimportPath

importcv2
importnumpyasnp
importopenvinoasov
importtorch
fromtorchvision.models.segmentationimport(
lraspp_mobilenet_v3_large,
LRASPP_MobileNet_V3_Large_Weights,
)

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimport(
segmentation_map_to_image,
viz_result_image,
SegmentationMap,
Label,
download_file,
)

Settings
~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Setanameforthemodel,thendefinewidthandheightoftheimagethat
willbeusedbythenetworkduringinference.Accordingtotheinput
transformsfunction,themodelispre-trainedonimageswithaheightof
520andwidthof780.

..code::ipython3

IMAGE_WIDTH=780
IMAGE_HEIGHT=520
DIRECTORY_NAME="model"
BASE_MODEL_NAME=DIRECTORY_NAME+"/lraspp_mobilenet_v3_large"
weights_path=Path(BASE_MODEL_NAME+".pt")

#PathswhereONNXandOpenVINOIRmodelswillbestored.
onnx_path=weights_path.with_suffix(".onnx")
ifnotonnx_path.parent.exists():
onnx_path.parent.mkdir()
ir_path=onnx_path.with_suffix(".xml")

LoadModel
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceof``torch.nn.Module``
class,initializedbyastatedictionarywithmodelweights.Typical
stepsforgettingapre-trainedmodel:1.Createinstanceofmodelclass
2.Loadcheckpointstatedict,whichcontainspre-trainedmodelweights
3.Turnmodeltoevaluationforswitchingsomeoperationstoinference
mode

The``torchvision``moduleprovidesareadytousesetoffunctionsfor
modelclassinitialization.Wewilluse
``torchvision.models.segmentation.lraspp_mobilenet_v3_large``.Youcan
directlypasspre-trainedmodelweightstothemodelinitialization
functionusingweightsenum
``LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1``.However,
fordemonstrationpurposes,wewillcreateitseparately.Downloadthe
pre-trainedweightsandloadthemodel.Thismaytakesometimeifyou
havenotdownloadedthemodelbefore.

..code::ipython3

print("DownloadingtheLRASPPMobileNetV3model(ifithasnotbeendownloadedalready)...")
download_file(
LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.url,
filename=weights_path.name,
directory=weights_path.parent,
)
#createmodelobject
model=lraspp_mobilenet_v3_large()
#readstatedict,usemap_locationargumenttoavoidasituationwhereweightsaresavedincuda(whichmaynotbeunavailableonthesystem)
state_dict=torch.load(weights_path,map_location="cpu")
#loadstatedicttomodel
model.load_state_dict(state_dict)
#switchmodelfromtrainingtoinferencemode
model.eval()
print("LoadedPyTorchLRASPPMobileNetV3model")


..parsed-literal::

DownloadingtheLRASPPMobileNetV3model(ifithasnotbeendownloadedalready)...



..parsed-literal::

model/lraspp_mobilenet_v3_large.pt:0%||0.00/12.5M[00:00<?,?B/s]


..parsed-literal::

LoadedPyTorchLRASPPMobileNetV3model


ONNXModelConversion
---------------------

`backtotop⬆️<#table-of-contents>`__

ConvertPyTorchmodeltoONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsthatareexportedinONNXformat.We
willusethe``torch.onnx.export``functiontoobtaintheONNXmodel,
youcanlearnmoreaboutthisfeatureinthe`PyTorch
documentation<https://pytorch.org/docs/stable/onnx.html>`__.Weneedto
provideamodelobject,exampleinputformodeltracingandpathwhere
themodelwillbesaved.Whenprovidingexampleinput,itisnot
necessarytouserealdata,dummyinputdatawithspecifiedshapeis
sufficient.Optionally,wecanprovideatargetonnxopsetfor
conversionand/orotherparametersspecifiedindocumentation
(e.g. inputandoutputnamesordynamicshapes).

Sometimesawarningwillbeshown,butinmostcasesitisharmless,so
letusjustfilteritout.Whentheconversionissuccessful,thelast
lineoftheoutputwillread:
``ONNXmodelexportedtomodel/lraspp_mobilenet_v3_large.onnx.``

..code::ipython3

withwarnings.catch_warnings():
warnings.filterwarnings("ignore")
ifnotonnx_path.exists():
dummy_input=torch.randn(1,3,IMAGE_HEIGHT,IMAGE_WIDTH)
torch.onnx.export(
model,
dummy_input,
onnx_path,
)
print(f"ONNXmodelexportedto{onnx_path}.")
else:
print(f"ONNXmodel{onnx_path}alreadyexists.")


..parsed-literal::

ONNXmodelexportedtomodel/lraspp_mobilenet_v3_large.onnx.


ConvertONNXModeltoOpenVINOIRFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ToconverttheONNXmodeltoOpenVINOIRwith``FP16``precision,use
modelconversionAPI.Themodelsaresavedinsidethecurrentdirectory.
Formoreinformationonhowtoconvertmodels,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

ifnotir_path.exists():
print("ExportingONNXmodeltoIR...Thismaytakeafewminutes.")
ov_model=ov.convert_model(onnx_path)
ov.save_model(ov_model,ir_path)
else:
print(f"IRmodel{ir_path}alreadyexists.")


..parsed-literal::

ExportingONNXmodeltoIR...Thismaytakeafewminutes.


ShowResults
------------

`backtotop⬆️<#table-of-contents>`__

Confirmthatthesegmentationresultslookasexpectedbycomparing
modelpredictionsontheONNX,OpenVINOIRandPyTorchmodels.

LoadandPreprocessanInputImage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Imagesneedtobenormalizedbeforepropagatingthroughthenetwork.

..code::ipython3

defnormalize(image:np.ndarray)->np.ndarray:
"""
Normalizetheimagetothegivenmeanandstandarddeviation
forCityScapesmodels.
"""
image=image.astype(np.float32)
mean=(0.485,0.456,0.406)
std=(0.229,0.224,0.225)
image/=255.0
image-=mean
image/=std
returnimage

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)

image=cv2.cvtColor(cv2.imread(str(image_filename)),cv2.COLOR_BGR2RGB)

resized_image=cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT))
normalized_image=normalize(resized_image)

#Converttheresizedimagestonetworkinputshape.
input_image=np.expand_dims(np.transpose(resized_image,(2,0,1)),0)
normalized_input_image=np.expand_dims(np.transpose(normalized_image,(2,0,1)),0)



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]


LoadtheOpenVINOIRNetworkandRunInferenceontheONNXmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINORuntimecanloadONNXmodelsdirectly.First,loadtheONNX
model,doinferenceandshowtheresults.Then,loadthemodelthatwas
convertedtoOpenVINOIntermediateRepresentation(OpenVINOIR)with
OpenVINOConverteranddoinferenceonthatmodel,andshowtheresults
onanimage.

1.ONNXModelinOpenVINORuntime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#InstantiateOpenVINOCore
core=ov.Core()

#ReadmodeltoOpenVINORuntime
model_onnx=core.read_model(model=onnx_path)

Selectinferencedevice
^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#Loadmodelondevice
compiled_model_onnx=core.compile_model(model=model_onnx,device_name=device.value)

#Runinferenceontheinputimage
res_onnx=compiled_model_onnx([normalized_input_image])[0]

Modelpredictsprobabilitiesforhowwelleachpixelcorrespondstoa
specificlabel.Togetthelabelwithhighestprobabilityforeach
pixel,operationargmaxshouldbeapplied.Afterthat,colorcodingcan
beappliedtoeachlabelformoreconvenientvisualization.

..code::ipython3

voc_labels=[
Label(index=0,color=(0,0,0),name="background"),
Label(index=1,color=(128,0,0),name="aeroplane"),
Label(index=2,color=(0,128,0),name="bicycle"),
Label(index=3,color=(128,128,0),name="bird"),
Label(index=4,color=(0,0,128),name="boat"),
Label(index=5,color=(128,0,128),name="bottle"),
Label(index=6,color=(0,128,128),name="bus"),
Label(index=7,color=(128,128,128),name="car"),
Label(index=8,color=(64,0,0),name="cat"),
Label(index=9,color=(192,0,0),name="chair"),
Label(index=10,color=(64,128,0),name="cow"),
Label(index=11,color=(192,128,0),name="diningtable"),
Label(index=12,color=(64,0,128),name="dog"),
Label(index=13,color=(192,0,128),name="horse"),
Label(index=14,color=(64,128,128),name="motorbike"),
Label(index=15,color=(192,128,128),name="person"),
Label(index=16,color=(0,64,0),name="pottedplant"),
Label(index=17,color=(128,64,0),name="sheep"),
Label(index=18,color=(0,192,0),name="sofa"),
Label(index=19,color=(128,192,0),name="train"),
Label(index=20,color=(0,64,128),name="tvmonitor"),
]
VOCLabels=SegmentationMap(voc_labels)

#Convertthenetworkresulttoasegmentationmapanddisplaytheresult.
result_mask_onnx=np.squeeze(np.argmax(res_onnx,axis=1)).astype(np.uint8)
viz_result_image(
image,
segmentation_map_to_image(result_mask_onnx,VOCLabels.get_colormap()),
resize=True,
)




..image::pytorch-onnx-to-openvino-with-output_files/pytorch-onnx-to-openvino-with-output_22_0.png



2.OpenVINOIRModelinOpenVINORuntime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Selectinferencedevice
^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#LoadthenetworkinOpenVINORuntime.
core=ov.Core()
model_ir=core.read_model(model=ir_path)
compiled_model_ir=core.compile_model(model=model_ir,device_name=device.value)

#Getinputandoutputlayers.
output_layer_ir=compiled_model_ir.output(0)

#Runinferenceontheinputimage.
res_ir=compiled_model_ir([normalized_input_image])[output_layer_ir]

..code::ipython3

result_mask_ir=np.squeeze(np.argmax(res_ir,axis=1)).astype(np.uint8)
viz_result_image(
image,
segmentation_map_to_image(result=result_mask_ir,colormap=VOCLabels.get_colormap()),
resize=True,
)




..image::pytorch-onnx-to-openvino-with-output_files/pytorch-onnx-to-openvino-with-output_27_0.png



PyTorchComparison
------------------

`backtotop⬆️<#table-of-contents>`__

DoinferenceonthePyTorchmodeltoverifythattheoutputvisually
looksthesameastheoutputontheONNX/OpenVINOIRmodels.

..code::ipython3

model.eval()
withtorch.no_grad():
result_torch=model(torch.as_tensor(normalized_input_image).float())

result_mask_torch=torch.argmax(result_torch["out"],dim=1).squeeze(0).numpy().astype(np.uint8)
viz_result_image(
image,
segmentation_map_to_image(result=result_mask_torch,colormap=VOCLabels.get_colormap()),
resize=True,
)




..image::pytorch-onnx-to-openvino-with-output_files/pytorch-onnx-to-openvino-with-output_29_0.png



PerformanceComparison
----------------------

`backtotop⬆️<#table-of-contents>`__

Measurethetimeittakestodoinferenceontwentyimages.Thisgives
anindicationofperformance.Formoreaccuratebenchmarking,usethe
`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
Keepinmindthatmanyoptimizationsarepossibletoimprovethe
performance.

..code::ipython3

num_images=100

withtorch.no_grad():
start=time.perf_counter()
for_inrange(num_images):
model(torch.as_tensor(input_image).float())
end=time.perf_counter()
time_torch=end-start
print(f"PyTorchmodelonCPU:{time_torch/num_images:.3f}secondsperimage,"f"FPS:{num_images/time_torch:.2f}")

compiled_model_onnx=core.compile_model(model=model_onnx,device_name=device.value)
start=time.perf_counter()
for_inrange(num_images):
compiled_model_onnx([normalized_input_image])
end=time.perf_counter()
time_onnx=end-start
print(f"ONNXmodelinOpenVINORuntime/{device.value}:{time_onnx/num_images:.3f}"f"secondsperimage,FPS:{num_images/time_onnx:.2f}")

compiled_model_ir=core.compile_model(model=model_ir,device_name=device.value)
start=time.perf_counter()
for_inrange(num_images):
compiled_model_ir([input_image])
end=time.perf_counter()
time_ir=end-start
print(f"OpenVINOIRmodelinOpenVINORuntime/{device.value}:{time_ir/num_images:.3f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}")


..parsed-literal::

PyTorchmodelonCPU:0.039secondsperimage,FPS:25.91
ONNXmodelinOpenVINORuntime/AUTO:0.018secondsperimage,FPS:54.55
OpenVINOIRmodelinOpenVINORuntime/AUTO:0.028secondsperimage,FPS:35.81


**ShowDeviceInformation**

..code::ipython3

devices=core.available_devices
fordeviceindevices:
device_name=core.get_property(device,"FULL_DEVICE_NAME")
print(f"{device}:{device_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)i9-10920XCPU@3.50GHz


References
----------

`backtotop⬆️<#table-of-contents>`__

-`Torchvision<https://pytorch.org/vision/stable/index.html>`__
-`PytorchONNX
Documentation<https://pytorch.org/docs/stable/onnx.html>`__
-`PIPinstallopenvino-dev<https://pypi.org/project/openvino-dev/>`__
-`OpenVINOONNX
support<https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_ONNX_Support.html>`__
-`ModelConversionAPI
documentation<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
-`ConvertingPytorch
model<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html>`__

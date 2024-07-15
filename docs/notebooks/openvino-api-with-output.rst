OpenVINO™RuntimeAPITutorial
==============================

ThisnotebookexplainsthebasicsoftheOpenVINORuntimeAPI.

Thenotebookisdividedintosectionswithheaders.Thenextcell
containsglobalrequirementsforinstallationandimports.Eachsection
isstandaloneanddoesnotdependonanyprevioussections.Allmodels
usedinthistutorialareprovidedasexamples.Thesemodelfilescanbe
replacedwithyourownmodels.Theexactoutputswillbedifferent,but
theprocessisthesame.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`LoadingOpenVINORuntimeandShowing
Info<#loading-openvino-runtime-and-showing-info>`__
-`LoadingaModel<#loading-a-model>`__

-`OpenVINOIRModel<#openvino-ir-model>`__
-`ONNXModel<#onnx-model>`__
-`PaddlePaddleModel<#paddlepaddle-model>`__
-`TensorFlowModel<#tensorflow-model>`__
-`TensorFlowLiteModel<#tensorflow-lite-model>`__
-`PyTorchModel<#pytorch-model>`__

-`GettingInformationabouta
Model<#getting-information-about-a-model>`__

-`ModelInputs<#model-inputs>`__
-`ModelOutputs<#model-outputs>`__

-`DoingInferenceonaModel<#doing-inference-on-a-model>`__
-`ReshapingandResizing<#reshaping-and-resizing>`__

-`ChangeImageSize<#change-image-size>`__
-`ChangeBatchSize<#change-batch-size>`__

-`CachingaModel<#caching-a-model>`__

..code::ipython3

#Requiredimports.Pleaseexecutethiscellfirst.
%pipinstall-q"openvino>=2023.1.0"
%pipinstall-qrequeststqdmipywidgets

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


LoadingOpenVINORuntimeandShowingInfo
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

InitializeOpenVINORuntimewith``ov.Core()``

..code::ipython3

importopenvinoasov

core=ov.Core()

OpenVINORuntimecanloadanetworkonadevice.Adeviceinthis
contextmeansaCPU,anIntelGPU,aNeuralComputeStick2,etc.The
``available_devices``propertyshowstheavailabledevicesinyour
system.The“FULL_DEVICE_NAME”optionto``core.get_property()``shows
thenameofthedevice.

..code::ipython3

devices=core.available_devices

fordeviceindevices:
device_name=core.get_property(device,"FULL_DEVICE_NAME")
print(f"{device}:{device_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)i9-10920XCPU@3.50GHz


Selectdeviceforinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Youcanspecifywhichdevicefromavailabledeviceswillbeusedfor
inferenceusingthiswidget

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices,
value=core.available_devices[0],
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU',),value='CPU')



LoadingaModel
---------------

`backtotop⬆️<#table-of-contents>`__

AfterinitializingOpenVINORuntime,firstreadthemodelfilewith
``read_model()``,thencompileittothespecifieddevicewiththe
``compile_model()``method.

`OpenVINO™supportsseveralmodel
formats<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html>`__
andenablesdeveloperstoconvertthemtoitsownOpenVINOIRformat
usingatooldedicatedtothistask.

OpenVINOIRModel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

AnOpenVINOIR(IntermediateRepresentation)modelconsistsofan
``.xml``file,containinginformationaboutnetworktopology,anda
``.bin``file,containingtheweightsandbiasesbinarydata.Modelsin
OpenVINOIRformatareobtainedbyusingmodelconversionAPI.The
``read_model()``functionexpectsthe``.bin``weightsfiletohavethe
samefilenameandbelocatedinthesamedirectoryasthe``.xml``file:
``model_weights_file==Path(model_xml).with_suffix(".bin")``.Ifthis
isthecase,specifyingtheweightsfileisoptional.Iftheweights
filehasadifferentfilename,itcanbespecifiedusingthe``weights``
parameterin``read_model()``.

TheOpenVINO`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
toolisusedtoconvertmodelstoOpenVINOIRformat.Modelconversion
APIreadstheoriginalmodelandcreatesanOpenVINOIRmodel(``.xml``
and``.bin``files)soinferencecanbeperformedwithoutdelaysdueto
formatconversion.Optionally,modelconversionAPIcanadjustthemodel
tobemoresuitableforinference,forexample,byalternatinginput
shapes,embeddingpreprocessingandcuttingtrainingpartsoff.For
informationonhowtoconvertyourexistingTensorFlow,PyTorchorONNX
modeltoOpenVINOIRformatwithmodelconversionAPI,refertothe
`tensorflow-to-openvino<tensorflow-classification-to-openvino-with-output.html>`__
and
`pytorch-onnx-to-openvino<pytorch-to-openvino-with-output.html>`__
notebooks.

..code::ipython3

ir_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml="classification.xml"
ir_model_name_bin="classification.bin"

download_file(ir_model_url+ir_model_name_xml,filename=ir_model_name_xml,directory="model")
download_file(ir_model_url+ir_model_name_bin,filename=ir_model_name_bin,directory="model")



..parsed-literal::

model/classification.xml:0%||0.00/179k[00:00<?,?B/s]



..parsed-literal::

model/classification.bin:0%||0.00/4.84M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/classification.bin')



..code::ipython3

importopenvinoasov

core=ov.Core()
classification_model_xml="model/classification.xml"

model=core.read_model(model=classification_model_xml)
compiled_model=core.compile_model(model=model,device_name=device.value)

ONNXModel
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`ONNX<https://onnx.ai/>`__isanopenformatbuilttorepresentmachine
learningmodels.ONNXdefinesacommonsetofoperators-thebuilding
blocksofmachinelearninganddeeplearningmodels-andacommonfile
formattoenableAIdeveloperstousemodelswithavarietyof
frameworks,tools,runtimes,andcompilers.OpenVINOsupportsreading
modelsinONNXformatdirectly,thatmeanstheycanbeusedwithOpenVINO
Runtimewithoutanypriorconversion.

ReadingandloadinganONNXmodel,whichisasingle``.onnx``file,
worksthesamewayaswithanOpenVINOIRmodel.The``model``argument
pointstothefilenameofanONNXmodel.

..code::ipython3

onnx_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/segmentation.onnx"
onnx_model_name="segmentation.onnx"

download_file(onnx_model_url,filename=onnx_model_name,directory="model")



..parsed-literal::

model/segmentation.onnx:0%||0.00/4.41M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/segmentation.onnx')



..code::ipython3

importopenvinoasov

core=ov.Core()
onnx_model_path="model/segmentation.onnx"

model_onnx=core.read_model(model=onnx_model_path)
compiled_model_onnx=core.compile_model(model=model_onnx,device_name=device.value)

TheONNXmodelcanbeexportedtoOpenVINOIRwith``save_model()``:

..code::ipython3

ov.save_model(model_onnx,output_model="model/exported_onnx_model.xml")

PaddlePaddleModel
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`PaddlePaddle<https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html>`__
modelssavedforinferencecanalsobepassedtoOpenVINORuntime
withoutanyconversionstep.Passthefilenamewithextensionto
``read_model``andexportedanOpenVINOIRwith``save_model``

..code::ipython3

paddle_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
paddle_model_name="inference.pdmodel"
paddle_params_name="inference.pdiparams"

download_file(paddle_model_url+paddle_model_name,filename=paddle_model_name,directory="model")
download_file(
paddle_model_url+paddle_params_name,
filename=paddle_params_name,
directory="model",
)



..parsed-literal::

model/inference.pdmodel:0%||0.00/1.03M[00:00<?,?B/s]



..parsed-literal::

model/inference.pdiparams:0%||0.00/21.0M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/inference.pdiparams')



..code::ipython3

importopenvinoasov

core=ov.Core()
paddle_model_path="model/inference.pdmodel"

model_paddle=core.read_model(model=paddle_model_path)
compiled_model_paddle=core.compile_model(model=model_paddle,device_name=device.value)

..code::ipython3

ov.save_model(model_paddle,output_model="model/exported_paddle_model.xml")

TensorFlowModel
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TensorFlowmodelssavedinfrozengraphformatcanalsobepassedto
``read_model``.

..code::ipython3

pb_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/classification.pb"
pb_model_name="classification.pb"

download_file(pb_model_url,filename=pb_model_name,directory="model")



..parsed-literal::

model/classification.pb:0%||0.00/9.88M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/classification.pb')



..code::ipython3

importopenvinoasov

core=ov.Core()
tf_model_path="model/classification.pb"

model_tf=core.read_model(model=tf_model_path)
compiled_model_tf=core.compile_model(model=model_tf,device_name=device.value)

..code::ipython3

ov.save_model(model_tf,output_model="model/exported_tf_model.xml")

TensorFlowLiteModel
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`TFLite<https://www.tensorflow.org/lite>`__modelssavedforinference
canalsobepassedtoOpenVINORuntime.Passthefilenamewithextension
``.tflite``to``read_model``andexportedanOpenVINOIRwith
``save_model``.

Thistutorialusestheimageclassificationmodel
`inception_v4_quant<https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/default/1>`__.
Itispre-trainedmodeloptimizedtoworkwithTensorFlowLite.

..code::ipython3

%pipinstall-qkagglehub


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath
importkagglehub

tflite_model_dir=kagglehub.model_download("tensorflow/inception/tfLite/v4-quant")
tflite_model_path=Path(tflite_model_dir)/"1.tflite"

..code::ipython3

importopenvinoasov

core=ov.Core()

model_tflite=core.read_model(tflite_model_path)
compiled_model_tflite=core.compile_model(model=model_tflite,device_name=device.value)

..code::ipython3

ov.save_model(model_tflite,output_model="model/exported_tflite_model.xml")

PyTorchModel
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`PyTorch<https://pytorch.org/>`__modelscannotbedirectlypassedto
``core.read_model``.``ov.Model``formodelobjectsfromthisframework
canbeobtainedusing``ov.convert_model``API.Youcanfindmore
detailsin`pytorch-to-openvino<../pytorch-to-openvino>`__notebook.In
thistutorialwewilluse
`resnet18<https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>`__
modelformtorchvisionlibrary.Afterconversionmodelusing
``ov.convert_model``,itcanbecompiledondeviceusing
``core.compile_model``orsavedondiskforthenextusageusing
``ov.save_model``

..code::ipython3

%pipinstall-q"torch>=2.1"torchvision--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importopenvinoasov
importtorch
fromtorchvision.modelsimportresnet18,ResNet18_Weights

core=ov.Core()

pt_model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
example_input=torch.zeros((1,3,224,224))
ov_model_pytorch=ov.convert_model(pt_model,example_input=example_input)

compiled_model_pytorch=core.compile_model(ov_model_pytorch,device_name=device.value)

ov.save_model(ov_model_pytorch,"model/exported_pytorch_model.xml")

GettingInformationaboutaModel
---------------------------------

`backtotop⬆️<#table-of-contents>`__

TheOpenVINOModelinstancestoresinformationaboutthemodel.
Informationabouttheinputsandoutputsofthemodelarein
``model.inputs``and``model.outputs``.Thesearealsopropertiesofthe
``CompiledModel``instance.Whileusing``model.inputs``and
``model.outputs``inthecellsbelow,youcanalsouse
``compiled_model.inputs``and``compiled_model.outputs``.

..code::ipython3

ir_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml="classification.xml"
ir_model_name_bin="classification.bin"

download_file(ir_model_url+ir_model_name_xml,filename=ir_model_name_xml,directory="model")
download_file(ir_model_url+ir_model_name_bin,filename=ir_model_name_bin,directory="model")


..parsed-literal::

'model/classification.xml'alreadyexists.
'model/classification.bin'alreadyexists.




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/classification.bin')



ModelInputs
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Informationaboutallinputlayersisstoredinthe``inputs``
dictionary.

..code::ipython3

importopenvinoasov

core=ov.Core()
classification_model_xml="model/classification.xml"
model=core.read_model(model=classification_model_xml)
model.inputs




..parsed-literal::

[<Output:names[input,input:0]shape[1,3,224,224]type:f32>]



Thecellaboveshowsthattheloadedmodelexpectsoneinputwiththe
name*input*.Ifyouloadedadifferentmodel,youmayseeadifferent
inputlayername,andyoumayseemoreinputs.Youmayalsoobtaininfo
abouteachinputlayerusing``model.input(index)``,whereindexisa
numericindexoftheinputlayersinthemodel.Ifamodelhasonlyone
input,indexcanbeomitted.

..code::ipython3

input_layer=model.input(0)

Itisoftenusefultohaveareferencetothenameofthefirstinput
layer.Foramodelwithoneinput,``model.input(0).any_name``getsthis
name.

..code::ipython3

input_layer.any_name




..parsed-literal::

'input'



Thenextcellprintstheinputlayout,precisionandshape.

..code::ipython3

print(f"inputprecision:{input_layer.element_type}")
print(f"inputshape:{input_layer.shape}")


..parsed-literal::

inputprecision:<Type:'float32'>
inputshape:[1,3,224,224]


Thiscellshowsthatthemodelexpectsinputswithashapeof
[1,3,224,224],andthatthisisinthe``NCHW``layout.Thismeansthat
themodelexpectsinputdatawiththebatchsizeof1(``N``),3
channels(``C``),andimageswithaheight(``H``)andwidth(``W``)
equalto224.Theinputdataisexpectedtobeof``FP32``(floating
point)precision.

ModelOutputs
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov

core=ov.Core()
classification_model_xml="model/classification.xml"
model=core.read_model(model=classification_model_xml)
model.outputs




..parsed-literal::

[<Output:names[MobilenetV3/Predictions/Softmax]shape[1,1001]type:f32>]



Modeloutputinfoisstoredin``model.outputs``.Thecellaboveshows
thatthemodelreturnsoneoutput,withthe
``MobilenetV3/Predictions/Softmax``name.Loadingadifferentmodelwill
resultindifferentoutputlayername,andmoreoutputsmightbe
returned.Similartoinput,youmayalsoobtaininformationabouteach
outputseparatelyusing``model.output(index)``

Sincethismodelhasoneoutput,followthesamemethodasfortheinput
layertogetitsname.

..code::ipython3

output_layer=model.output(0)
output_layer.any_name




..parsed-literal::

'MobilenetV3/Predictions/Softmax'



Gettingtheoutputprecisionandshapeissimilartogettingtheinput
precisionandshape.

..code::ipython3

print(f"outputprecision:{output_layer.element_type}")
print(f"outputshape:{output_layer.shape}")


..parsed-literal::

outputprecision:<Type:'float32'>
outputshape:[1,1001]


Thiscellshowsthatthemodelreturnsoutputswithashapeof[1,
1001],where1isthebatchsize(``N``)and1001isthenumberof
classes(``C``).Theoutputisreturnedas32-bitfloatingpoint.

DoingInferenceonaModel
--------------------------

`backtotop⬆️<#table-of-contents>`__

**NOTE**thisnotebookdemonstratesonlythebasicsynchronous
inferenceAPI.Foranasyncinferenceexample,pleasereferto`Async
APInotebook<async-api-with-output.html>`__

ThediagrambelowshowsatypicalinferencepipelinewithOpenVINO

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a91bc582-165b-41a2-ab08-12c812059936
:alt:image.png

image.png

CreatingOpenVINOCoreandmodelcompilationiscoveredintheprevious
steps.Thenextstepispreparinginputs.Youcanprovideinputsinone
ofthesupportedformat:dictionarywithnameofinputsaskeysand
``np.arrays``thatrepresentinputtensorsasvalues,listortupleof
``np.arrays``representedinputtensors(theirordershouldmatchwith
modelinputsorder).Ifamodelhasasingleinput,wrappingtoa
dictionaryorlistcanbeomitted.Todoinferenceonamodel,pass
preparedinputsintocompiledmodelobjectobtainedusing
``core.compile_model``.Theinferenceresultrepresentedasdictionary,
wherekeysaremodeloutputsand``np.arrays``representedtheir
produceddataasvalues.

..code::ipython3

#Installopencvpackageforimagehandling
%pipinstall-qopencv-python


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


**Loadthenetwork**

..code::ipython3

ir_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml="classification.xml"
ir_model_name_bin="classification.bin"

download_file(ir_model_url+ir_model_name_xml,filename=ir_model_name_xml,directory="model")
download_file(ir_model_url+ir_model_name_bin,filename=ir_model_name_bin,directory="model")


..parsed-literal::

'model/classification.xml'alreadyexists.
'model/classification.bin'alreadyexists.




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/classification.bin')



..code::ipython3

importopenvinoasov

core=ov.Core()
classification_model_xml="model/classification.xml"
model=core.read_model(model=classification_model_xml)
compiled_model=core.compile_model(model=model,device_name=device.value)
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)

**Loadanimageandconverttotheinputshape**

Topropagateanimagethroughthenetwork,itneedstobeloadedintoan
array,resizedtotheshapethatthenetworkexpects,andconvertedto
theinputlayoutofthenetwork.

..code::ipython3

importcv2

image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg",
directory="data",
)
image=cv2.imread(str(image_filename))
image.shape



..parsed-literal::

data/coco_hollywood.jpg:0%||0.00/485k[00:00<?,?B/s]




..parsed-literal::

(663,994,3)



Theimagehasashapeof(663,994,3).Itis663pixelsinheight,994
pixelsinwidth,andhas3colorchannels.Areferencetotheheightand
widthexpectedbythenetworkisobtainedandtheimageisresizedto
thesedimensions.

..code::ipython3

#N,C,H,W=batchsize,numberofchannels,height,width.
N,C,H,W=input_layer.shape
#OpenCVresizeexpectsthedestinationsizeas(width,height).
resized_image=cv2.resize(src=image,dsize=(W,H))
resized_image.shape




..parsed-literal::

(224,224,3)



Now,theimagehasthewidthandheightthatthenetworkexpects.This
isstillin``HWC``formatandmustbechangedto``NCHW``format.
First,callthe``np.transpose()``methodtochangeto``CHW``andthen
addthe``N``dimension(where``N``\=1)bycallingthe
``np.expand_dims()``method.Next,convertthedatato``FP32``with
``np.astype()``method.

..code::ipython3

importnumpyasnp

input_data=np.expand_dims(np.transpose(resized_image,(2,0,1)),0).astype(np.float32)
input_data.shape




..parsed-literal::

(1,3,224,224)



**Doinference**

Nowthattheinputdataisintherightshape,runinference.The
``CompiledModel``inferenceresultisadictionarywherekeysarethe
Outputclassinstances(thesamekeysin``compiled_model.outputs``that
canalsobeobtainedwith``compiled_model.output(index)``)andvalues-
predictedresultin``np.array``format.

..code::ipython3

#forsingleinputmodelsonly
result=compiled_model(input_data)[output_layer]

#formultipleinputsinalist
result=compiled_model([input_data])[output_layer]

#orusingadictionary,wherethekeyisinputtensornameorindex
result=compiled_model({input_layer.any_name:input_data})[output_layer]

Youcanalsocreate``InferRequest``andrun``infer``methodon
request.

..code::ipython3

request=compiled_model.create_infer_request()
request.infer(inputs={input_layer.any_name:input_data})
result=request.get_output_tensor(output_layer.index).data

The``.infer()``functionsetsoutputtensor,thatcanbereached,using
``get_output_tensor()``.Sincethisnetworkreturnsoneoutput,andthe
referencetotheoutputlayerisinthe``output_layer.index``
parameter,youcangetthedatawith
``request.get_output_tensor(output_layer.index)``.Togetanumpyarray
fromtheoutput,usethe``.data``parameter.

..code::ipython3

result.shape




..parsed-literal::

(1,1001)



Theoutputshapeis(1,1001),whichistheexpectedoutputshape.This
shapeindicatesthatthenetworkreturnsprobabilitiesfor1001classes.
Tolearnmoreaboutthisnotion,refertothe`helloworld
notebook<hello-world-with-output.html>`__.

ReshapingandResizing
----------------------

`backtotop⬆️<#table-of-contents>`__

ChangeImageSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Insteadofreshapingtheimagetofitthemodel,itisalsopossibleto
reshapethemodeltofittheimage.Beawarethatnotallmodelssupport
reshaping,andmodelsthatdo,maynotsupportallinputshapes.The
modelaccuracymayalsosufferifyoureshapethemodelinputshape.

Firstchecktheinputshapeofthemodel,thenreshapeittothenew
inputshape.

..code::ipython3

ir_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml="segmentation.xml"
ir_model_name_bin="segmentation.bin"

download_file(ir_model_url+ir_model_name_xml,filename=ir_model_name_xml,directory="model")
download_file(ir_model_url+ir_model_name_bin,filename=ir_model_name_bin,directory="model")



..parsed-literal::

model/segmentation.xml:0%||0.00/1.38M[00:00<?,?B/s]



..parsed-literal::

model/segmentation.bin:0%||0.00/1.09M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/segmentation.bin')



..code::ipython3

importopenvinoasov

core=ov.Core()
segmentation_model_xml="model/segmentation.xml"
segmentation_model=core.read_model(model=segmentation_model_xml)
segmentation_input_layer=segmentation_model.input(0)
segmentation_output_layer=segmentation_model.output(0)

print("~~~~ORIGINALMODEL~~~~")
print(f"inputshape:{segmentation_input_layer.shape}")
print(f"outputshape:{segmentation_output_layer.shape}")

new_shape=ov.PartialShape([1,3,544,544])
segmentation_model.reshape({segmentation_input_layer.any_name:new_shape})
segmentation_compiled_model=core.compile_model(model=segmentation_model,device_name=device.value)
#help(segmentation_compiled_model)
print("~~~~RESHAPEDMODEL~~~~")
print(f"modelinputshape:{segmentation_input_layer.shape}")
print(f"compiled_modelinputshape:"f"{segmentation_compiled_model.input(index=0).shape}")
print(f"compiled_modeloutputshape:{segmentation_output_layer.shape}")


..parsed-literal::

~~~~ORIGINALMODEL~~~~
inputshape:[1,3,512,512]
outputshape:[1,1,512,512]
~~~~RESHAPEDMODEL~~~~
modelinputshape:[1,3,544,544]
compiled_modelinputshape:[1,3,544,544]
compiled_modeloutputshape:[1,1,544,544]


Theinputshapeforthesegmentationnetworkis[1,3,512,512],withthe
``NCHW``layout:thenetworkexpects3-channelimageswithawidthand
heightof512andabatchsizeof1.Reshapethenetworkwiththe
``.reshape()``methodof``IENetwork``tomakeitacceptinputimages
withawidthandheightof544.Thissegmentationnetworkalwaysreturns
arrayswiththeinputwidthandheightofequalvalue.Therefore,
settingtheinputdimensionsto544x544alsomodifiestheoutput
dimensions.Afterreshaping,compilethenetworkonceagain.

ChangeBatchSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Usethe``.reshape()``methodtosetthebatchsize,byincreasingthe
firstelementof``new_shape``.Forexample,tosetabatchsizeoftwo,
set``new_shape=(2,3,544,544)``inthecellabove.

..code::ipython3

importopenvinoasov

segmentation_model_xml="model/segmentation.xml"
segmentation_model=core.read_model(model=segmentation_model_xml)
segmentation_input_layer=segmentation_model.input(0)
segmentation_output_layer=segmentation_model.output(0)
new_shape=ov.PartialShape([2,3,544,544])
segmentation_model.reshape({segmentation_input_layer.any_name:new_shape})
segmentation_compiled_model=core.compile_model(model=segmentation_model,device_name=device.value)

print(f"inputshape:{segmentation_input_layer.shape}")
print(f"outputshape:{segmentation_output_layer.shape}")


..parsed-literal::

inputshape:[2,3,544,544]
outputshape:[2,1,544,544]


Theoutputshowsthatbysettingthebatchsizeto2,thefirstelement
(``N``)oftheinputandoutputshapehasavalueof2.Propagatethe
inputimagethroughthenetworktoseetheresult:

..code::ipython3

importnumpyasnp
importopenvinoasov

core=ov.Core()
segmentation_model_xml="model/segmentation.xml"
segmentation_model=core.read_model(model=segmentation_model_xml)
segmentation_input_layer=segmentation_model.input(0)
segmentation_output_layer=segmentation_model.output(0)
new_shape=ov.PartialShape([2,3,544,544])
segmentation_model.reshape({segmentation_input_layer.any_name:new_shape})
segmentation_compiled_model=core.compile_model(model=segmentation_model,device_name=device.value)
input_data=np.random.rand(2,3,544,544)

output=segmentation_compiled_model([input_data])

print(f"inputdatashape:{input_data.shape}")
print(f"resultdatadatashape:{segmentation_output_layer.shape}")


..parsed-literal::

inputdatashape:(2,3,544,544)
resultdatadatashape:[2,1,544,544]


CachingaModel
---------------

`backtotop⬆️<#table-of-contents>`__

Forsomedevices,likeGPU,loadingamodelcantakesometime.Model
Cachingsolvesthisissuebycachingthemodelinacachedirectory.If
``core.compile_model(model=net,device_name=device_name,config=config_dict)``
isset,cachingwillbeused.Thisoptionchecksifamodelexistsin
thecache.Ifso,itloadsitfromthecache.Ifnot,itloadsthemodel
regularly,andstoresitinthecache,sothatthenexttimethemodel
isloadedwhenthisoptionisset,themodelwillbeloadedfromthe
cache.

Inthecellbelow,wecreatea*model_cache*directoryasasubdirectory
of*model*,wherethemodelwillbecachedforthespecifieddevice.The
modelwillbeloadedtotheGPU.Afterrunningthiscellonce,themodel
willbecached,sosubsequentrunsofthiscellwillloadthemodelfrom
thecache.

*Note:ModelCachingisalsoavailableonCPUdevices*

..code::ipython3

ir_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml="classification.xml"
ir_model_name_bin="classification.bin"

download_file(ir_model_url+ir_model_name_xml,filename=ir_model_name_xml,directory="model")
download_file(ir_model_url+ir_model_name_bin,filename=ir_model_name_bin,directory="model")


..parsed-literal::

'model/classification.xml'alreadyexists.
'model/classification.bin'alreadyexists.




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/openvino-api/model/classification.bin')



..code::ipython3

importtime
frompathlibimportPath

importopenvinoasov

core=ov.Core()

cache_path=Path("model/model_cache")
cache_path.mkdir(exist_ok=True)
#EnablecachingforOpenVINORuntime.Todisablecachingsetenable_caching=False
enable_caching=True
config_dict={"CACHE_DIR":str(cache_path)}ifenable_cachingelse{}

classification_model_xml="model/classification.xml"
model=core.read_model(model=classification_model_xml)

start_time=time.perf_counter()
compiled_model=core.compile_model(model=model,device_name=device.value,config=config_dict)
end_time=time.perf_counter()
print(f"Loadingthenetworktothe{device.value}devicetook{end_time-start_time:.2f}seconds.")


..parsed-literal::

LoadingthenetworktotheCPUdevicetook0.17seconds.


Afterrunningthepreviouscell,weknowthemodelexistsinthecache
directory.Then,wedeletethecompiledmodelandloaditagain.Now,we
measurethetimeittakesnow.

..code::ipython3

delcompiled_model
start_time=time.perf_counter()
compiled_model=core.compile_model(model=model,device_name=device.value,config=config_dict)
end_time=time.perf_counter()
print(f"Loadingthenetworktothe{device.value}devicetook{end_time-start_time:.2f}seconds.")


..parsed-literal::

LoadingthenetworktotheCPUdevicetook0.08seconds.


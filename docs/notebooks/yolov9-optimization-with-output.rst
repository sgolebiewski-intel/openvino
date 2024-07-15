ConvertandOptimizeYOLOv9withOpenVINO™
==========================================

YOLOv9marksasignificantadvancementinreal-timeobjectdetection,
introducinggroundbreakingtechniquessuchasProgrammableGradient
Information(PGI)andtheGeneralizedEfficientLayerAggregation
Network(GELAN).Thismodeldemonstratesremarkableimprovementsin
efficiency,accuracy,andadaptability,settingnewbenchmarksontheMS
COCOdataset.Moredetailsaboutmodelcanbefoundin
`paper<https://arxiv.org/abs/2402.13616>`__and`original
repository<https://github.com/WongKinYiu/yolov9>`__Thistutorial
demonstratesstep-by-stepinstructionsonhowtorunandoptimize
PyTorchYOLOV9withOpenVINO.

Thetutorialconsistsofthefollowingsteps:

-PreparePyTorchmodel
-ConvertPyTorchmodeltoOpenVINOIR
-RunmodelinferencewithOpenVINO
-Prepareandrunoptimizationpipeline
-CompareperformanceoftheFP32andquantizedmodels.
-Runoptimizedmodelinferenceonvideo####Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`GetPyTorchmodel<#get-pytorch-model>`__
-`ConvertPyTorchmodeltoOpenVINO
IR<#convert-pytorch-model-to-openvino-ir>`__
-`Verifymodelinference<#verify-model-inference>`__

-`Preprocessing<#preprocessing>`__
-`Postprocessing<#postprocessing>`__
-`Selectinferencedevice<#select-inference-device>`__

-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__

-`Preparedataset<#prepare-dataset>`__
-`Performmodelquantization<#perform-model-quantization>`__

-`Runquantizedmodelinference<#run-quantized-model-inference>`__
-`ComparePerformanceoftheOriginalandQuantized
Models<#compare-performance-of-the-original-and-quantized-models>`__
-`RunLiveObjectDetection<#run-live-object-detection>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__##Prerequisites

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.3.0""nncf>=2.8.1""opencv-python""seaborn""pandas""scikit-learn""torch""torchvision""tqdm"--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file,VideoPlayer

ifnotPath("yolov9").exists():
!gitclonehttps://github.com/WongKinYiu/yolov9
%cdyolov9


..parsed-literal::

Cloninginto'yolov9'...
remote:Enumeratingobjects:781,done.[K
remote:Countingobjects:100%(407/407),done.[K
remote:Compressingobjects:100%(168/168),done.[K
remote:Total781(delta280),reused279(delta227),pack-reused374[K
Receivingobjects:100%(781/781),3.30MiB|7.49MiB/s,done.
Resolvingdeltas:100%(325/325),done.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9


GetPyTorchmodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
`torch.nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class,initializedbyastatedictionarywithmodelweights.Wewilluse
the``gelan-c``(light-weightversionofyolov9)modelpre-trainedona
COCOdataset,whichisavailableinthis
`repo<https://github.com/WongKinYiu/yolov9>`__,butthesamestepsare
applicableforothermodelsfromYOLOV9family.

..code::ipython3

#Downloadpre-trainedmodelweights
MODEL_LINK="https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt"
DATA_DIR=Path("data/")
MODEL_DIR=Path("model/")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

download_file(MODEL_LINK,directory=MODEL_DIR,show_progress=True)



..parsed-literal::

model/gelan-c.pt:0%||0.00/49.1M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9/model/gelan-c.pt')



ConvertPyTorchmodeltoOpenVINOIR
------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelconversionviaModelConversionAPI.
``ov.convert_model``functionacceptsmodelobjectandexampleinputfor
tracingthemodelandreturnsaninstanceof``ov.Model``,representing
thismodelinOpenVINOformat.TheObtainedmodelisreadyforloading
onspecificdevicesorcanbesavedondiskforthenextdeployment
using``ov.save_model``.

..code::ipython3

frommodels.experimentalimportattempt_load
importtorch
importopenvinoasov
frommodels.yoloimportDetect,DualDDetect
fromutils.generalimportyaml_save,yaml_load

weights=MODEL_DIR/"gelan-c.pt"
ov_model_path=MODEL_DIR/weights.name.replace(".pt","_openvino_model")/weights.name.replace(".pt",".xml")

ifnotov_model_path.exists():
model=attempt_load(weights,device="cpu",inplace=True,fuse=True)
metadata={"stride":int(max(model.stride)),"names":model.names}

model.eval()
fork,minmodel.named_modules():
ifisinstance(m,(Detect,DualDDetect)):
m.inplace=False
m.dynamic=True
m.export=True

example_input=torch.zeros((1,3,640,640))
model(example_input)

ov_model=ov.convert_model(model,example_input=example_input)

#specifyinputandoutputnamesforcompatibilitywithyolov9repointerface
ov_model.outputs[0].get_tensor().set_names({"output0"})
ov_model.inputs[0].get_tensor().set_names({"images"})
ov.save_model(ov_model,ov_model_path)
#savemetadata
yaml_save(ov_model_path.parent/weights.name.replace(".pt",".yaml"),metadata)
else:
metadata=yaml_load(ov_model_path.parent/weights.name.replace(".pt",".yaml"))


..parsed-literal::

Fusinglayers...
Modelsummary:387layers,25288768parameters,0gradients,102.1GFLOPs
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9/models/yolo.py:108:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
elifself.dynamicorself.shape!=shape:


..parsed-literal::

['x']


Verifymodelinference
----------------------

`backtotop⬆️<#table-of-contents>`__

Totestmodelwork,wecreateinferencepipelinesimilarto
``detect.py``.Thepipelineconsistsofpreprocessingstep,inferenceof
OpenVINOmodel,andresultspost-processingtogetboundingboxes.

Preprocessing
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Modelinputisatensorwiththe``[1,3,640,640]``shapein
``N,C,H,W``format,where

-``N``-numberofimagesinbatch(batchsize)
-``C``-imagechannels
-``H``-imageheight
-``W``-imagewidth

ModelexpectsimagesinRGBchannelsformatandnormalizedin[0,1]
range.Toresizeimagestofitmodelsize``letterbox``resizeapproach
isusedwheretheaspectratioofwidthandheightispreserved.Itis
definedinyolov9repository.

Tokeepspecificshape,preprocessingautomaticallyenablespadding.

..code::ipython3

importnumpyasnp
importtorch
fromPILimportImage
fromutils.augmentationsimportletterbox

image_url="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/7b6af406-4ccb-4ded-a13d-62b7c0e42e96"
download_file(image_url,directory=DATA_DIR,filename="test_image.jpg",show_progress=True)


defpreprocess_image(img0:np.ndarray):
"""
PreprocessimageaccordingtoYOLOv9inputrequirements.
Takesimageinnp.arrayformat,resizesittospecificsizeusingletterboxresize,convertscolorspacefromBGR(defaultinOpenCV)toRGBandchangesdatalayoutfromHWCtoCHW.

Parameters:
img0(np.ndarray):imageforpreprocessing
Returns:
img(np.ndarray):imageafterpreprocessing
img0(np.ndarray):originalimage
"""
#resize
img=letterbox(img0,auto=False)[0]

#Convert
img=img.transpose(2,0,1)
img=np.ascontiguousarray(img)
returnimg,img0


defprepare_input_tensor(image:np.ndarray):
"""
ConvertspreprocessedimagetotensorformataccordingtoYOLOv9inputrequirements.
Takesimageinnp.arrayformatwithunit8datain[0,255]rangeandconvertsittotorch.Tensorobjectwithfloatdatain[0,1]range

Parameters:
image(np.ndarray):imageforconversiontotensor
Returns:
input_tensor(torch.Tensor):floattensorreadytouseforYOLOv9inference
"""
input_tensor=image.astype(np.float32)#uint8tofp16/32
input_tensor/=255.0#0-255to0.0-1.0

ifinput_tensor.ndim==3:
input_tensor=np.expand_dims(input_tensor,0)
returninput_tensor


NAMES=metadata["names"]



..parsed-literal::

data/test_image.jpg:0%||0.00/101k[00:00<?,?B/s]


Postprocessing
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Modeloutputcontainsdetectionboxescandidates.Itisatensorwith
the``[1,25200,85]``shapeinthe``B,N,85``format,where:

-``B``-batchsize
-``N``-numberofdetectionboxes

Detectionboxhasthe[``x``,``y``,``h``,``w``,``box_score``,
``class_no_1``,…,``class_no_80``]format,where:

-(``x``,``y``)-rawcoordinatesofboxcenter
-``h``,``w``-rawheightandwidthofbox
-``box_score``-confidenceofdetectionbox
-``class_no_1``,…,``class_no_80``-probabilitydistributionover
theclasses.

Forgettingfinalprediction,weneedtoapplynonmaximumsuppression
algorithmandrescaleboxescoordinatestooriginalimagesize.

..code::ipython3

fromutils.plotsimportAnnotator,colors

fromtypingimportList,Tuple
fromutils.generalimportscale_boxes,non_max_suppression


defdetect(
model:ov.Model,
image_path:Path,
conf_thres:float=0.25,
iou_thres:float=0.45,
classes:List[int]=None,
agnostic_nms:bool=False,
):
"""
OpenVINOYOLOv9modelinferencefunction.Readsimage,preprocessit,runsmodelinferenceandpostprocessresultsusingNMS.
Parameters:
model(Model):OpenVINOcompiledmodel.
image_path(Path):inputimagepath.
conf_thres(float,*optional*,0.25):minimalacceptedconfidenceforobjectfiltering
iou_thres(float,*optional*,0.45):minimaloverlapscoreforremovingobjectsduplicatesinNMS
classes(List[int],*optional*,None):labelsforpredictionfiltering,ifnotprovidedallpredictedlabelswillbeused
agnostic_nms(bool,*optional*,False):applyclassagnosticNMSapproachornot
Returns:
pred(List):listofdetectionswith(n,6)shape,wheren-numberofdetectedboxesinformat[x1,y1,x2,y2,score,label]
orig_img(np.ndarray):imagebeforepreprocessing,canbeusedforresultsvisualization
inpjut_shape(Tuple[int]):shapeofmodelinputtensor,canbeusedforoutputrescaling
"""
ifisinstance(image_path,np.ndarray):
img=image_path
else:
img=np.array(Image.open(image_path))
preprocessed_img,orig_img=preprocess_image(img)
input_tensor=prepare_input_tensor(preprocessed_img)
predictions=torch.from_numpy(model(input_tensor)[0])
pred=non_max_suppression(predictions,conf_thres,iou_thres,classes=classes,agnostic=agnostic_nms)
returnpred,orig_img,input_tensor.shape


defdraw_boxes(
predictions:np.ndarray,
input_shape:Tuple[int],
image:np.ndarray,
names:List[str],
):
"""
Utilityfunctionfordrawingpredictedboundingboxesonimage
Parameters:
predictions(np.ndarray):listofdetectionswith(n,6)shape,wheren-numberofdetectedboxesinformat[x1,y1,x2,y2,score,label]
image(np.ndarray):imageforboxesvisualization
names(List[str]):listofnamesforeachclassindataset
colors(Dict[str,int]):mappingbetweenclassnameanddrawingcolor
Returns:
image(np.ndarray):boxvisualizationresult
"""
ifnotlen(predictions):
returnimage

annotator=Annotator(image,line_width=1,example=str(names))
#Rescaleboxesfrominputsizetooriginalimagesize
predictions[:,:4]=scale_boxes(input_shape[2:],predictions[:,:4],image.shape).round()

#Writeresults
for*xyxy,conf,clsinreversed(predictions):
label=f"{names[int(cls)]}{conf:.2f}"
annotator.box_label(xyxy,label,color=colors(int(cls),True))
returnimage

..code::ipython3

core=ov.Core()
#readconvertedmodel
ov_model=core.read_model(ov_model_path)

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

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

#loadmodelonselecteddevice
ifdevice.value!="CPU":
ov_model.reshape({0:[1,3,640,640]})
compiled_model=core.compile_model(ov_model,device.value)

..code::ipython3

boxes,image,input_shape=detect(compiled_model,DATA_DIR/"test_image.jpg")
image_with_boxes=draw_boxes(boxes[0],input_shape,image,NAMES)
#visualizeresults
Image.fromarray(image_with_boxes)




..image::yolov9-optimization-with-output_files/yolov9-optimization-with-output_16_0.png



OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
YOLOv9.Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeanOpenVINOIRmodel,usingthe``ov.save_model``function.

Preparedataset
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThecodebelowdownloadsCOCOdatasetandpreparesadataloaderthatis
usedtoevaluatetheyolov9modelaccuracy.Wereuseitssubsetfor
quantization.

..code::ipython3

fromzipfileimportZipFile


DATA_URL="http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"

OUT_DIR=Path(".")

download_file(DATA_URL,directory=OUT_DIR,show_progress=True)
download_file(LABELS_URL,directory=OUT_DIR,show_progress=True)

ifnot(OUT_DIR/"coco/labels").exists():
withZipFile("coco2017labels-segments.zip","r")aszip_ref:
zip_ref.extractall(OUT_DIR)
withZipFile("val2017.zip","r")aszip_ref:
zip_ref.extractall(OUT_DIR/"coco/images")



..parsed-literal::

val2017.zip:0%||0.00/778M[00:00<?,?B/s]



..parsed-literal::

coco2017labels-segments.zip:0%||0.00/169M[00:00<?,?B/s]


..code::ipython3

fromcollectionsimportnamedtuple
importyaml
fromutils.dataloadersimportcreate_dataloader
fromutils.generalimportcolorstr

#readdatasetconfig
DATA_CONFIG="data/coco.yaml"
withopen(DATA_CONFIG)asf:
data=yaml.load(f,Loader=yaml.SafeLoader)

#Dataloader
TASK="val"#pathtotrain/val/testimages
Option=namedtuple("Options",["single_cls"])#imitationofcommandlineprovidedoptionsforsingleclassevaluation
opt=Option(False)
dataloader=create_dataloader(
str(Path("coco")/data[TASK]),
640,
1,
32,
opt,
pad=0.5,
prefix=colorstr(f"{TASK}:"),
)[0]


..parsed-literal::

val:Scanningcoco/val2017...4952images,48backgrounds,0corrupt:100%|██████████|5000/500000:00
val:Newcachecreated:coco/val2017.cache


NNCFprovides``nncf.Dataset``wrapperforusingnativeframework
dataloadersinquantizationpipeline.Additionally,wespecifytransform
functionthatwillberesponsibleforpreparinginputdatainmodel
expectedformat.

..code::ipython3

importnncf


deftransform_fn(data_item):
"""
Quantizationtransformfunction.Extractsandpreprocessinputdatafromdataloaderitemforquantization.
Parameters:
data_item:TuplewithdataitemproducedbyDataLoaderduringiteration
Returns:
input_tensor:Inputdataforquantization
"""
img=data_item[0].numpy()
input_tensor=prepare_input_tensor(img)
returninput_tensor


quantization_dataset=nncf.Dataset(dataloader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Performmodelquantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscopeetc.)canbeprovided.YOLOv9modelcontains
non-ReLUactivationfunctions,whichrequireasymmetricquantizationof
activations.Toachievebetterresults,wewillusea``mixed``
quantizationpreset.Itprovidessymmetricquantizationofweightsand
asymmetricquantizationofactivations.

..code::ipython3

ov_int8_model_path=MODEL_DIR/weights.name.replace(".pt","_int8_openvino_model")/weights.name.replace(".pt","_int8.xml")

ifnotov_int8_model_path.exists():
quantized_model=nncf.quantize(ov_model,quantization_dataset,preset=nncf.QuantizationPreset.MIXED)

ov.save_model(quantized_model,ov_int8_model_path)
yaml_save(ov_int8_model_path.parent/weights.name.replace(".pt","_int8.yaml"),metadata)


..parsed-literal::

2024-07-1304:25:19.627535:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:25:19.663330:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:25:20.258143:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.


Runquantizedmodelinference
-----------------------------

`backtotop⬆️<#table-of-contents>`__

Therearenochangesinmodelusageafterapplyingquantization.Let’s
checkthemodelworkonthepreviouslyusedimage.

..code::ipython3

quantized_model=core.read_model(ov_int8_model_path)

ifdevice.value!="CPU":
quantized_model.reshape({0:[1,3,640,640]})

compiled_model=core.compile_model(quantized_model,device.value)

..code::ipython3

boxes,image,input_shape=detect(compiled_model,DATA_DIR/"test_image.jpg")
image_with_boxes=draw_boxes(boxes[0],input_shape,image,NAMES)
#visualizeresults
Image.fromarray(image_with_boxes)




..image::yolov9-optimization-with-output_files/yolov9-optimization-with-output_27_0.png



ComparePerformanceoftheOriginalandQuantizedModels
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

WeusetheOpenVINO`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
tomeasuretheinferenceperformanceofthe``FP32``and``INT8``
models.

**NOTE**:Formoreaccurateperformance,itisrecommendedtorun
``benchmark_app``inaterminal/commandpromptafterclosingother
applications.Run``benchmark_app-mmodel.xml-dCPU``tobenchmark
asyncinferenceonCPUforoneminute.Change``CPU``to``GPU``to
benchmarkonGPU.Run``benchmark_app--help``toseeanoverviewof
allcommand-lineoptions.

..code::ipython3

!benchmark_app-m$ov_model_path-shape"[1,3,640,640]"-d$device.value-apiasync-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook26.21ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]images(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]output0(node:__module.model.22/aten::cat/Concat_5):f32/[...]/[?,84,8400]
[INFO]xi.1(node:__module.model.22/aten::cat/Concat_2):f32/[...]/[?,144,4..,4..]
[INFO]xi.3(node:__module.model.22/aten::cat/Concat_1):f32/[...]/[?,144,2..,2..]
[INFO]xi(node:__module.model.22/aten::cat/Concat):f32/[...]/[?,144,1..,1..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'images':[1,3,640,640]
[INFO]Reshapemodeltook7.85ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]images(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output0(node:__module.model.22/aten::cat/Concat_5):f32/[...]/[1,84,8400]
[INFO]xi.1(node:__module.model.22/aten::cat/Concat_2):f32/[...]/[1,144,80,80]
[INFO]xi.3(node:__module.model.22/aten::cat/Concat_1):f32/[...]/[1,144,40,40]
[INFO]xi(node:__module.model.22/aten::cat/Concat):f32/[...]/[1,144,20,20]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook490.27ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'images'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'images'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook186.95ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:228iterations
[INFO]Duration:15678.96ms
[INFO]Latency:
[INFO]Median:413.56ms
[INFO]Average:411.44ms
[INFO]Min:338.36ms
[INFO]Max:431.50ms
[INFO]Throughput:14.54FPS


..code::ipython3

!benchmark_app-m$ov_int8_model_path-shape"[1,3,640,640]"-d$device.value-apiasync-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook40.98ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]images(node:x):f32/[...]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output0(node:__module.model.22/aten::cat/Concat_5):f32/[...]/[1,84,8400]
[INFO]xi.1(node:__module.model.22/aten::cat/Concat_2):f32/[...]/[1,144,80,80]
[INFO]xi.3(node:__module.model.22/aten::cat/Concat_1):f32/[...]/[1,144,40,40]
[INFO]xi(node:__module.model.22/aten::cat/Concat):f32/[...]/[1,144,20,20]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'images':[1,3,640,640]
[INFO]Reshapemodeltook0.05ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]images(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output0(node:__module.model.22/aten::cat/Concat_5):f32/[...]/[1,84,8400]
[INFO]xi.1(node:__module.model.22/aten::cat/Concat_2):f32/[...]/[1,144,80,80]
[INFO]xi.3(node:__module.model.22/aten::cat/Concat_1):f32/[...]/[1,144,40,40]
[INFO]xi(node:__module.model.22/aten::cat/Concat):f32/[...]/[1,144,20,20]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook964.26ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'images'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'images'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook77.25ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:750iterations
[INFO]Duration:15181.84ms
[INFO]Latency:
[INFO]Median:121.39ms
[INFO]Average:121.02ms
[INFO]Min:93.56ms
[INFO]Max:133.28ms
[INFO]Throughput:49.40FPS


RunLiveObjectDetection
-------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importtime
fromIPythonimportdisplay
importcv2


#Mainprocessingfunctiontorunobjectdetection.
defrun_object_detection(
source=0,
flip=False,
use_popup=False,
skip_first_frames=0,
model=ov_model,
device=device.value,
):
player=None
compiled_model=core.compile_model(model,device)
try:
#Createavideoplayertoplaywithtargetfps.
player=VideoPlayer(source=source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
#Startcapturing.
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(winname=title,flags=cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)

processing_times=collections.deque()
whileTrue:
#Grabtheframe.
frame=player.next()
ifframeisNone:
print("Sourceended")
break
#IftheframeislargerthanfullHD,reducesizetoimprovetheperformance.
scale=1280/max(frame.shape)
ifscale<1:
frame=cv2.resize(
src=frame,
dsize=None,
fx=scale,
fy=scale,
interpolation=cv2.INTER_AREA,
)
#Gettheresults.
input_image=np.array(frame)

start_time=time.time()
#modelexpectsRGBimage,whilevideocapturinginBGR
detections,_,input_shape=detect(compiled_model,input_image[:,:,::-1])
stop_time=time.time()

image_with_boxes=draw_boxes(detections[0],input_shape,input_image,NAMES)
frame=image_with_boxes

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()

_,f_width=frame.shape[:2]
#Meanprocessingtime[ms].
processing_time=np.mean(processing_times)*1000
fps=1000/processing_time
cv2.putText(
img=frame,
text=f"Inferencetime:{processing_time:.1f}ms({fps:.1f}FPS)",
org=(20,40),
fontFace=cv2.FONT_HERSHEY_COMPLEX,
fontScale=f_width/1000,
color=(0,0,255),
thickness=1,
lineType=cv2.LINE_AA,
)
#Usethisworkaroundifthereisflickering.
ifuse_popup:
cv2.imshow(winname=title,mat=frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(ext=".jpg",img=frame,params=[cv2.IMWRITE_JPEG_QUALITY,100])
#CreateanIPythonimage.⬆️
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook.
display.clear_output(wait=True)
display.display(i)
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifplayerisnotNone:
#Stopcapturing.
player.stop()
ifuse_popup:
cv2.destroyAllWindows()

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with \``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set \``flip=True`` when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,
set \``use_popup=True``.

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
remoteserver(forexample,inBinderorGoogleColabservice),the
webcamwillnotwork.Bydefault,thelowercellwillrunmodel
inferenceonavideofile.Ifyouwanttotryliveinferenceonyour
webcamset``WEBCAM_INFERENCE=True``

Runtheobjectdetection:

..code::ipython3

WEBCAM_INFERENCE=False

ifWEBCAM_INFERENCE:
VIDEO_SOURCE=0#Webcam
else:
VIDEO_SOURCE="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

quantized_model=core.read_model(ov_int8_model_path)

run_object_detection(
source=VIDEO_SOURCE,
flip=True,
use_popup=False,
model=quantized_model,
device=device.value,
)



..image::yolov9-optimization-with-output_files/yolov9-optimization-with-output_36_0.png


..parsed-literal::

Sourceended


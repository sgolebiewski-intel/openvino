ConvertandOptimizeYOLOv7withOpenVINO™
==========================================

TheYOLOv7algorithmismakingbigwavesinthecomputervisionand
machinelearningcommunities.Itisareal-timeobjectdetection
algorithmthatperformsimagerecognitiontasksbytakinganimageas
inputandthenpredictingboundingboxesandclassprobabilitiesfor
eachobjectintheimage.

YOLOstandsfor“YouOnlyLookOnce”,itisapopularfamilyof
real-timeobjectdetectionalgorithms.TheoriginalYOLOobjectdetector
wasfirstreleasedin2016.Sincethen,differentversionsandvariants
ofYOLOhavebeenproposed,eachprovidingasignificantincreasein
performanceandefficiency.YOLOv7isnextstageofevolutionofYOLO
modelsfamily,whichprovidesagreatlyimprovedreal-timeobject
detectionaccuracywithoutincreasingtheinferencecosts.Moredetails
aboutitsrealizationcanbefoundinoriginalmodel
`paper<https://arxiv.org/abs/2207.02696>`__and
`repository<https://github.com/WongKinYiu/yolov7>`__

Real-timeobjectdetectionisoftenusedasakeycomponentincomputer
visionsystems.Applicationsthatusereal-timeobjectdetectionmodels
includevideoanalytics,robotics,autonomousvehicles,multi-object
trackingandobjectcounting,medicalimageanalysis,andmanyothers.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimizePyTorchYOLOV7withOpenVINO.

Thetutorialconsistsofthefollowingsteps:

-PreparePyTorchmodel
-Downloadandpreparedataset
-Validateoriginalmodel
-ConvertPyTorchmodeltoONNX
-ConvertONNXmodeltoOpenVINOIR
-Validateconvertedmodel
-Prepareandrunoptimizationpipeline
-CompareaccuracyoftheFP32andquantizedmodels.
-CompareperformanceoftheFP32andquantizedmodels.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`GetPytorchmodel<#get-pytorch-model>`__
-`Prerequisites<#prerequisites>`__
-`Checkmodelinference<#check-model-inference>`__
-`ExporttoONNX<#export-to-onnx>`__
-`ConvertONNXModeltoOpenVINOIntermediateRepresentation
(IR)<#convert-onnx-model-to-openvino-intermediate-representation-ir>`__
-`Verifymodelinference<#verify-model-inference>`__

-`Preprocessing<#preprocessing>`__
-`Postprocessing<#postprocessing>`__
-`Selectinferencedevice<#select-inference-device>`__

-`Verifymodelaccuracy<#verify-model-accuracy>`__

-`Downloaddataset<#download-dataset>`__
-`Createdataloader<#create-dataloader>`__
-`Definevalidationfunction<#define-validation-function>`__

-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__
-`ValidateQuantizedmodel
inference<#validate-quantized-model-inference>`__
-`Validatequantizedmodel
accuracy<#validate-quantized-model-accuracy>`__
-`ComparePerformanceoftheOriginalandQuantized
Models<#compare-performance-of-the-original-and-quantized-models>`__

GetPytorchmodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
`torch.nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class,initializedbyastatedictionarywithmodelweights.Wewilluse
theYOLOv7tinymodelpre-trainedonaCOCOdataset,whichisavailable
inthis`repo<https://github.com/WongKinYiu/yolov7>`__.Typicalsteps
toobtainpre-trainedmodel:

1.Createinstanceofmodelclass.
2.Loadcheckpointstatedict,whichcontainspre-trainedmodelweights.
3.Turnmodeltoevaluationforswitchingsomeoperationstoinference
mode.

Inthiscase,themodelcreatorsprovideatoolthatenablesconverting
theYOLOv7modeltoONNX,sowedonotneedtodothesestepsmanually.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0""nncf>=2.5.0""opencv-python""seaborn""onnx""Pillow""pandas""scikit-learn""torch""torchvision""PyYAML>=5.3.1""tqdm"--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
descript-audiotools0.7.2requiresprotobuf<3.20,>=3.9.2,butyouhaveprotobuf3.20.3whichisincompatible.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.3.1+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

..code::ipython3

#CloneYOLOv7repo
frompathlibimportPath

ifnotPath("yolov7").exists():
!gitclonehttps://github.com/WongKinYiu/yolov7
%cdyolov7


..parsed-literal::

Cloninginto'yolov7'...
remote:Enumeratingobjects:1197,done.[K
remote:Total1197(delta0),reused0(delta0),pack-reused1197[K
Receivingobjects:100%(1197/1197),74.23MiB|23.61MiB/s,done.
Resolvingdeltas:100%(520/520),done.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/yolov7-optimization/yolov7


..code::ipython3

#Downloadpre-trainedmodelweights
MODEL_LINK="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
DATA_DIR=Path("data/")
MODEL_DIR=Path("model/")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

download_file(MODEL_LINK,directory=MODEL_DIR,show_progress=True)



..parsed-literal::

model/yolov7-tiny.pt:0%||0.00/12.1M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/yolov7-optimization/yolov7/model/yolov7-tiny.pt')



Checkmodelinference
---------------------

`backtotop⬆️<#table-of-contents>`__

``detect.py``scriptrunpytorchmodelinferenceandsaveimageas
result,

..code::ipython3

!python-Wignoredetect.py--weightsmodel/yolov7-tiny.pt--conf0.25--img-size640--sourceinference/images/horses.jpg


..parsed-literal::

Namespace(agnostic_nms=False,augment=False,classes=None,conf_thres=0.25,device='',exist_ok=False,img_size=640,iou_thres=0.45,name='exp',no_trace=False,nosave=False,project='runs/detect',save_conf=False,save_txt=False,source='inference/images/horses.jpg',update=False,view_img=False,weights=['model/yolov7-tiny.pt'])
YOLOR🚀v0.1-128-ga207844torch2.3.1+cpuCPU

Fusinglayers...
ModelSummary:200layers,6219709parameters,229245gradients,13.7GFLOPS
ConvertmodeltoTraced-model...
traced_script_modulesaved!
modelistraced!

5horses,Done.(77.8ms)Inference,(0.9ms)NMS
Theimagewiththeresultissavedin:runs/detect/exp/horses.jpg
Done.(0.085s)


..code::ipython3

fromPILimportImage

#visualizepredictionresult
Image.open("runs/detect/exp/horses.jpg")




..image::yolov7-optimization-with-output_files/yolov7-optimization-with-output_10_0.png



ExporttoONNX
--------------

`backtotop⬆️<#table-of-contents>`__

ToexportanONNXformatofthemodel,wewilluse``export.py``script.
Letuscheckitsarguments.

..code::ipython3

!pythonexport.py--help


..parsed-literal::

Importonnx_graphsurgeonfailure:Nomodulenamed'onnx_graphsurgeon'
usage:export.py[-h][--weightsWEIGHTS][--img-sizeIMG_SIZE[IMG_SIZE...]]
[--batch-sizeBATCH_SIZE][--dynamic][--dynamic-batch]
[--grid][--end2end][--max-whMAX_WH][--topk-allTOPK_ALL]
[--iou-thresIOU_THRES][--conf-thresCONF_THRES]
[--deviceDEVICE][--simplify][--include-nms][--fp16]
[--int8]

optionalarguments:
-h,--helpshowthishelpmessageandexit
--weightsWEIGHTSweightspath
--img-sizeIMG_SIZE[IMG_SIZE...]
imagesize
--batch-sizeBATCH_SIZE
batchsize
--dynamicdynamicONNXaxes
--dynamic-batchdynamicbatchonnxfortensorrtandonnx-runtime
--gridexportDetect()layergrid
--end2endexportend2endonnx
--max-whMAX_WHNonefortensorrtnms,intvalueforonnx-runtimenms
--topk-allTOPK_ALLtopkobjectsforeveryimages
--iou-thresIOU_THRES
iouthresholdforNMS
--conf-thresCONF_THRES
confthresholdforNMS
--deviceDEVICEcudadevice,i.e.0or0,1,2,3orcpu
--simplifysimplifyonnxmodel
--include-nmsexportend2endonnx
--fp16CoreMLFP16half-precisionexport
--int8CoreMLINT8quantization


Themostimportantparameters:

-``--weights``-pathtomodelweightscheckpoint
-``--img-size``-sizeofinputimageforonnxtracing

WhenexportingtheONNXmodelfromPyTorch,thereisanopportunityto
setupconfigurableparametersforincludingpost-processingresultsin
model:

-``--end2end``-exportfullmodeltoonnxincludingpost-processing
-``--grid``-exportDetectlayeraspartofmodel
-``--topk-all``-topkelementsforallimages
-``--iou-thres``-intersectionoverunionthresholdforNMS
-``--conf-thres``-minimalconfidencethreshold
-``--max-wh``-maxboundingboxwidthandheightforNMS

Includingwholepost-processingtomodelcanhelptoachievemore
performantresults,butinthesametimeitmakesthemodelless
flexibleanddoesnotguaranteefullaccuracyreproducibility.Itisthe
reasonwhywewilladdonly``--grid``parametertopreserveoriginal
pytorchmodelresultformat.Ifyouwanttounderstandhowtoworkwith
anend2endONNXmodel,youcancheckthis
`notebook<https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb>`__.

..code::ipython3

!python-Wignoreexport.py--weightsmodel/yolov7-tiny.pt--grid


..parsed-literal::

Importonnx_graphsurgeonfailure:Nomodulenamed'onnx_graphsurgeon'
Namespace(batch_size=1,conf_thres=0.25,device='cpu',dynamic=False,dynamic_batch=False,end2end=False,fp16=False,grid=True,img_size=[640,640],include_nms=False,int8=False,iou_thres=0.45,max_wh=None,simplify=False,topk_all=100,weights='model/yolov7-tiny.pt')
YOLOR🚀v0.1-128-ga207844torch2.3.1+cpuCPU

Fusinglayers...
ModelSummary:200layers,6219709parameters,6219709gradients,13.7GFLOPS

StartingTorchScriptexportwithtorch2.3.1+cpu...
TorchScriptexportsuccess,savedasmodel/yolov7-tiny.torchscript.pt
CoreMLexportfailure:Nomodulenamed'coremltools'

StartingTorchScript-Liteexportwithtorch2.3.1+cpu...
TorchScript-Liteexportsuccess,savedasmodel/yolov7-tiny.torchscript.ptl

StartingONNXexportwithonnx1.16.1...
ONNXexportsuccess,savedasmodel/yolov7-tiny.onnx

Exportcomplete(2.69s).Visualizewithhttps://github.com/lutzroeder/netron.


ConvertONNXModeltoOpenVINOIntermediateRepresentation(IR)
---------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__WhileONNXmodelsaredirectly
supportedbyOpenVINOruntime,itcanbeusefultoconvertthemtoIR
formattotaketheadvantageofOpenVINOmodelconversionAPIfeatures.
The``ov.convert_model``pythonfunctionof`modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
canbeusedforconvertingthemodel.Thefunctionreturnsinstanceof
OpenVINOModelclass,whichisreadytouseinPythoninterface.
However,itcanalsobesaveondeviceinOpenVINOIRformatusing
``ov.save_model``forfutureexecution.

..code::ipython3

importopenvinoasov

model=ov.convert_model("model/yolov7-tiny.onnx")
#serializemodelforsavingIR
ov.save_model(model,"model/yolov7-tiny.xml")

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
definedinyolov7repository.

Tokeepspecificshape,preprocessingautomaticallyenablespadding.

..code::ipython3

importnumpyasnp
importtorch
fromPILimportImage
fromutils.datasetsimportletterbox
fromutils.plotsimportplot_one_box


defpreprocess_image(img0:np.ndarray):
"""
PreprocessimageaccordingtoYOLOv7inputrequirements.
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
ConvertspreprocessedimagetotensorformataccordingtoYOLOv7inputrequirements.
Takesimageinnp.arrayformatwithunit8datain[0,255]rangeandconvertsittotorch.Tensorobjectwithfloatdatain[0,1]range

Parameters:
image(np.ndarray):imageforconversiontotensor
Returns:
input_tensor(torch.Tensor):floattensorreadytouseforYOLOv7inference
"""
input_tensor=image.astype(np.float32)#uint8tofp16/32
input_tensor/=255.0#0-255to0.0-1.0

ifinput_tensor.ndim==3:
input_tensor=np.expand_dims(input_tensor,0)
returninput_tensor


#labelnamesforvisualization
DEFAULT_NAMES=[
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"trafficlight",
"firehydrant",
"stopsign",
"parkingmeter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sportsball",
"kite",
"baseballbat",
"baseballglove",
"skateboard",
"surfboard",
"tennisracket",
"bottle",
"wineglass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hotdog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cellphone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddybear",
"hairdrier",
"toothbrush",
]

#obtainclassnamesfrommodelcheckpoint
state_dict=torch.load("model/yolov7-tiny.pt",map_location="cpu")
ifhasattr(state_dict["model"],"module"):
NAMES=getattr(state_dict["model"].module,"names",DEFAULT_NAMES)
else:
NAMES=getattr(state_dict["model"],"names",DEFAULT_NAMES)

delstate_dict

#colorsforvisualization
COLORS={name:[np.random.randint(0,255)for_inrange(3)]fori,nameinenumerate(NAMES)}

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

fromtypingimportList,Tuple,Dict
fromutils.generalimportscale_coords,non_max_suppression


defdetect(
model:ov.Model,
image_path:Path,
conf_thres:float=0.25,
iou_thres:float=0.45,
classes:List[int]=None,
agnostic_nms:bool=False,
):
"""
OpenVINOYOLOv7modelinferencefunction.Readsimage,preprocessit,runsmodelinferenceandpostprocessresultsusingNMS.
Parameters:
model(Model):OpenVINOcompiledmodel.
image_path(Path):inputimagepath.
conf_thres(float,*optional*,0.25):minimalaccpetedconfidenceforobjectfiltering
iou_thres(float,*optional*,0.45):minimaloverlapscoreforremlovingobjectsduplicatesinNMS
classes(List[int],*optional*,None):labelsforpredictionfiltering,ifnotprovidedallpredictedlabelswillbeused
agnostic_nms(bool,*optiona*,False):applyclassagnostincNMSapproachornot
Returns:
pred(List):listofdetectionswith(n,6)shape,wheren-numberofdetectedboxesinformat[x1,y1,x2,y2,score,label]
orig_img(np.ndarray):imagebeforepreprocessing,canbeusedforresultsvisualization
inpjut_shape(Tuple[int]):shapeofmodelinputtensor,canbeusedforoutputrescaling
"""
output_blob=model.output(0)
img=np.array(Image.open(image_path))
preprocessed_img,orig_img=preprocess_image(img)
input_tensor=prepare_input_tensor(preprocessed_img)
predictions=torch.from_numpy(model(input_tensor)[output_blob])
pred=non_max_suppression(predictions,conf_thres,iou_thres,classes=classes,agnostic=agnostic_nms)
returnpred,orig_img,input_tensor.shape


defdraw_boxes(
predictions:np.ndarray,
input_shape:Tuple[int],
image:np.ndarray,
names:List[str],
colors:Dict[str,int],
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
#Rescaleboxesfrominputsizetooriginalimagesize
predictions[:,:4]=scale_coords(input_shape[2:],predictions[:,:4],image.shape).round()

#Writeresults
for*xyxy,conf,clsinreversed(predictions):
label=f"{names[int(cls)]}{conf:.2f}"
plot_one_box(xyxy,image,label=label,color=colors[names[int(cls)]],line_thickness=1)
returnimage

..code::ipython3

core=ov.Core()
#readconvertedmodel
model=core.read_model("model/yolov7-tiny.xml")

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

#loadmodelonCPUdevice
compiled_model=core.compile_model(model,device.value)

..code::ipython3

boxes,image,input_shape=detect(compiled_model,"inference/images/horses.jpg")
image_with_boxes=draw_boxes(boxes[0],input_shape,image,NAMES,COLORS)
#visualizeresults
Image.fromarray(image_with_boxes)




..image::yolov7-optimization-with-output_files/yolov7-optimization-with-output_27_0.png



Verifymodelaccuracy
---------------------

`backtotop⬆️<#table-of-contents>`__

Downloaddataset
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv7tinyispre-trainedontheCOCOdataset,soinordertoevaluate
themodelaccuracy,weneedtodownloadit.Accordingtothe
instructionsprovidedintheYOLOv7repo,wealsoneedtodownload
annotationsintheformatusedbytheauthorofthemodel,forusewith
theoriginalmodelevaluationscripts.

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


Createdataloader
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromcollectionsimportnamedtuple
importyaml
fromutils.datasetsimportcreate_dataloader
fromutils.generalimportcheck_dataset,box_iou,xywh2xyxy,colorstr

#readdatasetconfig
DATA_CONFIG="data/coco.yaml"
withopen(DATA_CONFIG)asf:
data=yaml.load(f,Loader=yaml.SafeLoader)

#Dataloader
TASK="val"#pathtotrain/val/testimages
Option=namedtuple("Options",["single_cls"])#imitationofcommandlineprovidedoptionsforsingleclassevaluation
opt=Option(False)
dataloader=create_dataloader(data[TASK],640,1,32,opt,pad=0.5,prefix=colorstr(f"{TASK}:"))[0]


..parsed-literal::

val:Scanning'coco/val2017'imagesandlabels...4952found,48missing,0empty,0corrupted:100%|██████████|5000/5000[00:02<00:00,2410.16it/s]


Definevalidationfunction
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WewillreusevalidationmetricsprovidedintheYOLOv7repowitha
modificationforthiscase(removingextrasteps).Theoriginalmodel
evaluationprocedurecanbefoundinthis
`file<https://github.com/WongKinYiu/yolov7/blob/main/test.py>`__

..code::ipython3

importnumpyasnp
fromtqdm.notebookimporttqdm
fromutils.metricsimportap_per_class


deftest(
data,
model:ov.Model,
dataloader:torch.utils.data.DataLoader,
conf_thres:float=0.001,
iou_thres:float=0.65,#forNMS
single_cls:bool=False,
v5_metric:bool=False,
names:List[str]=None,
num_samples:int=None,
):
"""
YOLOv7accuracyevaluation.Processesvalidationdatasetandcompitesmetrics.

Parameters:
model(ov.Model):OpenVINOcompiledmodel.
dataloader(torch.utils.DataLoader):validationdataset.
conf_thres(float,*optional*,0.001):minimalconfidencethresholdforkeepingdetections
iou_thres(float,*optional*,0.65):IOUthresholdforNMS
single_cls(bool,*optional*,False):classagnosticevaluation
v5_metric(bool,*optional*,False):useYOLOv5evaluationapproachformetricscalculation
names(List[str],*optional*,None):namesforeachclassindataset
num_samples(int,*optional*,None):numbersamplesfortesting
Returns:
mp(float):meanprecision
mr(float):meanrecall
map50(float):meanaverageprecisionat0.5IOUthreshold
map(float):meanaverageprecisionat0.5:0.95IOUthresholds
maps(Dict(int,float):averageprecisionperclass
seen(int):numberofevaluatedimages
labels(int):numberoflabels
"""

model_output=model.output(0)
check_dataset(data)#check
nc=1ifsingle_clselseint(data["nc"])#numberofclasses
iouv=torch.linspace(0.5,0.95,10)#iouvectorformAP@0.5:0.95
niou=iouv.numel()

ifv5_metric:
print("TestingwithYOLOv5APmetric...")

seen=0
p,r,mp,mr,map50,map=0.0,0.0,0.0,0.0,0.0,0.0
stats,ap,ap_class=[],[],[]
forsample_id,(img,targets,_,shapes)inenumerate(tqdm(dataloader)):
ifnum_samplesisnotNoneandsample_id==num_samples:
break
img=prepare_input_tensor(img.numpy())
targets=targets
height,width=img.shape[2:]

withtorch.no_grad():
#Runmodel
out=torch.from_numpy(model(ov.Tensor(img))[model_output])#inferenceoutput
#RunNMS
targets[:,2:]*=torch.Tensor([width,height,width,height])#topixels

out=non_max_suppression(
out,
conf_thres=conf_thres,
iou_thres=iou_thres,
labels=None,
multi_label=True,
)
#Statisticsperimage
forsi,predinenumerate(out):
labels=targets[targets[:,0]==si,1:]
nl=len(labels)
tcls=labels[:,0].tolist()ifnlelse[]#targetclass
seen+=1

iflen(pred)==0:
ifnl:
stats.append(
(
torch.zeros(0,niou,dtype=torch.bool),
torch.Tensor(),
torch.Tensor(),
tcls,
)
)
continue
#Predictions
predn=pred.clone()
scale_coords(img[si].shape[1:],predn[:,:4],shapes[si][0],shapes[si][1])#native-spacepred
#Assignallpredictionsasincorrect
correct=torch.zeros(pred.shape[0],niou,dtype=torch.bool,device="cpu")
ifnl:
detected=[]#targetindices
tcls_tensor=labels[:,0]
#targetboxes
tbox=xywh2xyxy(labels[:,1:5])
scale_coords(img[si].shape[1:],tbox,shapes[si][0],shapes[si][1])#native-spacelabels
#Pertargetclass
forclsintorch.unique(tcls_tensor):
ti=(cls==tcls_tensor).nonzero(as_tuple=False).view(-1)#predictionindices
pi=(cls==pred[:,5]).nonzero(as_tuple=False).view(-1)#targetindices
#Searchfordetections
ifpi.shape[0]:
#Predictiontotargetious
ious,i=box_iou(predn[pi,:4],tbox[ti]).max(1)#bestious,indices
#Appenddetections
detected_set=set()
forjin(ious>iouv[0]).nonzero(as_tuple=False):
d=ti[i[j]]#detectedtarget
ifd.item()notindetected_set:
detected_set.add(d.item())
detected.append(d)
correct[pi[j]]=ious[j]>iouv#iou_thresis1xn
iflen(detected)==nl:#alltargetsalreadylocatedinimage
break
#Appendstatistics(correct,conf,pcls,tcls)
stats.append((correct.cpu(),pred[:,4].cpu(),pred[:,5].cpu(),tcls))
#Computestatistics
stats=[np.concatenate(x,0)forxinzip(*stats)]#tonumpy
iflen(stats)andstats[0].any():
p,r,ap,f1,ap_class=ap_per_class(*stats,plot=True,v5_metric=v5_metric,names=names)
ap50,ap=ap[:,0],ap.mean(1)#AP@0.5,AP@0.5:0.95
mp,mr,map50,map=p.mean(),r.mean(),ap50.mean(),ap.mean()
nt=np.bincount(stats[3].astype(np.int64),minlength=nc)#numberoftargetsperclass
else:
nt=torch.zeros(1)
maps=np.zeros(nc)+map
fori,cinenumerate(ap_class):
maps[c]=ap[i]
returnmp,mr,map50,map,maps,seen,nt.sum()

Validationfunctionreportsfollowinglistofaccuracymetrics:

-``Precision``isthedegreeofexactnessofthemodelinidentifying
onlyrelevantobjects.
-``Recall``measurestheabilityofthemodeltodetectallground
truthsobjects.
-``mAP@t``-meanaverageprecision,representedasareaunderthe
Precision-Recallcurveaggregatedoverallclassesinthedataset,
where``t``isIntersectionOverUnion(IOU)threshold,degreeof
overlappingbetweengroundtruthandpredictedobjects.Therefore,
``mAP@.5``indicatesthatmeanaverageprecisioncalculatedat0.5
IOUthreshold,``mAP@.5:.95``-calculatedonrangeIOUthresholds
from0.5to0.95withstep0.05.

..code::ipython3

mp,mr,map50,map,maps,num_images,labels=test(data=data,model=compiled_model,dataloader=dataloader,names=NAMES)
#Printresults
s=("%20s"+"%12s"*6)%(
"Class",
"Images",
"Labels",
"Precision",
"Recall",
"mAP@.5",
"mAP@.5:.95",
)
print(s)
pf="%20s"+"%12i"*2+"%12.3g"*4#printformat
print(pf%("all",num_images,labels,mp,mr,map50,map))



..parsed-literal::

0%||0/5000[00:00<?,?it/s]


..parsed-literal::

ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all5000363350.6510.5070.5440.359


OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
YOLOv7.

**Note**:NNCFPost-trainingQuantizationisavailableasapreview
featureinOpenVINO2022.3release.Fullyfunctionalsupportwillbe
providedinthenextreleases.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeanOpenVINOIRmodel,usingthe
``openvino.runtime.serialize``function.

Reusevalidationdataloaderinaccuracytestingforquantization.For
that,itshouldbewrappedintothe``nncf.Dataset``objectanddefine
transformationfunctionforgettingonlyinputtensors.

..code::ipython3

importnncf#noqa:F811


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


The``nncf.quantize``functionprovidesinterfaceformodel
quantization.ItrequiresinstanceofOpenVINOModelandquantization
dataset.Optionally,someadditionalparametersforconfiguration
quantizationprocess(numberofsamplesforquantization,preset,
ignoredscopeetc.)canbeprovided.YOLOv7modelcontainsnon-ReLU
activationfunctions,whichrequireasymmetricquantizationof
activations.Toachievebetterresult,wewilluse``mixed``
quantizationpreset.Itprovidessymmetricquantizationofweightsand
asymmetricquantizationofactivations.

..code::ipython3

quantized_model=nncf.quantize(model,quantization_dataset,preset=nncf.QuantizationPreset.MIXED)

ov.save_model(quantized_model,"model/yolov7-tiny_int8.xml")


..parsed-literal::

2024-07-1304:17:49.896323:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:17:49.928585:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:17:50.540879:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



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

WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.
WARNING:openvino.runtime.opset13.ops:Convertingvalueoffloat32tofloat16.Memorysharingisdisabledbydefault.Setshared_memory=Falsetohidethiswarning.


ValidateQuantizedmodelinference
----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

int8_compiled_model=core.compile_model(quantized_model,device.value)
boxes,image,input_shape=detect(int8_compiled_model,"inference/images/horses.jpg")
image_with_boxes=draw_boxes(boxes[0],input_shape,image,NAMES,COLORS)
Image.fromarray(image_with_boxes)




..image::yolov7-optimization-with-output_files/yolov7-optimization-with-output_44_0.png



Validatequantizedmodelaccuracy
---------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

int8_result=test(data=data,model=int8_compiled_model,dataloader=dataloader,names=NAMES)



..parsed-literal::

0%||0/5000[00:00<?,?it/s]


..code::ipython3

mp,mr,map50,map,maps,num_images,labels=int8_result
#Printresults
s=("%20s"+"%12s"*6)%(
"Class",
"Images",
"Labels",
"Precision",
"Recall",
"mAP@.5",
"mAP@.5:.95",
)
print(s)
pf="%20s"+"%12i"*2+"%12.3g"*4#printformat
print(pf%("all",num_images,labels,mp,mr,map50,map))


..parsed-literal::

ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all5000363350.6430.5060.540.353


Aswecansee,modelaccuracyslightlychangedafterquantization.
However,ifwelookattheoutputimage,thesechangesarenot
significant.

ComparePerformanceoftheOriginalandQuantizedModels
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Finally,usetheOpenVINO`Benchmark
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

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#InferenceFP32model(OpenVINOIR)
!benchmark_app-mmodel/yolov7-tiny.xml-d$device.value-apiasync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
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
[INFO]Readmodeltook13.26ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]images(node:images):f32/[...]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output(node:output):f32/[...]/[1,25200,85]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]images(node:images):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output(node:output):f32/[...]/[1,25200,85]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook254.06ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:main_graph
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
[INFO]NETWORK_NAME:main_graph
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
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook46.77ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:11694iterations
[INFO]Duration:120060.72ms
[INFO]Latency:
[INFO]Median:61.36ms
[INFO]Average:61.45ms
[INFO]Min:34.72ms
[INFO]Max:81.34ms
[INFO]Throughput:97.40FPS


..code::ipython3

#InferenceINT8model(OpenVINOIR)
!benchmark_app-mmodel/yolov7-tiny_int8.xml-d$device.value-apiasync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
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
[INFO]Readmodeltook19.00ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]images(node:images):f32/[...]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output(node:output):f32/[...]/[1,25200,85]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]images(node:images):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]output(node:output):f32/[...]/[1,25200,85]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook402.55ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:main_graph
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
[INFO]NETWORK_NAME:main_graph
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
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook23.71ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:34356iterations
[INFO]Duration:120021.29ms
[INFO]Latency:
[INFO]Median:20.77ms
[INFO]Average:20.84ms
[INFO]Min:14.81ms
[INFO]Max:41.35ms
[INFO]Throughput:286.25FPS


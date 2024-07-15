ConvertandOptimizeYOLOv8keypointdetectionmodelwithOpenVINO™
===================================================================

Keypointdetection/Poseisataskthatinvolvesdetectingspecific
pointsinanimageorvideoframe.Thesepointsarereferredtoas
keypointsandareusedtotrackmovementorposeestimation.YOLOv8can
detectkeypointsinanimageorvideoframewithhighaccuracyand
speed.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimize`PyTorchYOLOv8Pose
model<https://docs.ultralytics.com/tasks/pose/>`__withOpenVINO.We
considerthestepsrequiredforkeypointdetectionscenario.

Thetutorialconsistsofthefollowingsteps:-PreparethePyTorch
model.-Downloadandprepareadataset.-Validatetheoriginalmodel.
-ConvertthePyTorchmodeltoOpenVINOIR.-Validatetheconverted
model.-Prepareandrunoptimizationpipeline.-Compareperformanceof
theFP32andquantizedmodels.-CompareaccuracyoftheFP32and
quantizedmodels.-Livedemo

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`GetPyTorchmodel<#get-pytorch-model>`__

-`Prerequisites<#prerequisites>`__

-`Instantiatemodel<#instantiate-model>`__

-`ConvertmodeltoOpenVINOIR<#convert-model-to-openvino-ir>`__
-`Verifymodelinference<#verify-model-inference>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Testonsingleimage<#test-on-single-image>`__

-`Checkmodelaccuracyonthe
dataset<#check-model-accuracy-on-the-dataset>`__

-`Downloadthevalidation
dataset<#download-the-validation-dataset>`__
-`Definevalidationfunction<#define-validation-function>`__
-`ConfigureValidatorhelperandcreate
DataLoader<#configure-validator-helper-and-create-dataloader>`__

-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__

-`ValidateQuantizedmodel
inference<#validate-quantized-model-inference>`__

-`ComparetheOriginalandQuantized
Models<#compare-the-original-and-quantized-models>`__

-`CompareperformanceoftheOriginalandQuantized
Models<#compare-performance-of-the-original-and-quantized-models>`__
-`CompareaccuracyoftheOriginalandQuantized
Models<#compare-accuracy-of-the-original-and-quantized-models>`__

-`Otherwaystooptimizemodel<#other-ways-to-optimize-model>`__
-`Livedemo<#live-demo>`__

-`RunKeypointDetectionon
video<#run-keypoint-detection-on-video>`__

GetPyTorchmodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
`torch.nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class,initializedbyastatedictionarywithmodelweights.Wewilluse
theYOLOv8nanomodel(alsoknownas``yolov8n``)pre-trainedonaCOCO
dataset,whichisavailableinthis
`repo<https://github.com/ultralytics/ultralytics>`__.Similarstepsare
alsoapplicabletootherYOLOv8models.Typicalstepstoobtaina
pre-trainedmodel:1.Createaninstanceofamodelclass.2.Loada
checkpointstatedict,whichcontainsthepre-trainedmodelweights.3.
Turnthemodeltoevaluationforswitchingsomeoperationstoinference
mode.

Inthiscase,thecreatorsofthemodelprovideanAPIthatenables
convertingtheYOLOv8modeltoOpenVINOIR.Therefore,wedonotneedto
dothesestepsmanually.

Prerequisites
^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Installnecessarypackages.

..code::ipython3

%pipinstall-q"openvino>=2024.0.0""nncf>=2.9.0"
%pipinstall-q"protobuf==3.20.*""torch>=2.1""torchvision>=0.16""ultralytics==8.2.24""onnx"tqdmopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu

Importrequiredutilityfunctions.Thelowercellwilldownloadthe
``notebook_utils``PythonmodulefromGitHub.

..code::ipython3

frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file,VideoPlayer

..code::ipython3

#Downloadatestsample
IMAGE_PATH=Path("./data/intel_rnb.jpg")
download_file(
url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg",
filename=IMAGE_PATH.name,
directory=IMAGE_PATH.parent,
)



..parsed-literal::

data/intel_rnb.jpg:0%||0.00/288k[00:00<?,?B/s]




..parsed-literal::

PosixPath('/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/intel_rnb.jpg')



Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Forloadingthemodel,requiredtospecifyapathtothemodel
checkpoint.Itcanbesomelocalpathornameavailableonmodelshub
(inthiscasemodelcheckpointwillbedownloadedautomatically).

Makingprediction,themodelacceptsapathtoinputimageandreturns
listwithResultsclassobject.Resultscontainsboxesandkeypoints.
Alsoitcontainsutilitiesforprocessingresults,forexample,
``plot()``methodfordrawing.

Letusconsidertheexamples:

..code::ipython3

models_dir=Path("./models")
models_dir.mkdir(exist_ok=True)

..code::ipython3

fromPILimportImage
fromultralyticsimportYOLO

POSE_MODEL_NAME="yolov8n-pose"

pose_model=YOLO(models_dir/f"{POSE_MODEL_NAME}.pt")
label_map=pose_model.model.names

res=pose_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::

Downloadinghttps://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.ptto'models/yolov8n-pose.pt'...


..parsed-literal::

100%|██████████████████████████████████████████████████████████████████████████████|6.51M/6.51M[00:01<00:00,3.93MB/s]


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/intel_rnb.jpg:480x6401person,45.3ms
Speed:1.5mspreprocess,45.3msinference,1.0mspostprocessperimageatshape(1,3,480,640)




..image::yolov8-keypoint-detection-with-output_files/yolov8-keypoint-detection-with-output_9_3.png



ConvertmodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8providesAPIforconvenientmodelexportingtodifferentformats
includingOpenVINOIR.``model.export``isresponsibleformodel
conversion.Weneedtospecifytheformat,andadditionally,wecan
preservedynamicshapesinthemodel.

..code::ipython3

#objectdetectionmodel
pose_model_path=models_dir/f"{POSE_MODEL_NAME}_openvino_model/{POSE_MODEL_NAME}.xml"
ifnotpose_model_path.exists():
pose_model.export(format="openvino",dynamic=True,half=True)


..parsed-literal::

UltralyticsYOLOv8.1.42🚀Python-3.10.12torch-2.2.2+cpuCPU(IntelCore(TM)i9-10980XE3.00GHz)

PyTorch:startingfrom'models/yolov8n-pose.pt'withinputshape(1,3,640,640)BCHWandoutputshape(s)(1,56,8400)(6.5MB)

OpenVINO:startingexportwithopenvino2024.0.0-14509-34caeefd078-releases/2024/0...
OpenVINO:exportsuccess✅1.7s,savedas'models/yolov8n-pose_openvino_model/'(6.7MB)

Exportcomplete(3.0s)
Resultssavedto/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/models
Predict:yolopredicttask=posemodel=models/yolov8n-pose_openvino_modelimgsz=640half
Validate:yolovaltask=posemodel=models/yolov8n-pose_openvino_modelimgsz=640data=/usr/src/app/ultralytics/datasets/coco-pose.yamlhalf
Visualize:https://netron.app


Verifymodelinference
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Wecanreusethebasemodelpipelineforpre-andpostprocessingjust
replacingtheinferencemethodwherewewillusetheIRmodelfor
inference.

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

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



Testonsingleimage
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,oncewehavedefinedpreprocessingandpostprocessingsteps,weare
readytocheckmodelprediction.

..code::ipython3

importtorch

core=ov.Core()
pose_ov_model=core.read_model(pose_model_path)

ov_config={}
ifdevice.value!="CPU":
pose_ov_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
pose_compiled_model=core.compile_model(pose_ov_model,device.value,ov_config)


definfer(*args):
result=pose_compiled_model(args)
returntorch.from_numpy(result[0])


pose_model.predictor.inference=infer
pose_model.predictor.model.pt=False

res=pose_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/intel_rnb.jpg:640x6402persons,23.7ms
Speed:5.0mspreprocess,23.7msinference,2.2mspostprocessperimageatshape(1,3,640,640)




..image::yolov8-keypoint-detection-with-output_files/yolov8-keypoint-detection-with-output_16_1.png



Great!Theresultisthesame,asproducedbyoriginalmodels.

Checkmodelaccuracyonthedataset
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

Forcomparingtheoptimizedmodelresultwiththeoriginal,itisgood
toknowsomemeasurableresultsintermsofmodelaccuracyonthe
validationdataset.

Downloadthevalidationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8ispre-trainedontheCOCOdataset,sotoevaluatethemodel
accuracyweneedtodownloadit.Accordingtotheinstructionsprovided
intheYOLOv8repo,wealsoneedtodownloadannotationsintheformat
usedbytheauthorofthemodel,forusewiththeoriginalmodel
evaluationfunction.

**Note**:Theinitialdatasetdownloadmaytakeafewminutesto
complete.Thedownloadspeedwillvarydependingonthequalityof
yourinternetconnection.

..code::ipython3

fromzipfileimportZipFile

fromultralytics.data.utilsimportDATASETS_DIR


DATA_URL="https://ultralytics.com/assets/coco8-pose.zip"
CFG_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco8-pose.yaml"

OUT_DIR=DATASETS_DIR

DATA_PATH=OUT_DIR/"val2017.zip"
CFG_PATH=OUT_DIR/"coco8-pose.yaml"

download_file(DATA_URL,DATA_PATH.name,DATA_PATH.parent)
download_file(CFG_URL,CFG_PATH.name,CFG_PATH.parent)

ifnot(OUT_DIR/"coco8-pose/labels").exists():
withZipFile(DATA_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR)



..parsed-literal::

/home/maleksandr/test_notebooks/ultrali/datasets/val2017.zip:0%||0.00/334k[00:00<?,?B/s]



..parsed-literal::

/home/maleksandr/test_notebooks/ultrali/datasets/coco8-pose.yaml:0%||0.00/552[00:00<?,?B/s]


Definevalidationfunction
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importnumpyasnp
fromtqdm.notebookimporttqdm
fromultralytics.utils.metricsimportConfusionMatrix


deftest(
model:ov.Model,
core:ov.Core,
data_loader:torch.utils.data.DataLoader,
validator,
num_samples:int=None,
):
"""
OpenVINOYOLOv8modelaccuracyvalidationfunction.Runsmodelvalidationondatasetandreturnsmetrics
Parameters:
model(Model):OpenVINOmodel
data_loader(torch.utils.data.DataLoader):datasetloader
validator:instanceofvalidatorclass
num_samples(int,*optional*,None):validatemodelonlyonspecifiednumbersamples,ifprovided
Returns:
stats:(Dict[str,float])-dictionarywithaggregatedaccuracymetricsstatistics,keyismetricname,valueismetricvalue
"""
validator.seen=0
validator.jdict=[]
validator.stats=dict(tp_p=[],tp=[],conf=[],pred_cls=[],target_cls=[])
validator.batch_i=1
validator.confusion_matrix=ConfusionMatrix(nc=validator.nc)
model.reshape({0:[1,3,-1,-1]})
compiled_model=core.compile_model(model)
forbatch_i,batchinenumerate(tqdm(data_loader,total=num_samples)):
ifnum_samplesisnotNoneandbatch_i==num_samples:
break
batch=validator.preprocess(batch)
results=compiled_model(batch["img"])
preds=torch.from_numpy(results[compiled_model.output(0)])
preds=validator.postprocess(preds)
validator.update_metrics(preds,batch)
stats=validator.get_stats()
returnstats


defprint_stats(stats:np.ndarray,total_images:int,total_objects:int):
"""
Helperfunctionforprintingaccuracystatistic
Parameters:
stats:(Dict[str,float])-dictionarywithaggregatedaccuracymetricsstatistics,keyismetricname,valueismetricvalue
total_images(int)-numberofevaluatedimages
totalobjects(int)
Returns:
None
"""
print("Boxes:")
mp,mr,map50,mean_ap=(
stats["metrics/precision(B)"],
stats["metrics/recall(B)"],
stats["metrics/mAP50(B)"],
stats["metrics/mAP50-95(B)"],
)
#Printresults
print("Bestmeanaverage:")
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
print(pf%("all",total_images,total_objects,mp,mr,map50,mean_ap))
if"metrics/precision(M)"instats:
s_mp,s_mr,s_map50,s_mean_ap=(
stats["metrics/precision(M)"],
stats["metrics/recall(M)"],
stats["metrics/mAP50(M)"],
stats["metrics/mAP50-95(M)"],
)
#Printresults
print("Macroaveragemean:")
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
print(pf%("all",total_images,total_objects,s_mp,s_mr,s_map50,s_mean_ap))

ConfigureValidatorhelperandcreateDataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Theoriginalmodelrepositoryusesa``Validator``wrapper,which
representstheaccuracyvalidationpipeline.Itcreatesdataloaderand
evaluationmetricsandupdatesmetricsoneachdatabatchproducedby
thedataloader.Besidesthat,itisresponsiblefordatapreprocessing
andresultspostprocessing.Forclassinitialization,theconfiguration
shouldbeprovided.Wewillusethedefaultsetup,butitcanbe
replacedwithsomeparametersoverridingtotestoncustomdata.The
modelhasconnectedthe``ValidatorClass``method,whichcreatesa
validatorclassinstance.

..code::ipython3

fromultralytics.utilsimportDEFAULT_CFG
fromultralytics.cfgimportget_cfg
fromultralytics.data.utilsimportcheck_det_dataset

args=get_cfg(cfg=DEFAULT_CFG)
args.data="coco8-pose.yaml"
args.model="yolov8n-pose.pt"

..code::ipython3

fromultralytics.models.yolo.poseimportPoseValidator

pose_validator=PoseValidator(args=args)

..code::ipython3

pose_validator.data=check_det_dataset(args.data)
pose_validator.stride=32
pose_data_loader=pose_validator.get_dataloader(OUT_DIR/"coco8-pose",1)


..parsed-literal::

val:Scanning/home/maleksandr/test_notebooks/ultrali/datasets/coco8-pose/labels/train.cache...8images,0backgrounds,


..code::ipython3

fromultralytics.utils.metricsimportOKS_SIGMA

pose_validator.is_coco=True
pose_validator.names=pose_model.model.names
pose_validator.metrics.names=pose_validator.names
pose_validator.nc=pose_model.model.model[-1].nc
pose_validator.sigma=OKS_SIGMA

Afterdefinitiontestfunctionandvalidatorcreation,wearereadyfor
gettingaccuracymetrics>\**Note**:Modelevaluationistimeconsuming
processandcantakeseveralminutes,dependingonthehardware.For
reducingcalculationtime,wedefine``num_samples``parameterwith
evaluationsubsetsize,butinthiscase,accuracycanbenoncomparable
withoriginallyreportedbytheauthorsofthemodel,duetovalidation
subsetdifference.*Tovalidatethemodelsonthefulldatasetset
``NUM_TEST_SAMPLES=None``.*

..code::ipython3

NUM_TEST_SAMPLES=300

..code::ipython3

fp_pose_stats=test(pose_ov_model,core,pose_data_loader,pose_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

print_stats(fp_pose_stats,pose_validator.seen,pose_validator.nt_per_class.sum())


..parsed-literal::

Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all82110.8990.9550.736


``print_stats``reportsthefollowinglistofaccuracymetrics:

-``Precision``isthedegreeofexactnessofthemodelinidentifying
onlyrelevantobjects.
-``Recall``measurestheabilityofthemodeltodetectallground
truthsobjects.
-``mAP@t``-meanaverageprecision,representedasareaunderthe
Precision-Recallcurveaggregatedoverallclassesinthedataset,
where``t``istheIntersectionOverUnion(IOU)threshold,degreeof
overlappingbetweengroundtruthandpredictedobjects.Therefore,
``mAP@.5``indicatesthatmeanaverageprecisioniscalculatedat0.5
IOUthreshold,``mAP@.5:.95``-iscalculatedonrangeIOUthresholds
from0.5to0.95withstep0.05.

OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
YOLOv8.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeOpenVINOIRmodel,usingthe``openvino.runtime.serialize``
function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

importipywidgetsaswidgets

int8_model_pose_path=models_dir/f"{POSE_MODEL_NAME}_openvino_int8_model/{POSE_MODEL_NAME}.xml"

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetchskip_kernel_extensionmodule
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Reusevalidationdataloaderinaccuracytestingforquantization.For
that,itshouldbewrappedintothe``nncf.Dataset``objectanddefinea
transformationfunctionforgettingonlyinputtensors.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
fromtypingimportDict


deftransform_fn(data_item:Dict):
"""
Quantizationtransformfunction.Extractsandpreprocessinputdatafromdataloaderitemforquantization.
Parameters:
data_item:DictwithdataitemproducedbyDataLoaderduringiteration
Returns:
input_tensor:Inputdataforquantization
"""
input_tensor=pose_validator.preprocess(data_item)['img'].numpy()
returninput_tensor


quantization_dataset=nncf.Dataset(pose_data_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscope,etc.)canbeprovided.YOLOv8modelcontains
non-ReLUactivationfunctions,whichrequireasymmetricquantizationof
activations.Toachieveabetterresult,wewillusea``mixed``
quantizationpreset.Itprovidessymmetricquantizationofweightsand
asymmetricquantizationofactivations.Formoreaccurateresults,we
shouldkeeptheoperationinthepostprocessingsubgraphinfloating
pointprecision,usingthe``ignored_scope``parameter.

**Note**:Modelpost-trainingquantizationistime-consumingprocess.
Bepatient,itcantakeseveralminutesdependingonyourhardware.

..code::ipython3

%%skipnot$to_quantize.value

ignored_scope=nncf.IgnoredScope(
names=[
"__module.model.22.cv3.0.0.conv/aten::_convolution/Convolution",#inthepost-processingsubgraph
	"__module.model.22.cv4.0.0.conv/aten::_convolution/Convolution",
	"__module.model.16.conv/aten::_convolution/Convolution",
	"__module.model.22.cv2.0.0.conv/aten::_convolution/Convolution",
	"__module.model.6.cv1.conv/aten::_convolution/Convolution",
	"__module.model.22.cv3.1.1.conv/aten::_convolution/Convolution",
	"__module.model.21.cv2.conv/aten::_convolution/Convolution",
	"__module.model.21.m.0.cv1.conv/aten::_convolution/Convolution",
	"__module.model.22/aten::add/Add_6",
	"__module.model.22/aten::sub/Subtract",
	"__module.model.7.conv/aten::_convolution/Convolution",
	"__module.model.12.cv1.conv/aten::_convolution/Convolution",
	"__module.model.4.cv1.conv/aten::_convolution/Convolution",
	"__module.model.22.cv2.2.1.conv/aten::_convolution/Convolution",
	"__module.model.22.cv2.0.1.conv/aten::_convolution/Convolution",
	"__module.model.22.cv4.2.1.conv/aten::_convolution/Convolution",
	"__module.model.22.dfl.conv/aten::_convolution/Convolution",
	"__module.model.22.cv3.2.2/aten::_convolution/Convolution",
	"__module.model.22.cv3.0.2/aten::_convolution/Convolution",
	"__module.model.15.cv1.conv/aten::_convolution/Convolution",
	"__module.model.5.conv/aten::_convolution/Convolution",
	"__module.model.0.conv/aten::_convolution/Convolution"
]
)


#Detectionmodel
quantized_pose_model=nncf.quantize(
pose_ov_model,
quantization_dataset,
preset=nncf.QuantizationPreset.MIXED,
ignored_scope=ignored_scope
)


..parsed-literal::

INFO:nncf:22ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:Notaddingactivationinputquantizerforoperation:1__module.model.0.conv/aten::_convolution/Convolution
2__module.model.0.conv/aten::_convolution/Add
3__module.model.22.cv4.2.1.act/aten::silu_/Swish

INFO:nncf:Notaddingactivationinputquantizerforoperation:25__module.model.4.cv1.conv/aten::_convolution/Convolution
26__module.model.4.cv1.conv/aten::_convolution/Add
27__module.model.22.cv4.2.1.act/aten::silu_/Swish_7

INFO:nncf:Notaddingactivationinputquantizerforoperation:43__module.model.5.conv/aten::_convolution/Convolution
47__module.model.5.conv/aten::_convolution/Add
51__module.model.22.cv4.2.1.act/aten::silu_/Swish_13

INFO:nncf:Notaddingactivationinputquantizerforoperation:54__module.model.6.cv1.conv/aten::_convolution/Convolution
56__module.model.6.cv1.conv/aten::_convolution/Add
59__module.model.22.cv4.2.1.act/aten::silu_/Swish_14

INFO:nncf:Notaddingactivationinputquantizerforoperation:99__module.model.7.conv/aten::_convolution/Convolution
112__module.model.7.conv/aten::_convolution/Add
123__module.model.22.cv4.2.1.act/aten::silu_/Swish_20

INFO:nncf:Notaddingactivationinputquantizerforoperation:111__module.model.12.cv1.conv/aten::_convolution/Convolution
122__module.model.12.cv1.conv/aten::_convolution/Add
131__module.model.22.cv4.2.1.act/aten::silu_/Swish_27

INFO:nncf:Notaddingactivationinputquantizerforoperation:46__module.model.15.cv1.conv/aten::_convolution/Convolution
50__module.model.15.cv1.conv/aten::_convolution/Add
53__module.model.22.cv4.2.1.act/aten::silu_/Swish_31

INFO:nncf:Notaddingactivationinputquantizerforoperation:74__module.model.16.conv/aten::_convolution/Convolution
83__module.model.16.conv/aten::_convolution/Add
92__module.model.22.cv4.2.1.act/aten::silu_/Swish_39

INFO:nncf:Notaddingactivationinputquantizerforoperation:75__module.model.22.cv2.0.0.conv/aten::_convolution/Convolution
84__module.model.22.cv2.0.0.conv/aten::_convolution/Add
93__module.model.22.cv4.2.1.act/aten::silu_/Swish_35

INFO:nncf:Notaddingactivationinputquantizerforoperation:103__module.model.22.cv2.0.1.conv/aten::_convolution/Convolution
116__module.model.22.cv2.0.1.conv/aten::_convolution/Add
126__module.model.22.cv4.2.1.act/aten::silu_/Swish_36

INFO:nncf:Notaddingactivationinputquantizerforoperation:76__module.model.22.cv3.0.0.conv/aten::_convolution/Convolution
85__module.model.22.cv3.0.0.conv/aten::_convolution/Add
94__module.model.22.cv4.2.1.act/aten::silu_/Swish_37

INFO:nncf:Notaddingactivationinputquantizerforoperation:135__module.model.22.cv3.0.2/aten::_convolution/Convolution
143__module.model.22.cv3.0.2/aten::_convolution/Add

INFO:nncf:Notaddingactivationinputquantizerforoperation:77__module.model.22.cv4.0.0.conv/aten::_convolution/Convolution
86__module.model.22.cv4.0.0.conv/aten::_convolution/Add
95__module.model.22.cv4.2.1.act/aten::silu_/Swish_57

INFO:nncf:Notaddingactivationinputquantizerforoperation:234__module.model.22.cv3.1.1.conv/aten::_convolution/Convolution
247__module.model.22.cv3.1.1.conv/aten::_convolution/Add
258__module.model.22.cv4.2.1.act/aten::silu_/Swish_47

INFO:nncf:Notaddingactivationinputquantizerforoperation:289__module.model.21.m.0.cv1.conv/aten::_convolution/Convolution
296__module.model.21.m.0.cv1.conv/aten::_convolution/Add
302__module.model.22.cv4.2.1.act/aten::silu_/Swish_50

INFO:nncf:Notaddingactivationinputquantizerforoperation:295__module.model.21.cv2.conv/aten::_convolution/Convolution
301__module.model.21.cv2.conv/aten::_convolution/Add
305__module.model.22.cv4.2.1.act/aten::silu_/Swish_52

INFO:nncf:Notaddingactivationinputquantizerforoperation:332__module.model.22.cv2.2.1.conv/aten::_convolution/Convolution
340__module.model.22.cv2.2.1.conv/aten::_convolution/Add
345__module.model.22.cv4.2.1.act/aten::silu_/Swish_54

INFO:nncf:Notaddingactivationinputquantizerforoperation:334__module.model.22.cv4.2.1.conv/aten::_convolution/Convolution
342__module.model.22.cv4.2.1.conv/aten::_convolution/Add
347__module.model.22.cv4.2.1.act/aten::silu_/Swish_62

INFO:nncf:Notaddingactivationinputquantizerforoperation:350__module.model.22.cv3.2.2/aten::_convolution/Convolution
354__module.model.22.cv3.2.2/aten::_convolution/Add

INFO:nncf:Notaddingactivationinputquantizerforoperation:243__module.model.22.dfl.conv/aten::_convolution/Convolution
INFO:nncf:Notaddingactivationinputquantizerforoperation:263__module.model.22/aten::sub/Subtract
INFO:nncf:Notaddingactivationinputquantizerforoperation:264__module.model.22/aten::add/Add_6



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/venv/lib/python3.10/site-packages/nncf/experimental/tensor/tensor.py:84:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

%%skipnot$to_quantize.value

print(f"Quantizedkeypointdetectionmodelwillbesavedto{int8_model_pose_path}")
ov.save_model(quantized_pose_model,str(int8_model_pose_path))


..parsed-literal::

Quantizedkeypointdetectionmodelwillbesavedtomodels/yolov8n-pose_openvino_int8_model/yolov8n-pose.xml


ValidateQuantizedmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

``nncf.quantize``returnstheOpenVINOModelclassinstance,whichis
suitableforloadingonadeviceformakingpredictions.``INT8``model
inputdataandoutputresultformatshavenodifferencefromthe
floatingpointmodelrepresentation.Therefore,wecanreusethesame
``detect``functiondefinedaboveforgettingthe``INT8``modelresult
ontheimage.

..code::ipython3

%%skipnot$to_quantize.value

device

..code::ipython3

%%skipnot$to_quantize.value

ov_config={}
ifdevice.value!="CPU":
quantized_pose_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
quantized_pose_compiled_model=core.compile_model(quantized_pose_model,device.value,ov_config)

definfer(*args):
result=quantized_pose_compiled_model(args)
returntorch.from_numpy(result[0])

pose_model.predictor.inference=infer

res=pose_model(IMAGE_PATH)
display(Image.fromarray(res[0].plot()[:,:,::-1]))


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/intel_rnb.jpg:640x6402persons,20.0ms
Speed:3.5mspreprocess,20.0msinference,1.1mspostprocessperimageatshape(1,3,640,640)



..image::yolov8-keypoint-detection-with-output_files/yolov8-keypoint-detection-with-output_44_1.png


ComparetheOriginalandQuantizedModels
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

CompareperformanceoftheOriginalandQuantizedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Finally,usetheOpenVINO
`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
tomeasuretheinferenceperformanceofthe``FP32``and``INT8``
models.

**Note**:Formoreaccurateperformance,itisrecommendedtorun
``benchmark_app``inaterminal/commandpromptafterclosingother
applications.Run
``benchmark_app-m<model_path>-dCPU-shape"<input_shape>"``to
benchmarkasyncinferenceonCPUonspecificinputdatashapeforone
minute.Change``CPU``to``GPU``tobenchmarkonGPU.Run
``benchmark_app--help``toseeanoverviewofallcommand-line
options.

..code::ipython3

%%skipnot$to_quantize.value

device

..code::ipython3

ifint8_model_pose_path.exists():
#InferenceFP32model(OpenVINOIR)
!benchmark_app-m$pose_model_path-d$device.value-apiasync-shape"[1,3,640,640]"


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
[INFO]OpenVINO:
[INFO]Build.................................2024.0.0-14509-34caeefd078-releases/2024/0
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.0.0-14509-34caeefd078-releases/2024/0
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook16.24ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[?,56,16..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook9.32ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[1,56,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook313.69ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:36
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook42.40ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:21612iterations
[INFO]Duration:120076.97ms
[INFO]Latency:
[INFO]Median:66.18ms
[INFO]Average:66.51ms
[INFO]Min:32.58ms
[INFO]Max:123.83ms
[INFO]Throughput:179.98FPS


..code::ipython3

ifint8_model_pose_path.exists():
#InferenceINT8model(OpenVINOIR)
!benchmark_app-m$int8_model_pose_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.0.0-14509-34caeefd078-releases/2024/0
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.0.0-14509-34caeefd078-releases/2024/0
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook23.83ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[1,56,21..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook13.45ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[1,56,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook540.15ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:36
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook34.70ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:4248iterations
[INFO]Duration:15064.81ms
[INFO]Latency:
[INFO]Median:42.11ms
[INFO]Average:42.37ms
[INFO]Min:24.13ms
[INFO]Max:69.95ms
[INFO]Throughput:281.98FPS


CompareaccuracyoftheOriginalandQuantizedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Aswecansee,thereisnosignificantdifferencebetween``INT8``and
floatmodelresultinasingleimagetest.Tounderstandhow
quantizationinfluencesmodelpredictionprecision,wecancomparemodel
accuracyonadataset.

..code::ipython3

%%skipnot$to_quantize.value

int8_pose_stats=test(quantized_pose_model,core,pose_data_loader,pose_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

%%skipnot$to_quantize.value

print("FP32modelaccuracy")
print_stats(fp_pose_stats,pose_validator.seen,pose_validator.nt_per_class.sum())

print("INT8modelaccuracy")
print_stats(int8_pose_stats,pose_validator.seen,pose_validator.nt_per_class.sum())


..parsed-literal::

FP32modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all82110.8990.9550.736
INT8modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all8210.9520.9380.9530.638


Great!Lookslikeaccuracywaschanged,butnotsignificantlyandit
meetspassingcriteria.

Otherwaystooptimizemodel
----------------------------

`backtotop⬆️<#table-of-contents>`__

TheperformancecouldbealsoimprovedbyanotherOpenVINOmethodsuch
asasyncinferencepipelineorpreprocessingAPI.

AsyncInferencepipelinehelptoutilizethedevicemoreoptimal.The
keyadvantageoftheAsyncAPIisthatwhenadeviceisbusywith
inference,theapplicationcanperformothertasksinparallel(for
example,populatinginputsorschedulingotherrequests)ratherthan
waitforthecurrentinferencetocompletefirst.Tounderstandhowto
performasyncinferenceusingopenvino,referto`AsyncAPI
tutorial<async-api-with-output.html>`__

PreprocessingAPIenablesmakingpreprocessingapartofthemodel
reducingapplicationcodeanddependencyonadditionalimageprocessing
libraries.ThemainadvantageofPreprocessingAPIisthatpreprocessing
stepswillbeintegratedintotheexecutiongraphandwillbeperformed
onaselecteddevice(CPU/GPUetc.)ratherthanalwaysbeingexecutedon
CPUaspartofanapplication.Thiswillalsoimproveselecteddevice
utilization.Formoreinformation,refertotheoverviewof
`PreprocessingAPI
tutorial<optimize-preprocessing-with-output.html>`__.To
see,howitcouldbeusedwithYOLOV8objectdetectionmodel,please,
see`ConvertandOptimizeYOLOv8real-timeobjectdetectionwith
OpenVINOtutorial<./yolov8-object-detection.ipynb>`__

Livedemo
---------

`backtotop⬆️<#table-of-contents>`__

Thefollowingcoderunsmodelinferenceonavideo:

..code::ipython3

importcollections
importtime
fromIPythonimportdisplay
importcv2


defrun_keypoint_detection(
source=0,
flip=False,
use_popup=False,
skip_first_frames=0,
model=pose_model,
device=device.value,
):
player=None

ov_config={}
ifdevice!="CPU":
model.reshape({0:[1,3,640,640]})
if"GPU"indeviceor("AUTO"indeviceand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
compiled_model=core.compile_model(model,device,ov_config)

definfer(*args):
result=compiled_model(args)
returntorch.from_numpy(result[0])

pose_model.predictor.inference=infer

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
#Gettheresults
input_image=np.array(frame)

start_time=time.time()

detections=pose_model(input_image)
stop_time=time.time()
frame=detections[0].plot()

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
#CreateanIPythonimage.
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

RunKeypointDetectiononvideo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

VIDEO_SOURCE="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

run_keypoint_detection(
source=VIDEO_SOURCE,
flip=True,
use_popup=False,
model=pose_ov_model,
device=device.value,
)



..image::yolov8-keypoint-detection-with-output_files/yolov8-keypoint-detection-with-output_60_0.png


..parsed-literal::

Sourceended


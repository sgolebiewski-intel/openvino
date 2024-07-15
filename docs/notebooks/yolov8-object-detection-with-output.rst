ConvertandOptimizeYOLOv8real-timeobjectdetectionwithOpenVINO™
=====================================================================

Real-timeobjectdetectionisoftenusedasakeycomponentincomputer
visionsystems.Applicationsthatusereal-timeobjectdetectionmodels
includevideoanalytics,robotics,autonomousvehicles,multi-object
trackingandobjectcounting,medicalimageanalysis,andmanyothers.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimizePyTorchYOLOv8withOpenVINO.Weconsiderthestepsrequired
forobjectdetectionscenario.

Thetutorialconsistsofthefollowingsteps:-PreparethePyTorch
model.-Downloadandprepareadataset.-Validatetheoriginalmodel.
-ConvertthePyTorchmodeltoOpenVINOIR.-Validatetheconverted
model.-Prepareandrunoptimizationpipeline.-Compareperformanceof
theFP32andquantizedmodels.-CompareaccuracyoftheFP32and
quantizedmodels.-OtheroptimizationpossibilitieswithOpenVINOapi-
Livedemo

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

-`Compareperformanceobjectdetection
models<#compare-performance-object-detection-models>`__
-`Validatequantizedmodel
accuracy<#validate-quantized-model-accuracy>`__

-`Nextsteps<#next-steps>`__

-`Asyncinferencepipeline<#async-inference-pipeline>`__
-`Integrationpreprocessingto
model<#integration-preprocessing-to-model>`__

-`InitializePrePostProcessing
API<#initialize-prepostprocessing-api>`__
-`Defineinputdataformat<#define-input-data-format>`__
-`Describepreprocessing
steps<#describe-preprocessing-steps>`__
-`IntegratingStepsintoa
Model<#integrating-steps-into-a-model>`__
-`Postprocessing<#postprocessing>`__

-`Livedemo<#live-demo>`__

-`RunLiveObjectDetection<#run-live-object-detection>`__

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
convertingtheYOLOv8modeltoONNXandthentoOpenVINOIR.Therefore,
wedonotneedtodothesestepsmanually.

Prerequisites
^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Installnecessarypackages.

..code::ipython3

%pipinstall-q"openvino>=2024.0.0""nncf>=2.9.0"
%pipinstall-q"torch>=2.1""torchvision>=0.16""ultralytics==8.2.24"onnxtqdmopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu

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
IMAGE_PATH=Path("./data/coco_bike.jpg")
download_file(
url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
filename=IMAGE_PATH.name,
directory=IMAGE_PATH.parent,
)

Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Thereare`several
models<https://docs.ultralytics.com/tasks/detect/>`__availableinthe
originalrepository,targetedfordifferenttasks.Forloadingthe
model,requiredtospecifyapathtothemodelcheckpoint.Itcanbe
somelocalpathornameavailableonmodelshub(inthiscasemodel
checkpointwillbedownloadedautomatically).

Makingprediction,themodelacceptsapathtoinputimageandreturns
listwithResultsclassobject.Resultscontainsboxesforobject
detectionmodel.Alsoitcontainsutilitiesforprocessingresults,for
example,``plot()``methodfordrawing.

Letusconsidertheexamples:

..code::ipython3

models_dir=Path("./models")
models_dir.mkdir(exist_ok=True)

..code::ipython3

fromPILimportImage
fromultralyticsimportYOLO

DET_MODEL_NAME="yolov8n"

det_model=YOLO(models_dir/f"{DET_MODEL_NAME}.pt")
label_map=det_model.model.names

res=det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::

Downloadinghttps://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.ptto'models/yolov8n.pt'...


..parsed-literal::

100%|██████████████████████████████████████████████████████████████████████████████|6.23M/6.23M[00:01<00:00,3.73MB/s]


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:480x6402bicycles,2cars,1dog,43.2ms
Speed:1.9mspreprocess,43.2msinference,0.9mspostprocessperimageatshape(1,3,480,640)




..image::yolov8-object-detection-with-output_files/yolov8-object-detection-with-output_9_3.png



ConvertmodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8providesAPIforconvenientmodelexportingtodifferentformats
includingOpenVINOIR.``model.export``isresponsibleformodel
conversion.Weneedtospecifytheformat,andadditionally,wecan
preservedynamicshapesinthemodel.

..code::ipython3

#objectdetectionmodel
det_model_path=models_dir/f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
ifnotdet_model_path.exists():
det_model.export(format="openvino",dynamic=True,half=True)


..parsed-literal::

UltralyticsYOLOv8.1.42🚀Python-3.10.12torch-2.2.2+cpuCPU(IntelCore(TM)i9-10980XE3.00GHz)

PyTorch:startingfrom'models/yolov8n.pt'withinputshape(1,3,640,640)BCHWandoutputshape(s)(1,84,8400)(6.2MB)

OpenVINO:startingexportwithopenvino2024.0.0-14509-34caeefd078-releases/2024/0...
OpenVINO:exportsuccess✅1.8s,savedas'models/yolov8n_openvino_model/'(6.4MB)

Exportcomplete(3.0s)
Resultssavedto/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/models
Predict:yolopredicttask=detectmodel=models/yolov8n_openvino_modelimgsz=640half
Validate:yolovaltask=detectmodel=models/yolov8n_openvino_modelimgsz=640data=coco.yamlhalf
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
readytocheckmodelpredictionforobjectdetection.

..code::ipython3

importtorch

core=ov.Core()

det_ov_model=core.read_model(det_model_path)

ov_config={}
ifdevice.value!="CPU":
det_ov_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
det_compiled_model=core.compile_model(det_ov_model,device.value,ov_config)


definfer(*args):
result=det_compiled_model(args)
returntorch.from_numpy(result[0])


det_model.predictor.inference=infer
det_model.predictor.model.pt=False

res=det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:640x6402bicycles,2cars,1dog,27.5ms
Speed:3.2mspreprocess,27.5msinference,1.2mspostprocessperimageatshape(1,3,640,640)




..image::yolov8-object-detection-with-output_files/yolov8-object-detection-with-output_16_1.png



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


DATA_URL="http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

OUT_DIR=DATASETS_DIR

DATA_PATH=OUT_DIR/"val2017.zip"
LABELS_PATH=OUT_DIR/"coco2017labels-segments.zip"
CFG_PATH=OUT_DIR/"coco.yaml"

download_file(DATA_URL,DATA_PATH.name,DATA_PATH.parent)
download_file(LABELS_URL,LABELS_PATH.name,LABELS_PATH.parent)
download_file(CFG_URL,CFG_PATH.name,CFG_PATH.parent)

ifnot(OUT_DIR/"coco/labels").exists():
withZipFile(LABELS_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR)
withZipFile(DATA_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR/"coco/images")

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
validator.stats=dict(tp=[],conf=[],pred_cls=[],target_cls=[])
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
fromultralytics.data.converterimportcoco80_to_coco91_class
fromultralytics.data.utilsimportcheck_det_dataset

args=get_cfg(cfg=DEFAULT_CFG)
args.data=str(CFG_PATH)

..code::ipython3

det_validator=det_model.task_map[det_model.task]["validator"](args=args)

..code::ipython3

det_validator.data=check_det_dataset(args.data)
det_validator.stride=32
det_data_loader=det_validator.get_dataloader(OUT_DIR/"coco",1)


..parsed-literal::

val:Scanning/home/maleksandr/test_notebooks/ultrali/datasets/coco/labels/val2017.cache...4952images,48backgrounds,


..code::ipython3

det_validator.is_coco=True
det_validator.class_map=coco80_to_coco91_class()
det_validator.names=det_model.model.names
det_validator.metrics.names=det_validator.names
det_validator.nc=det_model.model.model[-1].nc

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

fp_det_stats=test(det_ov_model,core,det_data_loader,det_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

print_stats(fp_det_stats,det_validator.seen,det_validator.nt_per_class.sum())


..parsed-literal::

Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021450.5940.5420.5790.417


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

int8_model_det_path=models_dir/f"{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml"

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
input_tensor=det_validator.preprocess(data_item)['img'].numpy()
returninput_tensor


quantization_dataset=nncf.Dataset(det_data_loader,transform_fn)


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
	"__module.model.22.dfl.conv/aten::_convolution/Convolution",
	"__module.model.22.cv3.2.2/aten::_convolution/Convolution",
	"__module.model.22.cv3.0.2/aten::_convolution/Convolution",
	"__module.model.15.cv1.conv/aten::_convolution/Convolution",
	"__module.model.5.conv/aten::_convolution/Convolution",
	"__module.model.0.conv/aten::_convolution/Convolution"
]
)


#Detectionmodel
quantized_det_model=nncf.quantize(
det_ov_model,
quantization_dataset,
preset=nncf.QuantizationPreset.MIXED,
ignored_scope=ignored_scope
)


..parsed-literal::

INFO:nncf:20ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:Notaddingactivationinputquantizerforoperation:1__module.model.0.conv/aten::_convolution/Convolution
2__module.model.0.conv/aten::_convolution/Add
3__module.model.22.cv3.2.1.act/aten::silu_/Swish

INFO:nncf:Notaddingactivationinputquantizerforoperation:25__module.model.4.cv1.conv/aten::_convolution/Convolution
26__module.model.4.cv1.conv/aten::_convolution/Add
27__module.model.22.cv3.2.1.act/aten::silu_/Swish_7

INFO:nncf:Notaddingactivationinputquantizerforoperation:43__module.model.5.conv/aten::_convolution/Convolution
47__module.model.5.conv/aten::_convolution/Add
51__module.model.22.cv3.2.1.act/aten::silu_/Swish_13

INFO:nncf:Notaddingactivationinputquantizerforoperation:54__module.model.6.cv1.conv/aten::_convolution/Convolution
56__module.model.6.cv1.conv/aten::_convolution/Add
59__module.model.22.cv3.2.1.act/aten::silu_/Swish_14

INFO:nncf:Notaddingactivationinputquantizerforoperation:92__module.model.7.conv/aten::_convolution/Convolution
99__module.model.7.conv/aten::_convolution/Add
106__module.model.22.cv3.2.1.act/aten::silu_/Swish_20

INFO:nncf:Notaddingactivationinputquantizerforoperation:98__module.model.12.cv1.conv/aten::_convolution/Convolution
105__module.model.12.cv1.conv/aten::_convolution/Add
111__module.model.22.cv3.2.1.act/aten::silu_/Swish_27

INFO:nncf:Notaddingactivationinputquantizerforoperation:46__module.model.15.cv1.conv/aten::_convolution/Convolution
50__module.model.15.cv1.conv/aten::_convolution/Add
53__module.model.22.cv3.2.1.act/aten::silu_/Swish_31

INFO:nncf:Notaddingactivationinputquantizerforoperation:74__module.model.16.conv/aten::_convolution/Convolution
81__module.model.16.conv/aten::_convolution/Add
88__module.model.22.cv3.2.1.act/aten::silu_/Swish_39

INFO:nncf:Notaddingactivationinputquantizerforoperation:75__module.model.22.cv2.0.0.conv/aten::_convolution/Convolution
82__module.model.22.cv2.0.0.conv/aten::_convolution/Add
89__module.model.22.cv3.2.1.act/aten::silu_/Swish_35

INFO:nncf:Notaddingactivationinputquantizerforoperation:76__module.model.22.cv3.0.0.conv/aten::_convolution/Convolution
83__module.model.22.cv3.0.0.conv/aten::_convolution/Add
90__module.model.22.cv3.2.1.act/aten::silu_/Swish_37

INFO:nncf:Notaddingactivationinputquantizerforoperation:96__module.model.22.cv2.0.1.conv/aten::_convolution/Convolution
103__module.model.22.cv2.0.1.conv/aten::_convolution/Add
109__module.model.22.cv3.2.1.act/aten::silu_/Swish_36

INFO:nncf:Notaddingactivationinputquantizerforoperation:115__module.model.22.cv3.0.2/aten::_convolution/Convolution
120__module.model.22.cv3.0.2/aten::_convolution/Add

INFO:nncf:Notaddingactivationinputquantizerforoperation:204__module.model.22.cv3.1.1.conv/aten::_convolution/Convolution
216__module.model.22.cv3.1.1.conv/aten::_convolution/Add
226__module.model.22.cv3.2.1.act/aten::silu_/Swish_47

INFO:nncf:Notaddingactivationinputquantizerforoperation:254__module.model.21.m.0.cv1.conv/aten::_convolution/Convolution
261__module.model.21.m.0.cv1.conv/aten::_convolution/Add
266__module.model.22.cv3.2.1.act/aten::silu_/Swish_50

INFO:nncf:Notaddingactivationinputquantizerforoperation:260__module.model.21.cv2.conv/aten::_convolution/Convolution
265__module.model.21.cv2.conv/aten::_convolution/Add
269__module.model.22.cv3.2.1.act/aten::silu_/Swish_52

INFO:nncf:Notaddingactivationinputquantizerforoperation:293__module.model.22.cv2.2.1.conv/aten::_convolution/Convolution
300__module.model.22.cv2.2.1.conv/aten::_convolution/Add
304__module.model.22.cv3.2.1.act/aten::silu_/Swish_54

INFO:nncf:Notaddingactivationinputquantizerforoperation:308__module.model.22.cv3.2.2/aten::_convolution/Convolution
311__module.model.22.cv3.2.2/aten::_convolution/Add

INFO:nncf:Notaddingactivationinputquantizerforoperation:212__module.model.22.dfl.conv/aten::_convolution/Convolution
INFO:nncf:Notaddingactivationinputquantizerforoperation:230__module.model.22/aten::sub/Subtract
INFO:nncf:Notaddingactivationinputquantizerforoperation:231__module.model.22/aten::add/Add_6



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()


..parsed-literal::

/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/venv/lib/python3.10/site-packages/nncf/experimental/tensor/tensor.py:84:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

%%skipnot$to_quantize.value

print(f"Quantizeddetectionmodelwillbesavedto{int8_model_det_path}")
ov.save_model(quantized_det_model,str(int8_model_det_path))


..parsed-literal::

Quantizeddetectionmodelwillbesavedtomodels/yolov8n_openvino_int8_model/yolov8n.xml


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
quantized_det_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
quantized_det_compiled_model=core.compile_model(quantized_det_model,device.value,ov_config)


definfer(*args):
result=quantized_det_compiled_model(args)
returntorch.from_numpy(result[0])

det_model.predictor.inference=infer

res=det_model(IMAGE_PATH)
display(Image.fromarray(res[0].plot()[:,:,::-1]))


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:640x6402bicycles,2cars,1dog,18.4ms
Speed:2.1mspreprocess,18.4msinference,0.9mspostprocessperimageatshape(1,3,640,640)



..image::yolov8-object-detection-with-output_files/yolov8-object-detection-with-output_43_1.png


ComparetheOriginalandQuantizedModels
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Compareperformanceobjectdetectionmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Finally,usetheOpenVINO`Benchmark
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

ifint8_model_det_path.exists():
#InferenceFP32model(OpenVINOIR)
!benchmark_app-m$det_model_path-d$device.value-apiasync-shape"[1,3,640,640]"


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
[INFO]Readmodeltook14.59ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_7):f32/[...]/[?,84,16..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook8.72ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_7):f32/[...]/[1,84,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook272.15ms
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
[INFO]Firstinferencetook41.00ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:21300iterations
[INFO]Duration:120060.45ms
[INFO]Latency:
[INFO]Median:67.21ms
[INFO]Average:67.48ms
[INFO]Min:31.90ms
[INFO]Max:143.04ms
[INFO]Throughput:177.41FPS


..code::ipython3

ifint8_model_det_path.exists():
#InferenceINT8model(OpenVINOIR)
!benchmark_app-m$int8_model_det_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


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
[INFO]Readmodeltook21.34ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_7):f32/[...]/[1,84,21..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook11.86ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_7):f32/[...]/[1,84,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook478.52ms
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
[INFO]Firstinferencetook35.17ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:4104iterations
[INFO]Duration:15062.51ms
[INFO]Latency:
[INFO]Median:43.53ms
[INFO]Average:43.85ms
[INFO]Min:24.58ms
[INFO]Max:70.57ms
[INFO]Throughput:272.46FPS


Validatequantizedmodelaccuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Aswecansee,thereisnosignificantdifferencebetween``INT8``and
floatmodelresultinasingleimagetest.Tounderstandhow
quantizationinfluencesmodelpredictionprecision,wecancomparemodel
accuracyonadataset.

..code::ipython3

%%skipnot$to_quantize.value

int8_det_stats=test(quantized_det_model,core,det_data_loader,det_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

%%skipnot$to_quantize.value

print("FP32modelaccuracy")
print_stats(fp_det_stats,det_validator.seen,det_validator.nt_per_class.sum())

print("INT8modelaccuracy")
print_stats(int8_det_stats,det_validator.seen,det_validator.nt_per_class.sum())


..parsed-literal::

FP32modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.5940.5420.5790.417
INT8modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.5970.5090.5620.389


Great!Lookslikeaccuracywaschanged,butnotsignificantlyandit
meetspassingcriteria.

Nextsteps
----------

`backtotop⬆️<#table-of-contents>`__Thissectioncontains
suggestionsonhowtoadditionallyimprovetheperformanceofyour
applicationusingOpenVINO.

Asyncinferencepipeline
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__ThekeyadvantageoftheAsync
APIisthatwhenadeviceisbusywithinference,theapplicationcan
performothertasksinparallel(forexample,populatinginputsor
schedulingotherrequests)ratherthanwaitforthecurrentinferenceto
completefirst.Tounderstandhowtoperformasyncinferenceusing
openvino,referto`AsyncAPItutorial<async-api-with-output.html>`__

Integrationpreprocessingtomodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

PreprocessingAPIenablesmakingpreprocessingapartofthemodel
reducingapplicationcodeanddependencyonadditionalimageprocessing
libraries.ThemainadvantageofPreprocessingAPIisthatpreprocessing
stepswillbeintegratedintotheexecutiongraphandwillbeperformed
onaselecteddevice(CPU/GPUetc.)ratherthanalwaysbeingexecutedon
CPUaspartofanapplication.Thiswillimproveselecteddevice
utilization.

Formoreinformation,refertotheoverviewof`Preprocessing
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details.html>`__.

Forexample,wecanintegrateconvertinginputdatalayoutand
normalizationdefinedin``image_to_tensor``function.

Theintegrationprocessconsistsofthefollowingsteps:1.Initializea
PrePostProcessingobject.2.Definetheinputdataformat.3.Describe
preprocessingsteps.4.IntegratingStepsintoaModel.

InitializePrePostProcessingAPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

The``openvino.preprocess.PrePostProcessor``classenablesspecifying
preprocessingandpostprocessingstepsforamodel.

..code::ipython3

fromopenvino.preprocessimportPrePostProcessor

ppp=PrePostProcessor(quantized_det_model)

Defineinputdataformat
^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Toaddressparticularinputofamodel/preprocessor,the
``input(input_id)``method,where``input_id``isapositionalindexor
inputtensornameforinputin``model.inputs``,ifamodelhasasingle
input,``input_id``canbeomitted.Afterreadingtheimagefromthe
disc,itcontainsU8pixelsinthe``[0,255]``rangeandisstoredin
the``NHWC``layout.Toperformapreprocessingconversion,weshould
providethistothetensordescription.

..code::ipython3

ppp.input(0).tensor().set_shape([1,640,640,3]).set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
pass

Toperformlayoutconversion,wealsoshouldprovideinformationabout
layoutexpectedbymodel

Describepreprocessingsteps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Ourpreprocessingfunctioncontainsthefollowingsteps:\*Convertthe
datatypefrom``U8``to``FP32``.\*Convertthedatalayoutfrom
``NHWC``to``NCHW``format.\*Normalizeeachpixelbydividingon
scalefactor255.

``ppp.input(input_id).preprocess()``isusedfordefiningasequenceof
preprocessingsteps:

..code::ipython3

ppp.input(0).preprocess().convert_element_type(ov.Type.f32).convert_layout(ov.Layout("NCHW")).scale([255.0,255.0,255.0])

print(ppp)


..parsed-literal::

Input"x":
User'sinputtensor:[1,640,640,3],[N,H,W,C],u8
Model'sexpectedtensor:[1,3,?,?],[N,C,H,W],f32
Pre-processingsteps(3):
converttype(f32):([1,640,640,3],[N,H,W,C],u8)->([1,640,640,3],[N,H,W,C],f32)
convertlayout[N,C,H,W]:([1,640,640,3],[N,H,W,C],f32)->([1,3,640,640],[N,C,H,W],f32)
scale(255,255,255):([1,3,640,640],[N,C,H,W],f32)->([1,3,640,640],[N,C,H,W],f32)



IntegratingStepsintoaModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Oncethepreprocessingstepshavebeenfinished,themodelcanbe
finallybuilt.Additionally,wecansaveacompletedmodeltoOpenVINO
IR,using``openvino.runtime.serialize``.

..code::ipython3

quantized_model_with_preprocess=ppp.build()
ov.save_model(
quantized_model_with_preprocess,
str(int8_model_det_path.with_name(f"{DET_MODEL_NAME}_with_preprocess.xml")),
)

Themodelwithintegratedpreprocessingisreadyforloadingtoa
device.

..code::ipython3

fromtypingimportTuple,Dict
importcv2
importnumpyasnp
fromultralytics.utils.plottingimportcolors


defplot_one_box(
box:np.ndarray,
img:np.ndarray,
color:Tuple[int,int,int]=None,
label:str=None,
line_thickness:int=5,
):
"""
Helperfunctionfordrawingsingleboundingboxonimage
Parameters:
x(np.ndarray):boundingboxcoordinatesinformat[x1,y1,x2,y2]
img(no.ndarray):inputimage
color(Tuple[int,int,int],*optional*,None):colorinBGRformatfordrawingbox,ifnotspecifiedwillbeselectedrandomly
label(str,*optonal*,None):boxlabelstring,ifnotprovidedwillnotbeprovidedasdrowingresult
line_thickness(int,*optional*,5):thicknessforboxdrawinglines
"""
#Plotsoneboundingboxonimageimg
tl=line_thicknessorround(0.002*(img.shape[0]+img.shape[1])/2)+1#line/fontthickness
color=coloror[random.randint(0,255)for_inrange(3)]
c1,c2=(int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
cv2.rectangle(img,c1,c2,color,thickness=tl,lineType=cv2.LINE_AA)
iflabel:
tf=max(tl-1,1)#fontthickness
t_size=cv2.getTextSize(label,0,fontScale=tl/3,thickness=tf)[0]
c2=c1[0]+t_size[0],c1[1]-t_size[1]-3
cv2.rectangle(img,c1,c2,color,-1,cv2.LINE_AA)#filled
cv2.putText(
img,
label,
(c1[0],c1[1]-2),
0,
tl/3,
[225,255,255],
thickness=tf,
lineType=cv2.LINE_AA,
)

returnimg


defdraw_results(results:Dict,source_image:np.ndarray,label_map:Dict):
"""
Helperfunctionfordrawingboundingboxesonimage
Parameters:
image_res(np.ndarray):detectionpredictionsinformat[x1,y1,x2,y2,score,label_id]
source_image(np.ndarray):inputimagefordrawing
label_map;(Dict[int,str]):label_idtoclassnamemapping
Returns:
Imagewithboxes
"""
boxes=results["det"]
foridx,(*xyxy,conf,lbl)inenumerate(boxes):
label=f"{label_map[int(lbl)]}{conf:.2f}"
source_image=plot_one_box(xyxy,source_image,label=label,color=colors(int(lbl)),line_thickness=1)
returnsource_image

Postprocessing
''''''''''''''

`backtotop⬆️<#table-of-contents>`__

Themodeloutputcontainsdetectionboxescandidates,itisatensor
withthe[-1,84,-1]shapeintheB,84,Nformat,where:

B-batchsizeN-numberofdetectionboxesForgettingthefinal
prediction,weneedtoapplyanon-maximumsuppressionalgorithmand
rescaleboxcoordinatestotheoriginalimagesize.

Finally,detectionboxhasthe[x,y,h,w,class_no_1,…,class_no_80]
format,where:

(x,y)-rawcoordinatesofboxcenterh,w-rawheightandwidthof
theboxclass_no_1,…,class_no_80-probabilitydistributionoverthe
classes.

..code::ipython3

fromtypingimportTuple
fromultralytics.utilsimportops
importtorch
importnumpyasnp


defletterbox(
img:np.ndarray,
new_shape:Tuple[int,int]=(640,640),
color:Tuple[int,int,int]=(114,114,114),
auto:bool=False,
scale_fill:bool=False,
scaleup:bool=False,
stride:int=32,
):
"""
Resizeimageandpaddingfordetection.Takesimageasinput,
resizesimagetofitintonewshapewithsavingoriginalaspectratioandpadsittomeetstride-multipleconstraints

Parameters:
img(np.ndarray):imageforpreprocessing
new_shape(Tuple(int,int)):imagesizeafterpreprocessinginformat[height,width]
color(Tuple(int,int,int)):colorforfillingpaddedarea
auto(bool):usedynamicinputsize,onlypaddingforstrideconstrinsapplied
scale_fill(bool):scaleimagetofillnew_shape
scaleup(bool):allowscaleimageifitislowerthendesiredinputsize,canaffectmodelaccuracy
stride(int):inputpaddingstride
Returns:
img(np.ndarray):imageafterpreprocessing
ratio(Tuple(float,float)):hightandwidthscalingratio
padding_size(Tuple(int,int)):heightandwidthpaddingsize


"""
#Resizeandpadimagewhilemeetingstride-multipleconstraints
shape=img.shape[:2]#currentshape[height,width]
ifisinstance(new_shape,int):
new_shape=(new_shape,new_shape)

#Scaleratio(new/old)
r=min(new_shape[0]/shape[0],new_shape[1]/shape[1])
ifnotscaleup:#onlyscaledown,donotscaleup(forbettertestmAP)
r=min(r,1.0)

#Computepadding
ratio=r,r#width,heightratios
new_unpad=int(round(shape[1]*r)),int(round(shape[0]*r))
dw,dh=new_shape[1]-new_unpad[0],new_shape[0]-new_unpad[1]#whpadding
ifauto:#minimumrectangle
dw,dh=np.mod(dw,stride),np.mod(dh,stride)#whpadding
elifscale_fill:#stretch
dw,dh=0.0,0.0
new_unpad=(new_shape[1],new_shape[0])
ratio=new_shape[1]/shape[1],new_shape[0]/shape[0]#width,heightratios

dw/=2#dividepaddinginto2sides
dh/=2

ifshape[::-1]!=new_unpad:#resize
img=cv2.resize(img,new_unpad,interpolation=cv2.INTER_LINEAR)
top,bottom=int(round(dh-0.1)),int(round(dh+0.1))
left,right=int(round(dw-0.1)),int(round(dw+0.1))
img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)#addborder
returnimg,ratio,(dw,dh)


defpostprocess(
pred_boxes:np.ndarray,
input_hw:Tuple[int,int],
orig_img:np.ndarray,
min_conf_threshold:float=0.25,
nms_iou_threshold:float=0.7,
agnosting_nms:bool=False,
max_detections:int=300,
):
"""
YOLOv8modelpostprocessingfunction.Appliednonmaximumsupressionalgorithmtodetectionsandrescaleboxestooriginalimagesize
Parameters:
pred_boxes(np.ndarray):modeloutputpredictionboxes
input_hw(np.ndarray):preprocessedimage
orig_image(np.ndarray):imagebeforepreprocessing
min_conf_threshold(float,*optional*,0.25):minimalacceptedconfidenceforobjectfiltering
nms_iou_threshold(float,*optional*,0.45):minimaloverlapscoreforremovingobjectsduplicatesinNMS
agnostic_nms(bool,*optiona*,False):applyclassagnostincNMSapproachornot
max_detections(int,*optional*,300):maximumdetectionsafterNMS
Returns:
pred(List[Dict[str,np.ndarray]]):listofdictionarywithdet-detectedboxesinformat[x1,y1,x2,y2,score,label]
"""
nms_kwargs={"agnostic":agnosting_nms,"max_det":max_detections}
preds=ops.non_max_suppression(torch.from_numpy(pred_boxes),min_conf_threshold,nms_iou_threshold,nc=80,**nms_kwargs)

results=[]
fori,predinenumerate(preds):
shape=orig_img[i].shapeifisinstance(orig_img,list)elseorig_img.shape
ifnotlen(pred):
results.append({"det":[],"segment":[]})
continue
pred[:,:4]=ops.scale_boxes(input_hw,pred[:,:4],shape).round()
results.append({"det":pred})

returnresults

Now,wecanskipthesepreprocessingstepsindetectfunction:

..code::ipython3

defdetect_without_preprocess(image:np.ndarray,model:ov.Model):
"""
OpenVINOYOLOv8modelwithintegratedpreprocessinginferencefunction.Preprocessimage,runsmodelinferenceandpostprocessresultsusingNMS.
Parameters:
image(np.ndarray):inputimage.
model(Model):OpenVINOcompiledmodel.
Returns:
detections(np.ndarray):detectedboxesinformat[x1,y1,x2,y2,score,label]
"""
output_layer=model.output(0)
img=letterbox(image)[0]
input_tensor=np.expand_dims(img,0)
input_hw=img.shape[:2]
result=model(input_tensor)[output_layer]
detections=postprocess(result,input_hw,image)
returndetections


compiled_model=core.compile_model(quantized_model_with_preprocess,device.value)
input_image=np.array(Image.open(IMAGE_PATH))
detections=detect_without_preprocess(input_image,compiled_model)[0]
image_with_boxes=draw_results(detections,input_image,label_map)

Image.fromarray(image_with_boxes)




..image::yolov8-object-detection-with-output_files/yolov8-object-detection-with-output_70_0.png



Livedemo
---------

`backtotop⬆️<#table-of-contents>`__

Thefollowingcoderunsmodelinferenceonavideo:

..code::ipython3

importcollections
importtime
fromIPythonimportdisplay


#Mainprocessingfunctiontorunobjectdetection.
defrun_object_detection(
source=0,
flip=False,
use_popup=False,
skip_first_frames=0,
model=det_model,
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

det_model.predictor.inference=infer

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
detections=det_model(input_image)
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

RunLiveObjectDetection
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

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

run_object_detection(
source=VIDEO_SOURCE,
flip=True,
use_popup=False,
model=det_ov_model,
device=device.value,
)



..image::yolov8-object-detection-with-output_files/yolov8-object-detection-with-output_76_0.png


..parsed-literal::

Sourceended


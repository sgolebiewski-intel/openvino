ConvertandOptimizeYOLOv8instancesegmentationmodelwithOpenVINO™
======================================================================

Instancesegmentationgoesastepfurtherthanobjectdetectionand
involvesidentifyingindividualobjectsinanimageandsegmentingthem
fromtherestoftheimage.Instancesegmentationasanobjectdetection
areoftenusedaskeycomponentsincomputervisionsystems.
Applicationsthatusereal-timeinstancesegmentationmodelsinclude
videoanalytics,robotics,autonomousvehicles,multi-objecttracking
andobjectcounting,medicalimageanalysis,andmanyothers.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimizePyTorchYOLOv8withOpenVINO.Weconsiderthestepsrequired
forinstancesegmentationscenario.

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
-`Validatequantizedmodel
accuracy<#validate-quantized-model-accuracy>`__

-`Otherwaystooptimizemodel<#other-ways-to-optimize-model>`__
-`Livedemo<#live-demo>`__

-`RunLiveObjectDetectionand
Segmentation<#run-live-object-detection-and-segmentation>`__

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
%pipinstall-q"torch>=2.1""torchvision>=0.16""ultralytics==8.2.24"onnxopencv-pythontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

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



..parsed-literal::

data/coco_bike.jpg:0%||0.00/182k[00:00<?,?B/s]




..parsed-literal::

PosixPath('/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg')



Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Forloadingthemodel,requiredtospecifyapathtothemodel
checkpoint.Itcanbesomelocalpathornameavailableonmodelshub
(inthiscasemodelcheckpointwillbedownloadedautomatically).

Makingprediction,themodelacceptsapathtoinputimageandreturns
listwithResultsclassobject.Resultscontainsboxesforobject
detectionmodelandboxesandmasksforsegmentationmodel.Alsoit
containsutilitiesforprocessingresults,forexample,``plot()``
methodfordrawing.

Letusconsidertheexamples:

..code::ipython3

models_dir=Path("./models")
models_dir.mkdir(exist_ok=True)

..code::ipython3

fromPILimportImage
fromultralyticsimportYOLO

SEG_MODEL_NAME="yolov8n-seg"

seg_model=YOLO(models_dir/f"{SEG_MODEL_NAME}.pt")
label_map=seg_model.model.names

res=seg_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::

Downloadinghttps://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.ptto'models/yolov8n-seg.pt'...


..parsed-literal::

100%|██████████████████████████████████████████████████████████████████████████████|6.73M/6.73M[00:01<00:00,3.89MB/s]


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:480x6401bicycle,2cars,1dog,55.6ms
Speed:1.8mspreprocess,55.6msinference,2.0mspostprocessperimageatshape(1,3,480,640)




..image::yolov8-instance-segmentation-with-output_files/yolov8-instance-segmentation-with-output_9_3.png



ConvertmodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8providesAPIforconvenientmodelexportingtodifferentformats
includingOpenVINOIR.``model.export``isresponsibleformodel
conversion.Weneedtospecifytheformat,andadditionally,wecan
preservedynamicshapesinthemodel.

..code::ipython3

#instancesegmentationmodel
seg_model_path=models_dir/f"{SEG_MODEL_NAME}_openvino_model/{SEG_MODEL_NAME}.xml"
ifnotseg_model_path.exists():
seg_model.export(format="openvino",dynamic=True,half=True)


..parsed-literal::

UltralyticsYOLOv8.1.42🚀Python-3.10.12torch-2.2.2+cpuCPU(IntelCore(TM)i9-10980XE3.00GHz)

PyTorch:startingfrom'models/yolov8n-seg.pt'withinputshape(1,3,640,640)BCHWandoutputshape(s)((1,116,8400),(1,32,160,160))(6.7MB)

OpenVINO:startingexportwithopenvino2024.0.0-14509-34caeefd078-releases/2024/0...
OpenVINO:exportsuccess✅1.8s,savedas'models/yolov8n-seg_openvino_model/'(6.9MB)

Exportcomplete(3.0s)
Resultssavedto/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/models
Predict:yolopredicttask=segmentmodel=models/yolov8n-seg_openvino_modelimgsz=640half
Validate:yolovaltask=segmentmodel=models/yolov8n-seg_openvino_modelimgsz=640data=coco.yamlhalf
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

..code::ipython3

core=ov.Core()
seg_ov_model=core.read_model(seg_model_path)

ov_config={}
ifdevice.value!="CPU":
seg_ov_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
seg_compiled_model=core.compile_model(seg_ov_model,device.value,ov_config)

..code::ipython3

importtorch


definfer(*args):
result=seg_compiled_model(args)
returntorch.from_numpy(result[0]),torch.from_numpy(result[1])


seg_model.predictor.inference=infer
seg_model.predictor.model.pt=False

..code::ipython3

res=seg_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:640x6401bicycle,2cars,1dog,27.6ms
Speed:3.5mspreprocess,27.6msinference,4.5mspostprocessperimageatshape(1,3,640,640)




..image::yolov8-instance-segmentation-with-output_files/yolov8-instance-segmentation-with-output_18_1.png



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
validator.stats=dict(tp_m=[],tp=[],conf=[],pred_cls=[],target_cls=[])
validator.batch_i=1
validator.confusion_matrix=ConfusionMatrix(nc=validator.nc)
model.reshape({0:[1,3,-1,-1]})
num_outputs=len(model.outputs)
compiled_model=core.compile_model(model)
forbatch_i,batchinenumerate(tqdm(data_loader,total=num_samples)):
ifnum_samplesisnotNoneandbatch_i==num_samples:
break
batch=validator.preprocess(batch)
results=compiled_model(batch["img"])
ifnum_outputs==1:
preds=torch.from_numpy(results[compiled_model.output(0)])
else:
preds=[
torch.from_numpy(results[compiled_model.output(0)]),
torch.from_numpy(results[compiled_model.output(1)]),
]
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
fromultralytics.utilsimportops

args=get_cfg(cfg=DEFAULT_CFG)
args.data=str(CFG_PATH)

..code::ipython3

seg_validator=seg_model.task_map[seg_model.task]["validator"](args=args)
seg_validator.data=check_det_dataset(args.data)
seg_validator.stride=32
seg_data_loader=seg_validator.get_dataloader(OUT_DIR/"coco/",1)

seg_validator.is_coco=True
seg_validator.class_map=coco80_to_coco91_class()
seg_validator.names=seg_model.model.names
seg_validator.metrics.names=seg_validator.names
seg_validator.nc=seg_model.model.model[-1].nc
seg_validator.nm=32
seg_validator.process=ops.process_mask
seg_validator.plot_masks=[]


..parsed-literal::

val:Scanning/home/maleksandr/test_notebooks/ultrali/datasets/coco/labels/val2017.cache...4952images,48backgrounds,


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

fp_seg_stats=test(seg_ov_model,core,seg_data_loader,seg_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

print_stats(fp_seg_stats,seg_validator.seen,seg_validator.nt_per_class.sum())


..parsed-literal::

Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021450.6090.5210.580.416
Macroaveragemean:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021450.6050.5010.5580.353


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

int8_model_seg_path=models_dir/f"{SEG_MODEL_NAME}_openvino_int8_model/{SEG_MODEL_NAME}.xml"

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
importrequests

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
input_tensor=seg_validator.preprocess(data_item)['img'].numpy()
returninput_tensor


quantization_dataset=nncf.Dataset(seg_data_loader,transform_fn)


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
	"__module.model.22.proto.cv1.conv/aten::_convolution/Convolution",
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

#Segmentationmodel
quantized_seg_model=nncf.quantize(
seg_ov_model,
quantization_dataset,
preset=nncf.QuantizationPreset.MIXED,
ignored_scope=ignored_scope
)


..parsed-literal::

INFO:nncf:23ignorednodeswerefoundbynameintheNNCFGraph
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

INFO:nncf:Notaddingactivationinputquantizerforoperation:98__module.model.7.conv/aten::_convolution/Convolution
107__module.model.7.conv/aten::_convolution/Add
116__module.model.22.cv4.2.1.act/aten::silu_/Swish_20

INFO:nncf:Notaddingactivationinputquantizerforoperation:106__module.model.12.cv1.conv/aten::_convolution/Convolution
115__module.model.12.cv1.conv/aten::_convolution/Add
123__module.model.22.cv4.2.1.act/aten::silu_/Swish_27

INFO:nncf:Notaddingactivationinputquantizerforoperation:46__module.model.15.cv1.conv/aten::_convolution/Convolution
50__module.model.15.cv1.conv/aten::_convolution/Add
53__module.model.22.cv4.2.1.act/aten::silu_/Swish_31

INFO:nncf:Notaddingactivationinputquantizerforoperation:74__module.model.16.conv/aten::_convolution/Convolution
83__module.model.16.conv/aten::_convolution/Add
92__module.model.22.cv4.2.1.act/aten::silu_/Swish_42

INFO:nncf:Notaddingactivationinputquantizerforoperation:75__module.model.22.cv2.0.0.conv/aten::_convolution/Convolution
84__module.model.22.cv2.0.0.conv/aten::_convolution/Add
93__module.model.22.cv4.2.1.act/aten::silu_/Swish_38

INFO:nncf:Notaddingactivationinputquantizerforoperation:102__module.model.22.cv2.0.1.conv/aten::_convolution/Convolution
111__module.model.22.cv2.0.1.conv/aten::_convolution/Add
119__module.model.22.cv4.2.1.act/aten::silu_/Swish_39

INFO:nncf:Notaddingactivationinputquantizerforoperation:76__module.model.22.cv3.0.0.conv/aten::_convolution/Convolution
85__module.model.22.cv3.0.0.conv/aten::_convolution/Add
94__module.model.22.cv4.2.1.act/aten::silu_/Swish_40

INFO:nncf:Notaddingactivationinputquantizerforoperation:127__module.model.22.cv3.0.2/aten::_convolution/Convolution
134__module.model.22.cv3.0.2/aten::_convolution/Add

INFO:nncf:Notaddingactivationinputquantizerforoperation:77__module.model.22.cv4.0.0.conv/aten::_convolution/Convolution
86__module.model.22.cv4.0.0.conv/aten::_convolution/Add
95__module.model.22.cv4.2.1.act/aten::silu_/Swish_60

INFO:nncf:Notaddingactivationinputquantizerforoperation:78__module.model.22.proto.cv1.conv/aten::_convolution/Convolution
87__module.model.22.proto.cv1.conv/aten::_convolution/Add
96__module.model.22.cv4.2.1.act/aten::silu_/Swish_35

INFO:nncf:Notaddingactivationinputquantizerforoperation:234__module.model.22.cv3.1.1.conv/aten::_convolution/Convolution
247__module.model.22.cv3.1.1.conv/aten::_convolution/Add
258__module.model.22.cv4.2.1.act/aten::silu_/Swish_50

INFO:nncf:Notaddingactivationinputquantizerforoperation:289__module.model.21.m.0.cv1.conv/aten::_convolution/Convolution
296__module.model.21.m.0.cv1.conv/aten::_convolution/Add
301__module.model.22.cv4.2.1.act/aten::silu_/Swish_53

INFO:nncf:Notaddingactivationinputquantizerforoperation:295__module.model.21.cv2.conv/aten::_convolution/Convolution
300__module.model.21.cv2.conv/aten::_convolution/Add
304__module.model.22.cv4.2.1.act/aten::silu_/Swish_55

INFO:nncf:Notaddingactivationinputquantizerforoperation:331__module.model.22.cv2.2.1.conv/aten::_convolution/Convolution
339__module.model.22.cv2.2.1.conv/aten::_convolution/Add
344__module.model.22.cv4.2.1.act/aten::silu_/Swish_57

INFO:nncf:Notaddingactivationinputquantizerforoperation:333__module.model.22.cv4.2.1.conv/aten::_convolution/Convolution
341__module.model.22.cv4.2.1.conv/aten::_convolution/Add
346__module.model.22.cv4.2.1.act/aten::silu_/Swish_65

INFO:nncf:Notaddingactivationinputquantizerforoperation:349__module.model.22.cv3.2.2/aten::_convolution/Convolution
353__module.model.22.cv3.2.2/aten::_convolution/Add

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

print(f"Quantizedsegmentationmodelwillbesavedto{int8_model_seg_path}")
ov.save_model(quantized_seg_model,str(int8_model_seg_path))


..parsed-literal::

Quantizedsegmentationmodelwillbesavedtomodels/yolov8n-seg_openvino_int8_model/yolov8n-seg.xml


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
quantized_seg_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}

quantized_seg_compiled_model=core.compile_model(quantized_seg_model,device.value,ov_config)

..code::ipython3

%%skipnot$to_quantize.value


definfer(*args):
result=quantized_seg_compiled_model(args)
returntorch.from_numpy(result[0]),torch.from_numpy(result[1])

seg_model.predictor.inference=infer

..code::ipython3

%%skipnot$to_quantize.value

res=seg_model(IMAGE_PATH)
display(Image.fromarray(res[0].plot()[:,:,::-1]))


..parsed-literal::


image1/1/home/maleksandr/test_notebooks/update_ultralytics/openvino_notebooks/notebooks/yolov8-optimization/data/coco_bike.jpg:640x6401bicycle,2cars,2dogs,26.8ms
Speed:2.8mspreprocess,26.8msinference,3.4mspostprocessperimageatshape(1,3,640,640)



..image::yolov8-instance-segmentation-with-output_files/yolov8-instance-segmentation-with-output_46_1.png


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

ifint8_model_seg_path.exists():
!benchmark_app-m$seg_model_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


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
[INFO]Readmodeltook15.84ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[?,116,16..]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[?,32,8..,8..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook9.23ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[1,116,8400]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[1,32,160,160]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook304.42ms
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
[INFO]Firstinferencetook50.67ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:2124iterations
[INFO]Duration:15076.69ms
[INFO]Latency:
[INFO]Median:84.69ms
[INFO]Average:84.95ms
[INFO]Min:43.23ms
[INFO]Max:184.81ms
[INFO]Throughput:140.88FPS


..code::ipython3

ifint8_model_seg_path.exists():
!benchmark_app-m$int8_model_seg_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


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
[INFO]Readmodeltook24.33ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[1,116,21..]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[1,32,8..,8..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook13.01ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[1,116,8400]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[1,32,160,160]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook574.36ms
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
[INFO]Firstinferencetook41.26ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:3048iterations
[INFO]Duration:15096.50ms
[INFO]Latency:
[INFO]Median:58.82ms
[INFO]Average:59.20ms
[INFO]Min:33.17ms
[INFO]Max:120.39ms
[INFO]Throughput:201.90FPS


Validatequantizedmodelaccuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Aswecansee,thereisnosignificantdifferencebetween``INT8``and
floatmodelresultinasingleimagetest.Tounderstandhow
quantizationinfluencesmodelpredictionprecision,wecancomparemodel
accuracyonadataset.

..code::ipython3

%%skipnot$to_quantize.value

int8_seg_stats=test(quantized_seg_model,core,seg_data_loader,seg_validator,num_samples=NUM_TEST_SAMPLES)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..code::ipython3

%%skipnot$to_quantize.value

print("FP32modelaccuracy")
print_stats(fp_seg_stats,seg_validator.seen,seg_validator.nt_per_class.sum())

print("INT8modelaccuracy")
print_stats(int8_seg_stats,seg_validator.seen,seg_validator.nt_per_class.sum())


..parsed-literal::

FP32modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.6090.5210.580.416
Macroaveragemean:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.6050.5010.5580.353
INT8modelaccuracy
Boxes:
Bestmeanaverage:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.5220.5380.5550.376
Macroaveragemean:
ClassImagesLabelsPrecisionRecallmAP@.5mAP@.5:.95
all30021530.6310.4630.5290.344


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
importcv2
fromIPythonimportdisplay


defrun_instance_segmentation(
source=0,
flip=False,
use_popup=False,
skip_first_frames=0,
model=seg_model,
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
returntorch.from_numpy(result[0]),torch.from_numpy(result[1])

seg_model.predictor.inference=infer

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
detections=seg_model(input_image)
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

RunLiveObjectDetectionandSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

run_instance_segmentation(
source=VIDEO_SOURCE,
flip=True,
use_popup=False,
model=seg_ov_model,
device=device.value,
)



..image::yolov8-instance-segmentation-with-output_files/yolov8-instance-segmentation-with-output_62_0.png


..parsed-literal::

Sourceended


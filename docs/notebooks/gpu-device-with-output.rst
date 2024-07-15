WorkingwithGPUsinOpenVINO™
==============================

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Introduction<#introduction>`__

-`Installrequiredpackages<#install-required-packages>`__

-`CheckingGPUswithQuery
Device<#checking-gpus-with-query-device>`__

-`ListGPUswith
core.available_devices<#list-gpus-with-core-available_devices>`__
-`CheckPropertieswith
core.get_property<#check-properties-with-core-get_property>`__
-`BriefDescriptionsofKey
Properties<#brief-descriptions-of-key-properties>`__

-`CompilingaModelonGPU<#compiling-a-model-on-gpu>`__

-`DownloadandConvertaModel<#download-and-convert-a-model>`__

-`Downloadandunpackthe
Model<#download-and-unpack-the-model>`__
-`ConverttheModeltoOpenVINOIR
format<#convert-the-model-to-openvino-ir-format>`__

-`CompilewithDefault
Configuration<#compile-with-default-configuration>`__
-`ReduceCompileTimethroughModel
Caching<#reduce-compile-time-through-model-caching>`__
-`ThroughputandLatencyPerformance
Hints<#throughput-and-latency-performance-hints>`__
-`UsingMultipleGPUswithMulti-DeviceandCumulative
Throughput<#using-multiple-gpus-with-multi-device-and-cumulative-throughput>`__

-`PerformanceComparisonwith
benchmark_app<#performance-comparison-with-benchmark_app>`__

-`CPUvsGPUwithLatencyHint<#cpu-vs-gpu-with-latency-hint>`__
-`CPUvsGPUwithThroughput
Hint<#cpu-vs-gpu-with-throughput-hint>`__
-`SingleGPUvsMultipleGPUs<#single-gpu-vs-multiple-gpus>`__

-`BasicApplicationUsingGPUs<#basic-application-using-gpus>`__

-`ImportNecessaryPackages<#import-necessary-packages>`__
-`CompiletheModel<#compile-the-model>`__
-`LoadandPreprocessVideo
Frames<#load-and-preprocess-video-frames>`__
-`DefineModelOutputClasses<#define-model-output-classes>`__
-`SetupAsynchronousPipeline<#set-up-asynchronous-pipeline>`__

-`CallbackDefinition<#callback-definition>`__
-`CreateAsyncPipeline<#create-async-pipeline>`__

-`PerformInference<#perform-inference>`__
-`ProcessResults<#process-results>`__

-`Conclusion<#conclusion>`__

Thistutorialprovidesahigh-leveloverviewofworkingwithIntelGPUs
inOpenVINO.ItshowshowtouseQueryDevicetolistsystemGPUsand
checktheirproperties,anditexplainssomeofthekeyproperties.It
showshowtocompileamodelonGPUwithperformancehintsandhowto
usemultipleGPUsusingMULTIorCUMULATIVE_THROUGHPUT.

Thetutorialalsoshowsexamplecommandsforbenchmark_appthatcanbe
runtocompareGPUperformanceindifferentconfigurations.Italso
providesthecodeforabasicend-to-endapplicationthatcompilesa
modelonGPUandusesittoruninference.

Introduction
------------

`backtotop⬆️<#table-of-contents>`__

Originally,graphicprocessingunits(GPUs)beganasspecializedchips,
developedtoacceleratetherenderingofcomputergraphics.Incontrast
toCPUs,whichhavefewbutpowerfulcores,GPUshavemanymore
specializedcores,makingthemidealforworkloadsthatcanbe
parallelizedintosimplertasks.Nowadays,onesuchworkloadisdeep
learning,whereGPUscaneasilyaccelerateinferenceofneuralnetworks
bysplittingoperationsacrossmultiplecores.

OpenVINOsupportsinferenceonIntelintegratedGPUs(whichareincluded
withmost`Intel®Core™desktopandmobile
processors<https://www.intel.com/content/www/us/en/products/details/processors/core.html>`__)
oronInteldiscreteGPUproductslikethe`Intel®Arc™A-Series
Graphics
cards<https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html>`__
and`Intel®DataCenterGPUFlex
Series<https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html>`__.
Togetstarted,first`install
OpenVINO<https://docs.openvino.ai/2024/get-started/install-openvino.html>`__
onasystemequippedwithoneormoreIntelGPUs.Followthe`GPU
configuration
instructions<https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
toconfigureOpenVINOtoworkwithyourGPU.Then,readontolearnhow
toaccelerateinferencewithGPUsinOpenVINO!

Installrequiredpackages
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino-dev>=2024.0.0""opencv-python""tqdm"
%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'andpython_version<='3.8'"

CheckingGPUswithQueryDevice
-------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthissection,wewillseehowtolisttheavailableGPUsandcheck
theirproperties.Someofthekeypropertieswillalsobedefined.

ListGPUswithcore.available_devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINORuntimeprovidesthe``available_devices``methodforchecking
whichdevicesareavailableforinference.Thefollowingcodewill
outputalistofcompatibleOpenVINOdevices,inwhichIntelGPUsshould
appear.

..code::ipython3

importopenvinoasov

core=ov.Core()
core.available_devices




..parsed-literal::

['CPU','GPU']



NotethatGPUdevicesarenumberedstartingat0,wheretheintegrated
GPUalwaystakestheid``0``ifthesystemhasone.Forinstance,if
thesystemhasaCPU,anintegratedanddiscreteGPU,weshouldexpect
toseealistlikethis:``['CPU','GPU.0','GPU.1']``.Tosimplifyits
use,the“GPU.0”canalsobeaddressedwithjust“GPU”.Formore
details,seethe`DeviceNaming
Convention<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#device-naming-convention>`__
section.

IftheGPUsareinstalledcorrectlyonthesystemandstilldonot
appearinthelist,followthestepsdescribed
`here<https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
toconfigureyourGPUdriverstoworkwithOpenVINO.Oncewehavethe
GPUsworkingwithOpenVINO,wecanproceedwiththenextsections.

CheckPropertieswithcore.get_property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TogetinformationabouttheGPUs,wecanusedeviceproperties.In
OpenVINO,deviceshavepropertiesthatdescribetheircharacteristics
andconfiguration.Eachpropertyhasanameandassociatedvaluethat
canbequeriedwiththe``get_property``method.

Togetthevalueofaproperty,suchasthedevicename,wecanusethe
``get_property``methodasfollows:

..code::ipython3

device="GPU"

core.get_property(device,"FULL_DEVICE_NAME")




..parsed-literal::

'Intel(R)Graphics[0x46a6](iGPU)'



Eachdevicealsohasaspecificpropertycalled
``SUPPORTED_PROPERTIES``,thatenablesviewingalltheavailable
propertiesinthedevice.Wecancheckthevalueforeachpropertyby
simplyloopingthroughthedictionaryreturnedby
``core.get_property("GPU","SUPPORTED_PROPERTIES")``andthenquerying
forthatproperty.

..code::ipython3

print(f"{device}SUPPORTED_PROPERTIES:\n")
supported_properties=core.get_property(device,"SUPPORTED_PROPERTIES")
indent=len(max(supported_properties,key=len))

forproperty_keyinsupported_properties:
ifproperty_keynotin(
"SUPPORTED_METRICS",
"SUPPORTED_CONFIG_KEYS",
"SUPPORTED_PROPERTIES",
):
try:
property_val=core.get_property(device,property_key)
exceptTypeError:
property_val="UNSUPPORTEDTYPE"
print(f"{property_key:<{indent}}:{property_val}")


..parsed-literal::

GPUSUPPORTED_PROPERTIES:

AVAILABLE_DEVICES:['0']
RANGE_FOR_ASYNC_INFER_REQUESTS:(1,2,1)
RANGE_FOR_STREAMS:(1,2)
OPTIMAL_BATCH_SIZE:1
MAX_BATCH_SIZE:1
CACHING_PROPERTIES:{'GPU_UARCH_VERSION':'RO','GPU_EXECUTION_UNITS_COUNT':'RO','GPU_DRIVER_VERSION':'RO','GPU_DEVICE_ID':'RO'}
DEVICE_ARCHITECTURE:GPU:v12.0.0
FULL_DEVICE_NAME:Intel(R)Graphics[0x46a6](iGPU)
DEVICE_UUID:UNSUPPORTEDTYPE
DEVICE_TYPE:Type.INTEGRATED
DEVICE_GOPS:UNSUPPORTEDTYPE
OPTIMIZATION_CAPABILITIES:['FP32','BIN','FP16','INT8']
GPU_DEVICE_TOTAL_MEM_SIZE:UNSUPPORTEDTYPE
GPU_UARCH_VERSION:12.0.0
GPU_EXECUTION_UNITS_COUNT:96
GPU_MEMORY_STATISTICS:UNSUPPORTEDTYPE
PERF_COUNT:False
MODEL_PRIORITY:Priority.MEDIUM
GPU_HOST_TASK_PRIORITY:Priority.MEDIUM
GPU_QUEUE_PRIORITY:Priority.MEDIUM
GPU_QUEUE_THROTTLE:Priority.MEDIUM
GPU_ENABLE_LOOP_UNROLLING:True
CACHE_DIR:
PERFORMANCE_HINT:PerformanceMode.UNDEFINED
COMPILATION_NUM_THREADS:20
NUM_STREAMS:1
PERFORMANCE_HINT_NUM_REQUESTS:0
INFERENCE_PRECISION_HINT:<Type:'undefined'>
DEVICE_ID:0


BriefDescriptionsofKeyProperties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Eachdevicehasseveralpropertiesasseeninthelastcommand.Someof
thekeypropertiesare:

-``FULL_DEVICE_NAME``- TheproductnameoftheGPUandwhetheritis
anintegratedordiscreteGPU(iGPUordGPU).
-``OPTIMIZATION_CAPABILITIES``-Themodeldatatypes(INT8,FP16,
FP32,etc)thataresupportedbythisGPU.
-``GPU_EXECUTION_UNITS_COUNT``-Theexecutioncoresavailableinthe
GPU’sarchitecture,whichisarelativemeasureoftheGPU’s
processingpower.
-``RANGE_FOR_STREAMS``-Thenumberofprocessingstreamsavailableon
theGPUthatcanbeusedtoexecuteparallelinferencerequests.When
compilingamodelinLATENCYorTHROUGHPUTmode,OpenVINOwill
automaticallyselectthebestnumberofstreamsforlowlatencyor
highthroughput.
-``PERFORMANCE_HINT``-Ahigh-levelwaytotunethedevicefora
specificperformancemetric,suchaslatencyorthroughput,without
worryingaboutdevice-specificsettings.
-``CACHE_DIR``-Thedirectorywherethemodelcachedataisstoredto
speedupcompilationtime.

Tolearnmoreaboutdevicesandproperties,seethe`QueryDevice
Properties<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
page.

CompilingaModelonGPU
------------------------

`backtotop⬆️<#table-of-contents>`__

Now,weknowhowtolisttheGPUsinthesystemandchecktheir
properties.Wecaneasilyuseoneforcompilingandrunningmodelswith
OpenVINO`GPU
plugin<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>`__.

DownloadandConvertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thistutorialusesthe``ssdlite_mobilenet_v2``model.The
``ssdlite_mobilenet_v2``modelisusedforobjectdetection.Themodel
wastrainedon`CommonObjectsinContext
(COCO)<https://cocodataset.org/#home>`__datasetversionwith91
categoriesofobject.Fordetails,seethe
`paper<https://arxiv.org/abs/1801.04381>`__.

DownloadandunpacktheModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Usethe``download_file``functionfromthe``notebook_utils``to
downloadanarchivewiththemodel.Itautomaticallycreatesadirectory
structureanddownloadstheselectedmodel.Thisstepisskippedifthe
packageisalreadydownloaded.

..code::ipython3

importtarfile
frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

#Adirectorywherethemodelwillbedownloaded.
base_model_dir=Path("./model").expanduser()

model_name="ssdlite_mobilenet_v2"
archive_name=Path(f"{model_name}_coco_2018_05_09.tar.gz")

#Downloadthearchive
downloaded_model_path=base_model_dir/archive_name
ifnotdownloaded_model_path.exists():
model_url=f"http://download.tensorflow.org/models/object_detection/{archive_name}"
download_file(model_url,downloaded_model_path.name,downloaded_model_path.parent)

#Unpackthemodel
tf_model_path=base_model_dir/archive_name.with_suffix("").stem/"frozen_inference_graph.pb"
ifnottf_model_path.exists():
withtarfile.open(downloaded_model_path)asfile:
file.extractall(base_model_dir)



..parsed-literal::

model/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz:0%||0.00/48.7M[00:00<?,?B/s]


..parsed-literal::

IOPubmessagerateexceeded.
Thenotebookserverwilltemporarilystopsendingoutput
totheclientinordertoavoidcrashingit.
Tochangethislimit,settheconfigvariable
`--NotebookApp.iopub_msg_rate_limit`.

Currentvalues:
NotebookApp.iopub_msg_rate_limit=1000.0(msgs/sec)
NotebookApp.rate_limit_window=3.0(secs)



ConverttheModeltoOpenVINOIRformat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ToconvertthemodeltoOpenVINOIRwith``FP16``precision,usemodel
conversionAPI.Themodelsaresavedtothe``model/ir_model/``
directory.Formoredetailsaboutmodelconversion,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

fromopenvino.tools.mo.frontimporttfasov_tf_front

precision="FP16"

#Theoutputpathfortheconversion.
model_path=base_model_dir/"ir_model"/f"{model_name}_{precision.lower()}.xml"

trans_config_path=Path(ov_tf_front.__file__).parent/"ssd_v2_support.json"
pipeline_config=base_model_dir/archive_name.with_suffix("").stem/"pipeline.config"

model=None
ifnotmodel_path.exists():
model=ov.tools.mo.convert_model(
input_model=tf_model_path,
input_shape=[1,300,300,3],
layout="NHWC",
transformations_config=trans_config_path,
tensorflow_object_detection_api_pipeline_config=pipeline_config,
reverse_input_channels=True,
)
ov.save_model(model,model_path,compress_to_fp16=(precision=="FP16"))
print("IRmodelsavedto{}".format(model_path))
else:
print("ReadIRmodelfrom{}".format(model_path))
model=core.read_model(model_path)


..parsed-literal::

[WARNING]ThePreprocessorblockhasbeenremoved.Onlynodesperformingmeanvaluesubtractionandscaling(ifapplicable)arekept.


..parsed-literal::

IRmodelsavedtomodel/ir_model/ssdlite_mobilenet_v2_fp16.xml


CompilewithDefaultConfiguration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Whenthemodelisready,firstweneedtoreadit,usingthe
``read_model``method.Then,wecanusethe``compile_model``methodand
specifythenameofthedevicewewanttocompilethemodelon,inthis
case,“GPU”.

..code::ipython3

compiled_model=core.compile_model(model,device)

IfyouhavemultipleGPUsinthesystem,youcanspecifywhichoneto
usebyusing“GPU.0”,“GPU.1”,etc.Anyofthedevicenamesreturnedby
the``available_devices``methodarevaliddevicespecifiers.Youmay
alsouse“AUTO”,whichwillautomaticallyselectthebestdevicefor
inference(whichisoftentheGPU).TolearnmoreaboutAUTOplugin,
visitthe`AutomaticDevice
Selection<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
pageaswellasthe`AUTOdevice
tutorial<auto-device-with-output.html>`__.

ReduceCompileTimethroughModelCaching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Dependingonthemodelused,device-specificoptimizationsandnetwork
compilationscancausethecompilesteptobetime-consuming,especially
withlargermodels,whichmayleadtobaduserexperienceinthe
application,inwhichtheyareused.Tosolvethis,OpenVINOcancache
themodelonceitiscompiledonsupporteddevicesandreuseitinlater
``compile_model``callsbysimplysettingacachefolderbeforehand.For
instance,tocachethesamemodelwecompiledabove,wecandothe
following:

..code::ipython3

importtime
frompathlibimportPath

#Createcachefolder
cache_folder=Path("cache")
cache_folder.mkdir(exist_ok=True)

start=time.time()
core=ov.Core()

#Setcachefolder
core.set_property({"CACHE_DIR":cache_folder})

#Compilethemodelasbefore
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cacheenabled(firsttime)-compiletime:{time.time()-start}s")


..parsed-literal::

Cacheenabled(firsttime)-compiletime:1.692436695098877s


Togetanideaoftheeffectthatcachingcanhave,wecanmeasurethe
compiletimeswithcachingenabledanddisabledasfollows:

..code::ipython3

start=time.time()
core=ov.Core()
core.set_property({"CACHE_DIR":"cache"})
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cacheenabled-compiletime:{time.time()-start}s")

start=time.time()
core=ov.Core()
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cachedisabled-compiletime:{time.time()-start}s")


..parsed-literal::

Cacheenabled-compiletime:0.26888394355773926s
Cachedisabled-compiletime:1.982884168624878s


Theactualtimeimprovementswilldependontheenvironmentaswellas
themodelbeingusedbutitisdefinitelysomethingtoconsiderwhen
optimizinganapplication.Toreadmoreaboutthis,seethe`Model
Caching<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__
docs.

ThroughputandLatencyPerformanceHints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tosimplifydeviceandpipelineconfiguration,OpenVINOprovides
high-levelperformancehintsthatautomaticallysetthebatchsizeand
numberofparallelthreadstouseforinference.The“LATENCY”
performancehintoptimizesforfastinferencetimeswhilethe
“THROUGHPUT”performancehintoptimizesforhighoverallbandwidthor
FPS.

Tousethe“LATENCY”performancehint,add
``{"PERFORMANCE_HINT":"LATENCY"}``whencompilingthemodelasshown
below.ForGPUs,thisautomaticallyminimizesthebatchsizeandnumber
ofparallelstreamssuchthatallofthecomputeresourcescanfocuson
completingasingleinferenceasfastaspossible.

..code::ipython3

compiled_model=core.compile_model(model,device,{"PERFORMANCE_HINT":"LATENCY"})

Tousethe“THROUGHPUT”performancehint,add
``{"PERFORMANCE_HINT":"THROUGHPUT"}``whencompilingthemodel.For
GPUs,thiscreatesmultipleprocessingstreamstoefficientlyutilize
alltheexecutioncoresandoptimizesthebatchsizetofillthe
availablememory.

..code::ipython3

compiled_model=core.compile_model(model,device,{"PERFORMANCE_HINT":"THROUGHPUT"})

UsingMultipleGPUswithMulti-DeviceandCumulativeThroughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thelatencyandthroughputhintsmentionedabovearegreatandcanmake
adifferencewhenusedadequatelybuttheyusuallyusejustonedevice,
eitherduetothe`AUTO
plugin<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html#how-auto-works>`__
orbymanualspecificationofthedevicenameasabove.Whenwehave
multipledevices,suchasanintegratedanddiscreteGPU,wemayuse
bothatthesametimetoimprovetheutilizationoftheresources.In
ordertodothis,OpenVINOprovidesavirtualdevicecalled
`MULTI<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html>`__,
whichisjustacombinationoftheexistentdevicesthatknowshowto
splitinferenceworkbetweenthem,leveragingthecapabilitiesofeach
device.

Asanexample,ifwewanttousebothintegratedanddiscreteGPUsand
theCPUatthesametime,wecancompilethemodelasfollows:

``compiled_model=core.compile_model(model=model,device_name="MULTI:GPU.1,GPU.0,CPU")``

NotethatwealwaysneedtoexplicitlyspecifythedevicelistforMULTI
towork,otherwiseMULTIdoesnotknowwhichdevicesareavailablefor
inference.However,thisisnottheonlywaytousemultipledevicesin
OpenVINO.Thereisanotherperformancehintcalled
“CUMULATIVE_THROUGHPUT”thatworkssimilartoMULTI,exceptitusesthe
devicesautomaticallyselectedbyAUTO.Thisway,wedonotneedto
manuallyspecifydevicestouse.Belowisanexampleshowinghowtouse
“CUMULATIVE_THROUGHPUT”,equivalenttotheMULTIone:

``compiled_model=core.compile_model(model=model,device_name="AUTO",config={"PERFORMANCE_HINT":"CUMULATIVE_THROUGHPUT"})``

**Important**:**The“THROUGHPUT”,“MULTI”,and
“CUMULATIVE_THROUGHPUT”modesareonlyapplicabletoasynchronous
inferencingpipelines.Theexampleattheendofthisarticleshows
howtosetupanasynchronouspipelinethattakesadvantageof
parallelismtoincreasethroughput.**Tolearnmore,see
`Asynchronous
Inferencing<https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request.html>`__
inOpenVINOaswellasthe`AsynchronousInference
notebook<async-api-with-output.html>`__.

PerformanceComparisonwithbenchmark_app
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Givenallthedifferentoptionsavailablewhencompilingamodel,itmay
bedifficulttoknowwhichsettingsworkbestforacertainapplication.
Thankfully,OpenVINOprovides``benchmark_app``-aperformance
benchmarkingtool.

Thebasicsyntaxof``benchmark_app``isasfollows:

``benchmark_app-mPATH_TO_MODEL-dTARGET_DEVICE-hint{throughput,cumulative_throughput,latency,none}``

where``TARGET_DEVICE``isanydeviceshownbythe``available_devices``
methodaswellastheMULTIandAUTOdeviceswesawpreviously,andthe
valueofhintshouldbeoneofthevaluesbetweenbrackets.

Notethatbenchmark_apponlyrequiresthemodelpathtorunbutboththe
deviceandhintargumentswillbeusefultous.Formoreadvanced
usages,thetoolitselfhasotheroptionsthatcanbecheckedbyrunning
``benchmark_app-h``orreadingthe
`docs<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
Thefollowingexampleshowshowtobenchmarkasimplemodel,usingaGPU
withalatencyfocus:

..code::ipython3

!benchmark_app-m{model_path}-dGPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook14.02ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1932.50ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:frozen_inference_graph
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERF_COUNT:False
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]GPU_HOST_TASK_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_THROTTLE:Priority.MEDIUM
[INFO]GPU_ENABLE_LOOP_UNROLLING:True
[INFO]CACHE_DIR:
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]COMPILATION_NUM_THREADS:20
[INFO]NUM_STREAMS:1
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]INFERENCE_PRECISION_HINT:<Type:'undefined'>
[INFO]DEVICE_ID:0
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'image_tensor'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'image_tensor'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook6.17ms
[Step11/11]Dumpingstatisticsreport
[INFO]Count:12710iterations
[INFO]Duration:60006.58ms
[INFO]Latency:
[INFO]Median:4.52ms
[INFO]Average:4.57ms
[INFO]Min:3.13ms
[INFO]Max:17.62ms
[INFO]Throughput:211.81FPS


Forcompleteness,letuslistheresomeofthecomparisonswemaywant
todobyvaryingthedeviceandhintused.Notethattheactual
performancemaydependonthehardwareused.Generally,weshouldexpect
GPUtobebetterthanCPU,whereasmultipleGPUsshouldbebetterthana
singleGPUaslongasthereisenoughworkforeachofthem.

CPUvsGPUwithLatencyHint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m{model_path}-dCPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook30.38ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook127.72ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:frozen_inference_graph
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]NUM_STREAMS:1
[INFO]AFFINITY:Affinity.CORE
[INFO]INFERENCE_NUM_THREADS:14
[INFO]PERF_COUNT:False
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'image_tensor'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'image_tensor'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook4.42ms
[Step11/11]Dumpingstatisticsreport
[INFO]Count:15304iterations
[INFO]Duration:60005.72ms
[INFO]Latency:
[INFO]Median:3.87ms
[INFO]Average:3.88ms
[INFO]Min:3.49ms
[INFO]Max:5.95ms
[INFO]Throughput:255.04FPS


..code::ipython3

!benchmark_app-m{model_path}-dGPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook14.65ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2254.81ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:frozen_inference_graph
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERF_COUNT:False
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]GPU_HOST_TASK_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_THROTTLE:Priority.MEDIUM
[INFO]GPU_ENABLE_LOOP_UNROLLING:True
[INFO]CACHE_DIR:
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]COMPILATION_NUM_THREADS:20
[INFO]NUM_STREAMS:1
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]INFERENCE_PRECISION_HINT:<Type:'undefined'>
[INFO]DEVICE_ID:0
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'image_tensor'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'image_tensor'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook8.79ms
[Step11/11]Dumpingstatisticsreport
[INFO]Count:11354iterations
[INFO]Duration:60007.21ms
[INFO]Latency:
[INFO]Median:4.57ms
[INFO]Average:5.16ms
[INFO]Min:3.18ms
[INFO]Max:34.87ms
[INFO]Throughput:189.21FPS


CPUvsGPUwithThroughputHint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m{model_path}-dCPU-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook29.56ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor:0,image_tensor(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor:0,image_tensor(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook158.91ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:frozen_inference_graph
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:5
[INFO]NUM_STREAMS:5
[INFO]AFFINITY:Affinity.CORE
[INFO]INFERENCE_NUM_THREADS:20
[INFO]PERF_COUNT:False
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'image_tensor'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'image_tensor'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,5inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook8.15ms
[Step11/11]Dumpingstatisticsreport
[INFO]Count:25240iterations
[INFO]Duration:60010.99ms
[INFO]Latency:
[INFO]Median:10.16ms
[INFO]Average:11.84ms
[INFO]Min:7.96ms
[INFO]Max:37.53ms
[INFO]Throughput:420.59FPS


..code::ipython3

!benchmark_app-m{model_path}-dGPU-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook15.45ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2249.04ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:frozen_inference_graph
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:4
[INFO]PERF_COUNT:False
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]GPU_HOST_TASK_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_PRIORITY:Priority.MEDIUM
[INFO]GPU_QUEUE_THROTTLE:Priority.MEDIUM
[INFO]GPU_ENABLE_LOOP_UNROLLING:True
[INFO]CACHE_DIR:
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]COMPILATION_NUM_THREADS:20
[INFO]NUM_STREAMS:2
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]INFERENCE_PRECISION_HINT:<Type:'undefined'>
[INFO]DEVICE_ID:0
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'image_tensor'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'image_tensor'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,4inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook9.17ms
[Step11/11]Dumpingstatisticsreport
[INFO]Count:19588iterations
[INFO]Duration:60023.47ms
[INFO]Latency:
[INFO]Median:11.31ms
[INFO]Average:12.15ms
[INFO]Min:9.26ms
[INFO]Max:36.04ms
[INFO]Throughput:326.34FPS


SingleGPUvsMultipleGPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m{model_path}-dGPU.1-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]DeviceGPU.1doesnotsupportperformancehintproperty(-hint).
[ERROR]Configfordevicewith1IDisnotregisteredinGPUplugin
Traceback(mostrecentcalllast):
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py",line329,inmain
benchmark.set_config(config)
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/benchmark.py",line57,inset_config
self.core.set_property(device,config[device])
RuntimeError:Configfordevicewith1IDisnotregisteredinGPUplugin


..code::ipython3

!benchmark_app-m{model_path}-dAUTO:GPU.1,GPU.0-hintcumulative_throughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]DeviceGPU.1doesnotsupportperformancehintproperty(-hint).
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook26.66ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor,image_tensor:0(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[ERROR]Configfordevicewith1IDisnotregisteredinGPUplugin
Traceback(mostrecentcalllast):
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py",line414,inmain
compiled_model=benchmark.core.compile_model(model,benchmark.device)
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/runtime/ie_api.py",line399,incompile_model
super().compile_model(model,device_name,{}ifconfigisNoneelseconfig),
RuntimeError:Configfordevicewith1IDisnotregisteredinGPUplugin


..code::ipython3

!benchmark_app-m{model_path}-dMULTI:GPU.1,GPU.0-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]Deviceinfo:
[INFO]GPU
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]MULTI
[INFO]Build.................................2022.3.0-9052-9752fafe8eb-releases/2022/3
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]DeviceGPU.1doesnotsupportperformancehintproperty(-hint).
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook14.84ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]image_tensor:0,image_tensor(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]image_tensor:0,image_tensor(node:image_tensor):u8/[N,H,W,C]/[1,300,300,3]
[INFO]Modeloutputs:
[INFO]detection_boxes:0(node:DetectionOutput):f32/[...]/[1,1,100,7]
[Step7/11]Loadingthemodeltothedevice
[ERROR]Configfordevicewith1IDisnotregisteredinGPUplugin
Traceback(mostrecentcalllast):
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py",line414,inmain
compiled_model=benchmark.core.compile_model(model,benchmark.device)
File"/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/runtime/ie_api.py",line399,incompile_model
super().compile_model(model,device_name,{}ifconfigisNoneelseconfig),
RuntimeError:Configfordevicewith1IDisnotregisteredinGPUplugin


BasicApplicationUsingGPUs
----------------------------

`backtotop⬆️<#table-of-contents>`__

Wewillnowshowanend-to-endobjectdetectionexampleusingGPUsin
OpenVINO.TheapplicationcompilesamodelonGPUwiththe“THROUGHPUT”
hint,thenloadsavideoandpreprocesseseveryframetoconvertthemto
theshapeexpectedbythemodel.Oncetheframesareloaded,itsetsup
anasynchronouspipeline,performsinferenceandsavesthedetections
foundineachframe.Thedetectionsarethendrawnontheir
correspondingframeandsavedasavideo,whichisdisplayedattheend
oftheapplication.

ImportNecessaryPackages
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtime
frompathlibimportPath

importcv2
importnumpyasnp
fromIPython.displayimportVideo
importopenvinoasov

#InstantiateOpenVINORuntime
core=ov.Core()
core.available_devices




..parsed-literal::

['CPU','GPU']



CompiletheModel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#ReadmodelandcompileitonGPUinTHROUGHPUTmode
model=core.read_model(model=model_path)
device_name="GPU"
compiled_model=core.compile_model(model=model,device_name=device_name,config={"PERFORMANCE_HINT":"THROUGHPUT"})

#Gettheinputandoutputnodes
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)

#Gettheinputsize
num,height,width,channels=input_layer.shape
print("Modelinputshape:",num,height,width,channels)


..parsed-literal::

Modelinputshape:13003003


LoadandPreprocessVideoFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Loadvideo
video_file="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
video=cv2.VideoCapture(video_file)
framebuf=[]

#Gothrougheveryframeofvideoandresizeit
print("Loadingvideo...")
whilevideo.isOpened():
ret,frame=video.read()
ifnotret:
print("Videoloaded!")
video.release()
break

#Preprocessframes-convertthemtoshapeexpectedbymodel
input_frame=cv2.resize(src=frame,dsize=(width,height),interpolation=cv2.INTER_AREA)
input_frame=np.expand_dims(input_frame,axis=0)

#Appendframetoframebuffer
framebuf.append(input_frame)


print("Frameshape:",framebuf[0].shape)
print("Numberofframes:",len(framebuf))

#Showoriginalvideofile
#Ifthevideodoesnotdisplaycorrectlyinsidethenotebook,pleaseopenitwithyourfavoritemediaplayer
Video(video_file)


..parsed-literal::

Loadingvideo...
Videoloaded!
Frameshape:(1,300,300,3)
Numberofframes:288


DefineModelOutputClasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Definethemodel'slabelmap(thismodelusesCOCOclasses)
classes=[
"background",
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
"streetsign",
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
"hat",
"backpack",
"umbrella",
"shoe",
"eyeglasses",
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
"plate",
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
"mirror",
"diningtable",
"window",
"desk",
"toilet",
"door",
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
"blender",
"book",
"clock",
"vase",
"scissors",
"teddybear",
"hairdrier",
"toothbrush",
"hairbrush",
]

SetupAsynchronousPipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

CallbackDefinition
^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Defineacallbackfunctionthatrunseverytimetheasynchronouspipelinecompletesinferenceonaframe
defcompletion_callback(infer_request:ov.InferRequest,frame_id:int)->None:
globalframe_number
stop_time=time.time()
frame_number+=1

predictions=next(iter(infer_request.results.values()))
results[frame_id]=predictions[:10]#Grabfirst10predictionsforthisframe

total_time=stop_time-start_time
frame_fps[frame_id]=frame_number/total_time

CreateAsyncPipeline
^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Createasynchronousinferencequeuewithoptimalnumberofinferrequests
infer_queue=ov.AsyncInferQueue(compiled_model)
infer_queue.set_callback(completion_callback)

PerformInference
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Performinferenceoneveryframeintheframebuffer
results={}
frame_fps={}
frame_number=0
start_time=time.time()
fori,input_frameinenumerate(framebuf):
infer_queue.start_async({0:input_frame},i)

infer_queue.wait_all()#WaituntilallinferencerequestsintheAsyncInferQueuearecompleted
stop_time=time.time()

#CalculatetotalinferencetimeandFPS
total_time=stop_time-start_time
fps=len(framebuf)/total_time
time_per_frame=1/fps
print(f"Totaltimetoinferallframes:{total_time:.3f}s")
print(f"Timeperframe:{time_per_frame:.6f}s({fps:.3f}FPS)")


..parsed-literal::

Totaltimetoinferallframes:1.366s
Timeperframe:0.004744s(210.774FPS)


ProcessResults
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Setminimumdetectionthreshold
min_thresh=0.6

#Loadvideo
video=cv2.VideoCapture(video_file)

#Getvideoparameters
frame_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(video.get(cv2.CAP_PROP_FPS))
fourcc=int(video.get(cv2.CAP_PROP_FOURCC))

#CreatefolderandVideoWritertosaveoutputvideo
Path("./output").mkdir(exist_ok=True)
output=cv2.VideoWriter("output/output.mp4",fourcc,fps,(frame_width,frame_height))

#Drawdetectionresultsoneveryframeofvideoandsaveasanewvideofile
whilevideo.isOpened():
current_frame=int(video.get(cv2.CAP_PROP_POS_FRAMES))
ret,frame=video.read()
ifnotret:
print("Videoloaded!")
output.release()
video.release()
break

#Drawinfoatthetopleftsuchascurrentfps,thedevicesandtheperformancehintbeingused
cv2.putText(
frame,
f"fps{str(round(frame_fps[current_frame],2))}",
(5,20),
cv2.FONT_ITALIC,
0.6,
(0,0,0),
1,
cv2.LINE_AA,
)
cv2.putText(
frame,
f"device{device_name}",
(5,40),
cv2.FONT_ITALIC,
0.6,
(0,0,0),
1,
cv2.LINE_AA,
)
cv2.putText(
frame,
f"hint{compiled_model.get_property('PERFORMANCE_HINT')}",
(5,60),
cv2.FONT_ITALIC,
0.6,
(0,0,0),
1,
cv2.LINE_AA,
)

#predictioncontains[image_id,label,conf,x_min,y_min,x_max,y_max]accordingtomodel
forpredictioninnp.squeeze(results[current_frame]):
ifprediction[2]>min_thresh:
x_min=int(prediction[3]*frame_width)
y_min=int(prediction[4]*frame_height)
x_max=int(prediction[5]*frame_width)
y_max=int(prediction[6]*frame_height)
label=classes[int(prediction[1])]

#Drawaboundingboxwithitslabelaboveit
cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),1,cv2.LINE_AA)
cv2.putText(
frame,
label,
(x_min,y_min-10),
cv2.FONT_ITALIC,
1,
(255,0,0),
1,
cv2.LINE_AA,
)

output.write(frame)

#Showoutputvideofile
#Ifthevideodoesnotdisplaycorrectlyinsidethenotebook,pleaseopenitwithyourfavoritemediaplayer
Video("output/output.mp4",width=800,embed=True)


..parsed-literal::

Videoloaded!




..raw::html

<videocontrolswidth="800">
<sourcesrc="data:None;base64,output/output.mp4"type="None">
Yourbrowserdoesnotsupportthevideotag.
</video>



Conclusion
----------

`backtotop⬆️<#table-of-contents>`__

ThistutorialdemonstrateshoweasyitistouseoneormoreGPUsin
OpenVINO,checktheirproperties,andeventailorthemodelperformance
throughthedifferentperformancehints.Italsoprovidesawalk-through
ofabasicobjectdetectionapplicationthatusesaGPUanddisplaysthe
detectedboundingboxes.

Toreadmoreaboutanyofthesetopics,feelfreetovisittheir
correspondingdocumentation:

-`GPU
Plugin<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>`__
-`AUTO
Plugin<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
-`Model
Caching<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__
-`MULTIDevice
Mode<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html>`__
-`QueryDevice
Properties<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
-`ConfigurationsforGPUswith
OpenVINO<https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
-`BenchmarkPython
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-`Asynchronous
Inferencing<https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request.html>`__

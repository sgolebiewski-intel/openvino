HelloNPU
=========

WorkingwithNPUinOpenVINO™
-----------------------------

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Introduction<#introduction>`__

-`Installrequiredpackages<#install-required-packages>`__

-`CheckingNPUwithQueryDevice<#checking-npu-with-query-device>`__

-`ListtheNPUwith
core.available_devices<#list-the-npu-with-core-available_devices>`__
-`CheckPropertieswith
core.get_property<#check-properties-with-core-get_property>`__
-`BriefDescriptionsofKey
Properties<#brief-descriptions-of-key-properties>`__

-`CompilingaModelonNPU<#compiling-a-model-on-npu>`__

-`DownloadandConvertaModel<#download-and-convert-a-model>`__

-`DownloadtheModel<#download-the-model>`__
-`ConverttheModeltoOpenVINOIR
format<#convert-the-model-to-openvino-ir-format>`__

-`CompilewithDefault
Configuration<#compile-with-default-configuration>`__
-`ReduceCompileTimethroughModel
Caching<#reduce-compile-time-through-model-caching>`__

-`UMDModelCaching<#umd-model-caching>`__
-`OpenVINOModelCaching<#openvino-model-caching>`__

-`ThroughputandLatencyPerformance
Hints<#throughput-and-latency-performance-hints>`__

-`PerformanceComparisonwith
benchmark_app<#performance-comparison-with-benchmark_app>`__

-`NPUvsCPUwithLatencyHint<#npu-vs-cpu-with-latency-hint>`__

-`EffectsofUMDModel
Caching<#effects-of-umd-model-caching>`__

-`NPUvsCPUwithThroughput
Hint<#npu-vs-cpu-with-throughput-hint>`__

-`Limitations<#limitations>`__
-`Conclusion<#conclusion>`__

Thistutorialprovidesahigh-leveloverviewofworkingwiththeNPU
device**Intel(R)AIBoost**(introducedwiththeIntel®Core™Ultra
generationofCPUs)inOpenVINO.Itexplainssomeofthekeyproperties
oftheNPUandshowshowtocompileamodelonNPUwithperformance
hints.

Thistutorialalsoshowsexamplecommandsforbenchmark_appthatcanbe
runtocompareNPUperformancewithCPUindifferentconfigurations.

Introduction
------------

`backtotop⬆️<#table-of-contents>`__

TheNeuralProcessingUnit(NPU)isalowpowerhardwaresolutionwhich
enablesyoutooffloadcertainneuralnetworkcomputationtasksfrom
otherdevices,formorestreamlinedresourcemanagement.

NotethattheNPUpluginisincludedinPIPinstallationofOpenVINO™
andyouneedto`installaproperNPU
driver<https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-npu.html>`__
touseitsuccessfully.

|**SupportedPlatforms**:
|Host:Intel®Core™Ultra
|NPUdevice:NPU3720
|OS:Ubuntu22.04(withLinuxKernel6.6+),MSWindows11(both64-bit)

TolearnmoreabouttheNPUDevice,seethe
`page<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html>`__.

Installrequiredpackages
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2024.1.0"torchtorchvision--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


CheckingNPUwithQueryDevice
------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthissection,wewillseehowtolisttheavailableNPUandcheckits
properties.Someofthekeypropertieswillbedefined.

ListtheNPUwithcore.available_devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINORuntimeprovidesthe``available_devices``methodforchecking
whichdevicesareavailableforinference.Thefollowingcodewill
outputalistacompatibleOpenVINOdevices,inwhichIntelNPUshould
appear(ensurethatthedriverisinstalledsuccessfully).

..code::ipython3

importopenvinoasov

core=ov.Core()
core.available_devices




..parsed-literal::

['CPU','GPU','NPU']



CheckPropertieswithcore.get_property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TogetinformationabouttheNPU,wecanusedeviceproperties.In
OpenVINO,deviceshavepropertiesthatdescribetheircharacteristics
andconfigurations.Eachpropertyhasanameandassociatedvaluethat
canbequeriedwiththe``get_property``method.

Togetthevalueofaproperty,suchasthedevicename,wecanusethe
``get_property``methodasfollows:

..code::ipython3

device="NPU"

core.get_property(device,"FULL_DEVICE_NAME")




..parsed-literal::

'Intel(R)AIBoost'



Eachdevicealsohasaspecificpropertycalled
``SUPPORTED_PROPERTIES``,thatenablesviewingalltheavailable
propertiesinthedevice.Wecancheckthevalueforeachpropertyby
simplyloopingthroughthedictionaryreturnedby
``core.get_property("NPU","SUPPORTED_PROPERTIES")``andthenquerying
forthatproperty.

..code::ipython3

print(f"{device}SUPPORTED_PROPERTIES:\n")
supported_properties=core.get_property(device,"SUPPORTED_PROPERTIES")
indent=len(max(supported_properties,key=len))

forproperty_keyinsupported_properties:
ifproperty_keynotin("SUPPORTED_METRICS","SUPPORTED_CONFIG_KEYS","SUPPORTED_PROPERTIES"):
try:
property_val=core.get_property(device,property_key)
exceptTypeError:
property_val="UNSUPPORTEDTYPE"
print(f"{property_key:<{indent}}:{property_val}")

BriefDescriptionsofKeyProperties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Eachdevicehasseveralpropertiesasseeninthelastcommand.Someof
thekeypropertiesare:-``FULL_DEVICE_NAME``-Theproductnameofthe
NPU.-``PERFORMANCE_HINT``-Ahigh-levelwaytotunethedevicefora
specificperformancemetric,suchaslatencyorthroughput,without
worryingaboutdevice-specificsettings.-``CACHE_DIR``-Thedirectory
wheretheOpenVINOmodelcachedataisstoredtospeedupthe
compilationtime.-``OPTIMIZATION_CAPABILITIES``-Themodeldatatypes
(INT8,FP16,FP32,etc)thataresupportedbythisNPU.

Tolearnmoreaboutdevicesandproperties,seethe`QueryDevice
Properties<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
page.

CompilingaModelonNPU
------------------------

`backtotop⬆️<#table-of-contents>`__

Now,weknowtheNPUpresentinthesystemandwehavecheckedits
properties.Wecaneasilyuseitforcompilingandrunningmodelswith
OpenVINONPUplugin.

DownloadandConvertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thistutorialusesthe``resnet50``model.The``resnet50``modelis
usedforimageclassificationtasks.Themodelwastrainedon
`ImageNet<https://www.image-net.org/index.php>`__datasetwhich
containsoveramillionimagescategorizedinto1000classes.Toread
moreaboutresnet50,seethe
`paper<https://ieeexplore.ieee.org/document/7780459>`__.

DownloadtheModel
^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Fetch`ResNet50
CV<https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
Classificationmodelfromtorchvision.

..code::ipython3

frompathlibimportPath

#createadirectoryforresnetmodelfile
MODEL_DIRECTORY_PATH=Path("model")
MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

model_name="resnet50"

..code::ipython3

fromtorchvision.modelsimportresnet50,ResNet50_Weights

#createmodelobject
pytorch_model=resnet50(weights=ResNet50_Weights.DEFAULT)

#switchmodelfromtrainingtoinferencemode
pytorch_model.eval();

ConverttheModeltoOpenVINOIRformat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ToconvertthisPytorchmodeltoOpenVINOIRwith``FP16``precision,
usemodelconversionAPI.Themodelsaresavedtothe
``model/ir_model/``directory.Formoredetailsaboutmodelconversion,
seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

precision="FP16"

model_path=MODEL_DIRECTORY_PATH/"ir_model"/f"{model_name}_{precision.lower()}.xml"

model=None
ifnotmodel_path.exists():
model=ov.convert_model(pytorch_model,input=[[1,3,224,224]])
ov.save_model(model,model_path,compress_to_fp16=(precision=="FP16"))
print("IRmodelsavedto{}".format(model_path))
else:
print("ReadIRmodelfrom{}".format(model_path))
model=core.read_model(model_path)


..parsed-literal::

ReadIRmodelfrommodel\ir_model\resnet50_fp16.xml


**Note:**NPUalsosupports``INT8``quantizedmodels.

CompilewithDefaultConfiguration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Whenthemodelisready,firstweneedtoreadit,usingthe
``read_model``method.Then,wecanusethe``compile_model``methodand
specifythenameofthedevicewewanttocompilethemodelon,inthis
case,“NPU”.

..code::ipython3

compiled_model=core.compile_model(model,device)

ReduceCompileTimethroughModelCaching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Dependingonthemodelused,device-specificoptimizationsandnetwork
compilationscancausethecompilesteptobetime-consuming,especially
withlargermodels,whichmayleadtobaduserexperienceinthe
application.Tosolvethis**ModelCaching**canbeused.

ModelCachinghelpsreduceapplicationstartupdelaysbyexportingand
reusingthecompiledmodelautomatically.Thefollowingtwo
compilation-relatedmetricsarecrucialinthisarea:

-**First-EverInferenceLatency(FEIL)**:
Measuresallstepsrequiredtocompileandexecuteamodelonthe
deviceforthefirsttime.Itincludesmodelcompilationtime,the
timerequiredtoloadandinitializethemodelonthedeviceandthe
firstinferenceexecution.
-**FirstInferenceLatency(FIL)**:
Measuresthetimerequiredtoloadandinitializethepre-compiled
modelonthedeviceandthefirstinferenceexecution.

InNPU,UMDmodelcachingisasolutionenabledbydefaultbythe
driver.Itimprovestimetofirstinference(FIL)bystoringthemodel
inthecacheaftercompilation(includedinFEIL).LearnmoreaboutUMD
Caching
`here<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#umd-dynamic-model-caching>`__.
Duetothiscaching,ittakeslessertimetoloadthemodelafterfirst
compilation.

|YoucanalsouseOpenVINOModelCaching,whichisacommonmechanism
forallOpenVINOdevicepluginsandcanbeenabledbysettingthe
``cache_dir``property.
|ByenablingOpenVINOModelCaching,theUMDcachingisautomatically
bypassedbytheNPUplugin,whichmeansthemodelwillonlybestored
intheOpenVINOcacheaftercompilation.Whenacachehitoccursfor
subsequentcompilationrequests,thepluginwillimportthemodel
insteadofrecompilingit.

UMDModelCaching
^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ToseehowUMDcachingseethefollowingexample:

..code::ipython3

importtime
frompathlibimportPath

start=time.time()
core=ov.Core()

#Compilethemodelasbefore
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"UMDCaching(firsttime)-compiletime:{time.time()-start}s")


..parsed-literal::

UMDCaching(firsttime)-compiletime:3.2854952812194824s


..code::ipython3

start=time.time()
core=ov.Core()

#CompilethemodelonceagaintoseeUMDCaching
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"UMDCaching-compiletime:{time.time()-start}s")


..parsed-literal::

UMDCaching-compiletime:2.269814968109131s


OpenVINOModelCaching
^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

TogetanideaofOpenVINOmodelcaching,wecanusetheOpenVINOcache
asfollow

..code::ipython3

#Createcachefolder
cache_folder=Path("cache")
cache_folder.mkdir(exist_ok=True)

start=time.time()
core=ov.Core()

#Setcachefolder
core.set_property({"CACHE_DIR":cache_folder})

#Compilethemodel
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cacheenabled(firsttime)-compiletime:{time.time()-start}s")

start=time.time()
core=ov.Core()

#Setcachefolder
core.set_property({"CACHE_DIR":cache_folder})

#Compilethemodelasbefore
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cacheenabled(secondtime)-compiletime:{time.time()-start}s")


..parsed-literal::

Cacheenabled(firsttime)-compiletime:0.6362860202789307s
Cacheenabled(secondtime)-compiletime:0.3032548427581787s


AndwhentheOpenVINOcacheisdisabled:

..code::ipython3

start=time.time()
core=ov.Core()
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model,device)
print(f"Cachedisabled-compiletime:{time.time()-start}s")


..parsed-literal::

Cachedisabled-compiletime:3.0127954483032227s


Theactualtimeimprovementswilldependontheenvironmentaswellas
themodelbeingusedbutitisdefinitelysomethingtoconsiderwhen
optimizinganapplication.Toreadmoreaboutthis,seethe`Model
Caching
docs<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__.

ThroughputandLatencyPerformanceHints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tosimplifydeviceandpipelineconfiguration,OpenVINOprovides
high-levelperformancehintsthatautomaticallysetthebatchsizeand
numberofparallelthreadsforinference.The“LATENCY”performancehint
optimizesforfastinferencetimeswhilethe“THROUGHPUT”performance
hintoptimizesforhighoverallbandwidthorFPS.

Tousethe“LATENCY”performancehint,add
``{"PERFORMANCE_HINT":"LATENCY"}``whencompilingthemodelasshown
below.ForNPU,thisautomaticallyminimizesthebatchsizeandnumber
ofparallelstreamssuchthatallofthecomputeresourcescanfocuson
completingasingleinferenceasfastaspossible.

..code::ipython3

compiled_model=core.compile_model(model,device,{"PERFORMANCE_HINT":"LATENCY"})

Tousethe“THROUGHPUT”performancehint,add
``{"PERFORMANCE_HINT":"THROUGHPUT"}``whencompilingthemodel.For
NPUs,thiscreatesmultipleprocessingstreamstoefficientlyutilize
alltheexecutioncoresandoptimizesthebatchsizetofillthe
availablememory.

..code::ipython3

compiled_model=core.compile_model(model,device,{"PERFORMANCE_HINT":"THROUGHPUT"})

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

Notethatbenchmark_apponlyrequiresthemodelpathtorunbutboth
deviceandhintargumentswillbeusefultous.Formoreadvanced
usages,thetoolitselfhasotheroptionsthatcanbecheckedbyrunning
``benchmark_app-h``orreadingthe
`docs<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
Thefollowingexampleshowsustobenchmarkasimplemodel,usingaNPU
withlatencyfocus:

``benchmark_app-m{model_path}-dNPU-hintlatency``

|Forcompleteness,letuslistheresomeofthecomparisonswemaywant
todobyvaryingthedeviceandhintused.Notethattheactual
performancemaydependonthehardwareused.Generally,weshould
expectNPUtobebetterthanCPU.
|Pleaserefertothe``benchmark_app``logentriesunder
``[Step11/11]Dumpingstatisticsreport``toobservethedifferences
inlatencyandthroughputbetweentheCPUandNPU..

NPUvsCPUwithLatencyHint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m{model_path}-dCPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook14.00ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook143.22ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model2
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]NUM_STREAMS:1
[INFO]AFFINITY:Affinity.HYBRID_AWARE
[INFO]INFERENCE_NUM_THREADS:12
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:LATENCY
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:False
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:False
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook28.95ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:1612iterations
[INFO]Duration:60039.72ms
[INFO]Latency:
[INFO]Median:39.99ms
[INFO]Average:37.13ms
[INFO]Min:19.13ms
[INFO]Max:71.94ms
[INFO]Throughput:26.85FPS


..code::ipython3

!benchmark_app-m{model_path}-dNPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]Deviceinfo:
[INFO]NPU
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook11.51ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2302.40ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]DEVICE_ID:
[INFO]ENABLE_CPU_PINNING:False
[INFO]EXECUTION_DEVICES:NPU.3720
[INFO]INFERENCE_PRECISION_HINT:<Type:'float16'>
[INFO]INTERNAL_SUPPORTED_PROPERTIES:{'CACHING_PROPERTIES':'RO'}
[INFO]LOADED_FROM_CACHE:False
[INFO]NETWORK_NAME:
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:1
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook7.94ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:NPU.3720
[INFO]Count:17908iterations
[INFO]Duration:60004.49ms
[INFO]Latency:
[INFO]Median:3.29ms
[INFO]Average:3.33ms
[INFO]Min:3.21ms
[INFO]Max:6.90ms
[INFO]Throughput:298.44FPS


EffectsofUMDModelCaching
''''''''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

ToseetheeffectsofUMDModelcaching,wearegoingtorunthe
benchmark_appandseethedifferenceinmodelreadtimeandcompilation
time:

..code::ipython3

!benchmark_app-m{model_path}-dNPU-hintlatency


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]Deviceinfo:
[INFO]NPU
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook11.00ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2157.58ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]DEVICE_ID:
[INFO]ENABLE_CPU_PINNING:False
[INFO]EXECUTION_DEVICES:NPU.3720
[INFO]INFERENCE_PRECISION_HINT:<Type:'float16'>
[INFO]INTERNAL_SUPPORTED_PROPERTIES:{'CACHING_PROPERTIES':'RO'}
[INFO]LOADED_FROM_CACHE:False
[INFO]NETWORK_NAME:
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:1
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,1inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook7.94ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:NPU.3720
[INFO]Count:17894iterations
[INFO]Duration:60004.76ms
[INFO]Latency:
[INFO]Median:3.29ms
[INFO]Average:3.33ms
[INFO]Min:3.21ms
[INFO]Max:14.38ms
[INFO]Throughput:298.21FPS


Asyoucanseefromthelogentries``[Step4/11]Readingmodelfiles``
and``[Step7/11]Loadingthemodeltothedevice``,ittakeslesstime
toreadandcompilethemodelaftertheinitialload.

NPUvsCPUwithThroughputHint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m{model_path}-dCPU-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook12.00ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook177.18ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model2
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:4
[INFO]NUM_STREAMS:4
[INFO]AFFINITY:Affinity.HYBRID_AWARE
[INFO]INFERENCE_NUM_THREADS:16
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:False
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,4inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook31.62ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:3212iterations
[INFO]Duration:60082.26ms
[INFO]Latency:
[INFO]Median:65.28ms
[INFO]Average:74.60ms
[INFO]Min:35.65ms
[INFO]Max:157.31ms
[INFO]Throughput:53.46FPS


..code::ipython3

!benchmark_app-m{model_path}-dNPU-hintthroughput


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]Deviceinfo:
[INFO]NPU
[INFO]Build.................................2024.1.0-14992-621b025bef4
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook11.50ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,224,224]
[INFO]Modeloutputs:
[INFO]x.45(node:aten::linear/Add):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2265.07ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]DEVICE_ID:
[INFO]ENABLE_CPU_PINNING:False
[INFO]EXECUTION_DEVICES:NPU.3720
[INFO]INFERENCE_PRECISION_HINT:<Type:'float16'>
[INFO]INTERNAL_SUPPORTED_PROPERTIES:{'CACHING_PROPERTIES':'RO'}
[INFO]LOADED_FROM_CACHE:False
[INFO]NETWORK_NAME:
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:4
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:1
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,4inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook7.95ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:NPU.3720
[INFO]Count:19080iterations
[INFO]Duration:60024.79ms
[INFO]Latency:
[INFO]Median:12.51ms
[INFO]Average:12.56ms
[INFO]Min:6.92ms
[INFO]Max:25.80ms
[INFO]Throughput:317.87FPS


Limitations
-----------

`backtotop⬆️<#table-of-contents>`__

1.Currently,onlythemodelswithstaticshapesaresupportedonNPU.
2.Ifthepathtothemodelfileincludesnon-Unicodesymbols,suchas
inChinese,themodelcannotbeusedforinferenceonNPU.Itwill
returnanerror.

Conclusion
----------

`backtotop⬆️<#table-of-contents>`__

ThistutorialdemonstrateshoweasyitistouseNPUinOpenVINO,check
itsproperties,andeventailorthemodelperformancethroughthe
differentperformancehints.

DiscoverthepowerofNeuralProcessingUnit(NPU)withOpenVINOthrough
theseinteractiveJupyternotebooks:#####Introduction-
`hello-world<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world>`__:
StartyourOpenVINOjourneybyperforminginferenceonanOpenVINOIR
model.-
`hello-segmentation<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-segmentation>`__:
Diveintoinferencewithasegmentationmodelandexploreimage
segmentationcapabilities.

ModelOptimizationandConversion
'''''''''''''''''''''''''''''''''

-`tflite-to-openvino<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tflite-to-openvino>`__:
LearntheprocessofconvertingTensorFlowLitemodelstoOpenVINOIR
format.
-`yolov7-optimization<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov7-optimization>`__:
OptimizetheYOLOv7modelforenhancedperformanceinOpenVINO.
-`yolov8-optimization<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization>`__:
ConvertandoptimizeYOLOv8modelsforefficientdeploymentwith
OpenVINO.

AdvancedComputerVisionTechniques
'''''''''''''''''''''''''''''''''''

-`vision-background-removal<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal>`__:
Implementadvancedimagesegmentationandbackgroundmanipulation
withU^2-Net.
-`handwritten-ocr<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/handwritten-ocr>`__:
ApplyopticalcharacterrecognitiontohandwrittenChineseand
Japanesetext.
-`vehicle-detection-and-recognition<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vehicle-detection-and-recognition>`__:
Usepre-trainedmodelsforvehicledetectionandrecognitionin
images.
-`vision-image-colorization<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-image-colorization>`__:
Bringblackandwhiteimagestolifebyaddingcolorwithneural
networks.

Real-TimeWebcamApplications
'''''''''''''''''''''''''''''

-`tflite-selfie-segmentation<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tflite-selfie-segmentation>`__:
ApplyTensorFlowLitemodelsforselfiesegmentationandbackground
processing.
-`object-detection-webcam<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/object-detection-webcam>`__:
Experiencereal-timeobjectdetectionusingyourwebcamandOpenVINO.
-`pose-estimation-webcam<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pose-estimation-webcam>`__:
Performhumanposeestimationinreal-timewithwebcamintegration.
-`action-recognition-webcam<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/action-recognition-webcam>`__:
Recognizeandclassifyhumanactionslivewithyourwebcam.
-`style-transfer-webcam<https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/style-transfer-webcam>`__:
Transformyourwebcamfeedwithartisticstylesinreal-timeusing
pre-trainedmodels.
-`3D-pose-estimation-webcam<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pose-estimation-webcam>`__:
Perform3Dmulti-personposeestimationwithOpenVINO.

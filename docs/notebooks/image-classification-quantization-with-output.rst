QuantizationofImageClassificationModels
===========================================

Thistutorialdemonstrateshowtoapply``INT8``quantizationtoImage
Classificationmodelusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__.Itusesthe
MobileNetV2model,trainedonCifar10dataset.Thecodeisdesignedto
beextendabletocustommodelsanddatasets.ThetutorialusesOpenVINO
backendforperformingmodelquantizationinNNCF,ifyouinterestedhow
toapplyquantizationonPyTorchmodel,pleasecheckthis
`tutorial<pytorch-post-training-quantization-nncf-with-output.html>`__.

Thistutorialconsistsofthefollowingsteps:

-Preparethemodelforquantization.
-Defineadataloadingfunctionality.
-Performquantization.
-Compareaccuracyoftheoriginalandquantizedmodels.
-Compareperformanceoftheoriginalandquantizedmodels.
-Compareresultsononepicture.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`PreparetheModel<#prepare-the-model>`__
-`PrepareDataset<#prepare-dataset>`__
-`PerformQuantization<#perform-quantization>`__

-`CreateDatasetforValidation<#create-dataset-for-validation>`__

-`Runnncf.quantizeforGettinganOptimized
Model<#run-nncf-quantize-for-getting-an-optimized-model>`__
-`SerializeanOpenVINOIRmodel<#serialize-an-openvino-ir-model>`__
-`CompareAccuracyoftheOriginalandQuantized
Models<#compare-accuracy-of-the-original-and-quantized-models>`__

-`Selectinferencedevice<#select-inference-device>`__

-`ComparePerformanceoftheOriginalandQuantized
Models<#compare-performance-of-the-original-and-quantized-models>`__
-`Compareresultsonfour
pictures<#compare-results-on-four-pictures>`__

..code::ipython3

importplatform

#Installrequiredpackages
%pipinstall-q"openvino>=2023.1.0""nncf>=2.6.0"torchtorchvisiontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

#Setthedataandmodeldirectories
DATA_DIR=Path("data")
MODEL_DIR=Path("model")
model_repo="pytorch-cifar-models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

PreparetheModel
-----------------

`backtotop⬆️<#table-of-contents>`__

Modelpreparationstagehasthefollowingsteps:

-DownloadaPyTorchmodel
-ConvertmodeltoOpenVINOIntermediateRepresentationformat(IR)
usingmodelconversionPythonAPI
-Serializeconvertedmodelondisk

..code::ipython3

importsys

ifnotPath(model_repo).exists():
!gitclonehttps://github.com/chenyaofo/pytorch-cifar-models.git

sys.path.append(model_repo)


..parsed-literal::

Cloninginto'pytorch-cifar-models'...
remote:Enumeratingobjects:282,done.[K
remote:Countingobjects:100%(281/281),done.[K
remote:Compressingobjects:100%(96/96),done.[K
remote:Total282(delta135),reused269(delta128),pack-reused1[K
Receivingobjects:100%(282/282),9.22MiB|24.58MiB/s,done.
Resolvingdeltas:100%(135/135),done.


..code::ipython3

frompytorch_cifar_modelsimportcifar10_mobilenetv2_x1_0

model=cifar10_mobilenetv2_x1_0(pretrained=True)

OpenVINOsupportsPyTorchmodelsviaconversiontoOpenVINOIntermediate
RepresentationformatusingmodelconversionPythonAPI.
``ov.convert_model``acceptPyTorchmodelinstanceandconvertitinto
``openvino.runtime.Model``representationofmodelinOpenVINO.
Optionally,youmayspecify``example_input``whichservesasahelper
formodeltracingand``input_shape``forconvertingthemodelwith
staticshape.Theconvertedmodelisreadytobeloadedonadevicefor
inferenceandcanbesavedonadiskfornextusageviathe
``save_model``function.MoredetailsaboutmodelconversionPythonAPI
canbefoundonthis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

importopenvinoasov

model.eval()

ov_model=ov.convert_model(model,input=[1,3,32,32])

ov.save_model(ov_model,MODEL_DIR/"mobilenet_v2.xml")

PrepareDataset
---------------

`backtotop⬆️<#table-of-contents>`__

Wewilluse`CIFAR10<https://www.cs.toronto.edu/~kriz/cifar.html>`__
datasetfrom
`torchvision<https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>`__.
Preprocessingformodelobtainedfromtraining
`config<https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf>`__

..code::ipython3

importtorch
fromtorchvisionimporttransforms
fromtorchvision.datasetsimportCIFAR10

transform=transforms.Compose(
[
transforms.ToTensor(),
transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
]
)
dataset=CIFAR10(root=DATA_DIR,train=False,transform=transform,download=True)
val_loader=torch.utils.data.DataLoader(
dataset,
batch_size=1,
shuffle=False,
num_workers=0,
pin_memory=True,
)


..parsed-literal::

Downloadinghttps://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gztodata/cifar-10-python.tar.gz


..parsed-literal::

100%|██████████|170498071/170498071[00:06<00:00,24813999.85it/s]


..parsed-literal::

Extractingdata/cifar-10-python.tar.gztodata


PerformQuantization
--------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
MobileNetV2.Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeanOpenVINOIRmodel,usingthe``openvino.save_model``
function.

CreateDatasetforValidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

NNCFiscompatiblewith``torch.utils.data.DataLoader``interface.For
performingquantizationitshouldbepassedinto``nncf.Dataset``object
withtransformationfunction,whichpreparesinputdatatofitinto
modelduringquantization,inourcase,topickinputtensorfrompair
(inputtensorandlabel)andconvertPyTorchtensortonumpy.

..code::ipython3

importnncf


deftransform_fn(data_item):
image_tensor=data_item[0]
returnimage_tensor.numpy()


quantization_dataset=nncf.Dataset(val_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Runnncf.quantizeforGettinganOptimizedModel
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

``nncf.quantize``functionacceptsmodelandpreparedquantization
datasetforperformingbasicquantization.Optionally,additional
parameterslike``subset_size``,``preset``,``ignored_scope``canbe
providedtoimprovequantizationresultifapplicable.Moredetails
aboutsupportedparameterscanbefoundonthis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters>`__

..code::ipython3

quant_ov_model=nncf.quantize(ov_model,quantization_dataset)


..parsed-literal::

2024-07-1300:36:24.894428:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1300:36:24.926464:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1300:36:25.567707:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



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



SerializeanOpenVINOIRmodel
------------------------------

`backtotop⬆️<#table-of-contents>`__

Similarto``ov.convert_model``,quantizedmodelis``ov.Model``object
whichreadytobeloadedintodeviceandcanbeserializedondiskusing
``ov.save_model``.

..code::ipython3

ov.save_model(quant_ov_model,MODEL_DIR/"quantized_mobilenet_v2.xml")

CompareAccuracyoftheOriginalandQuantizedModels
-----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtqdm.notebookimporttqdm
importnumpyasnp


deftest_accuracy(ov_model,data_loader):
correct=0
total=0
forbatch_imgs,batch_labelsintqdm(data_loader):
result=ov_model(batch_imgs)[0]
top_label=np.argmax(result)
correct+=top_label==batch_labels.numpy()
total+=1
returncorrect/total

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

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



..code::ipython3

core=ov.Core()
compiled_model=core.compile_model(ov_model,device.value)
optimized_compiled_model=core.compile_model(quant_ov_model,device.value)

orig_accuracy=test_accuracy(compiled_model,val_loader)
optimized_accuracy=test_accuracy(optimized_compiled_model,val_loader)



..parsed-literal::

0%||0/10000[00:00<?,?it/s]



..parsed-literal::

0%||0/10000[00:00<?,?it/s]


..code::ipython3

print(f"Accuracyoftheoriginalmodel:{orig_accuracy[0]*100:.2f}%")
print(f"Accuracyoftheoptimizedmodel:{optimized_accuracy[0]*100:.2f}%")


..parsed-literal::

Accuracyoftheoriginalmodel:93.61%
Accuracyoftheoptimizedmodel:93.57%


ComparePerformanceoftheOriginalandQuantizedModels
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Finally,measuretheinferenceperformanceofthe``FP32``and``INT8``
models,using`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-aninferenceperformancemeasurementtoolinOpenVINO.

**NOTE**:Formoreaccurateperformance,itisrecommendedtorun
benchmark_appinaterminal/commandpromptafterclosingother
applications.Run``benchmark_app-mmodel.xml-dCPU``tobenchmark
asyncinferenceonCPUforoneminute.ChangeCPUtoGPUtobenchmark
onGPU.Run``benchmark_app--help``toseeanoverviewofall
command-lineoptions.

..code::ipython3

#InferenceFP16model(OpenVINOIR)
!benchmark_app-m"model/mobilenet_v2.xml"-d$device.value-apiasync-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook9.87ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,32,32]
[INFO]Modeloutputs:
[INFO]x.17(node:aten::linear/Add):f32/[...]/[1,10]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,32,32]
[INFO]Modeloutputs:
[INFO]x.17(node:aten::linear/Add):f32/[...]/[1,10]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook210.66ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model2
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
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model2
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook3.59ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:88356iterations
[INFO]Duration:15003.01ms
[INFO]Latency:
[INFO]Median:1.85ms
[INFO]Average:1.85ms
[INFO]Min:1.20ms
[INFO]Max:9.01ms
[INFO]Throughput:5889.22FPS


..code::ipython3

#InferenceINT8model(OpenVINOIR)
!benchmark_app-m"model/quantized_mobilenet_v2.xml"-d$device.value-apiasync-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook14.98ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,32,32]
[INFO]Modeloutputs:
[INFO]x.17(node:aten::linear/Add):f32/[...]/[1,10]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,32,32]
[INFO]Modeloutputs:
[INFO]x.17(node:aten::linear/Add):f32/[...]/[1,10]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook318.23ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model2
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
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model2
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook1.88ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:166056iterations
[INFO]Duration:15001.00ms
[INFO]Latency:
[INFO]Median:1.01ms
[INFO]Average:1.04ms
[INFO]Min:0.73ms
[INFO]Max:7.22ms
[INFO]Throughput:11069.66FPS


Compareresultsonfourpictures
--------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#DefineallpossiblelabelsfromtheCIFAR10dataset
labels_names=[
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck",
]
all_pictures=[]
all_labels=[]

#Getallpicturesandtheirlabels.
fori,batchinenumerate(val_loader):
all_pictures.append(batch[0].numpy())
all_labels.append(batch[1].item())

..code::ipython3

importmatplotlib.pyplotasplt


defplot_pictures(indexes:list,all_pictures=all_pictures,all_labels=all_labels):
"""Plot4pictures.
:paramindexes:alistofindexesofpicturestobedisplayed.
:paramall_batches:batcheswithpictures.
"""
images,labels=[],[]
num_pics=len(indexes)
assertnum_pics==4,f"Noenoughindexesforpicturestobedisplayed,got{num_pics}"
foridxinindexes:
assertidx<10000,"Cannotgetsuchindex,thereareonly10000"
pic=np.rollaxis(all_pictures[idx].squeeze(),0,3)
images.append(pic)

labels.append(labels_names[all_labels[idx]])

f,axarr=plt.subplots(1,4)
axarr[0].imshow(images[0])
axarr[0].set_title(labels[0])

axarr[1].imshow(images[1])
axarr[1].set_title(labels[1])

axarr[2].imshow(images[2])
axarr[2].set_title(labels[2])

axarr[3].imshow(images[3])
axarr[3].set_title(labels[3])

..code::ipython3

definfer_on_pictures(model,indexes:list,all_pictures=all_pictures):
"""Inferencemodelonafewpictures.
:paramnet:modelonwhichdoinference
:paramindexes:listofindexes
"""
output_key=model.output(0)
predicted_labels=[]
foridxinindexes:
assertidx<10000,"Cannotgetsuchindex,thereareonly10000"
result=model(all_pictures[idx])[output_key]
result=labels_names[np.argmax(result[0])]
predicted_labels.append(result)
returnpredicted_labels

..code::ipython3

indexes_to_infer=[7,12,15,20]#Toplot,specify4indexes.

plot_pictures(indexes_to_infer)

results_float=infer_on_pictures(compiled_model,indexes_to_infer)
results_quanized=infer_on_pictures(optimized_compiled_model,indexes_to_infer)

print(f"Labelsforpicturefromfloatmodel:{results_float}.")
print(f"Labelsforpicturefromquantizedmodel:{results_quanized}.")


..parsed-literal::

ClippinginputdatatothevalidrangeforimshowwithRGBdata([0..1]forfloatsor[0..255]forintegers).
ClippinginputdatatothevalidrangeforimshowwithRGBdata([0..1]forfloatsor[0..255]forintegers).
ClippinginputdatatothevalidrangeforimshowwithRGBdata([0..1]forfloatsor[0..255]forintegers).
ClippinginputdatatothevalidrangeforimshowwithRGBdata([0..1]forfloatsor[0..255]forintegers).


..parsed-literal::

Labelsforpicturefromfloatmodel:['frog','dog','ship','horse'].
Labelsforpicturefromquantizedmodel:['frog','dog','ship','horse'].



..image::image-classification-quantization-with-output_files/image-classification-quantization-with-output_30_2.png


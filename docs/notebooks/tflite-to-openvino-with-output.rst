ConvertaTensorflowLiteModeltoOpenVINO™
============================================

`TensorFlowLite<https://www.tensorflow.org/lite/guide>`__,often
referredtoasTFLite,isanopensourcelibrarydevelopedfordeploying
machinelearningmodelstoedgedevices.

ThisshorttutorialshowshowtoconvertaTensorFlowLite
`EfficientNet-Lite-B0<https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
imageclassificationmodeltoOpenVINO`Intermediate
Representation<https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINOIR)format,usingModelConverter.AftercreatingtheOpenVINO
IR,loadthemodelin`OpenVINO
Runtime<https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
anddoinferencewithasampleimage.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Installrequirements<#install-requirements>`__
-`Imports<#imports>`__

-`DownloadTFLitemodel<#download-tflite-model>`__
-`ConvertaModeltoOpenVINOIR
Format<#convert-a-model-to-openvino-ir-format>`__
-`LoadmodelusingOpenVINOTensorFlowLite
Frontend<#load-model-using-openvino-tensorflow-lite-frontend>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectinferencedevice<#select-inference-device>`__

-`EstimateModelPerformance<#estimate-model-performance>`__

Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Installrequirements
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-qopencv-pythonrequeststqdmkagglehubPillow

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.




..parsed-literal::

23215



Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath
importnumpyasnp
fromPILimportImage
importopenvinoasov

fromnotebook_utilsimportdownload_file,load_image

DownloadTFLitemodel
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importkagglehub

model_dir=kagglehub.model_download("tensorflow/efficientnet/tfLite/lite0-fp32")
tflite_model_path=Path(model_dir)/"2.tflite"

ov_model_path=tflite_model_path.with_suffix(".xml")

ConvertaModeltoOpenVINOIRFormat
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

ToconverttheTFLitemodeltoOpenVINOIR,modelconversionPythonAPI
canbeused.``ov.convert_model``functionacceptsthepathtothe
TFLitemodelandreturnsanOpenVINOModelclassinstancewhich
representsthismodel.Theobtainedmodelisreadytouseandtobe
loadedonadeviceusing``ov.compile_model``orcanbesavedonadisk
using``ov.save_model``function,reducingloadingtimefornext
running.Bydefault,modelweightsarecompressedtoFP16during
serializationby``ov.save_model``.Formoreinformationaboutmodel
conversion,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
ForTensorFlowLitemodelssupport,refertothis
`tutorial<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow-lite.html>`__.

..code::ipython3

ov_model=ov.convert_model(tflite_model_path)
ov.save_model(ov_model,ov_model_path)
print(f"Model{tflite_model_path}successfullyconvertedandsavedto{ov_model_path}")


..parsed-literal::

Model/opt/home/k8sworker/.cache/kagglehub/models/tensorflow/efficientnet/tfLite/lite0-fp32/2/2.tflitesuccessfullyconvertedandsavedto/opt/home/k8sworker/.cache/kagglehub/models/tensorflow/efficientnet/tfLite/lite0-fp32/2/2.xml


LoadmodelusingOpenVINOTensorFlowLiteFrontend
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

TensorFlowLitemodelsaresupportedvia``FrontEnd``API.Youmayskip
conversiontoIRandreadmodelsdirectlybyOpenVINOruntimeAPI.For
moreexamplessupportedformatsreadingviaFrontendAPI,pleaselook
this`tutorial<../openvino-api>`__.

..code::ipython3

core=ov.Core()

ov_model=core.read_model(tflite_model_path)

RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Wecanfindinformationaboutmodelinputpreprocessinginits
`description<https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
on`TensorFlowHub<https://tfhub.dev/>`__.

..code::ipython3

image=load_image("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png")
#load_imagereadstheimageinBGRformat,[:,:,::-1]reshapetransfromsittoRGB
image=Image.fromarray(image[:,:,::-1])
resized_image=image.resize((224,224))
input_tensor=np.expand_dims((np.array(resized_image).astype(np.float32)-127)/128,0)

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

compiled_model=core.compile_model(ov_model,device.value)
predicted_scores=compiled_model(input_tensor)[0]

..code::ipython3

imagenet_classes_file_path=download_file("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt")
imagenet_classes=open(imagenet_classes_file_path).read().splitlines()

top1_predicted_cls_id=np.argmax(predicted_scores)
top1_predicted_score=predicted_scores[0][top1_predicted_cls_id]
predicted_label=imagenet_classes[top1_predicted_cls_id]

display(image.resize((640,512)))
print(f"Predictedlabel:{predicted_label}withprobability{top1_predicted_score:2f}")



..parsed-literal::

imagenet_2012.txt:0%||0.00/30.9k[00:00<?,?B/s]



..image::tflite-to-openvino-with-output_files/tflite-to-openvino-with-output_16_1.png


..parsed-literal::

Predictedlabel:n02109047GreatDanewithprobability0.715318


EstimateModelPerformance
--------------------------

`backtotop⬆️<#table-of-contents>`__`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
isusedtomeasuretheinferenceperformanceofthemodelonCPUand
GPU.

**NOTE**:Formoreaccurateperformance,itisrecommendedtorun
``benchmark_app``inaterminal/commandpromptafterclosingother
applications.Run``benchmark_app-mmodel.xml-dCPU``tobenchmark
asyncinferenceonCPUforoneminute.Change``CPU``to``GPU``to
benchmarkonGPU.Run``benchmark_app--help``toseeanoverviewof
allcommand-lineoptions.

..code::ipython3

print(f"Benchmarkmodelinferenceon{device.value}")
!benchmark_app-m$ov_model_path-d$device.value-t15


..parsed-literal::

BenchmarkmodelinferenceonAUTO
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
[INFO]Readmodeltook9.14ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]images(node:images):f32/[...]/[1,224,224,3]
[INFO]Modeloutputs:
[INFO]Softmax(node:61):f32/[...]/[1,1000]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]images(node:images):u8/[N,H,W,C]/[1,224,224,3]
[INFO]Modeloutputs:
[INFO]Softmax(node:61):f32/[...]/[1,1000]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook146.63ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:TensorFlow_Lite_Frontend_IR
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
[INFO]NETWORK_NAME:TensorFlow_Lite_Frontend_IR
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
[INFO]Firstinferencetook6.99ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:17430iterations
[INFO]Duration:15007.81ms
[INFO]Latency:
[INFO]Median:5.03ms
[INFO]Average:5.03ms
[INFO]Min:3.10ms
[INFO]Max:13.40ms
[INFO]Throughput:1161.40FPS


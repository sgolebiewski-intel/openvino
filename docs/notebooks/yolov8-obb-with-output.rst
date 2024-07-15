YOLOv8OrientedBoundingBoxesObjectDetectionwithOpenVINO™
==============================================================

`YOLOv8-OBB<https://docs.ultralytics.com/tasks/obb/>`__isintroduced
byUltralytics.

Orientedobjectdetectiongoesastepfurtherthanobjectdetectionand
introduceanextraangletolocateobjectsmoreaccurateinanimage.

Theoutputofanorientedobjectdetectorisasetofrotatedbounding
boxesthatexactlyenclosetheobjectsintheimage,alongwithclass
labelsandconfidencescoresforeachbox.Objectdetectionisagood
choicewhenyouneedtoidentifyobjectsofinterestinascene,but
don’tneedtoknowexactlywheretheobjectisoritsexactshape.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`GetPyTorchmodel<#get-pytorch-model>`__
-`Preparedatasetanddataloader<#prepare-dataset-and-dataloader>`__
-`Runinference<#run-inference>`__
-`ConvertPyTorchmodeltoOpenVINO
IR<#convert-pytorch-model-to-openvino-ir>`__

-`Selectinferencedevice<#select-inference-device>`__
-`Compilemodel<#compile-model>`__
-`Preparethemodelfor
inference<#prepare-the-model-for-inference>`__
-`Runinference<#run-inference>`__

-`Quantization<#quantization>`__
-`Compareinferencetimeandmodel
sizes.<#compare-inference-time-and-model-sizes>`__

Prerequisites
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"ultralytics==8.2.24""openvino>=2024.0.0""nncf>=2.9.0"tqdm


..parsed-literal::

DEPRECATION:torchsde0.2.5hasanon-standarddependencyspecifiernumpy>=1.19.*;python_version>="3.7".pip24.1willenforcethisbehaviourchange.Apossiblereplacementistoupgradetoanewerversionoftorchsdeorcontacttheauthortosuggestthattheyreleaseaversionwithaconformingdependencyspecifiers.Discussioncanbefoundathttps://github.com/pypa/pip/issues/12063
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Importrequiredutilityfunctions.Thelowercellwilldownloadthe
notebook_utilsPythonmodulefromGitHub.

..code::ipython3

frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

GetPyTorchmodel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
`torch.nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class,initializedbyastatedictionarywithmodelweights.Wewilluse
theYOLOv8pretrainedOBBlargemodel(alsoknownas``yolov8l-obbn``)
pre-trainedonaDOTAv1dataset,whichisavailableinthis
`repo<https://github.com/ultralytics/ultralytics>`__.Similarstepsare
alsoapplicabletootherYOLOv8models.

..code::ipython3

fromultralyticsimportYOLO

model=YOLO("yolov8l-obb.pt")

Preparedatasetanddataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8-obbispre-trainedontheDOTAdataset.Also,Ultralytics
providesDOTA8dataset.Itisasmall,butversatileorientedobject
detectiondatasetcomposedofthefirst8imagesof8imagesofthe
splitDOTAv1set,4fortrainingand4forvalidation.Thisdatasetis
idealfortestinganddebuggingobjectdetectionmodels,orfor
experimentingwithnewdetectionapproaches.With8images,itissmall
enoughtobeeasilymanageable,yetdiverseenoughtotesttraining
pipelinesforerrorsandactasasanitycheckbeforetraininglarger
datasets.

TheoriginalmodelrepositoryusesaValidatorwrapper,whichrepresents
theaccuracyvalidationpipeline.Itcreatesdataloaderandevaluation
metricsandupdatesmetricsoneachdatabatchproducedbythe
dataloader.Besidesthat,itisresponsiblefordatapreprocessingand
resultspostprocessing.Forclassinitialization,theconfiguration
shouldbeprovided.Wewillusethedefaultsetup,butitcanbe
replacedwithsomeparametersoverridingtotestoncustomdata.The
modelhasconnectedthetask_map,whichallowstogetavalidatorclass
instance.

..code::ipython3

fromultralytics.cfgimportget_cfg
fromultralytics.data.utilsimportcheck_det_dataset
fromultralytics.utilsimportDEFAULT_CFG,DATASETS_DIR


CFG_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/dota8.yaml"
OUT_DIR=Path("./datasets")
CFG_PATH=OUT_DIR/"dota8.yaml"

download_file(CFG_URL,CFG_PATH.name,CFG_PATH.parent)

args=get_cfg(cfg=DEFAULT_CFG)
args.data=CFG_PATH
args.task=model.task

validator=model.task_map[model.task]["validator"](args=args)

validator.stride=32
validator.data=check_det_dataset(str(args.data))
data_loader=validator.get_dataloader(DATASETS_DIR/"dota8",1)
example_image_path=list(data_loader)[1]["im_file"][0]



..parsed-literal::

datasets/dota8.yaml:0%||0.00/608[00:00<?,?B/s]


..parsed-literal::


Dataset'datasets/dota8.yaml'imagesnotfound⚠️,missingpath'/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/images/val'
Downloadinghttps://github.com/ultralytics/yolov5/releases/download/v1.0/dota8.zipto'/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8.zip'...


..parsed-literal::

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|1.24M/1.24M[00:00<00:00,1.63MB/s]
Unzipping/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8.zipto/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8...:100%|██████████|27/27[00:00<00:00,644.45file/s]

..parsed-literal::

Datasetdownloadsuccess✅(4.1s),savedto/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets


..parsed-literal::


val:Scanning/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/labels/train...8images,0backgrounds,0corrupt:100%|██████████|8/8[00:00<00:00,266.41it/s]

..parsed-literal::

val:Newcachecreated:/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/labels/train.cache


..parsed-literal::




Runinference
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromPILimportImage

res=model(example_image_path,device="cpu")
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/images/train/P1053__1024__0___90.jpg:1024x10244915.2ms
Speed:18.6mspreprocess,4915.2msinference,50.9mspostprocessperimageatshape(1,3,1024,1024)




..image::yolov8-obb-with-output_files/yolov8-obb-with-output_10_1.png



ConvertPyTorchmodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YOLOv8providesAPIforconvenientmodelexportingtodifferentformats
includingOpenVINOIR.``model.export``isresponsibleformodel
conversion.Weneedtospecifytheformat,andadditionally,wecan
preservedynamicshapesinthemodel.

..code::ipython3

frompathlibimportPath

models_dir=Path("./models")
models_dir.mkdir(exist_ok=True)


OV_MODEL_NAME="yolov8l-obb"


OV_MODEL_PATH=Path(f"{OV_MODEL_NAME}_openvino_model/{OV_MODEL_NAME}.xml")
ifnotOV_MODEL_PATH.exists():
model.export(format="openvino",dynamic=True,half=True)


..parsed-literal::

UltralyticsYOLOv8.1.24🚀Python-3.8.10torch-2.1.2+cpuCPU(IntelCore(TM)i9-10980XE3.00GHz)

PyTorch:startingfrom'yolov8l-obb.pt'withinputshape(1,3,1024,1024)BCHWandoutputshape(s)(1,20,21504)(85.4MB)

OpenVINO:startingexportwithopenvino2024.0.0-14509-34caeefd078-releases/2024/0...
OpenVINO:exportsuccess✅5.6s,savedas'yolov8l-obb_openvino_model/'(85.4MB)

Exportcomplete(18.7s)
Resultssavedto/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov8-optimization
Predict:yolopredicttask=obbmodel=yolov8l-obb_openvino_modelimgsz=1024half
Validate:yolovaltask=obbmodel=yolov8l-obb_openvino_modelimgsz=1024data=runs/DOTAv1.0-ms.yamlhalf
Visualize:https://netron.app


Selectinferencedevice
^^^^^^^^^^^^^^^^^^^^^^^

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



Compilemodel
^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_model=core.read_model(OV_MODEL_PATH)

ov_config={}
ifdevice.value!="CPU":
ov_model.reshape({0:[1,3,1024,1024]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}

compiled_ov_model=core.compile_model(ov_model,device.value,ov_config)

Preparethemodelforinference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Wecanreusethebasemodelpipelineforpre-andpostprocessingjust
replacingtheinferencemethodwherewewillusetheIRmodelfor
inference.

..code::ipython3

importtorch


definfer(*args):
result=compiled_ov_model(args)[0]
returntorch.from_numpy(result)


model.predictor.inference=infer

Runinference
^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

res=model(example_image_path,device="cpu")
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/images/train/P1053__1024__0___90.jpg:1024x1024338.0ms
Speed:4.7mspreprocess,338.0msinference,3.7mspostprocessperimageatshape(1,3,1024,1024)




..image::yolov8-obb-with-output_files/yolov8-obb-with-output_20_1.png



Quantization
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize()``toobtainquantizedmodel.
3.Savethe``INT8``modelusing``openvino.save_model()``function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

importipywidgetsaswidgets

INT8_OV_PATH=Path("model/int8_model.xml")

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

..code::ipython3

%%skipnot$to_quantize.value

fromtypingimportDict

importnncf


deftransform_fn(data_item:Dict):
input_tensor=validator.preprocess(data_item)["img"].numpy()
returninput_tensor


quantization_dataset=nncf.Dataset(data_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..

**NOTE**:WeusethetinyDOTA8datasetasacalibrationdataset.It
givesagoodenoughresultfortutorialpurpose.Forbatterresults,
useabiggerdataset.Usually300examplesareenough.

..code::ipython3

%%skipnot$to_quantize.value

ifINT8_OV_PATH.exists():
print("Loadingquantizedmodel")
quantized_model=core.read_model(INT8_OV_PATH)
else:
ov_model.reshape({0:[1,3,-1,-1]})
quantized_model=nncf.quantize(
ov_model,
quantization_dataset,
preset=nncf.QuantizationPreset.MIXED,
)
ov.save_model(quantized_model,INT8_OV_PATH)


ov_config={}
ifdevice.value!="CPU":
quantized_model.reshape({0:[1,3,1024,1024]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}

model_optimized=core.compile_model(quantized_model,device.value,ov_config)



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



WecanreusethebasemodelpipelineinthesamewayasforIRmodel.

..code::ipython3

%%skipnot$to_quantize.value

definfer(*args):
result=model_optimized(args)[0]
returntorch.from_numpy(result)

model.predictor.inference=infer

Runinference

..code::ipython3

%%skipnot$to_quantize.value

res=model(example_image_path,device='cpu')
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::


image1/1/home/ea/work/openvino_notebooks/notebooks/fast-segment-anything/datasets/dota8/images/train/P1053__1024__0___90.jpg:1024x1024240.5ms
Speed:3.2mspreprocess,240.5msinference,4.2mspostprocessperimageatshape(1,3,1024,1024)


Youcanseethattheresultisalmostthesamebutithasasmall
difference.Onesmallvehiclewasrecognizedastwovehicles.Butone
largecarwasalsoidentified,unliketheoriginalmodel.

Compareinferencetimeandmodelsizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=OV_MODEL_PATH.with_suffix(".bin").stat().st_size/1024
quantized_model_size=INT8_OV_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{quantized_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16modelsize:86849.05KB
INT8modelsize:43494.78KB
Modelcompressionrate:1.997


..code::ipython3

#InferenceFP32model(OpenVINOIR)
!benchmark_app-m$OV_MODEL_PATH-d$device.value-apiasync-shape"[1,3,640,640]"


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
[INFO]Readmodeltook25.07ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[?,20,16..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook10.42ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[1,20,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook645.51ms
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
[INFO]Firstinferencetook362.70ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:1620iterations
[INFO]Duration:121527.01ms
[INFO]Latency:
[INFO]Median:884.92ms
[INFO]Average:897.13ms
[INFO]Min:599.38ms
[INFO]Max:1131.46ms
[INFO]Throughput:13.33FPS


..code::ipython3

ifINT8_OV_PATH.exists():
#InferenceINT8model(Quantizedmodel)
!benchmark_app-m$INT8_OV_PATH-d$device.value-apiasync-shape"[1,3,640,640]"-t15


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
[INFO]Readmodeltook46.47ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[?,20,16..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook20.10ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_9):f32/[...]/[1,20,8400]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1201.42ms
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
[INFO]Firstinferencetook124.20ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:708iterations
[INFO]Duration:15216.46ms
[INFO]Latency:
[INFO]Median:252.23ms
[INFO]Average:255.76ms
[INFO]Min:176.97ms
[INFO]Max:344.41ms
[INFO]Throughput:46.53FPS


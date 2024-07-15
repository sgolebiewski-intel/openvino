ConvertandOptimizeYOLOv8withOpenVINO™
==========================================

TheYOLOv8algorithmdevelopedbyUltralyticsisacutting-edge,
state-of-the-art(SOTA)modelthatisdesignedtobefast,accurate,and
easytouse,makingitanexcellentchoiceforawiderangeofobject
detection,imagesegmentation,andimageclassificationtasks.More
detailsaboutitsrealizationcanbefoundintheoriginalmodel
`repository<https://github.com/ultralytics/ultralytics>`__.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunapply
quantizationwithaccuracycontroltoPyTorchYOLOv8.Theadvanced
quantizationflowallowstoapply8-bitquantizationtothemodelwith
controlofaccuracymetric.Thisisachievedbykeepingthemost
impactfuloperationswithinthemodelintheoriginalprecision.The
flowisbasedonthe`Basic8-bit
quantization<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__
andhasthefollowingdifferences:

-Besidesthecalibrationdataset,avalidationdatasetisrequiredto
computetheaccuracymetric.Bothdatasetscanrefertothesamedata
inthesimplestcase.
-Validationfunction,usedtocomputeaccuracymetricisrequired.It
canbeafunctionthatisalreadyavailableinthesourceframework
oracustomfunction.
-Sinceaccuracyvalidationisrunseveraltimesduringthe
quantizationprocess,quantizationwithaccuracycontrolcantake
moretimethantheBasic8-bitquantizationflow.
-Theresultedmodelcanprovidesmallerperformanceimprovementthan
theBasic8-bitquantizationflowbecausesomeoftheoperationsare
keptintheoriginalprecision.

..

**NOTE**:Currently,8-bitquantizationwithaccuracycontrolinNNCF
isavailableonlyformodelsinOpenVINOrepresentation.

Thestepsforthequantizationwithaccuracycontrolaredescribed
below.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`GetPytorchmodelandOpenVINOIR
model<#get-pytorch-model-and-openvino-ir-model>`__

-`Definevalidatoranddata
loader<#define-validator-and-data-loader>`__
-`Preparecalibrationandvalidation
datasets<#prepare-calibration-and-validation-datasets>`__
-`Preparevalidationfunction<#prepare-validation-function>`__

-`Runquantizationwithaccuracy
control<#run-quantization-with-accuracy-control>`__
-`CompareAccuracyandPerformanceoftheOriginalandQuantized
Models<#compare-accuracy-and-performance-of-the-original-and-quantized-models>`__

Prerequisites
^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Installnecessarypackages.

..code::ipython3

%pipinstall-q"openvino>=2024.0.0"
%pipinstall-q"nncf>=2.9.0"
%pipinstall-q"ultralytics==8.1.42"tqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

GetPytorchmodelandOpenVINOIRmodel
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
`torch.nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class,initializedbyastatedictionarywithmodelweights.Wewilluse
theYOLOv8nanomodel(alsoknownas``yolov8n``)pre-trainedonaCOCO
dataset,whichisavailableinthis
`repo<https://github.com/ultralytics/ultralytics>`__.Similarstepsare
alsoapplicabletootherYOLOv8models.Typicalstepstoobtaina
pre-trainedmodel:

1.Createaninstanceofamodelclass.
2.Loadacheckpointstatedict,whichcontainsthepre-trainedmodel
weights.

Inthiscase,thecreatorsofthemodelprovideanAPIthatenables
convertingtheYOLOv8modeltoONNXandthentoOpenVINOIR.Therefore,
wedonotneedtodothesestepsmanually.

..code::ipython3

importos
frompathlibimportPath

fromultralyticsimportYOLO
fromultralytics.cfgimportget_cfg
fromultralytics.data.utilsimportcheck_det_dataset
fromultralytics.engine.validatorimportBaseValidatorasValidator
fromultralytics.utilsimportDEFAULT_CFG
fromultralytics.utilsimportops
fromultralytics.utils.metricsimportConfusionMatrix

ROOT=os.path.abspath("")

MODEL_NAME="yolov8n-seg"

model=YOLO(f"{ROOT}/{MODEL_NAME}.pt")
args=get_cfg(cfg=DEFAULT_CFG)
args.data="coco128-seg.yaml"

..code::ipython3

#Fetchthenotebookutilsscriptfromtheopenvino_notebooksrepo
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

..code::ipython3

fromzipfileimportZipFile

fromultralytics.data.utilsimportDATASETS_DIR

DATA_URL="https://www.ultralytics.com/assets/coco128-seg.zip"
CFG_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/8ebe94d1e928687feaa1fee6d5668987df5e43be/ultralytics/datasets/coco128-seg.yaml"#lastcompatibleformatwithultralytics8.0.43

OUT_DIR=DATASETS_DIR

DATA_PATH=OUT_DIR/"coco128-seg.zip"
CFG_PATH=OUT_DIR/"coco128-seg.yaml"

download_file(DATA_URL,DATA_PATH.name,DATA_PATH.parent)
download_file(CFG_URL,CFG_PATH.name,CFG_PATH.parent)

ifnot(OUT_DIR/"coco128/labels").exists():
withZipFile(DATA_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR)


..parsed-literal::

'/home/maleksandr/test_notebooks/ultrali/datasets/coco128-seg.zip'alreadyexists.



..parsed-literal::

/home/maleksandr/test_notebooks/ultrali/datasets/coco128-seg.yaml:0%||0.00/0.98k[00:00<?,?B/s]


Loadmodel.

..code::ipython3

importopenvinoasov


model_path=Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
ifnotmodel_path.exists():
model.export(format="openvino",dynamic=True,half=False)

ov_model=ov.Core().read_model(model_path)

Definevalidatoranddataloader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

fromultralytics.data.converterimportcoco80_to_coco91_class


validator=model.task_map[model.task]["validator"](args=args)
validator.data=check_det_dataset(args.data)
validator.stride=3
data_loader=validator.get_dataloader(OUT_DIR/"coco128-seg",1)

validator.is_coco=True
validator.class_map=coco80_to_coco91_class()
validator.names=model.model.names
validator.metrics.names=validator.names
validator.nc=model.model.model[-1].nc
validator.nm=32
validator.process=ops.process_mask
validator.plot_masks=[]

Preparecalibrationandvalidationdatasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Wecanuseonedatasetascalibrationandvalidationdatasets.Nameit
``quantization_dataset``.

..code::ipython3

fromtypingimportDict

importnncf


deftransform_fn(data_item:Dict):
input_tensor=validator.preprocess(data_item)["img"].numpy()
returninput_tensor


quantization_dataset=nncf.Dataset(data_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,openvino


Preparevalidationfunction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromfunctoolsimportpartial

importtorch
fromnncf.quantization.advanced_parametersimportAdvancedAccuracyRestorerParameters


defvalidation_ac(
compiled_model:ov.CompiledModel,
validation_loader:torch.utils.data.DataLoader,
validator:Validator,
num_samples:int=None,
log=True,
)->float:
validator.seen=0
validator.jdict=[]
validator.stats=dict(tp_m=[],tp=[],conf=[],pred_cls=[],target_cls=[])
validator.batch_i=1
validator.confusion_matrix=ConfusionMatrix(nc=validator.nc)
num_outputs=len(compiled_model.outputs)

counter=0
forbatch_i,batchinenumerate(validation_loader):
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
counter+=1
stats=validator.get_stats()
ifnum_outputs==1:
stats_metrics=stats["metrics/mAP50-95(B)"]
else:
stats_metrics=stats["metrics/mAP50-95(M)"]
iflog:
print(f"Validate:datasetlength={counter},metricvalue={stats_metrics:.3f}")

returnstats_metrics


validation_fn=partial(validation_ac,validator=validator,log=False)

Runquantizationwithaccuracycontrol
--------------------------------------

`backtotop⬆️<#table-of-contents>`__

Youshouldprovidethecalibrationdatasetandthevalidationdataset.
Itcanbethesamedataset.-parameter``max_drop``definesthe
accuracydropthreshold.Thequantizationprocessstopswhenthe
degradationofaccuracymetriconthevalidationdatasetislessthan
the``max_drop``.Thedefaultvalueis0.01.NNCFwillstopthe
quantizationandreportanerrorifthe``max_drop``valuecan’tbe
reached.-``drop_type``defineshowtheaccuracydropwillbe
calculated:ABSOLUTE(usedbydefault)orRELATIVE.-
``ranking_subset_size``-sizeofasubsetthatisusedtoranklayers
bytheircontributiontotheaccuracydrop.Defaultvalueis300,and
themoresamplesithasthebetterranking,potentially.Hereweusethe
value25tospeeduptheexecution.

**NOTE**:Executioncantaketensofminutesandrequiresupto15GB
offreememory

..code::ipython3

quantized_model=nncf.quantize_with_accuracy_control(
ov_model,
quantization_dataset,
quantization_dataset,
validation_fn=validation_fn,
max_drop=0.01,
preset=nncf.QuantizationPreset.MIXED,
subset_size=128,
advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(ranking_subset_size=25),
)



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

/home/maleksandr/test_notebooks/ultrali/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control/venv/lib/python3.10/site-packages/nncf/experimental/tensor/tensor.py:84:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Validationofinitialmodelwasstarted
INFO:nncf:ElapsedTime:00:00:00
INFO:nncf:ElapsedTime:00:00:03
INFO:nncf:Metricofinitialmodel:0.3651327608484117
INFO:nncf:Collectingvaluesforeachdataitemusingtheinitialmodel
INFO:nncf:ElapsedTime:00:00:04
INFO:nncf:Validationofquantizedmodelwasstarted
INFO:nncf:ElapsedTime:00:00:00
INFO:nncf:ElapsedTime:00:00:03
INFO:nncf:Metricofquantizedmodel:0.34040251506886543
INFO:nncf:Collectingvaluesforeachdataitemusingthequantizedmodel
INFO:nncf:ElapsedTime:00:00:04
INFO:nncf:Accuracydrop:0.024730245779546245(absolute)
INFO:nncf:Accuracydrop:0.024730245779546245(absolute)
INFO:nncf:Totalnumberofquantizedoperationsinthemodel:92
INFO:nncf:Numberofparallelworkerstorankquantizedoperations:1
INFO:nncf:ORIGINALmetricisusedtorankquantizers



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:ElapsedTime:00:01:38
INFO:nncf:Changingthescopeofquantizernodeswasstarted
INFO:nncf:Reverted1operationstothefloating-pointprecision:
	__module.model.4.m.0.cv2.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.023408466397916217(absolute)
INFO:nncf:Reverted1operationstothefloating-pointprecision:
	__module.model.18.m.0.cv2.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.024749654890442174(absolute)
INFO:nncf:Re-calculatingrankingscoresforremaininggroups



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:ElapsedTime:00:01:36
INFO:nncf:Reverted1operationstothefloating-pointprecision:
	__module.model.22.proto.cv3.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.023229513575966754(absolute)
INFO:nncf:Reverted2operationstothefloating-pointprecision:
	__module.model.22/aten::add/Add_6
	__module.model.22/aten::sub/Subtract
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.02425608378963906(absolute)
INFO:nncf:Re-calculatingrankingscoresforremaininggroups



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:ElapsedTime:00:01:35
INFO:nncf:Reverted1operationstothefloating-pointprecision:
	__module.model.6.m.0.cv2.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.023297881500256024(absolute)
INFO:nncf:Reverted2operationstothefloating-pointprecision:
	__module.model.12.cv2.conv/aten::_convolution/Convolution
	__module.model.12.m.0.cv1.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.021779128052922092(absolute)
INFO:nncf:Reverted2operationstothefloating-pointprecision:
	__module.model.7.conv/aten::_convolution/Convolution
	__module.model.12.cv1.conv/aten::_convolution/Convolution
INFO:nncf:Accuracydropwiththenewquantizationscopeis0.01696486517685941(absolute)
INFO:nncf:Reverted2operationstothefloating-pointprecision:
	__module.model.22/aten::add/Add_7
	__module.model.22/aten::sub/Subtract_1
INFO:nncf:Algorithmcompleted:achievedrequiredaccuracydrop0.005923437521415831(absolute)
INFO:nncf:9outof92wererevertedbacktothefloating-pointprecision:
	__module.model.4.m.0.cv2.conv/aten::_convolution/Convolution
	__module.model.22.proto.cv3.conv/aten::_convolution/Convolution
	__module.model.6.m.0.cv2.conv/aten::_convolution/Convolution
	__module.model.12.cv2.conv/aten::_convolution/Convolution
	__module.model.12.m.0.cv1.conv/aten::_convolution/Convolution
	__module.model.7.conv/aten::_convolution/Convolution
	__module.model.12.cv1.conv/aten::_convolution/Convolution
	__module.model.22/aten::add/Add_7
	__module.model.22/aten::sub/Subtract_1


CompareAccuracyandPerformanceoftheOriginalandQuantizedModels
---------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

NowwecancomparemetricsoftheOriginalnon-quantizedOpenVINOIR
modelandQuantizedOpenVINOIRmodeltomakesurethatthe``max_drop``
isnotexceeded.

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

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



..code::ipython3

core=ov.Core()
ov_config={}
ifdevice.value!="CPU":
quantized_model.reshape({0:[1,3,640,640]})
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
quantized_compiled_model=core.compile_model(quantized_model,device.value,ov_config)
compiled_ov_model=core.compile_model(ov_model,device.value,ov_config)

pt_result=validation_ac(compiled_ov_model,data_loader,validator)
quantized_result=validation_ac(quantized_compiled_model,data_loader,validator)


print(f"[OriginalOpenVINO]:{pt_result:.4f}")
print(f"[QuantizedOpenVINO]:{quantized_result:.4f}")


..parsed-literal::

Validate:datasetlength=128,metricvalue=0.368
Validate:datasetlength=128,metricvalue=0.357
[OriginalOpenVINO]:0.3677
[QuantizedOpenVINO]:0.3570


Andcompareperformance.

..code::ipython3

frompathlibimportPath

#Setmodeldirectory
MODEL_DIR=Path("model")
MODEL_DIR.mkdir(exist_ok=True)

ir_model_path=MODEL_DIR/"ir_model.xml"
quantized_model_path=MODEL_DIR/"quantized_model.xml"

#Savemodelstousetheminthecommandlinebanchmarkapp
ov.save_model(ov_model,ir_model_path,compress_to_fp16=False)
ov.save_model(quantized_model,quantized_model_path,compress_to_fp16=False)

..code::ipython3

#InferenceOriginalmodel(OpenVINOIR)
!benchmark_app-m$ir_model_path-shape"[1,3,640,640]"-d$device.value-apiasync


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
[INFO]Readmodeltook13.54ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[?,116,16..]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[?,32,8..,8..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook8.56ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[1,116,8400]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[1,32,160,160]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook437.16ms
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
[INFO]Firstinferencetook46.51ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:16872iterations
[INFO]Duration:120117.37ms
[INFO]Latency:
[INFO]Median:85.10ms
[INFO]Average:85.27ms
[INFO]Min:53.55ms
[INFO]Max:108.50ms
[INFO]Throughput:140.46FPS


..code::ipython3

#InferenceQuantizedmodel(OpenVINOIR)
!benchmark_app-m$quantized_model_path-shape"[1,3,640,640]"-d$device.value-apiasync


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
[INFO]Readmodeltook20.52ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[?,116,16..]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[?,32,8..,8..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook11.74ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.22/aten::cat/Concat_8):f32/[...]/[1,116,8400]
[INFO]input.199(node:__module.model.22.cv4.2.1.act/aten::silu_/Swish_37):f32/[...]/[1,32,160,160]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook711.53ms
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
[INFO]Firstinferencetook35.64ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:33564iterations
[INFO]Duration:120059.16ms
[INFO]Latency:
[INFO]Median:42.72ms
[INFO]Average:42.76ms
[INFO]Min:23.29ms
[INFO]Max:67.71ms
[INFO]Throughput:279.56FPS


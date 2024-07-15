QuantizationAwareTrainingwithNNCF,usingTensorFlowFramework
=================================================================

ThegoalofthisnotebooktodemonstratehowtousetheNeuralNetwork
CompressionFramework`NNCF<https://github.com/openvinotoolkit/nncf>`__
8-bitquantizationtooptimizeaTensorFlowmodelforinferencewith
OpenVINO™Toolkit.Theoptimizationprocesscontainsthefollowing
steps:

-Transformingtheoriginal``FP32``modelto``INT8``
-Usingfine-tuningtorestoretheaccuracy.
-ExportingoptimizedandoriginalmodelstoFrozenGraphandthento
OpenVINO.
-Measuringandcomparingtheperformanceofmodels.

Formoreadvancedusage,refertothese
`examples<https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

ThistutorialusestheResNet-18modelwithImagenettedataset.
Imagenetteisasubsetof10easilyclassifiedclassesfromtheImageNet
dataset.Usingthesmallermodelanddatasetwillspeeduptrainingand
downloadtime.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`ImportsandSettings<#imports-and-settings>`__
-`DatasetPreprocessing<#dataset-preprocessing>`__
-`DefineaFloating-PointModel<#define-a-floating-point-model>`__
-`Pre-trainaFloating-Point
Model<#pre-train-a-floating-point-model>`__
-`CreateandInitialize
Quantization<#create-and-initialize-quantization>`__
-`Fine-tunetheCompressedModel<#fine-tune-the-compressed-model>`__
-`ExportModelstoOpenVINOIntermediateRepresentation
(IR)<#export-models-to-openvino-intermediate-representation-ir>`__
-`BenchmarkModelPerformancebyComputingInference
Time<#benchmark-model-performance-by-computing-inference-time>`__

ImportsandSettings
--------------------

`backtotop⬆️<#table-of-contents>`__

ImportNNCFandallauxiliarypackagesfromyourPythoncode.Setaname
forthemodel,inputimagesize,usedbatchsize,andthelearningrate.
Also,definepathswhereFrozenGraphandOpenVINOIRversionsofthe
modelswillbestored.

**NOTE**:AllNNCFloggingmessagesbelowERRORlevel(INFOand
WARNING)aredisabledtosimplifythetutorial.Forproductionuse,
itisrecommendedtoenableloggingbyremoving
``set_log_level(logging.ERROR)``.

..code::ipython3

%pipinstall-q"openvino>=2024.0.0""nncf>=2.9.0"
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'"
%pipinstall-q"tensorflow-datasets>=4.9.0,<4.9.3;platform_system=='Windows'"
%pipinstall-q"tensorflow-datasets>=4.9.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath
importlogging

importtensorflowastf
importtensorflow_datasetsastfds

fromnncfimportNNCFConfig
fromnncf.tensorflow.helpers.model_creationimportcreate_compressed_model
fromnncf.tensorflow.initializationimportregister_default_init_args
fromnncf.common.logging.loggerimportset_log_level
importopenvinoasov

set_log_level(logging.ERROR)

MODEL_DIR=Path("model")
OUTPUT_DIR=Path("output")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_MODEL_NAME="ResNet-18"

fp32_h5_path=Path(MODEL_DIR/(BASE_MODEL_NAME+"_fp32")).with_suffix(".h5")
fp32_ir_path=Path(OUTPUT_DIR/"saved_model").with_suffix(".xml")
int8_pb_path=Path(OUTPUT_DIR/(BASE_MODEL_NAME+"_int8")).with_suffix(".pb")
int8_ir_path=int8_pb_path.with_suffix(".xml")

BATCH_SIZE=128
IMG_SIZE=(64,64)#DefaultImagenetimagesize
NUM_CLASSES=10#ForImagenettedataset

LR=1e-5

MEAN_RGB=(0.485*255,0.456*255,0.406*255)#FromImagenetdataset
STDDEV_RGB=(0.229*255,0.224*255,0.225*255)#FromImagenetdataset

fp32_pth_url="https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/305_resnet18_imagenette_fp32_v1.h5"
_=tf.keras.utils.get_file(fp32_h5_path.resolve(),fp32_pth_url)
print(f"Absolutepathwherethemodelweightsaresaved:\n{fp32_h5_path.resolve()}")


..parsed-literal::

2024-07-1304:06:54.539917:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:06:54.575556:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:06:55.171044:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
WARNING:nncf:NNCFprovidesbestresultswithtorch==2.15.*,whilecurrenttorchversionis2.12.0.Ifyouencounterissues,considerswitchingtotorch==2.15.*
Downloadingdatafromhttps://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/305_resnet18_imagenette_fp32_v1.h5
134604992/134604992[==============================]-1s0us/step
Absolutepathwherethemodelweightsaresaved:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/tensorflow-quantization-aware-training/model/ResNet-18_fp32.h5


DatasetPreprocessing
---------------------

`backtotop⬆️<#table-of-contents>`__

DownloadandprepareImagenette160pxdataset.-Numberofclasses:10-
Downloadsize:94.18MiB

::

|Split|Examples|
|--------------|----------|
|'train'|12,894|
|'validation'|500|

..code::ipython3

datasets,datasets_info=tfds.load(
"imagenette/160px",
shuffle_files=True,
as_supervised=True,
with_info=True,
read_config=tfds.ReadConfig(shuffle_seed=0),
)
train_dataset,validation_dataset=datasets["train"],datasets["validation"]
fig=tfds.show_examples(train_dataset,datasets_info)


..parsed-literal::

2024-07-1304:07:01.446767:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1304:07:01.446799:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168]retrievingCUDAdiagnosticinformationforhost:iotg-dev-workstation-07
2024-07-1304:07:01.446804:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175]hostname:iotg-dev-workstation-07
2024-07-1304:07:01.446954:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199]libcudareportedversionis:470.223.2
2024-07-1304:07:01.446971:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203]kernelreportedversionis:470.182.3
2024-07-1304:07:01.446974:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration
2024-07-1304:07:01.553632:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_4'withdtypeint64andshape[1]
	[[{{nodePlaceholder/_4}}]]
2024-07-1304:07:01.553982:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_4'withdtypeint64andshape[1]
	[[{{nodePlaceholder/_4}}]]
2024-07-1304:07:01.624806:Wtensorflow/core/kernels/data/cache_dataset_ops.cc:856]Thecallingiteratordidnotfullyreadthedatasetbeingcached.Inordertoavoidunexpectedtruncationofthedataset,thepartiallycachedcontentsofthedatasetwillbediscarded.Thiscanhappenifyouhaveaninputpipelinesimilarto`dataset.cache().take(k).repeat()`.Youshoulduse`dataset.take(k).cache().repeat()`instead.



..image::tensorflow-quantization-aware-training-with-output_files/tensorflow-quantization-aware-training-with-output_6_1.png


..code::ipython3

defpreprocessing(image,label):
image=tf.image.resize(image,IMG_SIZE)
image=image-MEAN_RGB
image=image/STDDEV_RGB
label=tf.one_hot(label,NUM_CLASSES)
returnimage,label


train_dataset=train_dataset.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset=(
validation_dataset.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
)

DefineaFloating-PointModel
-----------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defresidual_conv_block(filters,stage,block,strides=(1,1),cut="pre"):
deflayer(input_tensor):
x=tf.keras.layers.BatchNormalization(epsilon=2e-5)(input_tensor)
x=tf.keras.layers.Activation("relu")(x)

#Definingshortcutconnection.
ifcut=="pre":
shortcut=input_tensor
elifcut=="post":
shortcut=tf.keras.layers.Conv2D(
filters,
(1,1),
strides=strides,
kernel_initializer="he_uniform",
use_bias=False,
)(x)

#Continuewithconvolutionlayers.
x=tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x=tf.keras.layers.Conv2D(
filters,
(3,3),
strides=strides,
kernel_initializer="he_uniform",
use_bias=False,
)(x)

x=tf.keras.layers.BatchNormalization(epsilon=2e-5)(x)
x=tf.keras.layers.Activation("relu")(x)
x=tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x=tf.keras.layers.Conv2D(filters,(3,3),kernel_initializer="he_uniform",use_bias=False)(x)

#Addresidualconnection.
x=tf.keras.layers.Add()([x,shortcut])
returnx

returnlayer


defResNet18(input_shape=None):
"""InstantiatestheResNet18architecture."""
img_input=tf.keras.layers.Input(shape=input_shape,name="data")

#ResNet18bottom
x=tf.keras.layers.BatchNormalization(epsilon=2e-5,scale=False)(img_input)
x=tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x=tf.keras.layers.Conv2D(64,(7,7),strides=(2,2),kernel_initializer="he_uniform",use_bias=False)(x)
x=tf.keras.layers.BatchNormalization(epsilon=2e-5)(x)
x=tf.keras.layers.Activation("relu")(x)
x=tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x=tf.keras.layers.MaxPooling2D((3,3),strides=(2,2),padding="valid")(x)

#ResNet18body
repetitions=(2,2,2,2)
forstage,repinenumerate(repetitions):
forblockinrange(rep):
filters=64*(2**stage)
ifblock==0andstage==0:
x=residual_conv_block(filters,stage,block,strides=(1,1),cut="post")(x)
elifblock==0:
x=residual_conv_block(filters,stage,block,strides=(2,2),cut="post")(x)
else:
x=residual_conv_block(filters,stage,block,strides=(1,1),cut="pre")(x)
x=tf.keras.layers.BatchNormalization(epsilon=2e-5)(x)
x=tf.keras.layers.Activation("relu")(x)

#ResNet18top
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(NUM_CLASSES)(x)
x=tf.keras.layers.Activation("softmax")(x)

#Createthemodel.
model=tf.keras.models.Model(img_input,x)

returnmodel

..code::ipython3

IMG_SHAPE=IMG_SIZE+(3,)
fp32_model=ResNet18(input_shape=IMG_SHAPE)

Pre-trainaFloating-PointModel
--------------------------------

`backtotop⬆️<#table-of-contents>`__

UsingNNCFformodelcompressionassumesthattheuserhasapre-trained
modelandatrainingpipeline.

**NOTE**Forthesakeofsimplicityofthetutorial,itis
recommendedtoskip``FP32``modeltrainingandloadtheweightsthat
areprovided.

..code::ipython3

#Loadthefloating-pointweights.
fp32_model.load_weights(fp32_h5_path)

#Compilethefloating-pointmodel.
fp32_model.compile(
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc@1")],
)

#Validatethefloating-pointmodel.
test_loss,acc_fp32=fp32_model.evaluate(
validation_dataset,
callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=["acc@1"]),
)
print(f"\nAccuracyofFP32model:{acc_fp32:.3f}")


..parsed-literal::

2024-07-1304:07:02.579874:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_0'withdtypestringandshape[1]
	[[{{nodePlaceholder/_0}}]]
2024-07-1304:07:02.580249:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_2'withdtypestringandshape[1]
	[[{{nodePlaceholder/_2}}]]


..parsed-literal::

4/4[==============================]-1s288ms/sample-loss:0.9807-acc@1:0.8220

AccuracyofFP32model:0.822


CreateandInitializeQuantization
----------------------------------

`backtotop⬆️<#table-of-contents>`__

NNCFenablescompression-awaretrainingbyintegratingintoregular
trainingpipelines.Theframeworkisdesignedsothatmodificationsto
youroriginaltrainingcodeareminor.Quantizationisthesimplest
scenarioandrequiresonly3modifications.

1.ConfigureNNCFparameterstospecifycompression

..code::ipython3

nncf_config_dict={
"input_info":{"sample_size":[1,3]+list(IMG_SIZE)},
"log_dir":str(OUTPUT_DIR),#ThelogdirectoryforNNCF-specificloggingoutputs.
"compression":{
"algorithm":"quantization",#Specifythealgorithmhere.
},
}
nncf_config=NNCFConfig.from_dict(nncf_config_dict)

2.Provideadataloadertoinitializethevaluesofquantizationranges
anddeterminewhichactivationshouldbesignedorunsignedfromthe
collectedstatistics,usingagivennumberofsamples.

..code::ipython3

nncf_config=register_default_init_args(nncf_config=nncf_config,data_loader=train_dataset,batch_size=BATCH_SIZE)

3.Createawrappedmodelreadyforcompressionfine-tuningfroma
pre-trained``FP32``modelandaconfigurationobject.

..code::ipython3

compression_ctrl,int8_model=create_compressed_model(fp32_model,nncf_config)


..parsed-literal::

2024-07-1304:07:05.362029:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_1'withdtypestringandshape[1]
	[[{{nodePlaceholder/_1}}]]
2024-07-1304:07:05.362414:Itensorflow/core/common_runtime/executor.cc:1197][/device:CPU:0](DEBUGINFO)Executorstartaborting(thisdoesnotindicateanerrorandyoucanignorethismessage):INVALID_ARGUMENT:Youmustfeedavalueforplaceholdertensor'Placeholder/_2'withdtypestringandshape[1]
	[[{{nodePlaceholder/_2}}]]
2024-07-1304:07:06.318865:Wtensorflow/core/kernels/data/cache_dataset_ops.cc:856]Thecallingiteratordidnotfullyreadthedatasetbeingcached.Inordertoavoidunexpectedtruncationofthedataset,thepartiallycachedcontentsofthedatasetwillbediscarded.Thiscanhappenifyouhaveaninputpipelinesimilarto`dataset.cache().take(k).repeat()`.Youshoulduse`dataset.take(k).cache().repeat()`instead.
2024-07-1304:07:07.067794:Wtensorflow/core/kernels/data/cache_dataset_ops.cc:856]Thecallingiteratordidnotfullyreadthedatasetbeingcached.Inordertoavoidunexpectedtruncationofthedataset,thepartiallycachedcontentsofthedatasetwillbediscarded.Thiscanhappenifyouhaveaninputpipelinesimilarto`dataset.cache().take(k).repeat()`.Youshoulduse`dataset.take(k).cache().repeat()`instead.
2024-07-1304:07:15.330371:Wtensorflow/core/kernels/data/cache_dataset_ops.cc:856]Thecallingiteratordidnotfullyreadthedatasetbeingcached.Inordertoavoidunexpectedtruncationofthedataset,thepartiallycachedcontentsofthedatasetwillbediscarded.Thiscanhappenifyouhaveaninputpipelinesimilarto`dataset.cache().take(k).repeat()`.Youshoulduse`dataset.take(k).cache().repeat()`instead.


Evaluatethenewmodelonthevalidationsetafterinitializationof
quantization.Theaccuracyshouldbenotfarfromtheaccuracyofthe
floating-point``FP32``modelforasimplecaseliketheonebeing
demonstratedhere.

..code::ipython3

#CompiletheINT8model.
int8_model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc@1")],
)

#ValidatetheINT8model.
test_loss,test_acc=int8_model.evaluate(
validation_dataset,
callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=["acc@1"]),
)


..parsed-literal::

4/4[==============================]-1s303ms/sample-loss:0.9766-acc@1:0.8120


Fine-tunetheCompressedModel
------------------------------

`backtotop⬆️<#table-of-contents>`__

Atthisstep,aregularfine-tuningprocessisappliedtofurther
improvequantizedmodelaccuracy.Normally,severalepochsoftuningare
requiredwithasmalllearningrate,thesamethatisusuallyusedat
theendofthetrainingoftheoriginalmodel.Nootherchangesinthe
trainingpipelinearerequired.Hereisasimpleexample.

..code::ipython3

print(f"\nAccuracyofINT8modelafterinitialization:{test_acc:.3f}")

#TraintheINT8model.
int8_model.fit(train_dataset,epochs=2)

#ValidatetheINT8model.
test_loss,acc_int8=int8_model.evaluate(
validation_dataset,
callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=["acc@1"]),
)
print(f"\nAccuracyofINT8modelafterfine-tuning:{acc_int8:.3f}")
print(f"\nAccuracydropoftunedINT8modeloverpre-trainedFP32model:{acc_fp32-acc_int8:.3f}")


..parsed-literal::


AccuracyofINT8modelafterinitialization:0.812
Epoch1/2
101/101[==============================]-49s415ms/step-loss:0.7134-acc@1:0.9299
Epoch2/2
101/101[==============================]-42s413ms/step-loss:0.6807-acc@1:0.9489
4/4[==============================]-1s139ms/sample-loss:0.9760-acc@1:0.8160

AccuracyofINT8modelafterfine-tuning:0.816

AccuracydropoftunedINT8modeloverpre-trainedFP32model:0.006


ExportModelstoOpenVINOIntermediateRepresentation(IR)
----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

UsemodelconversionPythonAPItoconvertthemodelstoOpenVINOIR.

Formoreinformationaboutmodelconversion,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

Executingthiscommandmaytakeawhile.

..code::ipython3

model_ir_fp32=ov.convert_model(fp32_model)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..code::ipython3

model_ir_int8=ov.convert_model(int8_model)

..code::ipython3

ov.save_model(model_ir_fp32,fp32_ir_path,compress_to_fp16=False)
ov.save_model(model_ir_int8,int8_ir_path,compress_to_fp16=False)

BenchmarkModelPerformancebyComputingInferenceTime
-------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Finally,measuretheinferenceperformanceofthe``FP32``and``INT8``
models,using`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-aninferenceperformancemeasurementtoolinOpenVINO.Bydefault,
BenchmarkToolrunsinferencefor60secondsinasynchronousmodeon
CPU.Itreturnsinferencespeedaslatency(millisecondsperimage)and
throughput(framespersecond)values.

**NOTE**:Thisnotebookruns``benchmark_app``for15secondstogive
aquickindicationofperformance.Formoreaccurateperformance,it
isrecommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.Run
``benchmark_app-mmodel.xml-dCPU``tobenchmarkasyncinferenceon
CPUforoneminute.ChangeCPUtoGPUtobenchmarkonGPU.Run
``benchmark_app--help``toseeanoverviewofallcommand-line
options.

Pleaseselectabenchmarkingdeviceusingthedropdownlist:

..code::ipython3

importipywidgetsaswidgets

#InitializeOpenVINOruntime
core=ov.Core()
device=widgets.Dropdown(
options=core.available_devices,
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU',),value='CPU')



..code::ipython3

defparse_benchmark_output(benchmark_output):
parsed_output=[lineforlineinbenchmark_outputif"FPS"inline]
print(*parsed_output,sep="\n")


print("BenchmarkFP32model(IR)")
benchmark_output=!benchmark_app-m$fp32_ir_path-d$device.value-apiasync-t15-shape[1,64,64,3]
parse_benchmark_output(benchmark_output)

print("\nBenchmarkINT8model(IR)")
benchmark_output=!benchmark_app-m$int8_ir_path-d$device.value-apiasync-t15-shape[1,64,64,3]
parse_benchmark_output(benchmark_output)


..parsed-literal::

BenchmarkFP32model(IR)
[INFO]Throughput:2839.00FPS

BenchmarkINT8model(IR)
[INFO]Throughput:11068.25FPS


ShowDeviceInformationforreference.

..code::ipython3

core=ov.Core()
core.get_property(device.value,"FULL_DEVICE_NAME")




..parsed-literal::

'Intel(R)Core(TM)i9-10920XCPU@3.50GHz'



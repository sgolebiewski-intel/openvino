QuantizeNLPmodelswithPost-TrainingQuantization​inNNCF
============================================================

Thistutorialdemonstrateshowtoapply``INT8``quantizationtothe
NaturalLanguageProcessingmodelknownas
`BERT<https://en.wikipedia.org/wiki/BERT_(language_model)>`__,using
the`Post-TrainingQuantization
API<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__
(NNCFlibrary).Afine-tuned`HuggingFace
BERT<https://huggingface.co/transformers/model_doc/bert.html>`__
`PyTorch<https://pytorch.org/>`__model,trainedonthe`Microsoft
ResearchParaphraseCorpus
(MRPC)<https://www.microsoft.com/en-us/download/details.aspx?id=52398>`__,
willbeused.Thetutorialisdesignedtobeextendabletocustommodels
anddatasets.Itconsistsofthefollowingsteps:

-DownloadandpreparetheBERTmodelandMRPCdataset.
-Definedataloadingandaccuracyvalidationfunctionality.
-Preparethemodelforquantization.
-Runoptimizationpipeline.
-Loadandtestquantizedmodel.
-Comparetheperformanceoftheoriginal,convertedandquantized
models.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Settings<#settings>`__
-`PreparetheModel<#prepare-the-model>`__
-`PreparetheDataset<#prepare-the-dataset>`__
-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__
-`LoadandTestOpenVINOModel<#load-and-test-openvino-model>`__

-`Selectinferencedevice<#select-inference-device>`__

-`CompareF1-scoreofFP32andINT8
models<#compare-f1-score-of-fp32-and-int8-models>`__
-`ComparePerformanceoftheOriginal,ConvertedandQuantized
Models<#compare-performance-of-the-original-converted-and-quantized-models>`__

..code::ipython3

%pipinstall-q"nncf>=2.5.0"
%pipinstall-qtorchtransformers"torch>=2.1"datasetsevaluatetqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos
importtime
frompathlibimportPath
fromzipfileimportZipFile
fromtypingimportIterable
fromtypingimportAny

importdatasets
importevaluate
importnumpyasnp
importnncf
fromnncf.parametersimportModelType
importopenvinoasov
importtorch
fromtransformersimportBertForSequenceClassification,BertTokenizer

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file


..parsed-literal::

2024-07-1300:54:14.849761:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1300:54:14.884698:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1300:54:15.454440:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Settings
--------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Setthedataandmodeldirectories,sourceURLandthefilenameofthemodel.
DATA_DIR="data"
MODEL_DIR="model"
MODEL_LINK="https://download.pytorch.org/tutorial/MRPC.zip"
FILE_NAME=MODEL_LINK.split("/")[-1]
PRETRAINED_MODEL_DIR=os.path.join(MODEL_DIR,"MRPC")

os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

PreparetheModel
-----------------

`backtotop⬆️<#table-of-contents>`__

Performthefollowing:

-Downloadandunpackpre-trainedBERTmodelforMRPCbyPyTorch.
-ConvertthemodeltotheOpenVINOIntermediateRepresentation
(OpenVINOIR)

..code::ipython3

download_file(MODEL_LINK,directory=MODEL_DIR,show_progress=True)
withZipFile(f"{MODEL_DIR}/{FILE_NAME}","r")aszip_ref:
zip_ref.extractall(MODEL_DIR)



..parsed-literal::

model/MRPC.zip:0%||0.00/387M[00:00<?,?B/s]


ConverttheoriginalPyTorchmodeltotheOpenVINOIntermediate
Representation.

FromOpenVINO2023.0,wecandirectlyconvertamodelfromthePyTorch
formattotheOpenVINOIRformatusingmodelconversionAPI.Following
PyTorchmodelformatsaresupported:

-``torch.nn.Module``
-``torch.jit.ScriptModule``
-``torch.jit.ScriptFunction``

..code::ipython3

MAX_SEQ_LENGTH=128
input_shape=ov.PartialShape([1,-1])
ir_model_xml=Path(MODEL_DIR)/"bert_mrpc.xml"
core=ov.Core()

torch_model=BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
torch_model.eval

input_info=[
("input_ids",input_shape,np.int64),
("attention_mask",input_shape,np.int64),
("token_type_ids",input_shape,np.int64),
]
default_input=torch.ones(1,MAX_SEQ_LENGTH,dtype=torch.int64)
inputs={
"input_ids":default_input,
"attention_mask":default_input,
"token_type_ids":default_input,
}

#ConvertthePyTorchmodeltoOpenVINOIRFP32.
ifnotir_model_xml.exists():
model=ov.convert_model(torch_model,example_input=inputs,input=input_info)
ov.save_model(model,str(ir_model_xml))
else:
model=core.read_model(ir_model_xml)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(


PreparetheDataset
-------------------

`backtotop⬆️<#table-of-contents>`__

Wedownloadthe`GeneralLanguageUnderstandingEvaluation
(GLUE)<https://gluebenchmark.com/>`__datasetfortheMRPCtaskfrom
HuggingFacedatasets.Then,wetokenizethedatawithapre-trainedBERT
tokenizerfromHuggingFace.

..code::ipython3

defcreate_data_source():
raw_dataset=datasets.load_dataset("glue","mrpc",split="validation")
tokenizer=BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)

def_preprocess_fn(examples):
texts=(examples["sentence1"],examples["sentence2"])
result=tokenizer(*texts,padding="max_length",max_length=MAX_SEQ_LENGTH,truncation=True)
result["labels"]=examples["label"]
returnresult

processed_dataset=raw_dataset.map(_preprocess_fn,batched=True,batch_size=1)

returnprocessed_dataset


data_source=create_data_source()

OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimizeBERT.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization
2.Run``nncf.quantize``forgettinganoptimizedmodel
3.SerializeOpenVINOIRmodelusing``openvino.save_model``function

..code::ipython3

INPUT_NAMES=[keyforkeyininputs.keys()]


deftransform_fn(data_item):
"""
Extractthemodel'sinputfromthedataitem.
Thedataitemhereisthedataitemthatisreturnedfromthedatasourceperiteration.
Thisfunctionshouldbepassedwhenthedataitemcannotbeusedasmodel'sinput.
"""
inputs={name:np.asarray([data_item[name]],dtype=np.int64)fornameinINPUT_NAMES}
returninputs


calibration_dataset=nncf.Dataset(data_source,transform_fn)
#Quantizethemodel.Byspecifyingmodel_type,wespecifyadditionaltransformerpatternsinthemodel.
quantized_model=nncf.quantize(model,calibration_dataset,model_type=ModelType.TRANSFORMER)



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



..parsed-literal::

INFO:nncf:50ignorednodeswerefoundbynameintheNNCFGraph



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



..code::ipython3

compressed_model_xml=Path(MODEL_DIR)/"quantized_bert_mrpc.xml"
ov.save_model(quantized_model,compressed_model_xml)

LoadandTestOpenVINOModel
----------------------------

`backtotop⬆️<#table-of-contents>`__

Toloadandtestconvertedmodel,performthefollowing:

-Loadthemodelandcompileitforselecteddevice.
-Preparetheinput.
-Runtheinference.
-Gettheanswerfromthemodeloutput.

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

#Compilethemodelforaspecificdevice.
compiled_quantized_model=core.compile_model(model=quantized_model,device_name=device.value)
output_layer=compiled_quantized_model.outputs[0]

TheDataSourcereturnsapairofsentences(indicatedby
``sample_idx``)andtheinferencecomparesthesesentencesandoutputs
whethertheirmeaningisthesame.Youcantestothersentencesby
changing``sample_idx``toanothervalue(from0to407).

..code::ipython3

sample_idx=5
sample=data_source[sample_idx]
inputs={k:torch.unsqueeze(torch.tensor(sample[k]),0)forkin["input_ids","token_type_ids","attention_mask"]}

result=compiled_quantized_model(inputs)[output_layer]
result=np.argmax(result)

print(f"Text1:{sample['sentence1']}")
print(f"Text2:{sample['sentence2']}")
print(f"Thesamemeaning:{'yes'ifresult==1else'no'}")


..parsed-literal::

Text1:Wal-Martsaiditwouldcheckallofitsmillion-plusdomesticworkerstoensuretheywerelegallyemployed.
Text2:Ithasalsosaiditwouldreviewallofitsdomesticemployeesmorethan1milliontoensuretheyhavelegalstatus.
Thesamemeaning:yes


CompareF1-scoreofFP32andINT8models
----------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defvalidate(model:ov.Model,dataset:Iterable[Any])->float:
"""
EvaluatethemodelonGLUEdataset.
ReturnsF1scoremetric.
"""
compiled_model=core.compile_model(model,device_name=device.value)
output_layer=compiled_model.output(0)

metric=evaluate.load("glue","mrpc")
forbatchindataset:
inputs=[np.expand_dims(np.asarray(batch[key],dtype=np.int64),0)forkeyinINPUT_NAMES]
outputs=compiled_model(inputs)[output_layer]
predictions=outputs[0].argmax(axis=-1)
metric.add_batch(predictions=[predictions],references=[batch["labels"]])
metrics=metric.compute()
f1_score=metrics["f1"]

returnf1_score


print("Checkingtheaccuracyoftheoriginalmodel:")
metric=validate(model,data_source)
print(f"F1score:{metric:.4f}")

print("Checkingtheaccuracyofthequantizedmodel:")
metric=validate(quantized_model,data_source)
print(f"F1score:{metric:.4f}")


..parsed-literal::

Checkingtheaccuracyoftheoriginalmodel:
F1score:0.9019
Checkingtheaccuracyofthequantizedmodel:
F1score:0.8969


ComparePerformanceoftheOriginal,ConvertedandQuantizedModels
-------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ComparetheoriginalPyTorchmodelwithOpenVINOconvertedandquantized
models(``FP32``,``INT8``)toseethedifferenceinperformance.Itis
expressedinSentencesPerSecond(SPS)measure,whichisthesameas
FramesPerSecond(FPS)forimages.

..code::ipython3

#Compilethemodelforaspecificdevice.
compiled_model=core.compile_model(model=model,device_name=device.value)

..code::ipython3

num_samples=50
sample=data_source[0]
inputs={k:torch.unsqueeze(torch.tensor(sample[k]),0)forkin["input_ids","token_type_ids","attention_mask"]}

withtorch.no_grad():
start=time.perf_counter()
for_inrange(num_samples):
torch_model(torch.vstack(list(inputs.values())))
end=time.perf_counter()
time_torch=end-start
print(f"PyTorchmodelonCPU:{time_torch/num_samples:.3f}secondspersentence,"f"SPS:{num_samples/time_torch:.2f}")

start=time.perf_counter()
for_inrange(num_samples):
compiled_model(inputs)
end=time.perf_counter()
time_ir=end-start
print(f"IRFP32modelinOpenVINORuntime/{device.value}:{time_ir/num_samples:.3f}"f"secondspersentence,SPS:{num_samples/time_ir:.2f}")

start=time.perf_counter()
for_inrange(num_samples):
compiled_quantized_model(inputs)
end=time.perf_counter()
time_ir=end-start
print(f"OpenVINOIRINT8modelinOpenVINORuntime/{device.value}:{time_ir/num_samples:.3f}"f"secondspersentence,SPS:{num_samples/time_ir:.2f}")


..parsed-literal::

Westronglyrecommendpassinginan`attention_mask`sinceyourinput_idsmaybepadded.Seehttps://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.


..parsed-literal::

PyTorchmodelonCPU:0.071secondspersentence,SPS:14.14
IRFP32modelinOpenVINORuntime/AUTO:0.021secondspersentence,SPS:47.94
OpenVINOIRINT8modelinOpenVINORuntime/AUTO:0.009secondspersentence,SPS:105.56


Finally,measuretheinferenceperformanceofOpenVINO``FP32``and
``INT8``models.Forthispurpose,use`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
inOpenVINO.

**Note**:The``benchmark_app``toolisabletomeasurethe
performanceoftheOpenVINOIntermediateRepresentation(OpenVINOIR)
modelsonly.Formoreaccurateperformance,run``benchmark_app``in
aterminal/commandpromptafterclosingotherapplications.Run
``benchmark_app-mmodel.xml-dCPU``tobenchmarkasyncinferenceon
CPUforoneminute.Change``CPU``to``GPU``tobenchmarkonGPU.
Run``benchmark_app--help``toseeanoverviewofallcommand-line
options.

..code::ipython3

#InferenceFP32model(OpenVINOIR)
!benchmark_app-m$ir_model_xml-shape[1,128],[1,128],[1,128]-d{device.value}-apisync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.LATENCY.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook17.97ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,?]
[INFO]attention_mask,63(node:attention_mask):i64/[...]/[1,?]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,?]
[INFO]Modeloutputs:
[INFO]logits(node:__module.classifier/aten::linear/Add):f32/[...]/[1,2]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'input_ids':[1,128],'63':[1,128],'token_type_ids':[1,128]
[INFO]Reshapemodeltook5.17ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,128]
[INFO]attention_mask,63(node:attention_mask):i64/[...]/[1,128]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,128]
[INFO]Modeloutputs:
[INFO]logits(node:__module.classifier/aten::linear/Add):f32/[...]/[1,2]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook412.90ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:False
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:12
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:1
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'input_ids'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'63'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'token_type_ids'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'input_ids'withrandomvalues
[INFO]Fillinput'63'withrandomvalues
[INFO]Fillinput'token_type_ids'withrandomvalues
[Step10/11]Measuringperformance(Startinferencesynchronously,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook31.38ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:6026iterations
[INFO]Duration:120012.51ms
[INFO]Latency:
[INFO]Median:19.74ms
[INFO]Average:19.82ms
[INFO]Min:18.76ms
[INFO]Max:22.82ms
[INFO]Throughput:50.21FPS


..code::ipython3

#InferenceINT8model(OpenVINOIR)
!benchmark_app-m$compressed_model_xml-shape[1,128],[1,128],[1,128]-d{device.value}-apisync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.LATENCY.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook23.50ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,?]
[INFO]63,attention_mask(node:attention_mask):i64/[...]/[1,?]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,?]
[INFO]Modeloutputs:
[INFO]logits(node:__module.classifier/aten::linear/Add):f32/[...]/[1,2]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'input_ids':[1,128],'63':[1,128],'token_type_ids':[1,128]
[INFO]Reshapemodeltook6.93ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,128]
[INFO]63,attention_mask(node:attention_mask):i64/[...]/[1,128]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,128]
[INFO]Modeloutputs:
[INFO]logits(node:__module.classifier/aten::linear/Add):f32/[...]/[1,2]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1164.57ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:False
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:12
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:1
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'input_ids'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'63'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'token_type_ids'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'input_ids'withrandomvalues
[INFO]Fillinput'63'withrandomvalues
[INFO]Fillinput'token_type_ids'withrandomvalues
[Step10/11]Measuringperformance(Startinferencesynchronously,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook16.55ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:12217iterations
[INFO]Duration:120007.78ms
[INFO]Latency:
[INFO]Median:9.78ms
[INFO]Average:9.73ms
[INFO]Min:8.31ms
[INFO]Max:10.71ms
[INFO]Throughput:101.80FPS


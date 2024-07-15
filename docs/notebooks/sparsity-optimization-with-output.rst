AccelerateInferenceofSparseTransformerModelswithOpenVINO™and4thGenIntel®Xeon®ScalableProcessors
=============================================================================================================

Thistutorialdemonstrateshowtoimproveperformanceofsparse
Transformermodelswith`OpenVINO<https://docs.openvino.ai/>`__on4th
GenIntel®Xeon®Scalableprocessors.

Thetutorialdownloads`aBERT-base
model<https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80>`__
whichhasbeenquantized,sparsified,andtunedfor`SST2
datasets<https://huggingface.co/datasets/sst2>`__using
`Optimum-Intel<https://github.com/huggingface/optimum-intel>`__.It
demonstratestheinferenceperformanceadvantageon4thGenIntel®Xeon®
ScalableProcessorsbyrunningitwith`SparseWeight
Decompression<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#sparse-weights-decompression-intel-x86-64>`__,
aruntimeoptionthatseizesmodelsparsityforefficiency.Thenotebook
consistsofthefollowingsteps:

-Installprerequisites
-DownloadandquantizesparsepublicBERTmodel,usingtheOpenVINO
integrationwithHuggingFaceOptimum.
-Comparesparse8-bitvs. dense8-bitinferenceperformance.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Imports<#imports>`__

-`Download,quantizeandsparsifythemodel,usingHuggingFace
Optimum
API<#download-quantize-and-sparsify-the-model-using-hugging-face-optimum-api>`__

-`Benchmarkquantizeddenseinference
performance<#benchmark-quantized-dense-inference-performance>`__
-`Benchmarkquantizedsparseinference
performance<#benchmark-quantized-sparse-inference-performance>`__
-`Whenthismightbehelpful<#when-this-might-be-helpful>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git""torch>=2.1"datasetsonnxtransformers>=4.33.0--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importshutil
frompathlibimportPath

fromoptimum.intel.openvinoimportOVModelForSequenceClassification
fromtransformersimportAutoTokenizer,pipeline
fromhuggingface_hubimporthf_hub_download


..parsed-literal::

2024-07-1303:25:40.698595:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1303:25:40.733249:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1303:25:41.315473:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(


Download,quantizeandsparsifythemodel,usingHuggingFaceOptimumAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thefirststepistodownloadaquantizedsparsetransformerswhichhas
beentranslatedtoOpenVINOIR.Then,itwillbeputthrougha
classificationasasimplevalidationofaworkingdownloadedmodel.To
findouthowthemodelisbeingquantizedandsparsified,refertothe
`OpenVINO/bert-base-uncased-sst2-int8-unstructured80<https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80>`__
modelcardonHuggingFace.

..code::ipython3

#Thefollowingmodelhasbeenquantized,sparsifiedusingOptimum-Intel1.7whichisenabledbyOpenVINOandNNCF
#forreproducibility,referhttps://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80
model_id="OpenVINO/bert-base-uncased-sst2-int8-unstructured80"

#ThefollowingtwostepswillsetupthemodelanddownloadthemtoHFCachefolder
ov_model=OVModelForSequenceClassification.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)

#Let'stakethemodelforaspin!
sentiment_classifier=pipeline("text-classification",model=ov_model,tokenizer=tokenizer)

text="He'sadreadfulmagician."
outputs=sentiment_classifier(text)

print(outputs)


..parsed-literal::

CompilingthemodeltoCPU...


..parsed-literal::

[{'label':'negative','score':0.9982142448425293}]


Forbenchmarking,wewilluseOpenVINO’sbenchmarkapplicationandput
theIRsintoasinglefolder.

..code::ipython3

#createafolder
quantized_sparse_dir=Path("bert_80pc_sparse_quantized_ir")
quantized_sparse_dir.mkdir(parents=True,exist_ok=True)

#followingreturnpathtospecifiedfilenameincachefolder(whichwe'vewiththe
ov_ir_xml_path=hf_hub_download(repo_id=model_id,filename="openvino_model.xml")
ov_ir_bin_path=hf_hub_download(repo_id=model_id,filename="openvino_model.bin")

#copyIRstothefolder
shutil.copy(ov_ir_xml_path,quantized_sparse_dir)
shutil.copy(ov_ir_bin_path,quantized_sparse_dir)




..parsed-literal::

'bert_80pc_sparse_quantized_ir/openvino_model.bin'



Benchmarkquantizeddenseinferenceperformance
-----------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Benchmarkdenseinferenceperformanceusingparallelexecutiononfour
CPUcorestosimulateasmallinstanceinthecloudinfrastructure.
Sequencelengthisdependentonusecases,16iscommonfor
conversationalAIwhile160forquestionansweringtask.Itissetto64
asanexample.Itisrecommendedtotunebasedonyourapplications.

..code::ipython3

#Dumpbenchmarkingconfigfordenseinference
with(quantized_sparse_dir/"perf_config.json").open("w")asoutfile:
outfile.write(
"""
{
"CPU":{"NUM_STREAMS":4,"INFERENCE_NUM_THREADS":4}
}
"""
)

..code::ipython3

!benchmark_app-m$quantized_sparse_dir/openvino_model.xml-shape"input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]"-load_config$quantized_sparse_dir/perf_config.json


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(CPU)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook62.05ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[?,?]
[INFO]attention_mask(node:attention_mask):i64/[...]/[?,?]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[?,?]
[INFO]Modeloutputs:
[INFO]logits(node:logits):f32/[...]/[?,2]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'input_ids':[1,64],'attention_mask':[1,64],'token_type_ids':[1,64]
[INFO]Reshapemodeltook28.43ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,64]
[INFO]attention_mask(node:attention_mask):i64/[...]/[1,64]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,64]
[INFO]Modeloutputs:
[INFO]logits(node:logits):f32/[...]/[1,2]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1005.52ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:torch_jit
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:4
[INFO]NUM_STREAMS:4
[INFO]INFERENCE_NUM_THREADS:4
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]AFFINITY:Affinity.CORE
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'input_ids'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'attention_mask'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'token_type_ids'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'input_ids'withrandomvalues
[INFO]Fillinput'attention_mask'withrandomvalues
[INFO]Fillinput'token_type_ids'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,4inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook27.14ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:9192iterations
[INFO]Duration:60045.59ms
[INFO]Latency:
[INFO]Median:25.82ms
[INFO]Average:25.87ms
[INFO]Min:24.44ms
[INFO]Max:40.26ms
[INFO]Throughput:153.08FPS


Benchmarkquantizedsparseinferenceperformance
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Toenablesparseweightdecompressionfeature,userscanadditto
runtimeconfiglikebelow.``CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE``
takesvaluesbetween0.5and1.0.Itisalayer-levelsparsitythreshold
forwhichalayerwillbeenabled.

..code::ipython3

#Dumpbenchmarkingconfigfordenseinference
#"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE"controlsminimumsparsityrateforweightstoconsider
#forsparseoptimizationattheruntime.
with(quantized_sparse_dir/"perf_config_sparse.json").open("w")asoutfile:
outfile.write(
"""
{
"CPU":{"NUM_STREAMS":4,"INFERENCE_NUM_THREADS":4,"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE":"0.75"}
}
"""
)

..code::ipython3

!benchmark_app-m$quantized_sparse_dir/openvino_model.xml-shape"input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]"-load_config$quantized_sparse_dir/perf_config_sparse.json


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(CPU)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook89.36ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[?,?]
[INFO]attention_mask(node:attention_mask):i64/[...]/[?,?]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[?,?]
[INFO]Modeloutputs:
[INFO]logits(node:logits):f32/[...]/[?,2]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'input_ids':[1,64],'attention_mask':[1,64],'token_type_ids':[1,64]
[INFO]Reshapemodeltook28.62ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]input_ids(node:input_ids):i64/[...]/[1,64]
[INFO]attention_mask(node:attention_mask):i64/[...]/[1,64]
[INFO]token_type_ids(node:token_type_ids):i64/[...]/[1,64]
[INFO]Modeloutputs:
[INFO]logits(node:logits):f32/[...]/[1,2]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1091.53ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:torch_jit
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:4
[INFO]NUM_STREAMS:4
[INFO]INFERENCE_NUM_THREADS:4
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:0.75
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]AFFINITY:Affinity.CORE
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'input_ids'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'attention_mask'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'token_type_ids'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'input_ids'withrandomvalues
[INFO]Fillinput'attention_mask'withrandomvalues
[INFO]Fillinput'token_type_ids'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,4inferencerequests,limits:60000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook28.28ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:9176iterations
[INFO]Duration:60035.45ms
[INFO]Latency:
[INFO]Median:25.86ms
[INFO]Average:25.90ms
[INFO]Min:23.07ms
[INFO]Max:41.68ms
[INFO]Throughput:152.84FPS


Whenthismightbehelpful
--------------------------

`backtotop⬆️<#table-of-contents>`__

Thisfeaturecanimproveinferenceperformanceformodelswithsparse
weightsinthescenarioswhenthemodelisdeployedtohandlemultiple
requestsinparallelasynchronously.Itisespeciallyhelpfulwitha
smallsequencelength,forexample,32andlower.

FormoredetailsaboutasynchronousinferencewithOpenVINO,referto
thefollowingdocumentation:

-`DeploymentOptimization
Guide<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html>`__
-`InferenceRequest
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html>`__

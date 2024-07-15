NamedentityrecognitionwithOpenVINO™
=======================================

TheNamedEntityRecognition(NER)isanaturallanguageprocessing
methodthatinvolvesthedetectingofkeyinformationinthe
unstructuredtextandcategorizingitintopre-definedcategories.These
categoriesornamedentitiesrefertothekeysubjectsoftext,suchas
names,locations,companiesandetc.

NERisagoodmethodforthesituationswhenahigh-leveloverviewofa
largeamountoftextisneeded.NERcanbehelpfulwithsuchtaskas
analyzingkeyinformationinunstructuredtextorautomatesthe
informationextractionoflargeamountsofdata.

Thistutorialshowshowtoperformnamedentityrecognitionusing
OpenVINO.Wewillusethepre-trainedmodel
`elastic/distilbert-base-cased-finetuned-conll03-english<https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__.
ItisDistilBERTbasedmodel,trainedon
`conll03englishdataset<https://huggingface.co/datasets/conll2003>`__.
Themodelcanrecognizefournamedentitiesintext:persons,locations,
organizationsandnamesofmiscellaneousentitiesthatdonotbelongto
thepreviousthreegroups.Themodelissensitivetocapitalletters.

Tosimplifytheuserexperience,the`HuggingFace
Optimum<https://huggingface.co/docs/optimum>`__libraryisusedto
convertthemodeltoOpenVINO™IRformatandquantizeit.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`DownloadtheNERmodel<#download-the-ner-model>`__
-`Quantizethemodel,usingHuggingFaceOptimum
API<#quantize-the-model-using-hugging-face-optimum-api>`__
-`ComparetheOriginalandQuantized
Models<#compare-the-original-and-quantized-models>`__

-`Compareperformance<#compare-performance>`__
-`Comparesizeofthemodels<#compare-size-of-the-models>`__

-`PreparedemoforNamedEntityRecognitionOpenVINO
Runtime<#prepare-demo-for-named-entity-recognition-openvino-runtime>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"diffusers>=0.17.1""openvino>=2023.1.0""nncf>=2.5.0""gradio>=4.19""onnx>=1.11.0""transformers>=4.33.0""torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


DownloadtheNERmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

Weloadthe
`distilbert-base-cased-finetuned-conll03-english<https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
modelfromthe`HuggingFaceHub<https://huggingface.co/models>`__with
`HuggingFaceTransformers
library<https://huggingface.co/docs/transformers/index>`__\andOptimum
IntelwithOpenVINOintegration.

``OVModelForTokenClassification``isrepresentmodelclassforNamed
EntityRecognitiontaskinOptimumIntel.Modelclassinitialization
startswithcalling``from_pretrained``method.Forconversionoriginal
PyTorchmodeltoOpenVINOformatonthefly,``export=True``parameter
shouldbeused.Toeasilysavethemodel,youcanusethe
``save_pretrained()``method.Aftersavingthemodelondisk,wecanuse
pre-convertedmodelfornextusage,andspeedupdeploymentprocess.

..code::ipython3

frompathlibimportPath
fromtransformersimportAutoTokenizer
fromoptimum.intelimportOVModelForTokenClassification

original_ner_model_dir=Path("original_ner_model")

model_id="elastic/distilbert-base-cased-finetuned-conll03-english"
ifnotoriginal_ner_model_dir.exists():
model=OVModelForTokenClassification.from_pretrained(model_id,export=True)

model.save_pretrained(original_ner_model_dir)
else:
model=OVModelForTokenClassification.from_pretrained(model_id,export=True)

tokenizer=AutoTokenizer.from_pretrained(model_id)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

NoCUDAruntimeisfound,usingCUDA_HOME='/usr/local/cuda'
2024-04-0518:35:04.594311:Itensorflow/core/util/port.cc:111]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-04-0518:35:04.596755:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-0518:35:04.628293:Etensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342]UnabletoregistercuDNNfactory:AttemptingtoregisterfactoryforplugincuDNNwhenonehasalreadybeenregistered
2024-04-0518:35:04.628326:Etensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609]UnabletoregistercuFFTfactory:AttemptingtoregisterfactoryforplugincuFFTwhenonehasalreadybeenregistered
2024-04-0518:35:04.628349:Etensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518]UnabletoregistercuBLASfactory:AttemptingtoregisterfactoryforplugincuBLASwhenonehasalreadybeenregistered
2024-04-0518:35:04.634704:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-0518:35:04.635314:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-04-0518:35:05.607762:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/utils/import_utils.py:519:FutureWarning:`is_torch_tpu_available`isdeprecatedandwillberemovedin4.41.0.Pleaseusethe`is_torch_xla_available`instead.
warnings.warn(
Frameworknotspecified.Usingpttoexportthemodel.
Usingtheexportvariantdefault.Availablevariantsare:
-default:ThedefaultONNXvariant.
UsingframeworkPyTorch:2.1.2+cpu
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:4225:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/home/ea/miniconda3/lib/python3.11/site-packages/nncf/torch/dynamic_graph/wrappers.py:80:TracerWarning:torch.tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
op1=operator(*args,**kwargs)
CompilingthemodeltoCPU...


Quantizethemodel,usingHuggingFaceOptimumAPI
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Post-trainingstaticquantizationintroducesanadditionalcalibration
stepwheredataisfedthroughthenetworkinordertocomputethe
activationsquantizationparameters.Forquantizationitwillbeused
`HuggingFaceOptimumIntel
API<https://huggingface.co/docs/optimum/intel/index>`__.

TohandletheNNCFquantizationprocessweuseclass
`OVQuantizer<https://huggingface.co/docs/optimum/intel/reference_ov#optimum.intel.OVQuantizer>`__.
ThequantizationwithHuggingFaceOptimumIntelAPIcontainsthenext
steps:\*Modelclassinitializationstartswithcalling
``from_pretrained()``method.\*Nextwecreatecalibrationdatasetwith
``get_calibration_dataset()``touseforthepost-trainingstatic
quantizationcalibrationstep.\*Afterwequantizeamodelandsavethe
resultingmodelintheOpenVINOIRformattosave_directorywith
``quantize()``method.\*Thenweloadthequantizedmodel.TheOptimum
InferencemodelsareAPIcompatiblewithHuggingFaceTransformers
modelsandwecanjustreplace``AutoModelForXxx``classwiththe
corresponding``OVModelForXxx``class.Soweuse
``OVModelForTokenClassification``toloadthemodel.

..code::ipython3

fromfunctoolsimportpartial
fromoptimum.intelimportOVQuantizer,OVConfig,OVQuantizationConfig

fromoptimum.intelimportOVModelForTokenClassification


defpreprocess_fn(data,tokenizer):
examples=[]
fordata_chunkindata["tokens"]:
examples.append("".join(data_chunk))

returntokenizer(examples,padding=True,truncation=True,max_length=128)


quantizer=OVQuantizer.from_pretrained(model)
calibration_dataset=quantizer.get_calibration_dataset(
"conll2003",
preprocess_function=partial(preprocess_fn,tokenizer=tokenizer),
num_samples=100,
dataset_split="train",
preprocess_batch=True,
trust_remote_code=True,
)

#Thedirectorywherethequantizedmodelwillbesaved
quantized_ner_model_dir="quantized_ner_model"

#ApplystaticquantizationandsavetheresultingmodelintheOpenVINOIRformat
ov_config=OVConfig(quantization_config=OVQuantizationConfig(num_samples=len(calibration_dataset)))
quantizer.quantize(
calibration_dataset=calibration_dataset,
save_directory=quantized_ner_model_dir,
ov_config=ov_config,
)


..parsed-literal::

/home/ea/miniconda3/lib/python3.11/site-packages/datasets/load.py:2516:FutureWarning:'use_auth_token'wasdeprecatedinfavorof'token'inversion2.14.0andwillberemovedin3.0.0.
Youcanremovethiswarningbypassing'token=<use_auth_token>'instead.
warnings.warn(



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

INFO:nncf:18ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:25ignorednodeswerefoundbynameintheNNCFGraph



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

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



..code::ipython3

#Loadthequantizedmodel
optimized_model=OVModelForTokenClassification.from_pretrained(quantized_ner_model_dir,device=device.value)


..parsed-literal::

CompilingthemodeltoAUTO...


ComparetheOriginalandQuantizedModels
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Comparetheoriginal
`distilbert-base-cased-finetuned-conll03-english<https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
modelwithquantizedandconvertedtoOpenVINOIRformatmodelstosee
thedifference.

Compareperformance
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

AstheOptimumInferencemodelsareAPIcompatiblewithHuggingFace
Transformersmodels,wecanjustuse``pipleine()``from`HuggingFace
TransformersAPI<https://huggingface.co/docs/transformers/index>`__for
inference.

..code::ipython3

fromtransformersimportpipeline

ner_pipeline_optimized=pipeline("token-classification",model=optimized_model,tokenizer=tokenizer)

ner_pipeline_original=pipeline("token-classification",model=model,tokenizer=tokenizer)

..code::ipython3

importtime
importnumpyasnp


defcalc_perf(ner_pipeline):
inference_times=[]

fordataincalibration_dataset:
text="".join(data["tokens"])
start=time.perf_counter()
ner_pipeline(text)
end=time.perf_counter()
inference_times.append(end-start)

returnnp.median(inference_times)


print(f"Medianinferencetimeofquantizedmodel:{calc_perf(ner_pipeline_optimized)}")

print(f"Medianinferencetimeoforiginalmodel:{calc_perf(ner_pipeline_original)}")


..parsed-literal::

Medianinferencetimeofquantizedmodel:0.0063508255407214165
Medianinferencetimeoforiginalmodel:0.007429798366501927


Comparesizeofthemodels
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath

fp_model_file=Path(original_ner_model_dir)/"openvino_model.bin"
print(f"SizeoforiginalmodelinBytesis{fp_model_file.stat().st_size}")
print(f'SizeofquantizedmodelinBytesis{Path(quantized_ner_model_dir,"openvino_model.bin").stat().st_size}')


..parsed-literal::

SizeoforiginalmodelinBytesis260795516
SizeofquantizedmodelinBytesis65802712


PreparedemoforNamedEntityRecognitionOpenVINORuntime
----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Now,youcantryNERmodelonowntext.Putyoursentencetoinputtext
box,clickSubmitbutton,themodellabeltherecognizedentitiesinthe
text.

..code::ipython3

importgradioasgr

examples=[
"MynameisWolfgangandIliveinBerlin.",
]


defrun_ner(text):
output=ner_pipeline_optimized(text)
return{"text":text,"entities":output}


demo=gr.Interface(
run_ner,
gr.Textbox(placeholder="Entersentencehere...",label="InputText"),
gr.HighlightedText(label="OutputText"),
examples=examples,
allow_flagging="never",
)

if__name__=="__main__":
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

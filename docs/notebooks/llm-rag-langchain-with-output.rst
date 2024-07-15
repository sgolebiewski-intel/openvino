CreateaRAGsystemusingOpenVINOandLangChain
================================================

**Retrieval-augmentedgeneration(RAG)**isatechniqueforaugmenting
LLMknowledgewithadditional,oftenprivateorreal-time,data.LLMs
canreasonaboutwide-rangingtopics,buttheirknowledgeislimitedto
thepublicdatauptoaspecificpointintimethattheyweretrained
on.IfyouwanttobuildAIapplicationsthatcanreasonaboutprivate
dataordataintroducedafteramodel’scutoffdate,youneedtoaugment
theknowledgeofthemodelwiththespecificinformationitneeds.The
processofbringingtheappropriateinformationandinsertingitinto
themodelpromptisknownasRetrievalAugmentedGeneration(RAG).

`LangChain<https://python.langchain.com/docs/get_started/introduction>`__
isaframeworkfordevelopingapplicationspoweredbylanguagemodels.
IthasanumberofcomponentsspecificallydesignedtohelpbuildRAG
applications.Inthistutorial,we’llbuildasimplequestion-answering
applicationoveratextdatasource.

Thetutorialconsistsofthefollowingsteps:

-Installprerequisites
-Downloadandconvertthemodelfromapublicsourceusingthe
`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-Compressmodelweightsto4-bitor8-bitdatatypesusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__
-CreateaRAGchainpipeline
-RunQ&Apipeline

Inthisexample,thecustomizedRAGpipelineconsistsoffollowing
componentsinorder,whereembedding,rerankandLLMwillbedeployed
withOpenVINOtooptimizetheirinferenceperformance.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/0076f6c7-75e4-4c2e-9015-87b355e5ca28
:alt:RAG

RAG

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Selectmodelforinference<#select-model-for-inference>`__
-`logintohuggingfacehubtogetaccesstopretrained
model<#login-to-huggingfacehub-to-get-access-to-pretrained-model>`__
-`Convertmodelandcompressmodel
weights<#convert-model-and-compress-model-weights>`__

-`LLMconversionandWeightsCompressionusing
Optimum-CLI<#llm-conversion-and-weights-compression-using-optimum-cli>`__

-`WeightcompressionwithAWQ<#weight-compression-with-awq>`__

-`Convertembeddingmodelusing
Optimum-CLI<#convert-embedding-model-using-optimum-cli>`__
-`Convertrerankmodelusing
Optimum-CLI<#convert-rerank-model-using-optimum-cli>`__

-`Selectdeviceforinferenceandmodel
variant<#select-device-for-inference-and-model-variant>`__

-`Selectdeviceforembeddingmodel
inference<#select-device-for-embedding-model-inference>`__
-`Selectdeviceforrerankmodel
inference<#select-device-for-rerank-model-inference>`__
-`SelectdeviceforLLMmodel
inference<#select-device-for-llm-model-inference>`__

-`Loadmodel<#load-model>`__

-`Loadembeddingmodel<#load-embedding-model>`__
-`Loadrerankmodel<#load-rerank-model>`__
-`LoadLLMmodel<#load-llm-model>`__

-`RunQAoverDocument<#run-qa-over-document>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequireddependencies

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipinstall-Uqpip
%pipuninstall-q-yoptimumoptimum-intel
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"git+https://github.com/huggingface/optimum-intel.git"\
"git+https://github.com/openvinotoolkit/nncf.git"\
"datasets"\
"accelerate"\
"gradio"\
"onnx""einops""transformers_stream_generator""tiktoken""transformers>=4.40""bitsandbytes""faiss-cpu""sentence_transformers""langchain>=0.2.0""langchain-community>=0.2.0""langchainhub""unstructured""scikit-learn""python-docx""pypdf"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
WARNING:typer0.12.3doesnotprovidetheextra'all'
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
llama-index-postprocessor-openvino-rerank0.1.3requireshuggingface-hub<0.21.0,>=0.20.3,butyouhavehuggingface-hub0.23.4whichisincompatible.
llama-index-llms-langchain0.1.4requireslangchain<0.2.0,>=0.1.3,butyouhavelangchain0.2.6whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importos
frompathlibimportPath
importrequests
importshutil
importio

#fetchmodelconfiguration

config_shared_path=Path("../../utils/llm_config.py")
config_dst_path=Path("llm_config.py")
text_example_en_path=Path("text_example_en.pdf")
text_example_cn_path=Path("text_example_cn.pdf")
text_example_en="https://github.com/openvinotoolkit/openvino_notebooks/files/15039728/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final.pdf"
text_example_cn="https://github.com/openvinotoolkit/openvino_notebooks/files/15039713/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final_CH.pdf"

ifnotconfig_dst_path.exists():
ifconfig_shared_path.exists():
try:
os.symlink(config_shared_path,config_dst_path)
exceptException:
shutil.copy(config_shared_path,config_dst_path)
else:
r=requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
withopen("llm_config.py","w",encoding="utf-8")asf:
f.write(r.text)
elifnotos.path.islink(config_dst_path):
print("LLMconfigwillbeupdated")
ifconfig_shared_path.exists():
shutil.copy(config_shared_path,config_dst_path)
else:
r=requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
withopen("llm_config.py","w",encoding="utf-8")asf:
f.write(r.text)


ifnottext_example_en_path.exists():
r=requests.get(url=text_example_en)
content=io.BytesIO(r.content)
withopen("text_example_en.pdf","wb")asf:
f.write(content.read())

ifnottext_example_cn_path.exists():
r=requests.get(url=text_example_cn)
content=io.BytesIO(r.content)
withopen("text_example_cn.pdf","wb")asf:
f.write(content.read())


..parsed-literal::

LLMconfigwillbeupdated


Selectmodelforinference
--------------------------

`backtotop⬆️<#table-of-contents>`__

Thetutorialsupportsdifferentmodels,youcanselectonefromthe
providedoptionstocomparethequalityofopensourceLLMsolutions.

**Note**:conversionofsomemodelscanrequireadditionalactions
fromusersideandatleast64GBRAMforconversion.

Theavailableembeddingmodeloptionsare:

-`bge-small-en-v1.5<https://huggingface.co/BAAI/bge-small-en-v1.5>`__
-`bge-small-zh-v1.5<https://huggingface.co/BAAI/bge-small-zh-v1.5>`__
-`bge-large-en-v1.5<https://huggingface.co/BAAI/bge-large-en-v1.5>`__
-`bge-large-zh-v1.5<https://huggingface.co/BAAI/bge-large-zh-v1.5>`__
-`bge-m3<https://huggingface.co/BAAI/bge-m3>`__

BGEembeddingisageneralEmbeddingModel.Themodelispre-trained
usingRetroMAEandtrainedonlarge-scalepairdatausingcontrastive
learning.

Theavailablererankmodeloptionsare:

-`bge-reranker-v2-m3<https://huggingface.co/BAAI/bge-reranker-v2-m3>`__
-`bge-reranker-large<https://huggingface.co/BAAI/bge-reranker-large>`__
-`bge-reranker-base<https://huggingface.co/BAAI/bge-reranker-base>`__

Rerankermodelwithcross-encoderwillperformfull-attentionoverthe
inputpair,whichismoreaccuratethanembeddingmodel(i.e.,
bi-encoder)butmoretime-consumingthanembeddingmodel.Therefore,it
canbeusedtore-rankthetop-kdocumentsreturnedbyembeddingmodel.

YoucanalsofindavailableLLMmodeloptionsin
`llm-chatbot<../llm-chatbot/README.md>`__notebook.

..code::ipython3

frompathlibimportPath
importopenvinoasov
importtorch
importipywidgetsaswidgets
fromtransformersimport(
TextIteratorStreamer,
StoppingCriteria,
StoppingCriteriaList,
)

Convertmodelandcompressmodelweights
----------------------------------------

`backtotop⬆️<#table-of-contents>`__

TheWeightsCompressionalgorithmisaimedatcompressingtheweightsof
themodelsandcanbeusedtooptimizethemodelfootprintand
performanceoflargemodelswherethesizeofweightsisrelatively
largerthanthesizeofactivations,forexample,LargeLanguageModels
(LLM).ComparedtoINT8compression,INT4compressionimproves
performanceevenmore,butintroducesaminordropinprediction
quality.

..code::ipython3

fromllm_configimport(
SUPPORTED_EMBEDDING_MODELS,
SUPPORTED_RERANK_MODELS,
SUPPORTED_LLM_MODELS,
)

model_languages=list(SUPPORTED_LLM_MODELS)

model_language=widgets.Dropdown(
options=model_languages,
value=model_languages[0],
description="ModelLanguage:",
disabled=False,
)

model_language




..parsed-literal::

Dropdown(description='ModelLanguage:',options=('English','Chinese','Japanese'),value='English')



..code::ipython3

llm_model_ids=[model_idformodel_id,model_configinSUPPORTED_LLM_MODELS[model_language.value].items()ifmodel_config.get("rag_prompt_template")]

llm_model_id=widgets.Dropdown(
options=llm_model_ids,
value=llm_model_ids[-1],
description="Model:",
disabled=False,
)

llm_model_id




..parsed-literal::

Dropdown(description='Model:',index=12,options=('tiny-llama-1b-chat','gemma-2b-it','red-pajama-3b-chat','…



..code::ipython3

llm_model_configuration=SUPPORTED_LLM_MODELS[model_language.value][llm_model_id.value]
print(f"SelectedLLMmodel{llm_model_id.value}")


..parsed-literal::

SelectedLLMmodelneural-chat-7b-v3-1


🤗`OptimumIntel<https://huggingface.co/docs/optimum/intel/index>`__is
theinterfacebetweenthe
`Transformers<https://huggingface.co/docs/transformers/index>`__and
`Diffusers<https://huggingface.co/docs/diffusers/index>`__libraries
andOpenVINOtoaccelerateend-to-endpipelinesonIntelarchitectures.
Itprovidesease-to-usecliinterfaceforexportingmodelsto`OpenVINO
IntermediateRepresentation
(IR)<https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
format.

Thecommandbellowdemonstratesbasiccommandformodelexportwith
``optimum-cli``

::

optimum-cliexportopenvino--model<model_id_or_path>--task<task><out_dir>

where``--model``argumentismodelidfromHuggingFaceHuborlocal
directorywithmodel(savedusing``.save_pretrained``method),
``--task``isoneof`supported
task<https://huggingface.co/docs/optimum/exporters/task_manager>`__
thatexportedmodelshouldsolve.ForLLMsitwillbe
``text-generation-with-past``.Ifmodelinitializationrequirestouse
remotecode,``--trust-remote-code``flagadditionallyshouldbepassed.

LLMconversionandWeightsCompressionusingOptimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Youcanalsoapplyfp16,8-bitor4-bitweightcompressiononthe
Linear,ConvolutionalandEmbeddinglayerswhenexportingyourmodel
withtheCLIbysetting``--weight-format``torespectivelyfp16,int8
orint4.Thistypeofoptimizationallowstoreducethememoryfootprint
andinferencelatency.Bydefaultthequantizationschemeforint8/int4
willbe
`asymmetric<https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization>`__,
tomakeit
`symmetric<https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization>`__
youcanadd``--sym``.

ForINT4quantizationyoucanalsospecifythefollowingarguments:

-The``--group-size``parameterwilldefinethegroupsizetousefor
quantization,-1itwillresultsinper-columnquantization.
-The``--ratio``parametercontrolstheratiobetween4-bitand8-bit
quantization.Ifsetto0.9,itmeansthat90%ofthelayerswillbe
quantizedtoint4while10%willbequantizedtoint8.

Smallergroup_sizeandratiovaluesusuallyimproveaccuracyatthe
sacrificeofthemodelsizeandinferencelatency.

**Note**:TheremaybenospeedupforINT4/INT8compressedmodelson
dGPU.

..code::ipython3

fromIPython.displayimportMarkdown,display

prepare_int4_model=widgets.Checkbox(
value=True,
description="PrepareINT4model",
disabled=False,
)
prepare_int8_model=widgets.Checkbox(
value=False,
description="PrepareINT8model",
disabled=False,
)
prepare_fp16_model=widgets.Checkbox(
value=False,
description="PrepareFP16model",
disabled=False,
)

display(prepare_int4_model)
display(prepare_int8_model)
display(prepare_fp16_model)



..parsed-literal::

Checkbox(value=True,description='PrepareINT4model')



..parsed-literal::

Checkbox(value=False,description='PrepareINT8model')



..parsed-literal::

Checkbox(value=False,description='PrepareFP16model')


WeightcompressionwithAWQ
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

`Activation-awareWeight
Quantization<https://arxiv.org/abs/2306.00978>`__(AWQ)isanalgorithm
thattunesmodelweightsformoreaccurateINT4compression.Itslightly
improvesgenerationqualityofcompressedLLMs,butrequiressignificant
additionaltimefortuningweightsonacalibrationdataset.Weuse
``wikitext-2-raw-v1/train``subsetofthe
`Wikitext<https://huggingface.co/datasets/Salesforce/wikitext>`__
datasetforcalibration.

BelowyoucanenableAWQtobeadditionallyappliedduringmodelexport
withINT4precision.

**Note**:ApplyingAWQrequiressignificantmemoryandtime.

..

**Note**:Itispossiblethattherewillbenomatchingpatternsin
themodeltoapplyAWQ,insuchcaseitwillbeskipped.

..code::ipython3

enable_awq=widgets.Checkbox(
value=False,
description="EnableAWQ",
disabled=notprepare_int4_model.value,
)
display(enable_awq)



..parsed-literal::

Checkbox(value=False,description='EnableAWQ')


..code::ipython3

pt_model_id=llm_model_configuration["model_id"]
pt_model_name=llm_model_id.value.split("-")[0]
fp16_model_dir=Path(llm_model_id.value)/"FP16"
int8_model_dir=Path(llm_model_id.value)/"INT8_compressed_weights"
int4_model_dir=Path(llm_model_id.value)/"INT4_compressed_weights"


defconvert_to_fp16():
if(fp16_model_dir/"openvino_model.xml").exists():
return
remote_code=llm_model_configuration.get("remote_code",False)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatfp16".format(pt_model_id)
ifremote_code:
export_command_base+="--trust-remote-code"
export_command=export_command_base+""+str(fp16_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int8():
if(int8_model_dir/"openvino_model.xml").exists():
return
int8_model_dir.mkdir(parents=True,exist_ok=True)
remote_code=llm_model_configuration.get("remote_code",False)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint8".format(pt_model_id)
ifremote_code:
export_command_base+="--trust-remote-code"
export_command=export_command_base+""+str(int8_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int4():
compression_configs={
"zephyr-7b-beta":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"mistral-7b":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"minicpm-2b-dpo":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"gemma-2b-it":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"notus-7b-v1":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"neural-chat-7b-v3-1":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"llama-2-chat-7b":{
"sym":True,
"group_size":128,
"ratio":0.8,
},
"llama-3-8b-instruct":{
"sym":True,
"group_size":128,
"ratio":0.8,
},
"gemma-7b-it":{
"sym":True,
"group_size":128,
"ratio":0.8,
},
"chatglm2-6b":{
"sym":True,
"group_size":128,
"ratio":0.72,
},
"qwen-7b-chat":{"sym":True,"group_size":128,"ratio":0.6},
"red-pajama-3b-chat":{
"sym":False,
"group_size":128,
"ratio":0.5,
},
"default":{
"sym":False,
"group_size":128,
"ratio":0.8,
},
}

model_compression_params=compression_configs.get(llm_model_id.value,compression_configs["default"])
if(int4_model_dir/"openvino_model.xml").exists():
return
remote_code=llm_model_configuration.get("remote_code",False)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint4".format(pt_model_id)
int4_compression_args="--group-size{}--ratio{}".format(model_compression_params["group_size"],model_compression_params["ratio"])
ifmodel_compression_params["sym"]:
int4_compression_args+="--sym"
ifenable_awq.value:
int4_compression_args+="--awq--datasetwikitext2--num-samples128"
export_command_base+=int4_compression_args
ifremote_code:
export_command_base+="--trust-remote-code"
export_command=export_command_base+""+str(int4_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


ifprepare_fp16_model.value:
convert_to_fp16()
ifprepare_int8_model.value:
convert_to_int8()
ifprepare_int4_model.value:
convert_to_int4()

Let’scomparemodelsizefordifferentcompressiontypes

..code::ipython3

fp16_weights=fp16_model_dir/"openvino_model.bin"
int8_weights=int8_model_dir/"openvino_model.bin"
int4_weights=int4_model_dir/"openvino_model.bin"

iffp16_weights.exists():
print(f"SizeofFP16modelis{fp16_weights.stat().st_size/1024/1024:.2f}MB")
forprecision,compressed_weightsinzip([8,4],[int8_weights,int4_weights]):
ifcompressed_weights.exists():
print(f"SizeofmodelwithINT{precision}compressedweightsis{compressed_weights.stat().st_size/1024/1024:.2f}MB")
ifcompressed_weights.exists()andfp16_weights.exists():
print(f"CompressionrateforINT{precision}model:{fp16_weights.stat().st_size/compressed_weights.stat().st_size:.3f}")


..parsed-literal::

SizeofmodelwithINT4compressedweightsis5069.90MB


ConvertembeddingmodelusingOptimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Sincesomeembeddingmodelscanonlysupportlimitedlanguages,wecan
filterthemoutaccordingtheLLMyouselected.

..code::ipython3

embedding_model_id=list(SUPPORTED_EMBEDDING_MODELS[model_language.value])

embedding_model_id=widgets.Dropdown(
options=embedding_model_id,
value=embedding_model_id[0],
description="EmbeddingModel:",
disabled=False,
)

embedding_model_id




..parsed-literal::

Dropdown(description='EmbeddingModel:',options=('bge-small-en-v1.5','bge-large-en-v1.5'),value='bge-small-…



..code::ipython3

embedding_model_configuration=SUPPORTED_EMBEDDING_MODELS[model_language.value][embedding_model_id.value]
print(f"Selected{embedding_model_id.value}model")


..parsed-literal::

Selectedbge-small-en-v1.5model


OpenVINOembeddingmodelandtokenizercanbeexportedby
``feature-extraction``taskwith``optimum-cli``.

..code::ipython3

export_command_base="optimum-cliexportopenvino--model{}--taskfeature-extraction".format(embedding_model_configuration["model_id"])
export_command=export_command_base+""+str(embedding_model_id.value)

ifnotPath(embedding_model_id.value).exists():
!$export_command

ConvertrerankmodelusingOptimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

rerank_model_id=list(SUPPORTED_RERANK_MODELS)

rerank_model_id=widgets.Dropdown(
options=rerank_model_id,
value=rerank_model_id[0],
description="RerankModel:",
disabled=False,
)

rerank_model_id




..parsed-literal::

Dropdown(description='RerankModel:',options=('bge-reranker-large','bge-reranker-base'),value='bge-reranker…



..code::ipython3

rerank_model_configuration=SUPPORTED_RERANK_MODELS[rerank_model_id.value]
print(f"Selected{rerank_model_id.value}model")


..parsed-literal::

Selectedbge-reranker-largemodel


Since``rerank``modelissortofsentenceclassificationtask,its
OpenVINOIRandtokenizercanbeexportedby``text-classification``
taskwith``optimum-cli``.

..code::ipython3

export_command_base="optimum-cliexportopenvino--model{}--tasktext-classification".format(rerank_model_configuration["model_id"])
export_command=export_command_base+""+str(rerank_model_id.value)

ifnotPath(rerank_model_id.value).exists():
!$export_command

Selectdeviceforinferenceandmodelvariant
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

**Note**:TheremaybenospeedupforINT4/INT8compressedmodelson
dGPU.

Selectdeviceforembeddingmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()

support_devices=core.available_devices

embedding_device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

embedding_device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

print(f"Embeddingmodelwillbeloadedto{embedding_device.value}devicefortextembedding")


..parsed-literal::

EmbeddingmodelwillbeloadedtoCPUdevicefortextembedding


OptimizetheBGEembeddingmodel’sparameterprecisionwhenloading
modeltoNPUdevice.

..code::ipython3

USING_NPU=embedding_device.value=="NPU"

npu_embedding_dir=embedding_model_id.value+"-npu"
npu_embedding_path=Path(npu_embedding_dir)/"openvino_model.xml"
ifUSING_NPUandnotPath(npu_embedding_dir).exists():
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
withopen("notebook_utils.py","w")asf:
f.write(r.text)
importnotebook_utilsasutils

shutil.copytree(embedding_model_id.value,npu_embedding_dir)
utils.optimize_bge_embedding(Path(embedding_model_id.value)/"openvino_model.xml",npu_embedding_path)

Selectdeviceforrerankmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

rerank_device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

rerank_device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

print(f"Rerenkmodelwillbeloadedto{rerank_device.value}devicefortextreranking")


..parsed-literal::

RerenkmodelwillbeloadedtoCPUdevicefortextreranking


SelectdeviceforLLMmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

llm_device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

llm_device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

print(f"LLMmodelwillbeloadedto{llm_device.value}deviceforresponsegeneration")


..parsed-literal::

LLMmodelwillbeloadedtoCPUdeviceforresponsegeneration


Loadmodels
-----------

`backtotop⬆️<#table-of-contents>`__

Loadembeddingmodel
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

NowaHuggingFaceembeddingmodelcanbesupportedbyOpenVINOthrough
`OpenVINOEmbeddings<https://python.langchain.com/docs/integrations/text_embedding/openvino>`__
and
`OpenVINOBgeEmbeddings<https://python.langchain.com/docs/integrations/text_embedding/openvino#bge-with-openvino>`__\classes
ofLangChain.

..code::ipython3

fromlangchain_community.embeddingsimportOpenVINOBgeEmbeddings

embedding_model_name=npu_embedding_dirifUSING_NPUelseembedding_model_id.value
batch_size=1ifUSING_NPUelse4
embedding_model_kwargs={"device":embedding_device.value,"compile":False}
encode_kwargs={
"mean_pooling":embedding_model_configuration["mean_pooling"],
"normalize_embeddings":embedding_model_configuration["normalize_embeddings"],
"batch_size":batch_size,
}

embedding=OpenVINOBgeEmbeddings(
model_name_or_path=embedding_model_name,
model_kwargs=embedding_model_kwargs,
encode_kwargs=encode_kwargs,
)
ifUSING_NPU:
embedding.ov_model.reshape(1,512)
embedding.ov_model.compile()

text="Thisisatestdocument."
embedding_result=embedding.embed_query(text)
embedding_result[:3]


..parsed-literal::

CompilingthemodeltoCPU...




..parsed-literal::

[-0.04208654910326004,0.06681869924068451,0.007916687056422234]



Loadrerankmodel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

NowaHuggingFaceembeddingmodelcanbesupportedbyOpenVINOthrough
`OpenVINOReranker<https://python.langchain.com/docs/integrations/document_transformers/openvino_rerank>`__
classofLangChain.

**Note**:RerankcanbeskippedinRAG.

..code::ipython3

fromlangchain_community.document_compressors.openvino_rerankimportOpenVINOReranker

rerank_model_name=rerank_model_id.value
rerank_model_kwargs={"device":rerank_device.value}
rerank_top_n=2

reranker=OpenVINOReranker(
model_name_or_path=rerank_model_name,
model_kwargs=rerank_model_kwargs,
top_n=rerank_top_n,
)


..parsed-literal::

CompilingthemodeltoCPU...


LoadLLMmodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINOmodelscanberunlocallythroughthe``HuggingFacePipeline``
class.TodeployamodelwithOpenVINO,youcanspecifythe
``backend="openvino"``parametertotriggerOpenVINOasbackend
inferenceframework.

..code::ipython3

available_models=[]
ifint4_model_dir.exists():
available_models.append("INT4")
ifint8_model_dir.exists():
available_models.append("INT8")
iffp16_model_dir.exists():
available_models.append("FP16")

model_to_run=widgets.Dropdown(
options=available_models,
value=available_models[0],
description="Modeltorun:",
disabled=False,
)

model_to_run




..parsed-literal::

Dropdown(description='Modeltorun:',options=('INT4',),value='INT4')



OpenVINOmodelscanberunlocallythroughthe``HuggingFacePipeline``
classin
`LangChain<https://python.langchain.com/docs/integrations/llms/openvino/>`__.
TodeployamodelwithOpenVINO,youcanspecifythe
``backend="openvino"``parametertotriggerOpenVINOasbackend
inferenceframework.

..code::ipython3

fromlangchain_community.llms.huggingface_pipelineimportHuggingFacePipeline

ifmodel_to_run.value=="INT4":
model_dir=int4_model_dir
elifmodel_to_run.value=="INT8":
model_dir=int8_model_dir
else:
model_dir=fp16_model_dir
print(f"Loadingmodelfrom{model_dir}")

ov_config={"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":""}

if"GPU"inllm_device.valueand"qwen2-7b-instruct"inllm_model_id.value:
ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"]="NO"

#OnaGPUdeviceamodelisexecutedinFP16precision.Forred-pajama-3b-chatmodelthereknownaccuracy
#issuescausedbythis,whichweavoidbysettingprecisionhintto"f32".
ifllm_model_id.value=="red-pajama-3b-chat"and"GPU"incore.available_devicesandllm_device.valuein["GPU","AUTO"]:
ov_config["INFERENCE_PRECISION_HINT"]="f32"

llm=HuggingFacePipeline.from_model_id(
model_id=str(model_dir),
task="text-generation",
backend="openvino",
model_kwargs={
"device":llm_device.value,
"ov_config":ov_config,
"trust_remote_code":True,
},
pipeline_kwargs={"max_new_tokens":2},
)

llm.invoke("2+2=")


..parsed-literal::

Theargument`trust_remote_code`istobeusedalongwithexport=True.Itwillbeignored.


..parsed-literal::

Loadingmodelfromneural-chat-7b-v3-1/INT4_compressed_weights


..parsed-literal::

CompilingthemodeltoCPU...




..parsed-literal::

'2+2=4'



RunQAoverDocument
--------------------

`backtotop⬆️<#table-of-contents>`__

Now,whenmodelcreated,wecansetupChatbotinterfaceusing
`Gradio<https://www.gradio.app/>`__.

AtypicalRAGapplicationhastwomaincomponents:

-**Indexing**:apipelineforingestingdatafromasourceand
indexingit.Thisusuallyhappenoffline.

-**Retrievalandgeneration**:theactualRAGchain,whichtakesthe
userqueryatruntimeandretrievestherelevantdatafromthe
index,thenpassesthattothemodel.

Themostcommonfullsequencefromrawdatatoanswerlookslike:

**Indexing**

1.``Load``:Firstweneedtoloadourdata.We’lluseDocumentLoaders
forthis.
2.``Split``:TextsplittersbreaklargeDocumentsintosmallerchunks.
Thisisusefulbothforindexingdataandforpassingitintoa
model,sincelargechunksarehardertosearchoverandwon’tina
model’sfinitecontextwindow.
3.``Store``:Weneedsomewheretostoreandindexoursplits,sothat
theycanlaterbesearchedover.Thisisoftendoneusinga
VectorStoreandEmbeddingsmodel.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/dfed2ba3-0c3a-4e0e-a2a7-01638730486a
:alt:Indexingpipeline

Indexingpipeline

**Retrievalandgeneration**

1.``Retrieve``:Givenauserinput,relevantsplitsareretrievedfrom
storageusingaRetriever.
2.``Generate``:ALLMproducesananswerusingapromptthatincludes
thequestionandtheretrieveddata.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/f0545ddc-c0cd-4569-8c86-9879fdab105a
:alt:Retrievalandgenerationpipeline

Retrievalandgenerationpipeline

..code::ipython3

importre
fromtypingimportList
fromlangchain.text_splitterimport(
CharacterTextSplitter,
RecursiveCharacterTextSplitter,
MarkdownTextSplitter,
)
fromlangchain.document_loadersimport(
CSVLoader,
EverNoteLoader,
PyPDFLoader,
TextLoader,
UnstructuredEPubLoader,
UnstructuredHTMLLoader,
UnstructuredMarkdownLoader,
UnstructuredODTLoader,
UnstructuredPowerPointLoader,
UnstructuredWordDocumentLoader,
)


classChineseTextSplitter(CharacterTextSplitter):
def__init__(self,pdf:bool=False,**kwargs):
super().__init__(**kwargs)
self.pdf=pdf

defsplit_text(self,text:str)->List[str]:
ifself.pdf:
text=re.sub(r"\n{3,}","\n",text)
text=text.replace("\n\n","")
sent_sep_pattern=re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
sent_list=[]
foreleinsent_sep_pattern.split(text):
ifsent_sep_pattern.match(ele)andsent_list:
sent_list[-1]+=ele
elifele:
sent_list.append(ele)
returnsent_list


TEXT_SPLITERS={
"Character":CharacterTextSplitter,
"RecursiveCharacter":RecursiveCharacterTextSplitter,
"Markdown":MarkdownTextSplitter,
"Chinese":ChineseTextSplitter,
}


LOADERS={
".csv":(CSVLoader,{}),
".doc":(UnstructuredWordDocumentLoader,{}),
".docx":(UnstructuredWordDocumentLoader,{}),
".enex":(EverNoteLoader,{}),
".epub":(UnstructuredEPubLoader,{}),
".html":(UnstructuredHTMLLoader,{}),
".md":(UnstructuredMarkdownLoader,{}),
".odt":(UnstructuredODTLoader,{}),
".pdf":(PyPDFLoader,{}),
".ppt":(UnstructuredPowerPointLoader,{}),
".pptx":(UnstructuredPowerPointLoader,{}),
".txt":(TextLoader,{"encoding":"utf8"}),
}

chinese_examples=[
["英特尔®酷睿™Ultra处理器可以降低多少功耗？"],
["相比英特尔之前的移动处理器产品，英特尔®酷睿™Ultra处理器的AI推理性能提升了多少？"],
["英特尔博锐®Enterprise系统提供哪些功能？"],
]

english_examples=[
["HowmuchpowerconsumptioncanIntel®Core™UltraProcessorshelpsave?"],
["ComparedtoIntel’spreviousmobileprocessor,whatistheadvantageofIntel®Core™UltraProcessorsforArtificialIntelligence?"],
["WhatcanIntelvPro®Enterprisesystemsoffer?"],
]

ifmodel_language.value=="English":
text_example_path="text_example_en.pdf"
else:
text_example_path="text_example_cn.pdf"

examples=chinese_examplesif(model_language.value=="Chinese")elseenglish_examples

WecanbuildaRAGpipelineofLangChainthrough
`create_retrieval_chain<https://python.langchain.com/docs/modules/chains/>`__,
whichwillhelptocreateachaintoconnectRAGcomponentsincluding:

-`Vectorstores<https://python.langchain.com/docs/modules/data_connection/vectorstores/>`__\，
-`Retrievers<https://python.langchain.com/docs/modules/data_connection/retrievers/>`__
-`LLM<https://python.langchain.com/docs/integrations/llms/>`__
-`Embedding<https://python.langchain.com/docs/integrations/text_embedding/>`__

..code::ipython3

fromlangchain.promptsimportPromptTemplate
fromlangchain_community.vectorstoresimportFAISS
fromlangchain.chains.retrievalimportcreate_retrieval_chain
fromlangchain.chains.combine_documentsimportcreate_stuff_documents_chain
fromlangchain.docstore.documentimportDocument
fromlangchain.retrieversimportContextualCompressionRetriever
fromthreadingimportThread
importgradioasgr

stop_tokens=llm_model_configuration.get("stop_tokens")
rag_prompt_template=llm_model_configuration["rag_prompt_template"]


classStopOnTokens(StoppingCriteria):
def__init__(self,token_ids):
self.token_ids=token_ids

def__call__(self,input_ids:torch.LongTensor,scores:torch.FloatTensor,**kwargs)->bool:
forstop_idinself.token_ids:
ifinput_ids[0][-1]==stop_id:
returnTrue
returnFalse


ifstop_tokensisnotNone:
ifisinstance(stop_tokens[0],str):
stop_tokens=llm.pipeline.tokenizer.convert_tokens_to_ids(stop_tokens)

stop_tokens=[StopOnTokens(stop_tokens)]


defload_single_document(file_path:str)->List[Document]:
"""
helperforloadingasingledocument

Params:
file_path:documentpath
Returns:
documentsloaded

"""
ext="."+file_path.rsplit(".",1)[-1]
ifextinLOADERS:
loader_class,loader_args=LOADERS[ext]
loader=loader_class(file_path,**loader_args)
returnloader.load()

raiseValueError(f"Filedoesnotexist'{ext}'")


defdefault_partial_text_processor(partial_text:str,new_text:str):
"""
helperforupdatingpartiallygeneratedanswer,usedbydefault

Params:
partial_text:textbufferforstoringprevioslygeneratedtext
new_text:textupdateforthecurrentstep
Returns:
updatedtextstring

"""
partial_text+=new_text
returnpartial_text


text_processor=llm_model_configuration.get("partial_text_processor",default_partial_text_processor)


defcreate_vectordb(
docs,spliter_name,chunk_size,chunk_overlap,vector_search_top_k,vector_rerank_top_n,run_rerank,search_method,score_threshold,progress=gr.Progress()
):
"""
Initializeavectordatabase

Params:
doc:orignaldocumentsprovidedbyuser
spliter_name:splitermethod
chunk_size:sizeofasinglesentencechunk
chunk_overlap:overlapsizebetween2chunks
vector_search_top_k:Vectorsearchtopk
vector_rerank_top_n:Searchreranktopn
run_rerank:whetherrunreranker
search_method:topksearchmethod
score_threshold:scorethresholdwhenselecting'similarity_score_threshold'method

"""
globaldb
globalretriever
globalcombine_docs_chain
globalrag_chain

ifvector_rerank_top_n>vector_search_top_k:
gr.Warning("Searchtopkmust>=Reranktopn")

documents=[]
fordocindocs:
iftype(doc)isnotstr:
doc=doc.name
documents.extend(load_single_document(doc))

text_splitter=TEXT_SPLITERS[spliter_name](chunk_size=chunk_size,chunk_overlap=chunk_overlap)

texts=text_splitter.split_documents(documents)
db=FAISS.from_documents(texts,embedding)
ifsearch_method=="similarity_score_threshold":
search_kwargs={"k":vector_search_top_k,"score_threshold":score_threshold}
else:
search_kwargs={"k":vector_search_top_k}
retriever=db.as_retriever(search_kwargs=search_kwargs,search_type=search_method)
ifrun_rerank:
reranker.top_n=vector_rerank_top_n
retriever=ContextualCompressionRetriever(base_compressor=reranker,base_retriever=retriever)
prompt=PromptTemplate.from_template(rag_prompt_template)
combine_docs_chain=create_stuff_documents_chain(llm,prompt)

rag_chain=create_retrieval_chain(retriever,combine_docs_chain)

return"VectordatabaseisReady"


defupdate_retriever(vector_search_top_k,vector_rerank_top_n,run_rerank,search_method,score_threshold):
"""
Updateretriever

Params:
vector_search_top_k:Vectorsearchtopk
vector_rerank_top_n:Searchreranktopn
run_rerank:whetherrunreranker
search_method:topksearchmethod
score_threshold:scorethresholdwhenselecting'similarity_score_threshold'method

"""
globaldb
globalretriever
globalcombine_docs_chain
globalrag_chain

ifvector_rerank_top_n>vector_search_top_k:
gr.Warning("Searchtopkmust>=Reranktopn")

ifsearch_method=="similarity_score_threshold":
search_kwargs={"k":vector_search_top_k,"score_threshold":score_threshold}
else:
search_kwargs={"k":vector_search_top_k}
retriever=db.as_retriever(search_kwargs=search_kwargs,search_type=search_method)
ifrun_rerank:
retriever=ContextualCompressionRetriever(base_compressor=reranker,base_retriever=retriever)
reranker.top_n=vector_rerank_top_n
rag_chain=create_retrieval_chain(retriever,combine_docs_chain)

return"VectordatabaseisReady"


defuser(message,history):
"""
callbackfunctionforupdatingusermessagesininterfaceonsubmitbuttonclick

Params:
message:currentmessage
history:conversationhistory
Returns:
None
"""
#Appendtheuser'smessagetotheconversationhistory
return"",history+[[message,""]]


defbot(history,temperature,top_p,top_k,repetition_penalty,hide_full_prompt,do_rag):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory
temperature:parameterforcontrolthelevelofcreativityinAI-generatedtext.
Byadjustingthe`temperature`,youcaninfluencetheAImodel'sprobabilitydistribution,makingthetextmorefocusedordiverse.
top_p:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability.
top_k:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability,selectingnumberoftokenswithhighestprobability.
repetition_penalty:parameterforpenalizingtokensbasedonhowfrequentlytheyoccurinthetext.
hide_full_prompt:whethertoshowsearchingresultsinpromopt.
do_rag:whetherdoRAGwhengeneratingtexts.

"""
streamer=TextIteratorStreamer(
llm.pipeline.tokenizer,
timeout=60.0,
skip_prompt=hide_full_prompt,
skip_special_tokens=True,
)
llm.pipeline._forward_params=dict(
max_new_tokens=512,
temperature=temperature,
do_sample=temperature>0.0,
top_p=top_p,
top_k=top_k,
repetition_penalty=repetition_penalty,
streamer=streamer,
)
ifstop_tokensisnotNone:
llm.pipeline._forward_params["stopping_criteria"]=StoppingCriteriaList(stop_tokens)

ifdo_rag:
t1=Thread(target=rag_chain.invoke,args=({"input":history[-1][0]},))
else:
input_text=rag_prompt_template.format(input=history[-1][0],context="")
t1=Thread(target=llm.invoke,args=(input_text,))
t1.start()

#Initializeanemptystringtostorethegeneratedtext
partial_text=""
fornew_textinstreamer:
partial_text=text_processor(partial_text,new_text)
history[-1][1]=partial_text
yieldhistory


defrequest_cancel():
llm.pipeline.model.request.cancel()


defclear_files():
return"VectorStoreisNotready"


#initializethevectorstorewithexampledocument
create_vectordb(
[text_example_path],
"RecursiveCharacter",
chunk_size=400,
chunk_overlap=50,
vector_search_top_k=10,
vector_rerank_top_n=2,
run_rerank=True,
search_method="similarity_score_threshold",
score_threshold=0.5,
)




..parsed-literal::

'VectordatabaseisReady'



NextwecancreateaGradioUIandrundemo.

..code::ipython3

withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asdemo:
gr.Markdown("""<h1><center>QAoverDocument</center></h1>""")
gr.Markdown(f"""<center>PoweredbyOpenVINOand{llm_model_id.value}</center>""")
withgr.Row():
withgr.Column(scale=1):
docs=gr.File(
label="Step1:Loadtextfiles",
value=[text_example_path],
file_count="multiple",
file_types=[
".csv",
".doc",
".docx",
".enex",
".epub",
".html",
".md",
".odt",
".pdf",
".ppt",
".pptx",
".txt",
],
)
load_docs=gr.Button("Step2:BuildVectorStore",variant="primary")
db_argument=gr.Accordion("VectorStoreConfiguration",open=False)
withdb_argument:
spliter=gr.Dropdown(
["Character","RecursiveCharacter","Markdown","Chinese"],
value="RecursiveCharacter",
label="TextSpliter",
info="Methodusedtosplitethedocuments",
multiselect=False,
)

chunk_size=gr.Slider(
label="Chunksize",
value=400,
minimum=50,
maximum=2000,
step=50,
interactive=True,
info="Sizeofsentencechunk",
)

chunk_overlap=gr.Slider(
label="Chunkoverlap",
value=50,
minimum=0,
maximum=400,
step=10,
interactive=True,
info=("Overlapbetween2chunks"),
)

langchain_status=gr.Textbox(
label="VectorStoreStatus",
value="VectorStoreisReady",
interactive=False,
)
do_rag=gr.Checkbox(
value=True,
label="RAGisON",
interactive=True,
info="WhethertodoRAGforgeneration",
)
withgr.Accordion("GenerationConfiguration",open=False):
withgr.Row():
withgr.Column():
withgr.Row():
temperature=gr.Slider(
label="Temperature",
value=0.1,
minimum=0.0,
maximum=1.0,
step=0.1,
interactive=True,
info="Highervaluesproducemorediverseoutputs",
)
withgr.Column():
withgr.Row():
top_p=gr.Slider(
label="Top-p(nucleussampling)",
value=1.0,
minimum=0.0,
maximum=1,
step=0.01,
interactive=True,
info=(
"Samplefromthesmallestpossiblesetoftokenswhosecumulativeprobability"
"exceedstop_p.Setto1todisableandsamplefromalltokens."
),
)
withgr.Column():
withgr.Row():
top_k=gr.Slider(
label="Top-k",
value=50,
minimum=0.0,
maximum=200,
step=1,
interactive=True,
info="Samplefromashortlistoftop-ktokens—0todisableandsamplefromalltokens.",
)
withgr.Column():
withgr.Row():
repetition_penalty=gr.Slider(
label="RepetitionPenalty",
value=1.1,
minimum=1.0,
maximum=2.0,
step=0.1,
interactive=True,
info="Penalizerepetition—1.0todisable.",
)
withgr.Column(scale=4):
chatbot=gr.Chatbot(
height=800,
label="Step3:InputQuery",
)
withgr.Row():
withgr.Column():
withgr.Row():
msg=gr.Textbox(
label="QAMessageBox",
placeholder="ChatMessageBox",
show_label=False,
container=False,
)
withgr.Column():
withgr.Row():
submit=gr.Button("Submit",variant="primary")
stop=gr.Button("Stop")
clear=gr.Button("Clear")
gr.Examples(examples,inputs=msg,label="Clickonanyexampleandpressthe'Submit'button")
retriever_argument=gr.Accordion("RetrieverConfiguration",open=True)
withretriever_argument:
withgr.Row():
withgr.Row():
do_rerank=gr.Checkbox(
value=True,
label="Reranksearchingresult",
interactive=True,
)
hide_context=gr.Checkbox(
value=True,
label="Hidesearchingresultinprompt",
interactive=True,
)
withgr.Row():
search_method=gr.Dropdown(
["similarity_score_threshold","similarity","mmr"],
value="similarity_score_threshold",
label="SearchingMethod",
info="Methodusedtosearchvectorstore",
multiselect=False,
interactive=True,
)
withgr.Row():
score_threshold=gr.Slider(
0.01,
0.99,
value=0.5,
step=0.01,
label="SimilarityThreshold",
info="Onlyworkingfor'similarityscorethreshold'method",
interactive=True,
)
withgr.Row():
vector_rerank_top_n=gr.Slider(
1,
10,
value=2,
step=1,
label="Reranktopn",
info="Numberofrerankresults",
interactive=True,
)
withgr.Row():
vector_search_top_k=gr.Slider(
1,
50,
value=10,
step=1,
label="Searchtopk",
info="Searchtopkmust>=Reranktopn",
interactive=True,
)
docs.clear(clear_files,outputs=[langchain_status],queue=False)
load_docs.click(
create_vectordb,
inputs=[docs,spliter,chunk_size,chunk_overlap,vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
queue=False,
)
submit_event=msg.submit(user,[msg,chatbot],[msg,chatbot],queue=False).then(
bot,
[chatbot,temperature,top_p,top_k,repetition_penalty,hide_context,do_rag],
chatbot,
queue=True,
)
submit_click_event=submit.click(user,[msg,chatbot],[msg,chatbot],queue=False).then(
bot,
[chatbot,temperature,top_p,top_k,repetition_penalty,hide_context,do_rag],
chatbot,
queue=True,
)
stop.click(
fn=request_cancel,
inputs=None,
outputs=None,
cancels=[submit_event,submit_click_event],
queue=False,
)
clear.click(lambda:None,None,chatbot,queue=False)
vector_search_top_k.release(
update_retriever,
[vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
)
vector_rerank_top_n.release(
update_retriever,
inputs=[vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
)
do_rerank.change(
update_retriever,
inputs=[vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
)
search_method.change(
update_retriever,
inputs=[vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
)
score_threshold.change(
update_retriever,
inputs=[vector_search_top_k,vector_rerank_top_n,do_rerank,search_method,score_threshold],
outputs=[langchain_status],
)


demo.queue()
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#ifyouhaveanyissuetolaunchonyourplatform,youcanpassshare=Truetolaunchmethod:
#demo.launch(share=True)
#itcreatesapubliclyshareablelinkfortheinterface.Readmoreinthedocs:https://gradio.app/docs/
demo.launch()

..code::ipython3

#pleaserunthiscellforstoppinggradiointerface
demo.close()

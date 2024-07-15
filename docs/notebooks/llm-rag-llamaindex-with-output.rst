CreateaRAGsystemusingOpenVINOandLlamaIndex
=================================================

**Retrieval-augmentedgeneration(RAG)**isatechniqueforaugmenting
LLMknowledgewithadditional,oftenprivateorreal-time,data.LLMs
canreasonaboutwide-rangingtopics,buttheirknowledgeislimitedto
thepublicdatauptoaspecificpointintimethattheyweretrained
on.IfyouwanttobuildAIapplicationsthatcanreasonaboutprivate
dataordataintroducedafteramodel’scutoffdate,youneedtoaugment
theknowledgeofthemodelwiththespecificinformationitneeds.The
processofbringingtheappropriateinformationandinsertingitinto
themodelpromptisknownasRetrievalAugmentedGeneration(RAG).

`LlamaIndex<https://docs.llamaindex.ai/en/stable/>`__isaframework
forbuildingcontext-augmentedgenerativeAIapplicationswith
LLMs.LlamaIndeximposesnorestrictiononhowyouuseLLMs.Youcanuse
LLMsasauto-complete,chatbots,semi-autonomousagents,andmore.It
justmakesusingthemeasier.

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
-`GradioDemo<#gradio-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequireddependencies

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipinstall-Uqpip
%pipuninstall-q-yoptimumoptimum-intel
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"llama-index""faiss-cpu""pymupdf""llama-index-readers-file""llama-index-vector-stores-faiss""llama-index-llms-langchain""llama-index-llms-openvino""llama-index-embeddings-openvino""llama-index-postprocessor-openvino-rerank""transformers>=4.40"\
"git+https://github.com/huggingface/optimum-intel.git"\
"git+https://github.com/openvinotoolkit/nncf.git"\
"datasets"\
"accelerate"\
"gradio"\
"langchain"
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
WARNING:Skippingoptimumasitisnotinstalled.
WARNING:Skippingoptimum-intelasitisnotinstalled.
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
Rerankermodelwithcross-encoderwillperformfull-attentionover
theinputpair,whichismoreaccuratethanembeddingmodel(i.e.,
bi-encoder)butmoretime-consumingthanembeddingmodel.Therefore,
itcanbeusedtore-rankthetop-kdocumentsreturnedbyembedding
model.

YoucanalsofindavailableLLMmodeloptionsin
`llm-chatbot<../llm-chatbot/README.md>`__notebook.

..code::ipython3

frompathlibimportPath
importopenvinoasov
importipywidgetsaswidgets

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

SelectedLLMmodelllama-3-8b-instruct


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

SizeofmodelwithINT4compressedweightsis5085.79MB


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

Dropdown(description='EmbeddingModel:',options=('bge-small-en-v1.5','bge-large-en-v1.5','bge-m3'),value='…



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

Dropdown(description='RerankModel:',options=('bge-reranker-v2-m3','bge-reranker-large','bge-reranker-base'…



..code::ipython3

rerank_model_configuration=SUPPORTED_RERANK_MODELS[rerank_model_id.value]
print(f"Selected{rerank_model_id.value}model")


..parsed-literal::

Selectedbge-reranker-v2-m3model


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
`OpenVINOEmbeddings<https://docs.llamaindex.ai/en/stable/examples/embeddings/openvino/>`__
classofLlamaIndex.

..code::ipython3

fromllama_index.embeddings.huggingface_openvinoimportOpenVINOEmbedding


embedding=OpenVINOEmbedding(folder_name=embedding_model_id.value,device=embedding_device.value)

embeddings=embedding.get_text_embedding("HelloWorld!")
print(len(embeddings))
print(embeddings[:5])


..parsed-literal::

CompilingthemodeltoCPU...


..parsed-literal::

384
[-0.003275666618719697,-0.01169075071811676,0.04155930131673813,-0.03814813867211342,0.02418304793536663]


Loadrerankmodel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

NowaHuggingFaceembeddingmodelcanbesupportedbyOpenVINOthrough
`OpenVINORerank<https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/openvino_rerank/>`__
classofLlamaIndex.

**Note**:RerankcanbeskippedinRAG.

..code::ipython3

fromllama_index.postprocessor.openvino_rerankimportOpenVINORerank

reranker=OpenVINORerank(model=rerank_model_id.value,device=rerank_device.value,top_n=2)


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



OpenVINOmodelscanberunlocallythroughthe``OpenVINOLLM``classin
`LlamaIndex<https://docs.llamaindex.ai/en/stable/examples/llm/openvino/>`__.
IfyouhaveanIntelGPU,youcanspecify``device_map="gpu"``torun
inferenceonit.

..code::ipython3

fromllama_index.llms.openvinoimportOpenVINOLLM

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

llm=OpenVINOLLM(
model_name=str(model_dir),
tokenizer_name=str(model_dir),
context_window=3900,
max_new_tokens=2,
model_kwargs={"ov_config":ov_config,"trust_remote_code":True},
generate_kwargs={"temperature":0.7,"top_k":50,"top_p":0.95},
device_map=llm_device.value,
)

response=llm.complete("2+2=")
print(str(response))


..parsed-literal::

Theargument`trust_remote_code`istobeusedalongwithexport=True.Itwillbeignored.


..parsed-literal::

Loadingmodelfromllama-3-8b-instruct/INT4_compressed_weights


..parsed-literal::

CompilingthemodeltoCPU...
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
Setting`pad_token_id`to`eos_token_id`:128001foropen-endgeneration.


..parsed-literal::

4


RunQAoverDocument
--------------------

`backtotop⬆️<#table-of-contents>`__

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

..code::ipython3

fromllama_index.coreimportVectorStoreIndex,StorageContext
fromllama_index.core.node_parserimportSentenceSplitter
fromllama_index.coreimportSettings
fromllama_index.readers.fileimportPyMuPDFReader
fromllama_index.vector_stores.faissimportFaissVectorStore
fromtransformersimportStoppingCriteria,StoppingCriteriaList
importfaiss
importtorch

ifmodel_language.value=="English":
text_example_path="text_example_en.pdf"
else:
text_example_path="text_example_cn.pdf"

stop_tokens=llm_model_configuration.get("stop_tokens")


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
stop_tokens=llm._tokenizer.convert_tokens_to_ids(stop_tokens)
stop_tokens=[StopOnTokens(stop_tokens)]

loader=PyMuPDFReader()
documents=loader.load(file_path=text_example_path)

#dimensionsofembeddingmodel
d=embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
faiss_index=faiss.IndexFlatL2(d)
Settings.embed_model=embedding

llm.max_new_tokens=2048
ifstop_tokensisnotNone:
llm._stopping_criteria=StoppingCriteriaList(stop_tokens)
Settings.llm=llm

vector_store=FaissVectorStore(faiss_index=faiss_index)
storage_context=StorageContext.from_defaults(vector_store=vector_store)
index=VectorStoreIndex.from_documents(
documents,
storage_context=storage_context,
transformations=[SentenceSplitter(chunk_size=200,chunk_overlap=40)],
)

**Retrievalandgeneration**

1.``Retrieve``:Givenauserinput,relevantsplitsareretrievedfrom
storageusingaRetriever.
2.``Generate``:ALLMproducesananswerusingapromptthatincludes
thequestionandtheretrieveddata.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/f0545ddc-c0cd-4569-8c86-9879fdab105a
:alt:Retrievalandgenerationpipeline

Retrievalandgenerationpipeline

..code::ipython3

query_engine=index.as_query_engine(streaming=True,similarity_top_k=10,node_postprocessors=[reranker])
ifmodel_language.value=="English":
query="WhatcanIntelvPro®Enterprisesystemsoffer?"
else:
query="英特尔博锐®Enterprise系统提供哪些功能？"

streaming_response=query_engine.query(query)
streaming_response.print_response_stream()


..parsed-literal::

Setting`pad_token_id`to`eos_token_id`:128001foropen-endgeneration.


..parsed-literal::

Accordingtotheprovidedcontextinformation,IntelvProEnterprisesystemscanoffer:
-Dynamicrootoftrust
-Systemmanagementmode(SMM)protections
-Memoryencryptionwithmulti-keysupport
-OSkernelprotection
-Out-of-bandmanagementwithremoteKVMcontrol
-Uniquedeviceidentifier
-Devicehistory
-In-bandmanageabilityplug-ins
Notethatthisinformationisbasedsolelyontheprovidedcontextanddoesnotrepresentanyexternalknowledgeorunderstanding.Theanswerisintendedtoaccuratelyreflectthecontentpresentedinthegiventext.

GradioDemo
-----------

`backtotop⬆️<#table-of-contents>`__

Now,whenmodelcreated,wecansetupChatbotinterfaceusing
`Gradio<https://www.gradio.app/>`__.

FirstwecancheckthedefaultprompttemplateinLlamaIndexpipeline.

..code::ipython3

prompts_dict=query_engine.get_prompts()


defdisplay_prompt_dict(prompts_dict):
fork,pinprompts_dict.items():
text_md=f"**PromptKey**:{k}<br>"f"**Text:**<br>"
display(Markdown(text_md))
print(p.get_template())
display(Markdown("<br><br>"))


display_prompt_dict(prompts_dict)



**PromptKey**:response_synthesizer:text_qa_template\**Text:**


..parsed-literal::

Contextinformationisbelow.
---------------------
{context_str}
---------------------
Giventhecontextinformationandnotpriorknowledge,answerthequery.
Query:{query_str}
Answer:







**PromptKey**:response_synthesizer:refine_template\**Text:**


..parsed-literal::

Theoriginalqueryisasfollows:{query_str}
Wehaveprovidedanexistinganswer:{existing_answer}
Wehavetheopportunitytorefinetheexistinganswer(onlyifneeded)withsomemorecontextbelow.
------------
{context_msg}
------------
Giventhenewcontext,refinetheoriginalanswertobetteranswerthequery.Ifthecontextisn'tuseful,returntheoriginalanswer.
RefinedAnswer:






..code::ipython3

fromlangchain.text_splitterimportRecursiveCharacterTextSplitter
fromllama_index.core.node_parserimportLangchainNodeParser
importgradioasgr

TEXT_SPLITERS={
"SentenceSplitter":SentenceSplitter,
"RecursiveCharacter":RecursiveCharacterTextSplitter,
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

examples=chinese_examplesif(model_language.value=="Chinese")elseenglish_examples


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


defcreate_vectordb(doc,spliter_name,chunk_size,chunk_overlap,vector_search_top_k,vector_rerank_top_n,run_rerank):
"""
Initializeavectordatabase

Params:
doc:orignaldocumentsprovidedbyuser
chunk_size:sizeofasinglesentencechunk
chunk_overlap:overlapsizebetween2chunks
vector_search_top_k:Vectorsearchtopk
vector_rerank_top_n:Rerranktopn
run_rerank:whethertorunreranker

"""
globalquery_engine
globalindex

ifvector_rerank_top_n>vector_search_top_k:
gr.Warning("Searchtopkmust>=Reranktopn")

loader=PyMuPDFReader()
documents=loader.load(file_path=doc.name)
spliter=TEXT_SPLITERS[spliter_name](chunk_size=chunk_size,chunk_overlap=chunk_overlap)
ifspliter_name=="RecursiveCharacter":
spliter=LangchainNodeParser(spliter)
faiss_index=faiss.IndexFlatL2(d)
vector_store=FaissVectorStore(faiss_index=faiss_index)
storage_context=StorageContext.from_defaults(vector_store=vector_store)

index=VectorStoreIndex.from_documents(
documents,
storage_context=storage_context,
transformations=[spliter],
)
ifrun_rerank:
reranker.top_n=vector_rerank_top_n
query_engine=index.as_query_engine(streaming=True,similarity_top_k=vector_search_top_k,node_postprocessors=[reranker])
else:
query_engine=index.as_query_engine(streaming=True,similarity_top_k=vector_search_top_k)

return"VectordatabaseisReady"


defupdate_retriever(vector_search_top_k,vector_rerank_top_n,run_rerank):
"""
Updateretriever

Params:
vector_search_top_k:sizeofsearchingresults
vector_rerank_top_n:sizeofrerankresults
run_rerank:whetherrunrerankstep

"""
globalquery_engine
globalindex

ifvector_rerank_top_n>vector_search_top_k:
gr.Warning("Searchtopkmust>=Reranktopn")

ifrun_rerank:
reranker.top_n=vector_rerank_top_n
query_engine=index.as_query_engine(streaming=True,similarity_top_k=vector_search_top_k,node_postprocessors=[reranker])
else:
query_engine=index.as_query_engine(streaming=True,similarity_top_k=vector_search_top_k)


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


defbot(history,temperature,top_p,top_k,repetition_penalty,do_rag):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory
temperature:parameterforcontrolthelevelofcreativityinAI-generatedtext.
Byadjustingthe`temperature`,youcaninfluencetheAImodel'sprobabilitydistribution,makingthetextmorefocusedordiverse.
top_p:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability.
top_k:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability,selectingnumberoftokenswithhighestprobability.
repetition_penalty:parameterforpenalizingtokensbasedonhowfrequentlytheyoccurinthetext.
do_rag:whetherdoRAGwhengeneratingtexts.

"""
llm.generate_kwargs=dict(
temperature=temperature,
do_sample=temperature>0.0,
top_p=top_p,
top_k=top_k,
repetition_penalty=repetition_penalty,
)

partial_text=""
ifdo_rag:
streaming_response=query_engine.query(history[-1][0])
fornew_textinstreaming_response.response_gen:
partial_text=text_processor(partial_text,new_text)
history[-1][1]=partial_text
yieldhistory
else:
streaming_response=llm.stream_complete(history[-1][0])
fornew_textinstreaming_response:
partial_text=text_processor(partial_text,new_text.delta)
history[-1][1]=partial_text
yieldhistory


defrequest_cancel():
llm._model.request.cancel()


defclear_files():
return"VectorStoreisNotready"


withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asdemo:
gr.Markdown("""<h1><center>QAoverDocument</center></h1>""")
gr.Markdown(f"""<center>PoweredbyOpenVINOand{llm_model_id.value}</center>""")
withgr.Row():
withgr.Column(scale=1):
docs=gr.File(
label="Step1:LoadaPDFfile",
value=text_example_path,
file_types=[
".pdf",
],
)
load_docs=gr.Button("Step2:BuildVectorStore",variant="primary")
db_argument=gr.Accordion("VectorStoreConfiguration",open=False)
withdb_argument:
spliter=gr.Dropdown(
["SentenceSplitter","RecursiveCharacter"],
value="SentenceSplitter",
label="TextSpliter",
info="Methodusedtosplitethedocuments",
multiselect=False,
)

chunk_size=gr.Slider(
label="Chunksize",
value=200,
minimum=50,
maximum=2000,
step=50,
interactive=True,
info="Sizeofsentencechunk",
)

chunk_overlap=gr.Slider(
label="Chunkoverlap",
value=20,
minimum=0,
maximum=400,
step=10,
interactive=True,
info=("Overlapbetween2chunks"),
)

vector_store_status=gr.Textbox(
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
height=600,
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
docs.clear(clear_files,outputs=[vector_store_status],queue=False)
load_docs.click(
create_vectordb,
inputs=[docs,spliter,chunk_size,chunk_overlap,vector_search_top_k,vector_rerank_top_n,do_rerank],
outputs=[vector_store_status],
queue=False,
)
submit_event=msg.submit(user,[msg,chatbot],[msg,chatbot],queue=False).then(
bot,
[chatbot,temperature,top_p,top_k,repetition_penalty,do_rag],
chatbot,
queue=True,
)
submit_click_event=submit.click(user,[msg,chatbot],[msg,chatbot],queue=False).then(
bot,
[chatbot,temperature,top_p,top_k,repetition_penalty,do_rag],
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
[vector_search_top_k,vector_rerank_top_n,do_rerank],
)
vector_rerank_top_n.release(
update_retriever,
[vector_search_top_k,vector_rerank_top_n,do_rerank],
)
do_rerank.change(
update_retriever,
[vector_search_top_k,vector_rerank_top_n,do_rerank],
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

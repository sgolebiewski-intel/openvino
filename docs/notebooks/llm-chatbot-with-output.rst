CreateanLLM-poweredChatbotusingOpenVINO
============================================

Intherapidlyevolvingworldofartificialintelligence(AI),chatbots
haveemergedaspowerfultoolsforbusinessestoenhancecustomer
interactionsandstreamlineoperations.LargeLanguageModels(LLMs)are
artificialintelligencesystemsthatcanunderstandandgeneratehuman
language.Theyusedeeplearningalgorithmsandmassiveamountsofdata
tolearnthenuancesoflanguageandproducecoherentandrelevant
responses.Whileadecentintent-basedchatbotcananswerbasic,
one-touchinquirieslikeordermanagement,FAQs,andpolicyquestions,
LLMchatbotscantacklemorecomplex,multi-touchquestions.LLMenables
chatbotstoprovidesupportinaconversationalmanner,similartohow
humansdo,throughcontextualmemory.Leveragingthecapabilitiesof
LanguageModels,chatbotsarebecomingincreasinglyintelligent,capable
ofunderstandingandrespondingtohumanlanguagewithremarkable
accuracy.

Previously,wealreadydiscussedhowtobuildaninstruction-following
pipelineusingOpenVINOandOptimumIntel,pleasecheckout`Dolly
example<../dolly-2-instruction-following>`__forreference.Inthis
tutorial,weconsiderhowtousethepowerofOpenVINOforrunningLarge
LanguageModelsforchat.Wewilluseapre-trainedmodelfromthe
`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__
library.Tosimplifytheuserexperience,the`HuggingFaceOptimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__libraryis
usedtoconvertthemodelstoOpenVINO™IRformatandtocreate
inferencepipeline.Theinferencepipelinecanalsobecreatedusing
`OpenVINOGenerate
API<https://github.com/openvinotoolkit/openvino.genai/tree/master/src>`__,
theexampleofthat,please,seeinthenotebook`LLMchatbotwith
OpenVINOGenerateAPI<./llm-chatbot-generate-api.ipynb>`__

Thetutorialconsistsofthefollowingsteps:

-Installprerequisites
-Downloadandconvertthemodelfromapublicsourceusingthe
`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-Compressmodelweightsto4-bitor8-bitdatatypesusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__
-Createachatinferencepipeline
-Runchatpipeline

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Selectmodelforinference<#select-model-for-inference>`__
-`ConvertmodelusingOptimum-CLI
tool<#convert-model-using-optimum-cli-tool>`__
-`Compressmodelweights<#compress-model-weights>`__

-`WeightsCompressionusing
Optimum-CLI<#weights-compression-using-optimum-cli>`__
-`WeightcompressionwithAWQ<#weight-compression-with-awq>`__

-`Selectdeviceforinferenceandmodel
variant<#select-device-for-inference-and-model-variant>`__
-`InstantiateModelusingOptimum
Intel<#instantiate-model-using-optimum-intel>`__
-`RunChatbot<#run-chatbot>`__

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
"torch>=2.1"\
"datasets"\
"accelerate"\
"gradio>=4.19"\
"onnx""einops""transformers_stream_generator""tiktoken""transformers>=4.40""bitsandbytes"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importos
frompathlibimportPath
importrequests
importshutil

#fetchmodelconfiguration

config_shared_path=Path("../../utils/llm_config.py")
config_dst_path=Path("llm_config.py")

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

Selectmodelforinference
--------------------------

`backtotop⬆️<#table-of-contents>`__

Thetutorialsupportsdifferentmodels,youcanselectonefromthe
providedoptionstocomparethequalityofopensourceLLMsolutions.
>\**Note**:conversionofsomemodelscanrequireadditionalactions
fromusersideandatleast64GBRAMforconversion.

Theavailableoptionsare:

-**tiny-llama-1b-chat**-Thisisthechatmodelfinetunedontopof
`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T<https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__.
TheTinyLlamaprojectaimstopretraina1.1BLlamamodelon3
trilliontokenswiththeadoptionofthesamearchitectureand
tokenizerasLlama2.ThismeansTinyLlamacanbepluggedandplayed
inmanyopen-sourceprojectsbuiltuponLlama.Besides,TinyLlamais
compactwithonly1.1Bparameters.Thiscompactnessallowsitto
catertoamultitudeofapplicationsdemandingarestricted
computationandmemoryfootprint.Moredetailsaboutmodelcanbe
foundin`model
card<https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
-**mini-cpm-2b-dpo**-MiniCPMisanEnd-SizeLLMdevelopedby
ModelBestInc. andTsinghuaNLP,withonly2.4Bparametersexcluding
embeddings.AfterDirectPreferenceOptimization(DPO)fine-tuning,
MiniCPMoutperformsmanypopular7b,13band70bmodels.Moredetails
canbefoundin
`model_card<https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16>`__.
-**gemma-2b-it**-Gemmaisafamilyoflightweight,state-of-the-art
openmodelsfromGoogle,builtfromthesameresearchandtechnology
usedtocreatetheGeminimodels.Theyaretext-to-text,decoder-only
largelanguagemodels,availableinEnglish,withopenweights,
pre-trainedvariants,andinstruction-tunedvariants.Gemmamodels
arewell-suitedforavarietyoftextgenerationtasks,including
questionanswering,summarization,andreasoning.Thismodelis
instruction-tunedversionof2Bparametersmodel.Moredetailsabout
modelcanbefoundin`model
card<https://huggingface.co/google/gemma-2b-it>`__.>\**Note**:run
modelwithdemo,youwillneedtoacceptlicenseagreement.>Youmust
bearegistereduserin🤗HuggingFaceHub.Pleasevisit`HuggingFace
modelcard<https://huggingface.co/google/gemma-2b-it>`__,carefully
readtermsofusageandclickacceptbutton.Youwillneedtousean
accesstokenforthecodebelowtorun.Formoreinformationon
accesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
>YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::python

##logintohuggingfacehubtogetaccesstopretrainedmodel

fromhuggingface_hubimportnotebook_login,whoami

try:
whoami()
print('Authorizationtokenalreadyprovided')
exceptOSError:
notebook_login()

-**phi3-mini-instruct**-ThePhi-3-Miniisa3.8Bparameters,
lightweight,state-of-the-artopenmodeltrainedwiththePhi-3
datasetsthatincludesbothsyntheticdataandthefilteredpublicly
availablewebsitesdatawithafocusonhigh-qualityandreasoning
denseproperties.Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__,
`Microsoftblog<https://aka.ms/phi3blog-april>`__and`technical
report<https://aka.ms/phi3-tech-report>`__.
-**red-pajama-3b-chat**-A2.8Bparameterpre-trainedlanguagemodel
basedonGPT-NEOXarchitecture.ItwasdevelopedbyTogetherComputer
andleadersfromtheopen-sourceAIcommunity.Themodelis
fine-tunedonOASST1andDolly2datasetstoenhancechattingability.
Moredetailsaboutmodelcanbefoundin`HuggingFacemodel
card<https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-**gemma-7b-it**-Gemmaisafamilyoflightweight,state-of-the-art
openmodelsfromGoogle,builtfromthesameresearchandtechnology
usedtocreatetheGeminimodels.Theyaretext-to-text,decoder-only
largelanguagemodels,availableinEnglish,withopenweights,
pre-trainedvariants,andinstruction-tunedvariants.Gemmamodels
arewell-suitedforavarietyoftextgenerationtasks,including
questionanswering,summarization,andreasoning.Thismodelis
instruction-tunedversionof7Bparametersmodel.Moredetailsabout
modelcanbefoundin`model
card<https://huggingface.co/google/gemma-7b-it>`__.>\**Note**:run
modelwithdemo,youwillneedtoacceptlicenseagreement.>Youmust
bearegistereduserin🤗HuggingFaceHub.Pleasevisit`HuggingFace
modelcard<https://huggingface.co/google/gemma-7b-it>`__,carefully
readtermsofusageandclickacceptbutton.Youwillneedtousean
accesstokenforthecodebelowtorun.Formoreinformationon
accesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
>YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::python

##logintohuggingfacehubtogetaccesstopretrainedmodel

fromhuggingface_hubimportnotebook_login,whoami

try:
whoami()
print('Authorizationtokenalreadyprovided')
exceptOSError:
notebook_login()

-**llama-2-7b-chat**-LLama2isthesecondgenerationofLLama
modelsdevelopedbyMeta.Llama2isacollectionofpre-trainedand
fine-tunedgenerativetextmodelsranginginscalefrom7billionto
70billionparameters.llama-2-7b-chatis7billionsparameters
versionofLLama2finetunedandoptimizedfordialogueusecase.
Moredetailsaboutmodelcanbefoundinthe
`paper<https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
`repository<https://github.com/facebookresearch/llama>`__and
`HuggingFacemodel
card<https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__.
>\**Note**:runmodelwithdemo,youwillneedtoacceptlicense
agreement.>Youmustbearegistereduserin🤗HuggingFaceHub.
Pleasevisit`HuggingFacemodel
card<https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
carefullyreadtermsofusageandclickacceptbutton.Youwillneed
touseanaccesstokenforthecodebelowtorun.Formore
informationonaccesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
>YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::python

##logintohuggingfacehubtogetaccesstopretrainedmodel

fromhuggingface_hubimportnotebook_login,whoami

try:
whoami()
print('Authorizationtokenalreadyprovided')
exceptOSError:
notebook_login()

-**llama-3-8b-instruct**-Llama3isanauto-regressivelanguage
modelthatusesanoptimizedtransformerarchitecture.Thetuned
versionsusesupervisedfine-tuning(SFT)andreinforcementlearning
withhumanfeedback(RLHF)toalignwithhumanpreferencesfor
helpfulnessandsafety.TheLlama3instructiontunedmodelsare
optimizedfordialogueusecasesandoutperformmanyoftheavailable
opensourcechatmodelsoncommonindustrybenchmarks.Moredetails
aboutmodelcanbefoundin`Metablog
post<https://ai.meta.com/blog/meta-llama-3/>`__,`model
website<https://llama.meta.com/llama3>`__and`model
card<https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__.
>\**Note**:runmodelwithdemo,youwillneedtoacceptlicense
agreement.>Youmustbearegistereduserin🤗HuggingFaceHub.
Pleasevisit`HuggingFacemodel
card<https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__,
carefullyreadtermsofusageandclickacceptbutton.Youwillneed
touseanaccesstokenforthecodebelowtorun.Formore
informationonaccesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
>YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::python

##logintohuggingfacehubtogetaccesstopretrainedmodel

fromhuggingface_hubimportnotebook_login,whoami

try:
whoami()
print('Authorizationtokenalreadyprovided')
exceptOSError:
notebook_login()

-**qwen2-1.5b-instruct/qwen2-7b-instruct**-Qwen2isthenewseries
ofQwenlargelanguagemodels.Comparedwiththestate-of-the-artopen
sourcelanguagemodels,includingthepreviousreleasedQwen1.5,
Qwen2hasgenerallysurpassedmostopensourcemodelsand
demonstratedcompetitivenessagainstproprietarymodelsacrossa
seriesofbenchmarkstargetingforlanguageunderstanding,language
generation,multilingualcapability,coding,mathematics,reasoning,
etc.Formoredetails,pleasereferto
`model_card<https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__,
`blog<https://qwenlm.github.io/blog/qwen2/>`__,
`GitHub<https://github.com/QwenLM/Qwen2>`__,and
`Documentation<https://qwen.readthedocs.io/en/latest/>`__.
-**qwen1.5-0.5b-chat/qwen1.5-1.8b-chat/qwen1.5-7b-chat**-Qwen1.5is
thebetaversionofQwen2,atransformer-baseddecoder-onlylanguage
modelpretrainedonalargeamountofdata.Qwen1.5isalanguage
modelseriesincludingdecoderlanguagemodelsofdifferentmodel
sizes.ItisbasedontheTransformerarchitecturewithSwiGLU
activation,attentionQKVbias,groupqueryattention,mixtureof
slidingwindowattentionandfullattention.Youcanfindmore
detailsaboutmodelinthe`model
repository<https://huggingface.co/Qwen>`__.
-**qwen-7b-chat**-Qwen-7Bisthe7B-parameterversionofthelarge
languagemodelseries,Qwen(abbr.TongyiQianwen),proposedby
AlibabaCloud.Qwen-7BisaTransformer-basedlargelanguagemodel,
whichispretrainedonalargevolumeofdata,includingwebtexts,
books,codes,etc.FormoredetailsaboutQwen,pleaserefertothe
`GitHub<https://github.com/QwenLM/Qwen>`__coderepository.
-**mpt-7b-chat**-MPT-7Bispartofthefamilyof
MosaicPretrainedTransformer(MPT)models,whichuseamodified
transformerarchitectureoptimizedforefficienttrainingand
inference.Thesearchitecturalchangesincludeperformance-optimized
layerimplementationsandtheeliminationofcontextlengthlimitsby
replacingpositionalembeddingswithAttentionwithLinearBiases
(`ALiBi<https://arxiv.org/abs/2108.12409>`__).Thankstothese
modifications,MPTmodelscanbetrainedwithhighthroughput
efficiencyandstableconvergence.MPT-7B-chatisachatbot-like
modelfordialoguegeneration.ItwasbuiltbyfinetuningMPT-7Bon
the
`ShareGPT-Vicuna<https://huggingface.co/datasets/jeffwan/sharegpt_vicuna>`__,
`HC3<https://huggingface.co/datasets/Hello-SimpleAI/HC3>`__,
`Alpaca<https://huggingface.co/datasets/tatsu-lab/alpaca>`__,
`HH-RLHF<https://huggingface.co/datasets/Anthropic/hh-rlhf>`__,and
`Evol-Instruct<https://huggingface.co/datasets/victor123/evol_instruct_70k>`__
datasets.Moredetailsaboutthemodelcanbefoundin`blog
post<https://www.mosaicml.com/blog/mpt-7b>`__,
`repository<https://github.com/mosaicml/llm-foundry/>`__and
`HuggingFacemodel
card<https://huggingface.co/mosaicml/mpt-7b-chat>`__.
-**chatglm3-6b**-ChatGLM3-6Bisthelatestopen-sourcemodelinthe
ChatGLMseries.Whileretainingmanyexcellentfeaturessuchas
smoothdialogueandlowdeploymentthresholdfromtheprevioustwo
generations,ChatGLM3-6Bemploysamorediversetrainingdataset,
moresufficienttrainingsteps,andamorereasonabletraining
strategy.ChatGLM3-6Badoptsanewlydesigned`Prompt
format<https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md>`__,
inadditiontothenormalmulti-turndialogue.Youcanfindmore
detailsaboutmodelinthe`model
card<https://huggingface.co/THUDM/chatglm3-6b>`__
-**mistral-7b**-TheMistral-7B-v0.1LargeLanguageModel(LLM)isa
pretrainedgenerativetextmodelwith7billionparameters.Youcan
findmoredetailsaboutmodelinthe`model
card<https://huggingface.co/mistralai/Mistral-7B-v0.1>`__,
`paper<https://arxiv.org/abs/2310.06825>`__and`releaseblog
post<https://mistral.ai/news/announcing-mistral-7b/>`__.
-**zephyr-7b-beta**-Zephyrisaseriesoflanguagemodelsthatare
trainedtoactashelpfulassistants.Zephyr-7B-betaisthesecond
modelintheseries,andisafine-tunedversionof
`mistralai/Mistral-7B-v0.1<https://huggingface.co/mistralai/Mistral-7B-v0.1>`__
thatwastrainedononamixofpubliclyavailable,synthetic
datasetsusing`DirectPreferenceOptimization
(DPO)<https://arxiv.org/abs/2305.18290>`__.Youcanfindmore
detailsaboutmodelin`technical
report<https://arxiv.org/abs/2310.16944>`__and`HuggingFacemodel
card<https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__.
-**neural-chat-7b-v3-1**-Mistral-7bmodelfine-tunedusingIntel
Gaudi.Themodelfine-tunedontheopensourcedataset
`Open-Orca/SlimOrca<https://huggingface.co/datasets/Open-Orca/SlimOrca>`__
andalignedwith`DirectPreferenceOptimization(DPO)
algorithm<https://arxiv.org/abs/2305.18290>`__.Moredetailscanbe
foundin`model
card<https://huggingface.co/Intel/neural-chat-7b-v3-1>`__and`blog
post<https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3>`__.
-**notus-7b-v1**-Notusisacollectionoffine-tunedmodelsusing
`DirectPreferenceOptimization
(DPO)<https://arxiv.org/abs/2305.18290>`__.andrelated
`RLHF<https://huggingface.co/blog/rlhf>`__techniques.Thismodelis
thefirstversion,fine-tunedwithDPOoverzephyr-7b-sft.Following
adata-firstapproach,theonlydifferencebetweenNotus-7B-v1and
Zephyr-7B-betaisthepreferencedatasetusedfordDPO.Proposed
approachfordatasetcreationhelpstoeffectivelyfine-tuneNotus-7b
thatsurpassesZephyr-7B-betaandClaude2on
`AlpacaEval<https://tatsu-lab.github.io/alpaca_eval/>`__.More
detailsaboutmodelcanbefoundin`model
card<https://huggingface.co/argilla/notus-7b-v1>`__.
-**youri-7b-chat**-Youri-7b-chatisaLlama2basedmodel.`Rinna
Co.,Ltd.<https://rinna.co.jp/>`__conductedfurtherpre-training
fortheLlama2modelwithamixtureofEnglishandJapanesedatasets
toimproveJapanesetaskcapability.Themodelispubliclyreleased
onHuggingFacehub.Youcanfinddetailedinformationatthe
`rinna/youri-7b-chatproject
page<https://huggingface.co/rinna/youri-7b>`__.
-**baichuan2-7b-chat**-Baichuan2isthenewgenerationof
large-scaleopen-sourcelanguagemodelslaunchedby`Baichuan
Intelligenceinc<https://www.baichuan-ai.com/home>`__.Itistrained
onahigh-qualitycorpuswith2.6trilliontokensandhasachieved
thebestperformanceinauthoritativeChineseandEnglishbenchmarks
ofthesamesize.
-**internlm2-chat-1.8b**-InternLM2isthesecondgenerationInternLM
series.Comparedtothepreviousgenerationmodel,itshows
significantimprovementsinvariouscapabilities,including
reasoning,mathematics,andcoding.Moredetailsaboutmodelcanbe
foundin`modelrepository<https://huggingface.co/internlm>`__.
-**glm-4-9b-chat**-GLM-4-9Bistheopen-sourceversionofthelatest
generationofpre-trainedmodelsintheGLM-4serieslaunchedby
ZhipuAI.Intheevaluationofdatasetsinsemantics,mathematics,
reasoning,code,andknowledge,GLM-4-9Banditshuman
preference-alignedversionGLM-4-9B-Chathaveshownsuperior
performancebeyondLlama-3-8B.Inadditiontomulti-round
conversations,GLM-4-9B-Chatalsohasadvancedfeaturessuchasweb
browsing,codeexecution,customtoolcalls(FunctionCall),andlong
textreasoning(supportingupto128Kcontext).Moredetailsabout
modelcanbefoundin`model
card<https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README_en.md>`__,
`technicalreport<https://arxiv.org/pdf/2406.12793>`__and
`repository<https://github.com/THUDM/GLM-4>`__

..code::ipython3

fromllm_configimportSUPPORTED_LLM_MODELS
importipywidgetsaswidgets

..code::ipython3

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

model_ids=list(SUPPORTED_LLM_MODELS[model_language.value])

model_id=widgets.Dropdown(
options=model_ids,
value=model_ids[0],
description="Model:",
disabled=False,
)

model_id




..parsed-literal::

Dropdown(description='Model:',index=2,options=('tiny-llama-1b-chat','gemma-2b-it','phi-3-mini-instruct','…



..code::ipython3

model_configuration=SUPPORTED_LLM_MODELS[model_language.value][model_id.value]
print(f"Selectedmodel{model_id.value}")


..parsed-literal::

Selectedmodelqwen2-7b-instruct


ConvertmodelusingOptimum-CLItool
------------------------------------

`backtotop⬆️<#table-of-contents>`__

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

Compressmodelweights
----------------------

The`Weights
Compression<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
algorithmisaimedatcompressingtheweightsofthemodelsandcanbe
usedtooptimizethemodelfootprintandperformanceoflargemodels
wherethesizeofweightsisrelativelylargerthanthesizeof
activations,forexample,LargeLanguageModels(LLM).ComparedtoINT8
compression,INT4compressionimprovesperformanceevenmore,but
introducesaminordropinpredictionquality.

WeightsCompressionusingOptimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

ForINT4quantizationyoucanalsospecifythefollowingarguments:-
The``--group-size``parameterwilldefinethegroupsizetousefor
quantization,-1itwillresultsinper-columnquantization.-The
``--ratio``parametercontrolstheratiobetween4-bitand8-bit
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Wecannowsavefloatingpointandcompressedmodelvariants

..code::ipython3

frompathlibimportPath

pt_model_id=model_configuration["model_id"]
pt_model_name=model_id.value.split("-")[0]
fp16_model_dir=Path(model_id.value)/"FP16"
int8_model_dir=Path(model_id.value)/"INT8_compressed_weights"
int4_model_dir=Path(model_id.value)/"INT4_compressed_weights"


defconvert_to_fp16():
if(fp16_model_dir/"openvino_model.xml").exists():
return
remote_code=model_configuration.get("remote_code",False)
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
remote_code=model_configuration.get("remote_code",False)
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

model_compression_params=compression_configs.get(model_id.value,compression_configs["default"])
if(int4_model_dir/"openvino_model.xml").exists():
return
remote_code=model_configuration.get("remote_code",False)
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



**Exportcommand:**



``optimum-cliexportopenvino--modelQwen/Qwen2-7B-Instruct--tasktext-generation-with-past--weight-formatint4--group-size128--ratio0.8qwen2-7b-instruct/INT4_compressed_weights``


..parsed-literal::



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

SizeofmodelwithINT4compressedweightsis4929.13MB


Selectdeviceforinferenceandmodelvariant
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

**Note**:TheremaybenospeedupforINT4/INT8compressedmodelson
dGPU.

..code::ipython3

importopenvinoasov

core=ov.Core()

support_devices=core.available_devices
if"NPU"insupport_devices:
support_devices.remove("NPU")

device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','AUTO'),value='CPU')



Thecellbelowdemonstrateshowtoinstantiatemodelbasedonselected
variantofmodelweightsandinferencedevice

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



InstantiateModelusingOptimumIntel
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.TheOptimumInferencemodelsareAPIcompatiblewithHugging
FaceTransformersmodels.Thismeanswejustneedtoreplace
``AutoModelForXxx``classwiththecorresponding``OVModelForXxx``
class.

BelowisanexampleoftheRedPajamamodel

..code::diff

-fromtransformersimportAutoModelForCausalLM
+fromoptimum.intel.openvinoimportOVModelForCausalLM
fromtransformersimportAutoTokenizer,pipeline

model_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1"
-model=AutoModelForCausalLM.from_pretrained(model_id)
+model=OVModelForCausalLM.from_pretrained(model_id,export=True)

Modelclassinitializationstartswithcalling``from_pretrained``
method.WhendownloadingandconvertingTransformersmodel,the
parameter``export=True``shouldbeadded(aswealreadyconvertedmodel
before,wedonotneedtoprovidethisparameter).Wecansavethe
convertedmodelforthenextusagewiththe``save_pretrained``method.
TokenizerclassandpipelinesAPIarecompatiblewithOptimummodels.

YoucanfindmoredetailsaboutOpenVINOLLMinferenceusingHuggingFace
OptimumAPIin`LLMinference
guide<https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html>`__.

..code::ipython3

fromtransformersimportAutoConfig,AutoTokenizer
fromoptimum.intel.openvinoimportOVModelForCausalLM

ifmodel_to_run.value=="INT4":
model_dir=int4_model_dir
elifmodel_to_run.value=="INT8":
model_dir=int8_model_dir
else:
model_dir=fp16_model_dir
print(f"Loadingmodelfrom{model_dir}")

ov_config={"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":""}

if"GPU"indevice.valueand"qwen2-7b-instruct"inmodel_id.value:
ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"]="NO"

#OnaGPUdeviceamodelisexecutedinFP16precision.Forred-pajama-3b-chatmodelthereknownaccuracy
#issuescausedbythis,whichweavoidbysettingprecisionhintto"f32".
ifmodel_id.value=="red-pajama-3b-chat"and"GPU"incore.available_devicesanddevice.valuein["GPU","AUTO"]:
ov_config["INFERENCE_PRECISION_HINT"]="f32"

model_name=model_configuration["model_id"]
tok=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

ov_model=OVModelForCausalLM.from_pretrained(
model_dir,
device=device.value,
ov_config=ov_config,
config=AutoConfig.from_pretrained(model_dir,trust_remote_code=True),
trust_remote_code=True,
)


..parsed-literal::

Loadingmodelfromqwen2-7b-instruct/INT4_compressed_weights


..parsed-literal::

Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
Theargument`trust_remote_code`istobeusedalongwithexport=True.Itwillbeignored.
CompilingthemodeltoCPU...


..code::ipython3

tokenizer_kwargs=model_configuration.get("tokenizer_kwargs",{})
test_string="2+2="
input_tokens=tok(test_string,return_tensors="pt",**tokenizer_kwargs)
answer=ov_model.generate(**input_tokens,max_new_tokens=2)
print(tok.batch_decode(answer,skip_special_tokens=True)[0])


..parsed-literal::

2+2=4


RunChatbot
-----------

`backtotop⬆️<#table-of-contents>`__

Now,whenmodelcreated,wecansetupChatbotinterfaceusing
`Gradio<https://www.gradio.app/>`__.Thediagrambelowillustrateshow
thechatbotpipelineworks

..figure::https://user-images.githubusercontent.com/29454499/255523209-d9336491-c7ba-4dc1-98f0-07f23743ce89.png
:alt:generationpipeline

generationpipeline

Ascanbeseen,thepipelineverysimilartoinstruction-followingwith
onlychangesthatpreviousconversationhistoryadditionallypassedas
inputwithnextuserquestionforgettingwiderinputcontext.Onthe
firstiteration,theuserprovidedinstructionsjoinedtoconversation
history(ifexists)convertedtotokenidsusingatokenizer,then
preparedinputprovidedtothemodel.Themodelgeneratesprobabilities
foralltokensinlogitsformatThewaythenexttokenwillbeselected
overpredictedprobabilitiesisdrivenbytheselecteddecoding
methodology.Youcanfindmoreinformationaboutthemostpopular
decodingmethodsinthis
`blog<https://huggingface.co/blog/how-to-generate>`__.Theresult
generationupdatesconversationhistoryfornextconversationstep.it
makesstrongerconnectionofnextquestionwithpreviouslyprovidedand
allowsusertomakeclarificationsregardingpreviouslyprovided
answers.https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html

|Thereareseveralparametersthatcancontroltextgenerationquality:
\*``Temperature``isaparameterusedtocontrolthelevelof
creativityinAI-generatedtext.Byadjustingthe``temperature``,you
caninfluencetheAImodel’sprobabilitydistribution,makingthetext
morefocusedordiverse.
|Considerthefollowingexample:TheAImodelhastocompletethe
sentence“Thecatis\____.”withthefollowingtokenprobabilities:

::

playing:0.5
sleeping:0.25
eating:0.15
driving:0.05
flying:0.05

-**Lowtemperature**(e.g.,0.2):TheAImodelbecomesmorefocusedanddeterministic,choosingtokenswiththehighestprobability,suchas"playing."
-**Mediumtemperature**(e.g.,1.0):TheAImodelmaintainsabalancebetweencreativityandfocus,selectingtokensbasedontheirprobabilitieswithoutsignificantbias,suchas"playing,""sleeping,"or"eating."
-**Hightemperature**(e.g.,2.0):TheAImodelbecomesmoreadventurous,increasingthechancesofselectinglesslikelytokens,suchas"driving"and"flying."

-``Top-p``,alsoknownasnucleussampling,isaparameterusedto
controltherangeoftokensconsideredbytheAImodelbasedontheir
cumulativeprobability.Byadjustingthe``top-p``value,youcan
influencetheAImodel’stokenselection,makingitmorefocusedor
diverse.Usingthesameexamplewiththecat,considerthefollowing
top_psettings:

-**Lowtop_p**(e.g.,0.5):TheAImodelconsidersonlytokenswith
thehighestcumulativeprobability,suchas“playing.”
-**Mediumtop_p**(e.g.,0.8):TheAImodelconsiderstokenswitha
highercumulativeprobability,suchas“playing,”“sleeping,”and
“eating.”
-**Hightop_p**(e.g.,1.0):TheAImodelconsidersalltokens,
includingthosewithlowerprobabilities,suchas“driving”and
“flying.”

-``Top-k``isananotherpopularsamplingstrategy.Incomparisonwith
Top-P,whichchoosesfromthesmallestpossiblesetofwordswhose
cumulativeprobabilityexceedstheprobabilityP,inTop-KsamplingK
mostlikelynextwordsarefilteredandtheprobabilitymassis
redistributedamongonlythoseKnextwords.Inourexamplewithcat,
ifk=3,thenonly“playing”,“sleeping”and“eating”willbetaken
intoaccountaspossiblenextword.
-``RepetitionPenalty``Thisparametercanhelppenalizetokensbased
onhowfrequentlytheyoccurinthetext,includingtheinputprompt.
Atokenthathasalreadyappearedfivetimesispenalizedmore
heavilythanatokenthathasappearedonlyonetime.Avalueof1
meansthatthereisnopenaltyandvalueslargerthan1discourage
repeated
tokens.https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html

..code::ipython3

importtorch
fromthreadingimportEvent,Thread
fromuuidimportuuid4
fromtypingimportList,Tuple
importgradioasgr
fromtransformersimport(
AutoTokenizer,
StoppingCriteria,
StoppingCriteriaList,
TextIteratorStreamer,
)


model_name=model_configuration["model_id"]
start_message=model_configuration["start_message"]
history_template=model_configuration.get("history_template")
current_message_template=model_configuration.get("current_message_template")
stop_tokens=model_configuration.get("stop_tokens")
tokenizer_kwargs=model_configuration.get("tokenizer_kwargs",{})

chinese_examples=[
["你好!"],
["你是谁?"],
["请介绍一下上海"],
["请介绍一下英特尔公司"],
["晚上睡不着怎么办？"],
["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
["给这个故事起一个标题。"],
]

english_examples=[
["Hellothere!Howareyoudoing?"],
["WhatisOpenVINO?"],
["Whoareyou?"],
["CanyouexplaintomebrieflywhatisPythonprogramminglanguage?"],
["ExplaintheplotofCinderellainasentence."],
["Whataresomecommonmistakestoavoidwhenwritingcode?"],
["Writea100-wordblogposton“BenefitsofArtificialIntelligenceandOpenVINO“"],
]

japanese_examples=[
["こんにちは！調子はどうですか?"],
["OpenVINOとは何ですか?"],
["あなたは誰ですか?"],
["Pythonプログラミング言語とは何か簡単に説明してもらえますか?"],
["シンデレラのあらすじを一文で説明してください。"],
["コードを書くときに避けるべきよくある間違いは何ですか?"],
["人工知能と「OpenVINOの利点」について100語程度のブログ記事を書いてください。"],
]

examples=chinese_examplesif(model_language.value=="Chinese")elsejapanese_examplesif(model_language.value=="Japanese")elseenglish_examples

max_new_tokens=256


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
stop_tokens=tok.convert_tokens_to_ids(stop_tokens)

stop_tokens=[StopOnTokens(stop_tokens)]


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


text_processor=model_configuration.get("partial_text_processor",default_partial_text_processor)


defconvert_history_to_token(history:List[Tuple[str,str]]):
"""
functionforconversionhistorystoredaslistpairsofuserandassistantmessagestotokensaccordingtomodelexpectedconversationtemplate
Params:
history:dialoguehistory
Returns:
historyintokenformat
"""
ifpt_model_name=="baichuan2":
system_tokens=tok.encode(start_message)
history_tokens=[]
forold_query,responseinhistory[:-1]:
round_tokens=[]
round_tokens.append(195)
round_tokens.extend(tok.encode(old_query))
round_tokens.append(196)
round_tokens.extend(tok.encode(response))
history_tokens=round_tokens+history_tokens
input_tokens=system_tokens+history_tokens
input_tokens.append(195)
input_tokens.extend(tok.encode(history[-1][0]))
input_tokens.append(196)
input_token=torch.LongTensor([input_tokens])
elifhistory_templateisNone:
messages=[{"role":"system","content":start_message}]
foridx,(user_msg,model_msg)inenumerate(history):
ifidx==len(history)-1andnotmodel_msg:
messages.append({"role":"user","content":user_msg})
break
ifuser_msg:
messages.append({"role":"user","content":user_msg})
ifmodel_msg:
messages.append({"role":"assistant","content":model_msg})

input_token=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors="pt")
else:
text=start_message+"".join(
["".join([history_template.format(num=round,user=item[0],assistant=item[1])])forround,iteminenumerate(history[:-1])]
)
text+="".join(
[
"".join(
[
current_message_template.format(
num=len(history)+1,
user=history[-1][0],
assistant=history[-1][1],
)
]
)
]
)
input_token=tok(text,return_tensors="pt",**tokenizer_kwargs).input_ids
returninput_token


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


defbot(history,temperature,top_p,top_k,repetition_penalty,conversation_id):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory
temperature:parameterforcontrolthelevelofcreativityinAI-generatedtext.
Byadjustingthe`temperature`,youcaninfluencetheAImodel'sprobabilitydistribution,makingthetextmorefocusedordiverse.
top_p:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability.
top_k:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability,selectingnumberoftokenswithhighestprobability.
repetition_penalty:parameterforpenalizingtokensbasedonhowfrequentlytheyoccurinthetext.
conversation_id:uniqueconversationidentifier.

"""

#Constructtheinputmessagestringforthemodelbyconcatenatingthecurrentsystemmessageandconversationhistory
#Tokenizethemessagesstring
input_ids=convert_history_to_token(history)
ifinput_ids.shape[1]>2000:
history=[history[-1]]
input_ids=convert_history_to_token(history)
streamer=TextIteratorStreamer(tok,timeout=30.0,skip_prompt=True,skip_special_tokens=True)
generate_kwargs=dict(
input_ids=input_ids,
max_new_tokens=max_new_tokens,
temperature=temperature,
do_sample=temperature>0.0,
top_p=top_p,
top_k=top_k,
repetition_penalty=repetition_penalty,
streamer=streamer,
)
ifstop_tokensisnotNone:
generate_kwargs["stopping_criteria"]=StoppingCriteriaList(stop_tokens)

stream_complete=Event()

defgenerate_and_signal_complete():
"""
genrationfunctionforsinglethread
"""
globalstart_time
ov_model.generate(**generate_kwargs)
stream_complete.set()

t1=Thread(target=generate_and_signal_complete)
t1.start()

#Initializeanemptystringtostorethegeneratedtext
partial_text=""
fornew_textinstreamer:
partial_text=text_processor(partial_text,new_text)
history[-1][1]=partial_text
yieldhistory


defrequest_cancel():
ov_model.request.cancel()


defget_uuid():
"""
universaluniqueidentifierforthread
"""
returnstr(uuid4())


withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asdemo:
conversation_id=gr.State(get_uuid)
gr.Markdown(f"""<h1><center>OpenVINO{model_id.value}Chatbot</center></h1>""")
chatbot=gr.Chatbot(height=500)
withgr.Row():
withgr.Column():
msg=gr.Textbox(
label="ChatMessageBox",
placeholder="ChatMessageBox",
show_label=False,
container=False,
)
withgr.Column():
withgr.Row():
submit=gr.Button("Submit")
stop=gr.Button("Stop")
clear=gr.Button("Clear")
withgr.Row():
withgr.Accordion("AdvancedOptions:",open=False):
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
gr.Examples(examples,inputs=msg,label="Clickonanyexampleandpressthe'Submit'button")

submit_event=msg.submit(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
temperature,
top_p,
top_k,
repetition_penalty,
conversation_id,
],
outputs=chatbot,
queue=True,
)
submit_click_event=submit.click(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
temperature,
top_p,
top_k,
repetition_penalty,
conversation_id,
],
outputs=chatbot,
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

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#ifyouhaveanyissuetolaunchonyourplatform,youcanpassshare=Truetolaunchmethod:
#demo.launch(share=True)
#itcreatesapubliclyshareablelinkfortheinterface.Readmoreinthedocs:https://gradio.app/docs/
demo.launch()

..code::ipython3

#pleaseuncommentandrunthiscellforstoppinggradiointerface
#demo.close()

NextStep
~~~~~~~~~

Besideschatbot,wecanuseLangChaintoaugmentingLLMknowledgewith
additionaldata,whichallowyoutobuildAIapplicationsthatcan
reasonaboutprivatedataordataintroducedafteramodel’scutoff
date.Youcanfindthissolutionin`Retrieval-augmentedgeneration
(RAG)example<../llm-rag-langchain/>`__.

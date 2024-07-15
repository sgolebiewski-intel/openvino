InstructionfollowingusingDatabricksDolly2.0andOpenVINO
=============================================================

Theinstructionfollowingisoneofthecornerstonesofthecurrent
generationoflargelanguagemodels(LLMs).Reinforcementlearningwith
humanpreferences(`RLHF<https://arxiv.org/abs/1909.08593>`__)and
techniquessuchas`InstructGPT<https://arxiv.org/abs/2203.02155>`__
hasbeenthecorefoundationofbreakthroughssuchasChatGPTandGPT-4.
However,thesepowerfulmodelsremainhiddenbehindAPIsandweknow
verylittleabouttheirunderlyingarchitecture.Instruction-following
modelsarecapableofgeneratingtextinresponsetopromptsandare
oftenusedfortaskslikewritingassistance,chatbots,andcontent
generation.Manyusersnowinteractwiththesemodelsregularlyandeven
usethemforworkbutthemajorityofsuchmodelsremainclosed-source
andrequiremassiveamountsofcomputationalresourcestoexperiment
with.

`Dolly
2.0<https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm>`__
isthefirstopen-source,instruction-followingLLMfine-tunedby
Databricksonatransparentandfreelyavailabledatasetthatisalso
open-sourcedtouseforcommercialpurposes.ThatmeansDolly2.0is
availableforcommercialapplicationswithouttheneedtopayforAPI
accessorsharedatawiththirdparties.Dolly2.0exhibitssimilar
characteristicssoChatGPTdespitebeingmuchsmaller.

Inthistutorial,weconsiderhowtorunaninstruction-followingtext
generationpipelineusingDolly2.0andOpenVINO.Wewillusea
pre-trainedmodelfromthe`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__
library.Tosimplifytheuserexperience,the`HuggingFaceOptimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__libraryis
usedtoconvertthemodelstoOpenVINO™IRformat.

Thetutorialconsistsofthefollowingsteps:

-Installprerequisites
-Downloadandconvertthemodelfromapublicsourceusingthe
`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-CompressmodelweightstoINT8with`OpenVINO
NNCF<https://github.com/openvinotoolkit/nncf>`__
-Createaninstruction-followinginferencepipeline
-Runinstruction-followingpipeline

AboutDolly2.0
---------------

Dolly2.0isaninstruction-followinglargelanguagemodeltrainedon
theDatabricksmachine-learningplatformthatislicensedforcommercial
use.Itisbasedon`Pythia<https://github.com/EleutherAI/pythia>`__
andistrainedon~15kinstruction/responsefine-tuningrecords
generatedbyDatabricksemployeesinvariouscapabilitydomains,
includingbrainstorming,classification,closedQA,generation,
informationextraction,openQA,andsummarization.Dolly2.0worksby
processingnaturallanguageinstructionsandgeneratingresponsesthat
followthegiveninstructions.Itcanbeusedforawiderangeof
applications,includingclosedquestion-answering,summarization,and
generation.

Themodeltrainingprocesswasinspiredby
`InstructGPT<https://arxiv.org/abs/2203.02155>`__.TotrainInstructGPT
models,thecoretechniqueisreinforcementlearningfromhumanfeedback
(RLHF),Thistechniqueuseshumanpreferencesasarewardsignalto
fine-tunemodels,whichisimportantasthesafetyandalignment
problemsrequiredtobesolvedarecomplexandsubjective,andaren’t
fullycapturedbysimpleautomaticmetrics.Moredetailsaboutthe
InstructGPTapproachcanbefoundinOpenAI`blog
post<https://openai.com/research/instruction-following>`__The
breakthroughdiscoveredwithInstructGPTisthatlanguagemodelsdon’t
needlargerandlargertrainingsets.Byusinghuman-evaluated
question-and-answertraining,authorswereabletotrainabetter
languagemodelusingonehundredtimesfewerparametersthanthe
previousmodel.Databricksusedasimilarapproachtocreateaprompt
andresponsedatasetcalledtheycall
`databricks-dolly-15k<https://huggingface.co/datasets/databricks/databricks-dolly-15k>`__,
acorpusofmorethan15,000recordsgeneratedbythousandsof
Databricksemployeestoenablelargelanguagemodelstoexhibitthe
magicalinteractivityofInstructGPT.Moredetailsaboutthemodeland
datasetcanbefoundin`Databricksblog
post<https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm>`__
and`repo<https://github.com/databrickslabs/dolly>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`ConvertmodelusingOptimum-CLI
tool<#convert-model-using-optimum-cli-tool>`__
-`Compressmodelweights<#compress-model-weights>`__

-`WeightsCompressionusing
Optimum-CLI<#weights-compression-using-optimum-cli>`__

-`Selectmodelvariantandinference
device<#select-model-variant-and-inference-device>`__
-`InstantiateModelusingOptimum
Intel<#instantiate-model-using-optimum-intel>`__
-`Createaninstruction-followinginference
pipeline<#create-an-instruction-following-inference-pipeline>`__

-`Setupimports<#setup-imports>`__
-`Preparetemplateforuser
prompt<#prepare-template-for-user-prompt>`__
-`Helpersforoutputparsing<#helpers-for-output-parsing>`__
-`Maingenerationfunction<#main-generation-function>`__
-`Helpersforapplication<#helpers-for-application>`__

-`Runinstruction-following
pipeline<#run-instruction-following-pipeline>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

First,weshouldinstallthe`HuggingFace
Optimum<https://huggingface.co/docs/optimum/installation>`__library
acceleratedbyOpenVINOintegration.TheHuggingFaceOptimumIntelAPI
isahigh-levelAPIthatenablesustoconvertandquantizemodelsfrom
theHuggingFaceTransformerslibrarytotheOpenVINO™IRformat.For
moredetails,refertothe`HuggingFaceOptimumIntel
documentation<https://huggingface.co/docs/optimum/intel/inference>`__.

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipuninstall-q-yoptimumoptimum-intel
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q"diffusers>=0.16.1""transformers>=4.33.0""torch>=2.1""nncf>=2.10.0"onnx"gradio>=4.19"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git"

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

..code::bash

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

`backtotop⬆️<#table-of-contents>`__

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
importipywidgetsaswidgets

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


..code::ipython3

frompathlibimportPath

model_id="databricks/dolly-v2-3b"
model_path=Path("dolly-v2-3b")

fp16_model_dir=model_path/"FP16"
int8_model_dir=model_path/"INT8_compressed_weights"
int4_model_dir=model_path/"INT4_compressed_weights"


defconvert_to_fp16():
if(fp16_model_dir/"openvino_model.xml").exists():
return
fp16_model_dir.mkdir(parents=True,exist_ok=True)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatfp16".format(model_id)
export_command=export_command_base+""+str(fp16_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int8():
if(int8_model_dir/"openvino_model.xml").exists():
return
int8_model_dir.mkdir(parents=True,exist_ok=True)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint8".format(model_id)
export_command=export_command_base+""+str(int8_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int4():
if(int4_model_dir/"openvino_model.xml").exists():
return
int4_model_dir.mkdir(parents=True,exist_ok=True)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint4".format(model_id)
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

SizeofmodelwithINT4compressedweightsis2154.54MB


Selectmodelvariantandinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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



..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','AUTO'),value='CPU')



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

BelowisanexampleoftheDollymodel

..code::diff

-fromtransformersimportAutoModelForCausalLM
+fromoptimum.intel.openvinoimportOVModelForCausalLM
fromtransformersimportAutoTokenizer,pipeline

model_id="databricks/dolly-v2-3b"
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

frompathlibimportPath
fromtransformersimportAutoTokenizer
fromoptimum.intel.openvinoimportOVModelForCausalLM

ifmodel_to_run.value=="INT4":
model_dir=int4_model_dir
elifmodel_to_run.value=="INT8":
model_dir=int8_model_dir
else:
model_dir=fp16_model_dir
print(f"Loadingmodelfrom{model_dir}")

tokenizer=AutoTokenizer.from_pretrained(model_dir)

current_device=device.value

ov_config={"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":""}

ov_model=OVModelForCausalLM.from_pretrained(model_dir,device=current_device,ov_config=ov_config)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

NoCUDAruntimeisfound,usingCUDA_HOME='/usr/local/cuda'
2024-05-0110:43:29.010748:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-05-0110:43:29.012724:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-05-0110:43:29.047558:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-05-0110:43:29.048434:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-05-0110:43:29.742257:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34:UserWarning:TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
warn("TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport."


..parsed-literal::

/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so:undefinedsymbol:cadam32bit_grad_fp32


..parsed-literal::

WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.0.1+cu118withCUDA1108(youhave2.1.2+cpu)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


..parsed-literal::

Loadingmodelfromdolly-v2-3b/INT4_compressed_weights


..parsed-literal::

CompilingthemodeltoCPU...


Createaninstruction-followinginferencepipeline
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

The``run_generation``functionacceptsuser-providedtextinput,
tokenizesit,andrunsthegenerationprocess.Textgenerationisan
iterativeprocess,whereeachnexttokendependsonpreviouslygenerated
untilamaximumnumberoftokensorstopgenerationconditionisnot
reached.Toobtainintermediategenerationresultswithoutwaitinguntil
whengenerationisfinished,wewilluse
`TextIteratorStreamer<https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer>`__,
providedaspartofHuggingFace`Streaming
API<https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming>`__.

Thediagrambelowillustrateshowtheinstruction-followingpipeline
works

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e881f4a4-fcc8-427a-afe1-7dd80aebd66e
:alt:generationpipeline)

generationpipeline)

Ascanbeseen,onthefirstiteration,theuserprovidedinstructions
convertedtotokenidsusingatokenizer,thenpreparedinputprovided
tothemodel.Themodelgeneratesprobabilitiesforalltokensinlogits
formatThewaythenexttokenwillbeselectedoverpredicted
probabilitiesisdrivenbytheselecteddecodingmethodology.Youcan
findmoreinformationaboutthemostpopulardecodingmethodsinthis
`blog<https://huggingface.co/blog/how-to-generate>`__.

Thereareseveralparametersthatcancontroltextgenerationquality:

-|``Temperature``isaparameterusedtocontrolthelevelof
creativityinAI-generatedtext.Byadjustingthe``temperature``,
youcaninfluencetheAImodel’sprobabilitydistribution,making
thetextmorefocusedordiverse.
|Considerthefollowingexample:TheAImodelhastocompletethe
sentence“Thecatis\____.”withthefollowingtoken
probabilities:

|playing:0.5
|sleeping:0.25
|eating:0.15
|driving:0.05
|flying:0.05

-**Lowtemperature**(e.g.,0.2):TheAImodelbecomesmorefocused
anddeterministic,choosingtokenswiththehighestprobability,
suchas“playing.”
-**Mediumtemperature**(e.g.,1.0):TheAImodelmaintainsa
balancebetweencreativityandfocus,selectingtokensbasedon
theirprobabilitieswithoutsignificantbias,suchas“playing,”
“sleeping,”or“eating.”
-**Hightemperature**(e.g.,2.0):TheAImodelbecomesmore
adventurous,increasingthechancesofselectinglesslikely
tokens,suchas“driving”and“flying.”

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

-``Top-k``isanotherpopularsamplingstrategy.Incomparisonwith
Top-P,whichchoosesfromthesmallestpossiblesetofwordswhose
cumulativeprobabilityexceedstheprobabilityP,inTop-KsamplingK
mostlikelynextwordsarefilteredandtheprobabilitymassis
redistributedamongonlythoseKnextwords.Inourexamplewithcat,
ifk=3,thenonly“playing”,“sleeping”and“eating”willbetaken
intoaccountaspossiblenextword.

Tooptimizethegenerationprocessandusememorymoreefficiently,the
``use_cache=True``optionisenabled.Sincetheoutputsideis
auto-regressive,anoutputtokenhiddenstateremainsthesameonce
computedforeveryfurthergenerationstep.Therefore,recomputingit
everytimeyouwanttogenerateanewtokenseemswasteful.Withthe
cache,themodelsavesthehiddenstateonceithasbeencomputed.The
modelonlycomputestheoneforthemostrecentlygeneratedoutputtoken
ateachtimestep,re-usingthesavedonesforhiddentokens.This
reducesthegenerationcomplexityfromO(n^3)toO(n^2)fora
transformermodel.Moredetailsabouthowitworkscanbefoundinthis
`article<https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
Withthisoption,themodelgetsthepreviousstep’shiddenstates
(cachedattentionkeysandvalues)asinputandadditionallyprovides
hiddenstatesforthecurrentstepasoutput.Itmeansforallnext
iterations,itisenoughtoprovideonlyanewtokenobtainedfromthe
previousstepandcachedkeyvaluestogetthenexttokenprediction.

Thegenerationcyclerepeatsuntiltheendofthesequencetokenis
reachedoritalsocanbeinterruptedwhenmaximumtokenswillbe
generated.Asalreadymentionedbefore,wecanenableprintingcurrent
generatedtokenswithoutwaitinguntilwhenthewholegenerationis
finishedusingStreamingAPI,itaddsanewtokentotheoutputqueue
andthenprintsthemwhentheyareready.

Setupimports
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromthreadingimportThread
fromtimeimportperf_counter
fromtypingimportList
importgradioasgr
fromtransformersimportAutoTokenizer,TextIteratorStreamer
importnumpyasnp

Preparetemplateforuserprompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Foreffectivegeneration,modelexpectstohaveinputinspecific
format.Thecodebelowpreparetemplateforpassinguserinstruction
intomodelwithprovidingadditionalcontext.

..code::ipython3

INSTRUCTION_KEY="###Instruction:"
RESPONSE_KEY="###Response:"
END_KEY="###End"
INTRO_BLURB="Belowisaninstructionthatdescribesatask.Writearesponsethatappropriatelycompletestherequest."

#Thisisthepromptthatisusedforgeneratingresponsesusinganalreadytrainedmodel.Itendswiththeresponse
#key,wherethejobofthemodelistoprovidethecompletionthatfollowsit(i.e.theresponseitself).
PROMPT_FOR_GENERATION_FORMAT="""{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
intro=INTRO_BLURB,
instruction_key=INSTRUCTION_KEY,
instruction="{instruction}",
response_key=RESPONSE_KEY,
)

Helpersforoutputparsing
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Modelwasretrainedtofinishgenerationusingspecialtoken``###End``
thecodebelowfinditsidforusingitasgenerationstop-criteria.

..code::ipython3

defget_special_token_id(tokenizer:AutoTokenizer,key:str)->int:
"""
GetsthetokenIDforagivenstringthathasbeenaddedtothetokenizerasaspecialtoken.

Whentraining,weconfigurethetokenizersothatthesequenceslike"###Instruction:"and"###End"are
treatedspeciallyandconvertedtoasingle,newtoken.ThisretrievesthetokenIDeachofthesekeysmapto.

Args:
tokenizer(PreTrainedTokenizer):thetokenizer
key(str):thekeytoconverttoasingletoken

Raises:
RuntimeError:ifmorethanoneIDwasgenerated

Returns:
int:thetokenIDforthegivenkey
"""
token_ids=tokenizer.encode(key)
iflen(token_ids)>1:
raiseValueError(f"Expectedonlyasingletokenfor'{key}'butfound{token_ids}")
returntoken_ids[0]


tokenizer_response_key=next(
(tokenfortokenintokenizer.additional_special_tokensiftoken.startswith(RESPONSE_KEY)),
None,
)

end_key_token_id=None
iftokenizer_response_key:
try:
end_key_token_id=get_special_token_id(tokenizer,END_KEY)
#Ensuregenerationstopsonceitgenerates"###End"
exceptValueError:
pass

Maingenerationfunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Asitwasdiscussedabove,``run_generation``functionistheentry
pointforstartinggeneration.Itgetsprovidedinputinstructionas
parameterandreturnsmodelresponse.

..code::ipython3

defrun_generation(
user_text:str,
top_p:float,
temperature:float,
top_k:int,
max_new_tokens:int,
perf_text:str,
):
"""
Textgenerationfunction

Parameters:
user_text(str):User-providedinstructionforageneration.
top_p(float):Nucleussampling.Ifsetto<1,onlythesmallestsetofmostprobabletokenswithprobabilitiesthatadduptotop_porhigherarekeptforageneration.
temperature(float):Thevalueusedtomodulethelogitsdistribution.
top_k(int):Thenumberofhighestprobabilityvocabularytokenstokeepfortop-k-filtering.
max_new_tokens(int):Maximumlengthofgeneratedsequence.
perf_text(str):Contentoftextfieldforprintingperformanceresults.
Returns:
model_output(str)-model-generatedtext
perf_text(str)-updatedperftextfiledcontent
"""

#Prepareinputpromptaccordingtomodelexpectedtemplate
prompt_text=PROMPT_FOR_GENERATION_FORMAT.format(instruction=user_text)

#Tokenizetheusertext.
model_inputs=tokenizer(prompt_text,return_tensors="pt")

#Startgenerationonaseparatethread,sothatwedon'tblocktheUI.Thetextispulledfromthestreamer
#inthemainthread.Addstimeouttothestreamertohandleexceptionsinthegenerationthread.
streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
generate_kwargs=dict(
model_inputs,
streamer=streamer,
max_new_tokens=max_new_tokens,
do_sample=True,
top_p=top_p,
temperature=float(temperature),
top_k=top_k,
eos_token_id=end_key_token_id,
)
t=Thread(target=ov_model.generate,kwargs=generate_kwargs)
t.start()

#Pullthegeneratedtextfromthestreamer,andupdatethemodeloutput.
model_output=""
per_token_time=[]
num_tokens=0
start=perf_counter()
fornew_textinstreamer:
current_time=perf_counter()-start
model_output+=new_text
perf_text,num_tokens=estimate_latency(current_time,perf_text,new_text,per_token_time,num_tokens)
yieldmodel_output,perf_text
start=perf_counter()
returnmodel_output,perf_text

Helpersforapplication
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

FormakinginteractiveuserinterfacewewilluseGradiolibrary.The
codebellowprovidesusefulfunctionsusedforcommunicationwithUI
elements.

..code::ipython3

defestimate_latency(
current_time:float,
current_perf_text:str,
new_gen_text:str,
per_token_time:List[float],
num_tokens:int,
):
"""
Helperfunctionforperformanceestimation

Parameters:
current_time(float):Thissteptimeinseconds.
current_perf_text(str):CurrentcontentofperformanceUIfield.
new_gen_text(str):Newgeneratedtext.
per_token_time(List[float]):historyofperformancefromprevioussteps.
num_tokens(int):Totalnumberofgeneratedtokens.

Returns:
updateforperformancetextfield
updateforatotalnumberoftokens
"""
num_current_toks=len(tokenizer.encode(new_gen_text))
num_tokens+=num_current_toks
per_token_time.append(num_current_toks/current_time)
iflen(per_token_time)>10andlen(per_token_time)%4==0:
current_bucket=per_token_time[:-10]
return(
f"Averagegenerationspeed:{np.mean(current_bucket):.2f}tokens/s.Totalgeneratedtokens:{num_tokens}",
num_tokens,
)
returncurrent_perf_text,num_tokens


defreset_textbox(instruction:str,response:str,perf:str):
"""
Helperfunctionforresettingcontentofalltextfields

Parameters:
instruction(str):Contentofuserinstructionfield.
response(str):Contentofmodelresponsefield.
perf(str):Contentofperformanceinfofiled

Returns:
emptystringforeachplaceholder
"""
return"","",""


defselect_device(device_str:str,current_text:str="",progress:gr.Progress=gr.Progress()):
"""
Helperfunctionforuploadingmodelonthedevice.

Parameters:
device_str(str):Devicename.
current_text(str):Currentcontentofuserinstructionfield(usedonlyforbackuppurposes,temporallyreplacingitontheprogressbarduringmodelloading).
progress(gr.Progress):gradioprogresstracker
Returns:
current_text
"""
ifdevice_str!=ov_model._device:
ov_model.request=None
ov_model._device=device_str

foriinprogress.tqdm(range(1),desc=f"Modelloadingon{device_str}"):
ov_model.compile()
returncurrent_text

Runinstruction-followingpipeline
----------------------------------

`backtotop⬆️<#table-of-contents>`__

Now,wearereadytoexploremodelcapabilities.Thisdemoprovidesa
simpleinterfacethatallowscommunicationwithamodelusingtext
instruction.Typeyourinstructionintothe``Userinstruction``field
orselectonefrompredefinedexamplesandclickonthe``Submit``
buttontostartgeneration.Additionally,youcanmodifyadvanced
generationparameters:

-``Device``-allowsswitchinginferencedevice.Pleasenote,every
timewhennewdeviceisselected,modelwillberecompiledandthis
takessometime.
-``MaxNewTokens``-maximumsizeofgeneratedtext.
-``Top-p(nucleussampling)``-ifsetto<1,onlythesmallestset
ofmostprobabletokenswithprobabilitiesthatadduptotop_por
higherarekeptforageneration.
-``Top-k``-thenumberofhighestprobabilityvocabularytokensto
keepfortop-k-filtering.
-``Temperature``-thevalueusedtomodulethelogitsdistribution.

..code::ipython3

available_devices=ov.Core().available_devices+["AUTO"]

examples=[
"Givemerecipeforpizzawithpineapple",
"WritemeatweetaboutnewOpenVINOrelease",
"ExplaindifferencebetweenCPUandGPU",
"Givefiveideasforgreatweekendwithfamily",
"DoAndroidsdreamofElectricsheep?",
"WhoisDolly?",
"Pleasegivemeadvicehowtowriteresume?",
"Name3advantagestobeacat",
"WriteinstructionsonhowtobecomeagoodAIengineer",
"Writealovelettertomybestfriend",
]

withgr.Blocks()asdemo:
gr.Markdown(
"#InstructionfollowingusingDatabricksDolly2.0andOpenVINO.\n"
"Provideinsturctionwhichdescribesataskbeloworselectamongpredefinedexamplesandmodelwritesresponsethatperformsrequestedtask."
)

withgr.Row():
withgr.Column(scale=4):
user_text=gr.Textbox(
placeholder="Writeanemailaboutanalpacathatlikesflan",
label="Userinstruction",
)
model_output=gr.Textbox(label="Modelresponse",interactive=False)
performance=gr.Textbox(label="Performance",lines=1,interactive=False)
withgr.Column(scale=1):
button_clear=gr.Button(value="Clear")
button_submit=gr.Button(value="Submit")
gr.Examples(examples,user_text)
withgr.Column(scale=1):
device=gr.Dropdown(choices=available_devices,value=current_device,label="Device")
max_new_tokens=gr.Slider(
minimum=1,
maximum=1000,
value=256,
step=1,
interactive=True,
label="MaxNewTokens",
)
top_p=gr.Slider(
minimum=0.05,
maximum=1.0,
value=0.92,
step=0.05,
interactive=True,
label="Top-p(nucleussampling)",
)
top_k=gr.Slider(
minimum=0,
maximum=50,
value=0,
step=1,
interactive=True,
label="Top-k",
)
temperature=gr.Slider(
minimum=0.1,
maximum=5.0,
value=0.8,
step=0.1,
interactive=True,
label="Temperature",
)

user_text.submit(
run_generation,
[user_text,top_p,temperature,top_k,max_new_tokens,performance],
[model_output,performance],
)
button_submit.click(select_device,[device,user_text],[user_text])
button_submit.click(
run_generation,
[user_text,top_p,temperature,top_k,max_new_tokens,performance],
[model_output,performance],
)
button_clear.click(
reset_textbox,
[user_text,model_output,performance],
[user_text,model_output,performance],
)
device.change(select_device,[device,user_text],[user_text])

if__name__=="__main__":
try:
demo.queue().launch(debug=False,height=800)
exceptException:
demo.queue().launch(debug=False,share=True,height=800)

#Ifyouarelaunchingremotely,specifyserver_nameandserver_port
#EXAMPLE:`demo.launch(server_name='yourservername',server_port='serverportinint')`
#TolearnmorepleaserefertotheGradiodocs:https://gradio.app/docs/

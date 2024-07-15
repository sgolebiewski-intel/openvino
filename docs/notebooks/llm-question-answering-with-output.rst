LLMInstruction-followingpipelinewithOpenVINO
================================================

LLMstandsfor“LargeLanguageModel,”whichreferstoatypeof
artificialintelligencemodelthatisdesignedtounderstandand
generatehuman-liketextbasedontheinputitreceives.LLMsare
trainedonlargedatasetsoftexttolearnpatterns,grammar,and
semanticrelationships,allowingthemtogeneratecoherentand
contextuallyrelevantresponses.OnecorecapabilityofLargeLanguage
Models(LLMs)istofollownaturallanguageinstructions.
Instruction-followingmodelsarecapableofgeneratingtextinresponse
topromptsandareoftenusedfortaskslikewritingassistance,
chatbots,andcontentgeneration.

Inthistutorial,weconsiderhowtorunaninstruction-followingtext
generationpipelineusingpopularLLMsandOpenVINO.Wewilluse
pre-trainedmodelsfromthe`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__
library.The`HuggingFaceOptimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__library
convertsthemodelstoOpenVINO™IRformat.Tosimplifytheuser
experience,wewilluse`OpenVINOGenerate
API<https://github.com/openvinotoolkit/openvino.genai>`__for
generationofinstruction-followinginferencepipeline.

Thetutorialconsistsofthefollowingsteps:

-Installprerequisites
-Downloadandconvertthemodelfromapublicsourceusingthe
`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-CompressmodelweightstoINT8andINT4with`OpenVINO
NNCF<https://github.com/openvinotoolkit/nncf>`__
-Createaninstruction-followinginferencepipelinewith`Generate
API<https://github.com/openvinotoolkit/openvino.genai>`__
-Runinstruction-followingpipeline

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Selectmodelforinference<#select-model-for-inference>`__
-`DownloadandconvertmodeltoOpenVINOIRviaOptimumIntel
CLI<#download-and-convert-model-to-openvino-ir-via-optimum-intel-cli>`__
-`Compressmodelweights<#compress-model-weights>`__

-`WeightsCompressionusingOptimumIntel
CLI<#weights-compression-using-optimum-intel-cli>`__
-`WeightsCompressionusing
NNCF<#weights-compression-using-nncf>`__

-`Selectdeviceforinferenceandmodel
variant<#select-device-for-inference-and-model-variant>`__
-`Createaninstruction-followinginference
pipeline<#create-an-instruction-following-inference-pipeline>`__

-`Setupimports<#setup-imports>`__
-`Preparetextstreamertogetresults
runtime<#prepare-text-streamer-to-get-results-runtime>`__
-`Maingenerationfunction<#main-generation-function>`__
-`Helpersforapplication<#helpers-for-application>`__

-`Runinstruction-following
pipeline<#run-instruction-following-pipeline>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipuninstall-q-yoptimumoptimum-intel
%pipinstall-q"openvino-genai>=2024.2"
%pipinstall-q"torch>=2.1""nncf>=2.7""transformers>=4.40.0"onnx"optimum>=1.16.1""accelerate""datasets>=2.14.6""gradio>=4.19""git+https://github.com/huggingface/optimum-intel.git"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

WARNING:Skippingoptimumasitisnotinstalled.
WARNING:Skippingoptimum-intelasitisnotinstalled.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


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
-**phi-2**-Phi-2isaTransformerwith2.7billionparameters.It
wastrainedusingthesamedatasourcesas
`Phi-1.5<https://huggingface.co/microsoft/phi-1_5>`__,augmented
withanewdatasourcethatconsistsofvariousNLPsynthetictexts
andfilteredwebsites(forsafetyandeducationalvalue).When
assessedagainstbenchmarkstestingcommonsense,language
understanding,andlogicalreasoning,Phi-2showcasedanearly
state-of-the-artperformanceamongmodelswithlessthan13billion
parameters.Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/microsoft/phi-2#limitations-of-phi-2>`__.
-**dolly-v2-3b**-Dolly2.0isaninstruction-followinglarge
languagemodeltrainedontheDatabricksmachine-learningplatform
thatislicensedforcommercialuse.Itisbasedon
`Pythia<https://github.com/EleutherAI/pythia>`__andistrainedon
~15kinstruction/responsefine-tuningrecordsgeneratedbyDatabricks
employeesinvariouscapabilitydomains,includingbrainstorming,
classification,closedQA,generation,informationextraction,open
QA,andsummarization.Dolly2.0worksbyprocessingnaturallanguage
instructionsandgeneratingresponsesthatfollowthegiven
instructions.Itcanbeusedforawiderangeofapplications,
includingclosedquestion-answering,summarization,andgeneration.
Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/databricks/dolly-v2-3b>`__.
-**red-pajama-3b-instruct**-A2.8Bparameterpre-trainedlanguage
modelbasedonGPT-NEOXarchitecture.Themodelwasfine-tunedfor
few-shotapplicationsonthedataof
`GPT-JT<https://huggingface.co/togethercomputer/GPT-JT-6B-v1>`__,
withexclusionoftasksthatoverlapwiththeHELMcore
scenarios.Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__.
-**mistral-7b**-TheMistral-7B-v0.2LargeLanguageModel(LLM)isa
pretrainedgenerativetextmodelwith7billionparameters.Youcan
findmoredetailsaboutmodelinthe`model
card<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__,
`paper<https://arxiv.org/abs/2310.06825>`__and`releaseblog
post<https://mistral.ai/news/announcing-mistral-7b/>`__.
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

..code::ipython3

frompathlibimportPath
importrequests

#Fetch`notebook_utils`module
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

ifnotPath("./config.py").exists():
download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-question-answering/config.py")
fromconfigimportSUPPORTED_LLM_MODELS
importipywidgetsaswidgets

..code::ipython3

model_ids=list(SUPPORTED_LLM_MODELS)

model_id=widgets.Dropdown(
options=model_ids,
value=model_ids[1],
description="Model:",
disabled=False,
)

model_id




..parsed-literal::

Dropdown(description='Model:',index=1,options=('tiny-llama-1b','phi-2','dolly-v2-3b','red-pajama-instruct…



..code::ipython3

model_configuration=SUPPORTED_LLM_MODELS[model_id.value]
print(f"Selectedmodel{model_id.value}")


..parsed-literal::

Selectedmodeldolly-v2-3b


DownloadandconvertmodeltoOpenVINOIRviaOptimumIntelCLI
---------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Listedmodelareavailablefordownloadingviathe`HuggingFace
hub<https://huggingface.co/models>`__.Wewilluseoptimum-cli
interfaceforexportingitintoOpenVINOIntermediateRepresentation
(IR)format.

OptimumCLIinterfaceforconvertingmodelssupportsexporttoOpenVINO
(supportedstartingoptimum-intel1.12version).Generalcommandformat:

..code::bash

optimum-cliexportopenvino--model<model_id_or_path>--task<task><output_dir>

where``--model``argumentismodelidfromHuggingFaceHuborlocal
directorywithmodel(savedusing``.save_pretrained``method),
``--task``isoneof`supported
task<https://huggingface.co/docs/optimum/exporters/task_manager>`__
thatexportedmodelshouldsolve.ForLLMsitwillbe
``text-generation-with-past``.Ifmodelinitializationrequirestouse
remotecode,``--trust-remote-code``flagadditionallyshouldbepassed.
Fulllistofsupportedargumentsavailablevia``--help``Formore
detailsandexamplesofusage,pleasecheck`optimum
documentation<https://huggingface.co/docs/optimum/intel/inference#export>`__.

Compressmodelweights
----------------------

`backtotop⬆️<#table-of-contents>`__

TheWeightsCompressionalgorithmisaimedatcompressingtheweightsof
themodelsandcanbeusedtooptimizethemodelfootprintand
performanceoflargemodelswherethesizeofweightsisrelatively
largerthanthesizeofactivations,forexample,LargeLanguageModels
(LLM).ComparedtoINT8compression,INT4compressionimproves
performanceevenmorebutintroducesaminordropinpredictionquality.

WeightsCompressionusingOptimumIntelCLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OptimumIntelsupportsweightcompressionviaNNCFoutofthebox.For
8-bitcompressionwepass``--weight-formatint8``to``optimum-cli``
commandline.For4bitcompressionweprovide``--weight-formatint4``
andsomeotheroptionscontainingnumberofbitsandothercompression
parameters.Anexampleofthisapproachusageyoucanfindin
`llm-chatbotnotebook<../llm-chatbot>`__

WeightsCompressionusingNNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

YoualsocanperformweightscompressionforOpenVINOmodelsusingNNCF
directly.``nncf.compress_weights``functionacceptstheOpenVINOmodel
instanceandcompressesitsweightsforLinearandEmbeddinglayers.We
willconsiderthisvariantinthisnotebookforbothint4andint8
compression.

**Note**:ThistutorialinvolvesconversionmodelforFP16and
INT4/INT8weightscompressionscenarios.Itmaybememoryand
time-consuminginthefirstrun.Youcanmanuallycontrolthe
compressionprecisionbelow.**Note**:Theremaybenospeedupfor
INT4/INT8compressedmodelsondGPU

..code::ipython3

fromIPython.displayimportdisplay,Markdown

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
importlogging
importopenvinoasov
importnncf

nncf.set_log_level(logging.ERROR)

pt_model_id=model_configuration["model_id"]
fp16_model_dir=Path(model_id.value)/"FP16"
int8_model_dir=Path(model_id.value)/"INT8_compressed_weights"
int4_model_dir=Path(model_id.value)/"INT4_compressed_weights"

core=ov.Core()


defconvert_to_fp16():
if(fp16_model_dir/"openvino_model.xml").exists():
return
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatfp16".format(pt_model_id)
export_command=export_command_base+""+str(fp16_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int8():
if(int8_model_dir/"openvino_model.xml").exists():
return
int8_model_dir.mkdir(parents=True,exist_ok=True)
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint8".format(pt_model_id)
export_command=export_command_base+""+str(int8_model_dir)
display(Markdown("**Exportcommand:**"))
display(Markdown(f"`{export_command}`"))
!$export_command


defconvert_to_int4():
compression_configs={
"mistral-7b":{
"sym":True,
"group_size":64,
"ratio":0.6,
},
"red-pajama-3b-instruct":{
"sym":False,
"group_size":128,
"ratio":0.5,
},
"dolly-v2-3b":{"sym":False,"group_size":32,"ratio":0.5},
"llama-3-8b-instruct":{"sym":True,"group_size":128,"ratio":1.0},
"default":{
"sym":False,
"group_size":128,
"ratio":0.8,
},
}

model_compression_params=compression_configs.get(model_id.value,compression_configs["default"])
if(int4_model_dir/"openvino_model.xml").exists():
return
export_command_base="optimum-cliexportopenvino--model{}--tasktext-generation-with-past--weight-formatint4".format(pt_model_id)
int4_compression_args="--group-size{}--ratio{}".format(model_compression_params["group_size"],model_compression_params["ratio"])
ifmodel_compression_params["sym"]:
int4_compression_args+="--sym"
export_command_base+=int4_compression_args
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


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


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

SizeofFP16modelis5297.21MB
SizeofmodelwithINT8compressedweightsis2656.29MB
CompressionrateforINT8model:1.994
SizeofmodelwithINT4compressedweightsis2154.54MB
CompressionrateforINT4model:2.459


Selectdeviceforinferenceandmodelvariant
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

**Note**:TheremaybenospeedupforINT4/INT8compressedmodelson
dGPU.

..code::ipython3

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

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



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

Dropdown(description='Modeltorun:',options=('INT4','INT8','FP16'),value='INT4')



..code::ipython3

fromtransformersimportAutoTokenizer
fromopenvino_tokenizersimportconvert_tokenizer

ifmodel_to_run.value=="INT4":
model_dir=int4_model_dir
elifmodel_to_run.value=="INT8":
model_dir=int8_model_dir
else:
model_dir=fp16_model_dir
print(f"Loadingmodelfrom{model_dir}")

#optionallyconverttokenizerifusedcachedmodelwithoutit
ifnot(model_dir/"openvino_tokenizer.xml").exists()ornot(model_dir/"openvino_detokenizer.xml").exists():
hf_tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
ov_tokenizer,ov_detokenizer=convert_tokenizer(hf_tokenizer,with_detokenizer=True)
ov.save_model(ov_tokenizer,model_dir/"openvino_tokenizer.xml")
ov.save_model(ov_tokenizer,model_dir/"openvino_detokenizer.xml")


..parsed-literal::

Loadingmodelfromdolly-v2-3b/INT8_compressed_weights


Createaninstruction-followinginferencepipeline
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

The``run_generation``functionacceptsuser-providedtextinput,
tokenizesit,andrunsthegenerationprocess.Textgenerationisan
iterativeprocess,whereeachnexttokendependsonpreviouslygenerated
untilamaximumnumberoftokensorstopgenerationconditionisnot
reached.

Thediagrambelowillustrateshowtheinstruction-followingpipeline
works

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/e881f4a4-fcc8-427a-afe1-7dd80aebd66e
:alt:generationpipeline)

generationpipeline)

Ascanbeseen,onthefirstiteration,theuserprovidedinstructions.
Instructionsisconvertedtotokenidsusingatokenizer,thenprepared
inputprovidedtothemodel.Themodelgeneratesprobabilitiesforall
tokensinlogitsformat.Thewaythenexttokenwillbeselectedover
predictedprobabilitiesisdrivenbytheselecteddecodingmethodology.
Youcanfindmoreinformationaboutthemostpopulardecodingmethodsin
this`blog<https://huggingface.co/blog/how-to-generate>`__.

Tosimplifyuserexperiencewewilluse`OpenVINOGenerate
API<https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
Firstlywewillcreatepipelinewith``LLMPipeline``.``LLMPipeline``is
themainobjectusedfordecoding.Youcanconstructitstraightaway
fromthefolderwiththeconvertedmodel.Itwillautomaticallyloadthe
``mainmodel``,``tokenizer``,``detokenizer``anddefault
``generationconfiguration``.Afterthatwewillconfigureparameters
fordecoding.Wecangetdefaultconfigwith
``get_generation_config()``,setupparametersandapplytheupdated
versionwith``set_generation_config(config)``orputconfigdirectlyto
``generate()``.It’salsopossibletospecifytheneededoptionsjustas
inputsinthe``generate()``method,asshownbelow.Thenwejustrun
``generate``methodandgettheoutputintextformat.Wedonotneedto
encodeinputpromptaccordingtomodelexpectedtemplateorwrite
post-processingcodeforlogitsdecoder,itwillbedoneeasilywith
LLMPipeline.

Toobtainintermediategenerationresultswithoutwaitinguntilwhen
generationisfinished,wewillwriteclass-iteratorbasedon
``StreamerBase``classof``openvino_genai``.

..code::ipython3

fromopenvino_genaiimportLLMPipeline

pipe=LLMPipeline(model_dir.as_posix(),device.value)
print(pipe.generate("TheSunisyellowbacause",temperature=1.2,top_k=4,do_sample=True,max_new_tokens=150))


..parsed-literal::

ofthepresenceofchlorophyll
initsleaves.Chlorophyllabsorbsall
visiblesunlightandthiscausesitto
turnfromagreentoyellowcolour.
TheSunisyellowbacauseofthepresenceofchlorophyllinitsleaves.Chlorophyllabsorbsall
visiblesunlightandthiscausesitto
turnfromagreentoyellowcolour.
TheyellowcolouroftheSunisthe
colourweperceiveasthecolourofthe
sun.Italsocausesustoperceivethe
sunasyellow.Thispropertyiscalled
theyellowcolourationoftheSunandit
iscausedbythepresenceofchlorophyll
intheleavesofplants.
Chlorophyllisalsoresponsibleforthegreencolourofplants


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
importnumpyasnp
fromopenvino_genaiimportStreamerBase
fromqueueimportQueue
importre

Preparetextstreamertogetresultsruntime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Loadthe``detokenizer``,useittoconverttoken_idtostringoutput
format.Wewillcollectprint-readytextinaqueueandgivethetext
whenitisneeded.Itwillhelpestimateperformance.

..code::ipython3

core=ov.Core()

detokinizer_dir=Path(model_dir,"openvino_detokenizer.xml")


classTextIteratorStreamer(StreamerBase):
def__init__(self,tokenizer):
super().__init__()
self.tokenizer=tokenizer
self.compiled_detokenizer=core.compile_model(detokinizer_dir.as_posix())
self.text_queue=Queue()
self.stop_signal=None

def__iter__(self):
returnself

def__next__(self):
value=self.text_queue.get()
ifvalue==self.stop_signal:
raiseStopIteration()
else:
returnvalue

defput(self,token_id):
openvino_output=self.compiled_detokenizer(np.array([[token_id]],dtype=int))
text=str(openvino_output["string_output"][0])
#removelabels/specialsymbols
text=text.lstrip("!")
text=re.sub("<.*>","",text)
self.text_queue.put(text)

defend(self):
self.text_queue.put(self.stop_signal)

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

#setupconfigfordecodingstage
config=pipe.get_generation_config()
config.temperature=temperature
iftop_k>0:
config.top_k=top_k
config.top_p=top_p
config.do_sample=True
config.max_new_tokens=max_new_tokens

#Startgenerationonaseparatethread,sothatwedon'tblocktheUI.Thetextispulledfromthestreamer
#inthemainthread.
streamer=TextIteratorStreamer(pipe.get_tokenizer())
t=Thread(target=pipe.generate,args=(user_text,config,streamer))
t.start()

model_output=""
per_token_time=[]
num_tokens=0
start=perf_counter()
fornew_textinstreamer:
current_time=perf_counter()-start
model_output+=new_text
perf_text,num_tokens=estimate_latency(current_time,perf_text,per_token_time,num_tokens)
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
per_token_time:List[float],
num_tokens:int,
):
"""
Helperfunctionforperformanceestimation

Parameters:
current_time(float):Thissteptimeinseconds.
current_perf_text(str):CurrentcontentofperformanceUIfield.
per_token_time(List[float]):historyofperformancefromprevioussteps.
num_tokens(int):Totalnumberofgeneratedtokens.

Returns:
updateforperformancetextfield
updateforatotalnumberoftokens
"""
num_tokens+=1
per_token_time.append(1/current_time)
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

examples=[
"Givemearecipeforpizzawithpineapple",
"WritemeatweetaboutthenewOpenVINOrelease",
"ExplainthedifferencebetweenCPUandGPU",
"Givefiveideasforagreatweekendwithfamily",
"DoAndroidsdreamofElectricsheep?",
"WhoisDolly?",
"Pleasegivemeadviceonhowtowriteresume?",
"Name3advantagestobeingacat",
"WriteinstructionsonhowtobecomeagoodAIengineer",
"Writealovelettertomybestfriend",
]


withgr.Blocks()asdemo:
gr.Markdown(
"#QuestionAnsweringwith"+model_id.value+"andOpenVINO.\n"
"Provideinstructionwhichdescribesataskbeloworselectamongpredefinedexamplesandmodelwritesresponsethatperformsrequestedtask."
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
if__name__=="__main__":
demo.queue()
try:
demo.launch(height=800)
exceptException:
demo.launch(share=True,height=800)

#Ifyouarelaunchingremotely,specifyserver_nameandserver_port
#EXAMPLE:`demo.launch(server_name='yourservername',server_port='serverportinint')`
#TolearnmorepleaserefertotheGradiodocs:https://gradio.app/docs/

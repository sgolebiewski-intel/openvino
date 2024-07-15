TypoDetectorwithOpenVINO™
============================

TypodetectioninAIisaprocessofidentifyingandcorrecting
typographicalerrorsintextdatausingmachinelearningalgorithms.The
goaloftypodetectionistoimprovetheaccuracy,readability,and
usabilityoftextbyidentifyingandindicatingmistakesmadeduringthe
writingprocess.Todetecttypos,AI-basedtypodetectorsusevarious
techniques,suchasnaturallanguageprocessing(NLP),machinelearning
(ML),anddeeplearning(DL).

Atypodetectortakesasentenceasaninputandidentifyall
typographicalerrorssuchasmisspellingsandhomophoneerrors.

Thistutorialprovideshowtousethe`Typo
Detector<https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en>`__
fromthe`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__library
intheOpenVINOenvironmenttoperformtheabovetask.

Themodeldetectstyposinagiventextwithahighaccuracy,
performancesofwhicharelistedbelow,-Precisionscoreof0.9923-
Recallscoreof0.9859-f1-scoreof0.9891

`Sourceforabove
metrics<https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en>`__

Thesemetricsindicatethatthemodelcancorrectlyidentifyahigh
proportionofbothcorrectandincorrecttext,minimizingbothfalse
positivesandfalsenegatives.

Themodelhasbeenpretrainedonthe
`NeuSpell<https://github.com/neuspell/neuspell>`__dataset.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Methods<#methods>`__

-`1.UsingtheHuggingFaceOptimum
library<#1--using-the-hugging-face-optimum-library>`__

-`2.ConvertingthemodeltoOpenVINO
IR<#2--converting-the-model-to-openvino-ir>`__

-`Selectinferencedevice<#select-inference-device>`__
-`1.HuggingFaceOptimumIntel
library<#1--hugging-face-optimum-intel-library>`__

-`Loadthemodel<#load-the-model>`__
-`Loadthetokenizer<#load-the-tokenizer>`__

-`2.ConvertingthemodeltoOpenVINO
IR<#2--converting-the-model-to-openvino-ir>`__

-`LoadthePytorchmodel<#load-the-pytorch-model>`__
-`ConvertingtoOpenVINOIR<#converting-to-openvino-ir>`__
-`Inference<#inference>`__

-`HelperFunctions<#helper-functions>`__

..code::ipython3

%pipinstall-q"diffusers>=0.17.1""openvino>=2023.1.0""nncf>=2.5.0""gradio>=4.19""onnx>=1.11.0""transformers>=4.39.0""torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtransformersimport(
AutoConfig,
AutoTokenizer,
AutoModelForTokenClassification,
pipeline,
)
frompathlibimportPath
importnumpyasnp
importre
fromtypingimportList,Dict
importtime


..parsed-literal::

2024-07-1304:12:55.289699:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:12:55.325068:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:12:55.925329:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


Methods
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thenotebookprovidestwomethodstoruntheinferenceoftypodetector
withOpenVINOruntime,sothatyoucanexperiencebothcallingtheAPI
ofOptimumwithOpenVINORuntimeincluded,andloadingmodelsinother
frameworks,convertingthemtoOpenVINOIRformat,andrunninginference
withOpenVINORuntime.

1.Usingthe`HuggingFaceOptimum<https://huggingface.co/docs/optimum/index>`__library
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

TheHuggingFaceOptimumAPIisahigh-levelAPIthatallowsusto
convertmodelsfromtheHuggingFaceTransformerslibrarytothe
OpenVINO™IRformat.CompiledmodelsinOpenVINOIRformatcanbeloaded
usingOptimum.Optimumallowstheuseofoptimizationontargeted
hardware.

2.ConvertingthemodeltoOpenVINOIR
''''''''''''''''''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

ThePytorchmodelisconvertedto`OpenVINOIR
format<https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__.
Thismethodprovidesmuchmoreinsighttohowtosetupapipelinefrom
modelloadingtomodelconverting,compilingandrunninginferencewith
OpenVINO,sothatyoucouldconvenientlyuseOpenVINOtooptimizeand
accelerateinferenceforotherdeep-learningmodels.Theoptimizationof
targetedhardwareisalsousedhere.

Thefollowingtablesummarizesthemajordifferencesbetweenthetwo
methods

+-----------------------------------+----------------------------------+
|Method1|Method2|
+===================================+==================================+
|LoadmodelsfromOptimum,an|Loadmodelfromtransformers|
|extensionoftransformers||
+-----------------------------------+----------------------------------+
|LoadthemodelinOpenVINOIR|ConverttoOpenVINOIR|
|formatonthefly||
+-----------------------------------+----------------------------------+
|Loadthecompiledmodelby|CompiletheOpenVINOIRandrun|
|default|inferencewithOpenVINORuntime|
+-----------------------------------+----------------------------------+
|Pipelineiscreatedtorun|Manuallyruninference.|
|inferencewithOpenVINORuntime||
+-----------------------------------+----------------------------------+

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



1.HuggingFaceOptimumIntellibrary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forthismethod,weneedtoinstallthe
``HuggingFaceOptimumIntellibrary``acceleratedbyOpenVINO
integration.

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.TheOptimumInferencemodelsareAPIcompatiblewithHugging
FaceTransformersmodels.Thismeansweneedjustreplace
``AutoModelForXxx``classwiththecorresponding``OVModelForXxx``
class.

Importrequiredmodelclass

..code::ipython3

fromoptimum.intel.openvinoimportOVModelForTokenClassification


..parsed-literal::

TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.


Loadthemodel
''''''''''''''

`backtotop⬆️<#table-of-contents>`__

Fromthe``OVModelForTokenCLassification``classwewillimportthe
relevantpre-trainedmodel.ToloadaTransformersmodelandconvertit
totheOpenVINOformaton-the-fly,weset``export=True``whenloading
yourmodel.

..code::ipython3

#Thepretrainedmodelweareusing
model_id="m3hrdadfi/typo-detector-distilbert-en"

model_dir=Path("optimum_model")

#Savethemodeltothepathifnotexisting
ifmodel_dir.exists():
model=OVModelForTokenClassification.from_pretrained(model_dir,device=device.value)
else:
model=OVModelForTokenClassification.from_pretrained(model_id,export=True,device=device.value)
model.save_pretrained(model_dir)


..parsed-literal::

Frameworknotspecified.Usingpttoexportthemodel.
UsingframeworkPyTorch:2.2.2+cpu


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:86:TracerWarning:torch.tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
op1=operator(*args,**kwargs)


..parsed-literal::

['input_ids','attention_mask']


..parsed-literal::

CompilingthemodeltoAUTO...


Loadthetokenizer
''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

TextPreprocessingcleansthetext-basedinputdatasoitcanbefed
intothemodel.Tokenizationsplitsparagraphsandsentencesinto
smallerunitsthatcanbemoreeasilyassignedmeaning.Itinvolves
cleaningthedataandassigningtokensorIDstothewords,sotheyare
representedinavectorspacewheresimilarwordshavesimilarvectors.
Thishelpsthemodelunderstandthecontextofasentence.We’remaking
useofan
`AutoTokenizer<https://huggingface.co/docs/transformers/main_classes/tokenizer>`__
fromHuggingFace,whichisessentiallyapretrainedtokenizer.

..code::ipython3

tokenizer=AutoTokenizer.from_pretrained(model_id)

Thenweusetheinferencepipelinefor``token-classification``task.
YoucanfindmoreinformationaboutusageHuggingFaceinference
pipelinesinthis
`tutorial<https://huggingface.co/docs/transformers/pipeline_tutorial>`__

..code::ipython3

nlp=pipeline(
"token-classification",
model=model,
tokenizer=tokenizer,
aggregation_strategy="average",
)

Functiontofindtyposinasentenceandwritethemtotheterminal

..code::ipython3

defshow_typos(sentence:str):
"""
Detecttyposfromthegivensentence.
Writesboththeoriginalinputandtypo-taggedversiontotheterminal.

Arguments:
sentence--Sentencetobeevaluated(string)
"""

typos=[sentence[r["start"]:r["end"]]forrinnlp(sentence)]

detected=sentence
fortypointypos:
detected=detected.replace(typo,f"<i>{typo}</i>")

print("[Input]:",sentence)
print("[Detected]:",detected)
print("-"*130)

Let’srunademousingtheHuggingFaceOptimumAPI.

..code::ipython3

sentences=[
"HehadalsostgruggledwithaddictionduringhistimeinCongress.",
"ThereviewthoroughlaassessedallaspectsofJLENSSuRandCPGesignmaturitandconfidence.",
"Lettermaalsoapologizedtwohisstaffforthesatyation.",
"VincentJayhadearlierwonFrance'sfirstgoldingthe10kmbiathlonsprint.",
"Itislefttothedirectorstofigureouthpwtobringthestryacrosstotyeaudience.",
"Iwnettotheparkyestredaytoplayfoorballwithmyfiends,butitstatredtorainveryhevailyandwehadtostop.",
"Myfaoriterestuarantservsthebestspahgettiinthetown,buttheyarealwayssobuzythatyouhavetomakearesrvationinadvnace.",
"IwasgoigtowatchamvoieonNetflxlastnight,butthestramingwassoslowthatIdecidedtocancledmysubscrpition.",
"MyfreindandIwentcampignintheforestlastweekendandsawabeutifulsunstthatwassoamzingittookourbrethaway.",
"Ihavebeenstuyingformymathexamallweek,butI'mstilnotveryconfidetthatIwillpassit,becausetherearesomanyformualstoremeber.",
]

start=time.time()

forsentenceinsentences:
show_typos(sentence)

print(f"Timeelapsed:{time.time()-start}")


..parsed-literal::

[Input]:HehadalsostgruggledwithaddictionduringhistimeinCongress.
[Detected]:Hehadalso<i>stgruggled</i>withaddictionduringhistimeinCongress.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:ThereviewthoroughlaassessedallaspectsofJLENSSuRandCPGesignmaturitandconfidence.
[Detected]:Thereview<i>thoroughla</i>assessedallaspectsofJLENSSuRandCPG<i>esignmaturit</i>andconfidence.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Lettermaalsoapologizedtwohisstaffforthesatyation.
[Detected]:<i>Letterma</i>alsoapologized<i>two</i>hisstaffforthe<i>satyation</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:VincentJayhadearlierwonFrance'sfirstgoldingthe10kmbiathlonsprint.
[Detected]:VincentJayhadearlierwonFrance'sfirstgoldin<i>gthe</i>10kmbiathlonsprint.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Itislefttothedirectorstofigureouthpwtobringthestryacrosstotyeaudience.
[Detected]:Itislefttothedirectorstofigureout<i>hpw</i>tobringthe<i>stry</i>acrossto<i>tye</i>audience.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Iwnettotheparkyestredaytoplayfoorballwithmyfiends,butitstatredtorainveryhevailyandwehadtostop.
[Detected]:I<i>wnet</i>tothepark<i>yestreday</i>toplay<i>foorball</i>withmy<i>fiends</i>,butit<i>statred</i>torainvery<i>hevaily</i>andwehadtostop.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Myfaoriterestuarantservsthebestspahgettiinthetown,buttheyarealwayssobuzythatyouhavetomakearesrvationinadvnace.
[Detected]:My<i>faoriterestuarantservs</i>thebest<i>spahgetti</i>inthetown,buttheyarealwaysso<i>buzy</i>thatyouhavetomakea<i>resrvation</i>in<i>advnace</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:IwasgoigtowatchamvoieonNetflxlastnight,butthestramingwassoslowthatIdecidedtocancledmysubscrpition.
[Detected]:Iwas<i>goig</i>towatcha<i>mvoie</i>on<i>Netflx</i>lastnight,butthe<i>straming</i>wassoslowthatIdecidedto<i>cancled</i>my<i>subscrpition</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:MyfreindandIwentcampignintheforestlastweekendandsawabeutifulsunstthatwassoamzingittookourbrethaway.
[Detected]:My<i>freind</i>andIwent<i>campign</i>intheforestlastweekendandsawa<i>beutifulsunst</i>thatwasso<i>amzing</i>ittookour<i>breth</i>away.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Ihavebeenstuyingformymathexamallweek,butI'mstilnotveryconfidetthatIwillpassit,becausetherearesomanyformualstoremeber.
[Detected]:Ihavebeen<i>stuying</i>formymathexamallweek,butI'm<i>stil</i>notvery<i>confidet</i>thatIwillpassit,becausetherearesomanyformualsto<i>remeber</i>.
----------------------------------------------------------------------------------------------------------------------------------
Timeelapsed:0.15727782249450684


2.ConvertingthemodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LoadthePytorchmodel
''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

Usethe``AutoModelForTokenClassification``classtoloadthepretrained
pytorchmodel.

..code::ipython3

model_id="m3hrdadfi/typo-detector-distilbert-en"
model_dir=Path("pytorch_model")

tokenizer=AutoTokenizer.from_pretrained(model_id)
config=AutoConfig.from_pretrained(model_id)

#Savethemodeltothepathifnotexisting
ifmodel_dir.exists():
model=AutoModelForTokenClassification.from_pretrained(model_dir)
else:
model=AutoModelForTokenClassification.from_pretrained(model_id,config=config)
model.save_pretrained(model_dir)

ConvertingtoOpenVINOIR
'''''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_model_path=Path(model_dir)/"typo_detect.xml"

dummy_model_input=tokenizer("Thisisasample",return_tensors="pt")
ov_model=ov.convert_model(model,example_input=dict(dummy_model_input))
ov.save_model(ov_model,ov_model_path)


..parsed-literal::

['input_ids','attention_mask']


Inference
'''''''''

`backtotop⬆️<#table-of-contents>`__

OpenVINO™RuntimePythonAPIisusedtocompilethemodelinOpenVINOIR
format.TheCoreclassfromthe``openvino``moduleisimportedfirst.
ThisclassprovidesaccesstotheOpenVINORuntimeAPI.The``core``
object,whichisaninstanceofthe``Core``class,representstheAPI
anditisusedtocompilethemodel.Theoutputlayerisextractedfrom
thecompiledmodelasitisneededforinference.

..code::ipython3

compiled_model=core.compile_model(ov_model,device.value)
output_layer=compiled_model.output(0)

HelperFunctions
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

deftoken_to_words(tokens:List[str])->Dict[str,int]:
"""
Mapsthelistoftokenstowordsintheoriginaltext.
Builtonthefeaturethattokensstartingwith'##'isattachedtotheprevioustokenastokensderivedfromthesameword.

Arguments:
tokens--Listoftokens

Returns:
map_to_words--Dictionarymappingtokenstowordsinoriginaltext
"""

word_count=-1
map_to_words={}
fortokenintokens:
iftoken.startswith("##"):
map_to_words[token]=word_count
continue
word_count+=1
map_to_words[token]=word_count
returnmap_to_words

..code::ipython3

definfer(input_text:str)->Dict[np.ndarray,np.ndarray]:
"""
Creatingagenericinferencefunctiontoreadtheinputandinfertheresult

Arguments:
input_text--Thetexttobeinfered(String)

Returns:
result--Resultinglistfrominference
"""

tokens=tokenizer(
input_text,
return_tensors="np",
)
inputs=dict(tokens)
result=compiled_model(inputs)[output_layer]
returnresult

..code::ipython3

defget_typo_indexes(
result:Dict[np.ndarray,np.ndarray],
map_to_words:Dict[str,int],
tokens:List[str],
)->List[int]:
"""
Givenresultsfromtheinferenceandtokens-map-to-words,identifiestheindexesofthewordswithtypos.

Arguments:
result--Resultfrominference(tensor)
map_to_words--Dictionarymappingtokenstowords(Dictionary)

Results:
wrong_words--Listofindexesofwordswithtypos
"""

wrong_words=[]
c=0
result_list=result[0][1:-1]
foriinresult_list:
prob=np.argmax(i)
ifprob==1:
ifmap_to_words[tokens[c]]notinwrong_words:
wrong_words.append(map_to_words[tokens[c]])
c+=1
returnwrong_words

..code::ipython3

defsentence_split(sentence:str)->List[str]:
"""
Splitthesentenceintowordsandcharacters

Arguments:
sentence-Sentencetobesplit(string)

Returns:
splitted--Listofwordsandcharacters
"""

splitted=re.split("([',.])",sentence)
splitted=[xforxinsplittedifx!=""andx!=""]
returnsplitted

..code::ipython3

defshow_typos(sentence:str):
"""
Detecttyposfromthegivensentence.
Writesboththeoriginalinputandtypo-taggedversiontotheterminal.

Arguments:
sentence--Sentencetobeevaluated(string)
"""

tokens=tokenizer.tokenize(sentence)
map_to_words=token_to_words(tokens)
result=infer(sentence)
typo_indexes=get_typo_indexes(result,map_to_words,tokens)

sentence_words=sentence_split(sentence)

typos=[sentence_words[i]foriintypo_indexes]

detected=sentence
fortypointypos:
detected=detected.replace(typo,f"<i>{typo}</i>")

print("[Input]:",sentence)
print("[Detected]:",detected)
print("-"*130)

Let’srunademousingtheconvertedOpenVINOIRmodel.

..code::ipython3

sentences=[
"HehadalsostgruggledwithaddictionduringhistimeinCongress.",
"ThereviewthoroughlaassessedallaspectsofJLENSSuRandCPGesignmaturitandconfidence.",
"Lettermaalsoapologizedtwohisstaffforthesatyation.",
"VincentJayhadearlierwonFrance'sfirstgoldingthe10kmbiathlonsprint.",
"Itislefttothedirectorstofigureouthpwtobringthestryacrosstotyeaudience.",
"Iwnettotheparkyestredaytoplayfoorballwithmyfiends,butitstatredtorainveryhevailyandwehadtostop.",
"Myfaoriterestuarantservsthebestspahgettiinthetown,buttheyarealwayssobuzythatyouhavetomakearesrvationinadvnace.",
"IwasgoigtowatchamvoieonNetflxlastnight,butthestramingwassoslowthatIdecidedtocancledmysubscrpition.",
"MyfreindandIwentcampignintheforestlastweekendandsawabeutifulsunstthatwassoamzingittookourbrethaway.",
"Ihavebeenstuyingformymathexamallweek,butI'mstilnotveryconfidetthatIwillpassit,becausetherearesomanyformualstoremeber.",
]

start=time.time()

forsentenceinsentences:
show_typos(sentence)

print(f"Timeelapsed:{time.time()-start}")


..parsed-literal::

[Input]:HehadalsostgruggledwithaddictionduringhistimeinCongress.
[Detected]:Hehadalso<i>stgruggled</i>withaddictionduringhistimeinCongress.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:ThereviewthoroughlaassessedallaspectsofJLENSSuRandCPGesignmaturitandconfidence.
[Detected]:Thereview<i>thoroughla</i>assessedallaspectsofJLENSSuRandCPG<i>esign</i><i>maturit</i>andconfidence.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Lettermaalsoapologizedtwohisstaffforthesatyation.
[Detected]:<i>Letterma</i>alsoapologized<i>two</i>hisstaffforthe<i>satyation</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:VincentJayhadearlierwonFrance'sfirstgoldingthe10kmbiathlonsprint.
[Detected]:VincentJayhadearlierwonFrance'sfirstgoldin<i>gthe</i>10kmbiathlonsprint.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Itislefttothedirectorstofigureouthpwtobringthestryacrosstotyeaudience.
[Detected]:Itislefttothedirectorstofigureout<i>hpw</i>tobringthe<i>stry</i>acrossto<i>tye</i>audience.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Iwnettotheparkyestredaytoplayfoorballwithmyfiends,butitstatredtorainveryhevailyandwehadtostop.
[Detected]:I<i>wnet</i>tothepark<i>yestreday</i>toplay<i>foorball</i>withmy<i>fiends</i>,butit<i>statred</i>torainvery<i>hevaily</i>andwehadtostop.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Myfaoriterestuarantservsthebestspahgettiinthetown,buttheyarealwayssobuzythatyouhavetomakearesrvationinadvnace.
[Detected]:My<i>faorite</i><i>restuarant</i><i>servs</i>thebest<i>spahgetti</i>inthetown,buttheyarealwaysso<i>buzy</i>thatyouhavetomakea<i>resrvation</i>in<i>advnace</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:IwasgoigtowatchamvoieonNetflxlastnight,butthestramingwassoslowthatIdecidedtocancledmysubscrpition.
[Detected]:Iwas<i>goig</i>towatcha<i>mvoie</i>on<i>Netflx</i>lastnight,butthe<i>straming</i>wassoslowthatIdecidedto<i>cancled</i>my<i>subscrpition</i>.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:MyfreindandIwentcampignintheforestlastweekendandsawabeutifulsunstthatwassoamzingittookourbrethaway.
[Detected]:My<i>freind</i>andIwent<i>campign</i>intheforestlastweekendandsawa<i>beutiful</i><i>sunst</i>thatwasso<i>amzing</i>ittookour<i>breth</i>away.
----------------------------------------------------------------------------------------------------------------------------------
[Input]:Ihavebeenstuyingformymathexamallweek,butI'mstilnotveryconfidetthatIwillpassit,becausetherearesomanyformualstoremeber.
[Detected]:Ihavebeen<i>stuying</i>formymathexamallweek,butI'm<i>stil</i>notvery<i>confidet</i>thatIwillpassit,becausetherearesomanyformualsto<i>remeber</i>.
----------------------------------------------------------------------------------------------------------------------------------
Timeelapsed:0.10021185874938965


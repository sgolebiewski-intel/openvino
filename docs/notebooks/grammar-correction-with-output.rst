GrammaticalErrorCorrectionwithOpenVINO
==========================================

AI-basedauto-correctionproductsarebecomingincreasinglypopulardue
totheireaseofuse,editingspeed,andaffordability.Theseproducts
improvethequalityofwrittentextinemails,blogs,andchats.

GrammaticalErrorCorrection(GEC)isthetaskofcorrectingdifferent
typesoferrorsintextsuchasspelling,punctuation,grammaticaland
wordchoiceerrors.GECistypicallyformulatedasasentencecorrection
task.AGECsystemtakesapotentiallyerroneoussentenceasinputand
isexpectedtotransformitintoamorecorrectversion.Seetheexample
givenbelow:

=====================================================
Input(Erroneous)Output(Corrected)
=====================================================
Iliketoridesmybicycle.Iliketoridemybicycle.
=====================================================

Asshownintheimagebelow,differenttypesoferrorsinwritten
languagecanbecorrected.

..figure::https://cdn-images-1.medium.com/max/540/1*Voez5hEn5MU8Knde3fIZfw.png
:alt:error_types

error_types

Thistutorialshowshowtoperformgrammaticalerrorcorrectionusing
OpenVINO.Wewillusepre-trainedmodelsfromthe`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__
library.Tosimplifytheuserexperience,the`HuggingFace
Optimum<https://huggingface.co/docs/optimum>`__libraryisusedto
convertthemodelstoOpenVINO™IRformat.

Itconsistsofthefollowingsteps:

-Installprerequisites
-Downloadandconvertmodelsfromapublicsourceusingthe`OpenVINO
integrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-Createaninferencepipelineforgrammaticalerrorchecking
-Optimizegrammarcorrectionpipelinewith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__quantization
-Compareoriginalandoptimizedpipelinesfromperformanceand
accuracystandpoints

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Howdoesitwork?<#how-does-it-work>`__
-`Prerequisites<#prerequisites>`__
-`DownloadandConvertModels<#download-and-convert-models>`__

-`Selectinferencedevice<#select-inference-device>`__
-`GrammarChecker<#grammar-checker>`__
-`GrammarCorrector<#grammar-corrector>`__

-`PrepareDemoPipeline<#prepare-demo-pipeline>`__
-`Quantization<#quantization>`__

-`RunQuantization<#run-quantization>`__
-`Comparemodelsize,performanceand
accuracy<#compare-model-size-performance-and-accuracy>`__

-`Interactivedemo<#interactive-demo>`__

Howdoesitwork?
-----------------

`backtotop⬆️<#table-of-contents>`__

AGrammaticalErrorCorrectiontaskcanbethoughtofasa
sequence-to-sequencetaskwhereamodelistrainedtotakea
grammaticallyincorrectsentenceasinputandreturnagrammatically
correctsentenceasoutput.Wewillusethe
`FLAN-T5<https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis>`__
modelfinetunedonanexpandedversionofthe
`JFLEG<https://paperswithcode.com/dataset/jfleg>`__dataset.

TheversionofFLAN-T5releasedwiththe`ScalingInstruction-Finetuned
LanguageModels<https://arxiv.org/pdf/2210.11416.pdf>`__paperisan
enhancedversionof`T5<https://huggingface.co/t5-large>`__thathas
beenfinetunedonacombinationoftasks.Thepaperexploresinstruction
finetuningwithaparticularfocusonscalingthenumberoftasks,
scalingthemodelsize,andfinetuningonchain-of-thoughtdata.The
paperdiscoversthatoverallinstructionfinetuningisageneralmethod
thatimprovestheperformanceandusabilityofpre-trainedlanguage
models.

..figure::https://production-media.paperswithcode.com/methods/a04cb14e-e6b8-449e-9487-bc4262911d74.png
:alt:flan-t5_training

flan-t5_training

Formoredetailsaboutthemodel,pleasecheckout
`paper<https://arxiv.org/abs/2210.11416>`__,original
`repository<https://github.com/google-research/t5x>`__,andHugging
Face`modelcard<https://huggingface.co/google/flan-t5-large>`__

Additionally,toreducethenumberofsentencesrequiredtobe
processed,youcanperformgrammaticalcorrectnesschecking.Thistask
shouldbeconsideredasasimplebinarytextclassification,wherethe
modelgetsinputtextandpredictslabel1ifatextcontainsany
grammaticalerrorsand0ifitdoesnot.Youwillusethe
`roberta-base-CoLA<https://huggingface.co/textattack/roberta-base-CoLA>`__
model,theRoBERTaBasemodelfinetunedontheCoLAdataset.TheRoBERTa
modelwasproposedin`RoBERTa:ARobustlyOptimizedBERTPretraining
Approachpaper<https://arxiv.org/abs/1907.11692>`__.ItbuildsonBERT
andmodifieskeyhyperparameters,removingthenext-sentence
pre-trainingobjectiveandtrainingwithmuchlargermini-batchesand
learningrates.Additionaldetailsaboutthemodelcanbefoundina
`blog
post<https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/>`__
byMetaAIandinthe`HuggingFace
documentation<https://huggingface.co/docs/transformers/model_doc/roberta>`__

NowthatweknowmoreaboutFLAN-T5andRoBERTa,letusgetstarted.🚀

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

First,weneedtoinstallthe`HuggingFace
Optimum<https://huggingface.co/docs/transformers/index>`__library
acceleratedbyOpenVINOintegration.TheHuggingFaceOptimumAPIisa
high-levelAPIthatenablesustoconvertandquantizemodelsfromthe
HuggingFaceTransformerslibrarytotheOpenVINO™IRformat.Formore
details,refertothe`HuggingFaceOptimum
documentation<https://huggingface.co/docs/optimum/intel/inference>`__.

..code::ipython3

%pipinstall-q"torch>=2.1.0""git+https://github.com/huggingface/optimum-intel.git""openvino>=2024.0.0"onnxtqdm"gradio>=4.19""transformers>=4.33.0"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"nncf>=2.9.0"datasetsjiwer


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


DownloadandConvertModels
---------------------------

`backtotop⬆️<#table-of-contents>`__

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.TheOptimumInferencemodelsareAPIcompatiblewithHugging
FaceTransformersmodels.Thismeanswejustneedtoreplace
``AutoModelForXxx``classwiththecorresponding``OVModelForXxx``
class.

BelowisanexampleoftheRoBERTatextclassificationmodel

..code::diff

-fromtransformersimportAutoModelForSequenceClassification
+fromoptimum.intel.openvinoimportOVModelForSequenceClassification
fromtransformersimportAutoTokenizer,pipeline

model_id="textattack/roberta-base-CoLA"
-model=AutoModelForSequenceClassification.from_pretrained(model_id)
+model=OVModelForSequenceClassification.from_pretrained(model_id,from_transformers=True)

Modelclassinitializationstartswithcalling``from_pretrained``
method.WhendownloadingandconvertingTransformersmodel,the
parameter``from_transformers=True``shouldbeadded.Wecansavethe
convertedmodelforthenextusagewiththe``save_pretrained``method.
TokenizerclassandpipelinesAPIarecompatiblewithOptimummodels.

..code::ipython3

frompathlibimportPath
fromtransformersimportpipeline,AutoTokenizer
fromoptimum.intel.openvinoimportOVModelForSeq2SeqLM,OVModelForSequenceClassification


..parsed-literal::

2024-03-2511:56:04.043628:Itensorflow/core/util/port.cc:111]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-2511:56:04.045940:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-2511:56:04.079112:Etensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342]UnabletoregistercuDNNfactory:AttemptingtoregisterfactoryforplugincuDNNwhenonehasalreadybeenregistered
2024-03-2511:56:04.079147:Etensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609]UnabletoregistercuFFTfactory:AttemptingtoregisterfactoryforplugincuFFTwhenonehasalreadybeenregistered
2024-03-2511:56:04.079167:Etensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518]UnabletoregistercuBLASfactory:AttemptingtoregisterfactoryforplugincuBLASwhenonehasalreadybeenregistered
2024-03-2511:56:04.085243:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-2511:56:04.085971:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-03-2511:56:05.314633:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


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

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



GrammarChecker
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

grammar_checker_model_id="textattack/roberta-base-CoLA"
grammar_checker_dir=Path("roberta-base-cola")
grammar_checker_tokenizer=AutoTokenizer.from_pretrained(grammar_checker_model_id)

ifgrammar_checker_dir.exists():
grammar_checker_model=OVModelForSequenceClassification.from_pretrained(grammar_checker_dir,device=device.value)
else:
grammar_checker_model=OVModelForSequenceClassification.from_pretrained(grammar_checker_model_id,export=True,device=device.value,load_in_8bit=False)
grammar_checker_model.save_pretrained(grammar_checker_dir)


..parsed-literal::

Frameworknotspecified.Usingpttoexportthemodel.
Someweightsofthemodelcheckpointattextattack/roberta-base-CoLAwerenotusedwheninitializingRobertaForSequenceClassification:['roberta.pooler.dense.bias','roberta.pooler.dense.weight']
-ThisISexpectedifyouareinitializingRobertaForSequenceClassificationfromthecheckpointofamodeltrainedonanothertaskorwithanotherarchitecture(e.g.initializingaBertForSequenceClassificationmodelfromaBertForPreTrainingmodel).
-ThisISNOTexpectedifyouareinitializingRobertaForSequenceClassificationfromthecheckpointofamodelthatyouexpecttobeexactlyidentical(initializingaBertForSequenceClassificationmodelfromaBertForSequenceClassificationmodel).
Usingtheexportvariantdefault.Availablevariantsare:
-default:ThedefaultONNXvariant.
UsingframeworkPyTorch:2.2.1+cpu
Overriding1configurationitem(s)
	-use_cache->False
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:4225:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
CompilingthemodeltoAUTO...


Letuscheckmodelwork,usinginferencepipelinefor
``text-classification``task.Youcanfindmoreinformationaboutusage
HuggingFaceinferencepipelinesinthis
`tutorial<https://huggingface.co/docs/transformers/pipeline_tutorial>`__

..code::ipython3

input_text="Theyaremovedbysalarenergy"
grammar_checker_pipe=pipeline(
"text-classification",
model=grammar_checker_model,
tokenizer=grammar_checker_tokenizer,
)
result=grammar_checker_pipe(input_text)[0]
print(f"inputtext:{input_text}")
print(f'predictedlabel:{"contains_errors"ifresult["label"]=="LABEL_1"else"noerrors"}')
print(f'predictedscore:{result["score"]:.2}')


..parsed-literal::

inputtext:Theyaremovedbysalarenergy
predictedlabel:contains_errors
predictedscore:0.88


Great!Lookslikethemodelcandetecterrorsinthesample.

GrammarCorrector
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThestepsforloadingtheGrammarCorrectormodelareverysimilar,
exceptforthemodelclassthatisused.BecauseFLAN-T5isa
sequence-to-sequencetextgenerationmodel,weshouldusethe
``OVModelForSeq2SeqLM``classandthe``text2text-generation``pipeline
torunit.

..code::ipython3

grammar_corrector_model_id="pszemraj/flan-t5-large-grammar-synthesis"
grammar_corrector_dir=Path("flan-t5-large-grammar-synthesis")
grammar_corrector_tokenizer=AutoTokenizer.from_pretrained(grammar_corrector_model_id)

ifgrammar_corrector_dir.exists():
grammar_corrector_model=OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_dir,device=device.value)
else:
grammar_corrector_model=OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_model_id,export=True,device=device.value)
grammar_corrector_model.save_pretrained(grammar_corrector_dir)


..parsed-literal::

Frameworknotspecified.Usingpttoexportthemodel.
Usingtheexportvariantdefault.Availablevariantsare:
-default:ThedefaultONNXvariant.
Somenon-defaultgenerationparametersaresetinthemodelconfig.TheseshouldgointoaGenerationConfigfile(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)instead.Thiswarningwillberaisedtoanexceptioninv4.41.
Non-defaultgenerationparameters:{'max_length':512,'min_length':8,'num_beams':2,'no_repeat_ngram_size':4}
UsingframeworkPyTorch:2.2.1+cpu
Overriding1configurationitem(s)
	-use_cache->False
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:4225:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
UsingframeworkPyTorch:2.2.1+cpu
Overriding1configurationitem(s)
	-use_cache->True
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:943:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_mask.shape[1]<attention_mask.shape[1]:
UsingframeworkPyTorch:2.2.1+cpu
Overriding1configurationitem(s)
	-use_cache->True
/home/ea/miniconda3/lib/python3.11/site-packages/transformers/models/t5/modeling_t5.py:509:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
elifpast_key_value.shape[2]!=key_value_states.shape[1]:
Somenon-defaultgenerationparametersaresetinthemodelconfig.TheseshouldgointoaGenerationConfigfile(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)instead.Thiswarningwillberaisedtoanexceptioninv4.41.
Non-defaultgenerationparameters:{'max_length':512,'min_length':8,'num_beams':2,'no_repeat_ngram_size':4}
CompilingtheencodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...
Somenon-defaultgenerationparametersaresetinthemodelconfig.TheseshouldgointoaGenerationConfigfile(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)instead.Thiswarningwillberaisedtoanexceptioninv4.41.
Non-defaultgenerationparameters:{'max_length':512,'min_length':8,'num_beams':2,'no_repeat_ngram_size':4}


..code::ipython3

grammar_corrector_pipe=pipeline(
"text2text-generation",
model=grammar_corrector_model,
tokenizer=grammar_corrector_tokenizer,
)

..code::ipython3

result=grammar_corrector_pipe(input_text)[0]
print(f"inputtext:{input_text}")
print(f'generatedtext:{result["generated_text"]}')


..parsed-literal::

inputtext:Theyaremovedbysalarenergy
generatedtext:Theyarepoweredbysolarenergy.


Nice!Theresultlooksprettygood!

PrepareDemoPipeline
---------------------

`backtotop⬆️<#table-of-contents>`__

Nowletusputeverythingtogetherandcreatethepipelineforgrammar
correction.Thepipelineacceptsinputtext,verifiesitscorrectness,
andgeneratesthecorrectversionifrequired.Itwillconsistof
severalsteps:

1.Splittextonsentences.
2.CheckgrammaticalcorrectnessforeachsentenceusingGrammar
Checker.
3.Generateanimprovedversionofthesentenceifrequired.

..code::ipython3

importre
importtransformers
fromtqdm.notebookimporttqdm


defsplit_text(text:str)->list:
"""
Splitastringoftextintoalistofsentencebatches.

Parameters:
text(str):Thetexttobesplitintosentencebatches.

Returns:
list:Alistofsentencebatches.Eachsentencebatchisalistofsentences.
"""
#Splitthetextintosentencesusingregex
sentences=re.split(r"(?<=[^A-Z].[.?])+(?=[A-Z])",text)

#Initializealisttostorethesentencebatches
sentence_batches=[]

#Initializeatemporarylisttostorethecurrentbatchofsentences
temp_batch=[]

#Iteratethroughthesentences
forsentenceinsentences:
#Addthesentencetothetemporarybatch
temp_batch.append(sentence)

#Ifthelengthofthetemporarybatchisbetween2and3sentences,orifitisthelastbatch,addittothelistofsentencebatches
iflen(temp_batch)>=2andlen(temp_batch)<=3orsentence==sentences[-1]:
sentence_batches.append(temp_batch)
temp_batch=[]

returnsentence_batches


defcorrect_text(
text:str,
checker:transformers.pipelines.Pipeline,
corrector:transformers.pipelines.Pipeline,
separator:str="",
)->str:
"""
Correctthegrammarinastringoftextusingatext-classificationandtext-generationpipeline.

Parameters:
text(str):Theinpurtexttobecorrected.
checker(transformers.pipelines.Pipeline):Thetext-classificationpipelinetouseforcheckingthegrammarqualityofthetext.
corrector(transformers.pipelines.Pipeline):Thetext-generationpipelinetouseforcorrectingthetext.
separator(str,optional):Theseparatortousewhenjoiningthecorrectedtextintoasinglestring.Defaultisaspacecharacter.

Returns:
str:Thecorrectedtext.
"""
#Splitthetextintosentencebatches
sentence_batches=split_text(text)

#Initializealisttostorethecorrectedtext
corrected_text=[]

#Iteratethroughthesentencebatches
forbatchintqdm(sentence_batches,total=len(sentence_batches),desc="correctingtext.."):
#Jointhesentencesinthebatchintoasinglestring
raw_text="".join(batch)

#Checkthegrammarqualityofthetextusingthetext-classificationpipeline
results=checker(raw_text)

#Onlycorrectthetextiftheresultsofthetext-classificationarenotLABEL_1orareLABEL_1withascorebelow0.9
ifresults[0]["label"]!="LABEL_1"or(results[0]["label"]=="LABEL_1"andresults[0]["score"]<0.9):
#Correctthetextusingthetext-generationpipeline
corrected_batch=corrector(raw_text)
corrected_text.append(corrected_batch[0]["generated_text"])
else:
corrected_text.append(raw_text)

#Jointhecorrectedtextintoasinglestring
corrected_text=separator.join(corrected_text)

returncorrected_text

Letusseeitinaction.

..code::ipython3

default_text=(
"Mostofthecourseisaboutsemanticorcontentoflanguagebuttherearealsointeresting"
"topicstobelearnedfromtheservicefeaturesexceptstatisticsincharactersindocuments.At"
"thispoint,HeintroducesherselfashisnativeEnglishspeakerandgoesontosaythatif"
"youcontinetoworkonsocialscnce"
)

corrected_text=correct_text(default_text,grammar_checker_pipe,grammar_corrector_pipe)



..parsed-literal::

correctingtext..:0%||0/1[00:00<?,?it/s]


..code::ipython3

print(f"inputtext:{default_text}\n")
print(f"generatedtext:{corrected_text}")


..parsed-literal::

inputtext:Mostofthecourseisaboutsemanticorcontentoflanguagebuttherearealsointerestingtopicstobelearnedfromtheservicefeaturesexceptstatisticsincharactersindocuments.Atthispoint,HeintroducesherselfashisnativeEnglishspeakerandgoesontosaythatifyoucontinetoworkonsocialscnce

generatedtext:Mostofthecourseisaboutthesemanticcontentoflanguagebuttherearealsointerestingtopicstobelearnedfromtheservicefeaturesexceptstatisticsincharactersindocuments.Atthispoint,sheintroducesherselfasanativeEnglishspeakerandgoesontosaythatifyoucontinuetoworkonsocialscience,youwillcontinuetobesuccessful.


Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Grammarcheckermodeltakesupatinyportionofthewholetext
correctionpipelinesoweoptimizeonlythegrammarcorrectormodel.
Grammarcorrectoritselfconsistsofthreemodels:encoder,firstcall
decoderanddecoderwithpast.Thelastmodel’sshareofinference
dominatestheotherones.Becauseofthiswequantizeonlyit.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize()``toobtainquantizedmodels.
3.Serializethe``INT8``modelusing``openvino.save_model()``
function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



RunQuantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Belowweretrievethequantizedmodel.Pleasesee``utils.py``for
sourcecode.Quantizationisrelativelytime-consumingandwilltake
sometimetocomplete.

..code::ipython3

fromutilsimportget_quantized_pipeline,CALIBRATION_DATASET_SIZE

grammar_corrector_pipe_fp32=grammar_corrector_pipe
grammar_corrector_pipe_int8=None
ifto_quantize.value:
quantized_model_path=Path("quantized_decoder_with_past")/"openvino_model.xml"
grammar_corrector_pipe_int8=get_quantized_pipeline(
grammar_corrector_pipe_fp32,
grammar_corrector_tokenizer,
core,
grammar_corrector_dir,
quantized_model_path,
device.value,
calibration_dataset_size=CALIBRATION_DATASET_SIZE,
)



..parsed-literal::

Downloadingreadme:0%||0.00/5.94k[00:00<?,?B/s]


..parsed-literal::

Downloadingdata:100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|148k/148k[00:01<00:00,79.1kB/s]
Downloadingdata:100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|141k/141k[00:01<00:00,131kB/s]



..parsed-literal::

Generatingvalidationsplit:0%||0/755[00:00<?,?examples/s]



..parsed-literal::

Generatingtestsplit:0%||0/748[00:00<?,?examples/s]



..parsed-literal::

Collectingcalibrationdata:0%||0/10[00:00<?,?it/s]



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

INFO:nncf:72ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:145ignorednodeswerefoundbynameintheNNCFGraph



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

CompilingtheencodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...


Let’sseecorrectionresults.ThegeneratedtextsforquantizedINT8
modelandoriginalFP32modelshouldbealmostthesame.

..code::ipython3

ifto_quantize.value:
corrected_text_int8=correct_text(default_text,grammar_checker_pipe,grammar_corrector_pipe_int8)
print(f"Inputtext:{default_text}\n")
print(f"GeneratedtextbyINT8model:{corrected_text_int8}")



..parsed-literal::

correctingtext..:0%||0/1[00:00<?,?it/s]


..parsed-literal::

Inputtext:Mostofthecourseisaboutsemanticorcontentoflanguagebuttherearealsointerestingtopicstobelearnedfromtheservicefeaturesexceptstatisticsincharactersindocuments.Atthispoint,HeintroducesherselfashisnativeEnglishspeakerandgoesontosaythatifyoucontinetoworkonsocialscnce

GeneratedtextbyINT8model:Mostofthecourseisaboutsemanticsorcontentoflanguagebuttherearealsointerestingtopicstobelearnedfromtheservicefeaturesexceptstatisticsincharactersindocuments.Atthispoint,sheintroduceshimselfasanativeEnglishspeakerandgoesontosaythatifyoucontinuetoworkonsocialscience,youwillcontinuetodoso.


Comparemodelsize,performanceandaccuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,wecomparefilesizeof``FP32``and``INT8``models.

..code::ipython3

fromutilsimportcalculate_compression_rate

ifto_quantize.value:
model_size_fp32,model_size_int8=calculate_compression_rate(
grammar_corrector_dir/"openvino_decoder_with_past_model.xml",
quantized_model_path,
)


..parsed-literal::

Modelfootprintcomparison:
*FP32IRmodelsize:1658150.25KB
*INT8IRmodelsize:415711.39KB


Second,wecomparetwogrammarcorrectionpipelinesfromperformanceand
accuracystandpoints.

Testsplitof\`jfleg<https://huggingface.co/datasets/jfleg>`__\
datasetisusedfortesting.Onedatasetsampleconsistsofatextwith
errorsasinputandseveralcorrectedversionsaslabels.Whenmeasuring
accuracyweusemean``(1-WER)``againstcorrectedtextversions,
whereWERisWordErrorRatemetric.

..code::ipython3

fromutilsimportcalculate_inference_time_and_accuracy

TEST_SUBSET_SIZE=50

ifto_quantize.value:
inference_time_fp32,accuracy_fp32=calculate_inference_time_and_accuracy(grammar_corrector_pipe_fp32,TEST_SUBSET_SIZE)
print(f"EvaluationresultsofFP32grammarcorrectionpipeline.Accuracy:{accuracy_fp32:.2f}%.Time:{inference_time_fp32:.2f}sec.")
inference_time_int8,accuracy_int8=calculate_inference_time_and_accuracy(grammar_corrector_pipe_int8,TEST_SUBSET_SIZE)
print(f"EvaluationresultsofINT8grammarcorrectionpipeline.Accuracy:{accuracy_int8:.2f}%.Time:{inference_time_int8:.2f}sec.")
print(f"Performancespeedup:{inference_time_fp32/inference_time_int8:.3f}")
print(f"Accuracydrop:{accuracy_fp32-accuracy_int8:.2f}%.")
print(f"Modelfootprintreduction:{model_size_fp32/model_size_int8:.3f}")



..parsed-literal::

Evaluation:0%||0/50[00:00<?,?it/s]


..parsed-literal::

EvaluationresultsofFP32grammarcorrectionpipeline.Accuracy:58.04%.Time:62.44sec.



..parsed-literal::

Evaluation:0%||0/50[00:00<?,?it/s]


..parsed-literal::

EvaluationresultsofINT8grammarcorrectionpipeline.Accuracy:59.04%.Time:40.32sec.
Performancespeedup:1.549
Accuracydrop:-0.99%.
Modelfootprintreduction:3.989


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
importtime


defcorrect(text,quantized,progress=gr.Progress(track_tqdm=True)):
grammar_corrector=grammar_corrector_pipe_int8ifquantizedelsegrammar_corrector_pipe

start_time=time.perf_counter()
corrected_text=correct_text(text,grammar_checker_pipe,grammar_corrector)
end_time=time.perf_counter()

returncorrected_text,f"{end_time-start_time:.2f}"


defcreate_demo_block(quantized:bool,show_model_type:bool):
model_type=("optimized"ifquantizedelse"original")ifshow_model_typeelse""
withgr.Row():
gr.Markdown(f"##Run{model_type}grammarcorrectionpipeline")
withgr.Row():
withgr.Column():
input_text=gr.Textbox(label="Text")
withgr.Column():
output_text=gr.Textbox(label="Correction")
correction_time=gr.Textbox(label="Time(seconds)")
withgr.Row():
gr.Examples(examples=[default_text],inputs=[input_text])
withgr.Row():
button=gr.Button(f"Run{model_type}")
button.click(
correct,
inputs=[input_text,gr.Number(quantized,visible=False)],
outputs=[output_text,correction_time],
)


withgr.Blocks()asdemo:
gr.Markdown("#Interactivedemo")
quantization_is_present=grammar_corrector_pipe_int8isnotNone
create_demo_block(quantized=False,show_model_type=quantization_is_present)
ifquantization_is_present:
create_demo_block(quantized=True,show_model_type=True)


#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)

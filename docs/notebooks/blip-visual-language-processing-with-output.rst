VisualQuestionAnsweringandImageCaptioningusingBLIPandOpenVINO
======================================================================

Humansperceivetheworldthroughvisionandlanguage.Alongtimegoal
ofAIistobuildintelligentagentsthatcanunderstandtheworld
throughvisionandlanguageinputstocommunicatewithhumansthrough
naturallanguage.Inordertoachievethisgoal,vision-language
pre-traininghasemergedasaneffectiveapproach,wheredeepneural
networkmodelsarepre-trainedonlargescaleimage-textdatasetsto
improveperformanceondownstreamvision-languagetasks,suchas
image-textretrieval,imagecaptioning,andvisualquestionanswering.

`BLIP<https://github.com/salesforce/BLIP>`__isalanguage-image
pre-trainingframeworkforunifiedvision-languageunderstandingand
generation.BLIPachievesstate-of-the-artresultsonawiderangeof
vision-languagetasks.ThistutorialdemonstrateshowtouseBLIPfor
visualquestionansweringandimagecaptioning.Anadditionalpartof
tutorialdemonstrateshowtospeedupthemodelbyapplying8-bit
post-trainingquantizationanddatafreeint8weightcompressionfrom
`NNCF<https://github.com/openvinotoolkit/nncf/>`__(NeuralNetwork
CompressionFramework)toOpenVINOIRmodelsandinferoptimizedBLIP
modelviaOpenVINO™Toolkit.

Thetutorialconsistsofthefollowingparts:

1.InstantiateaBLIPmodel.
2.ConverttheBLIPmodeltoOpenVINOIR.
3.RunvisualquestionansweringandimagecaptioningwithOpenVINO.
4.OptimizeBLIPmodelusingNNCF
5.Compareoriginalandoptimizedmodels
6.Launchinteractivedemo

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Background<#background>`__

-`ImageCaptioning<#image-captioning>`__
-`VisualQuestionAnswering<#visual-question-answering>`__

-`InstantiateModel<#instantiate-model>`__
-`ConvertModelstoOpenVINOIR<#convert-models-to-openvino-ir>`__

-`VisionModel<#vision-model>`__
-`TextEncoder<#text-encoder>`__
-`TextDecoder<#text-decoder>`__

-`RunOpenVINOModel<#run-openvino-model>`__

-`PrepareInferencePipeline<#prepare-inference-pipeline>`__
-`Selectinferencedevice<#select-inference-device>`__
-`ImageCaptioning<#image-captioning>`__
-`QuestionAnswering<#question-answering>`__

-`OptimizemodelusingNNCF<#optimize-model-using-nncf>`__

-`Preparedataset<#prepare-dataset>`__
-`Quantizevisionmodel<#quantize-vision-model>`__
-`Quantizetextencoder<#quantize-text-encoder>`__
-`Compressweightsoftext
decoder<#compress-weights-of-text-decoder>`__
-`RunoptimizedOpenVINOmodel<#run-optimized-openvino-model>`__

-`Imagecaptioning<#image-captioning>`__
-`Questionanswering<#question-answering>`__

-`Comparefilesizes<#compare-file-sizes>`__
-`CompareinferencetimeoftheFP16andoptimized
models<#compare-inference-time-of-the-fp16-and-optimized-models>`__

-`Interactivedemo<#interactive-demo>`__

Background
----------

`backtotop⬆️<#table-of-contents>`__

Visuallanguageprocessingisabranchofartificialintelligencethat
focusesoncreatingalgorithmsdesignedtoenablecomputerstomore
accuratelyunderstandimagesandtheircontent.

Populartasksinclude:

-**TexttoImageRetrieval**-asemantictaskthataimstofindthe
mostrelevantimageforagiventextdescription.
-**ImageCaptioning**-asemantictaskthataimstoprovideatext
descriptionforimagecontent.
-**VisualQuestionAnswering**-asemantictaskthataimstoanswer
questionsbasedonimagecontent.

Asshowninthediagrambelow,thesethreetasksdifferintheinput
providedtotheAIsystem.Fortext-to-imageretrieval,youhavea
predefinedgalleryofimagesforsearchandauser-requestedtext
description(query).Imagecaptioningcanberepresentedasaparticular
caseofvisualquestionanswering,whereyouhaveapredefinedquestion
“Whatisinthepicture?”andvariousimagesprovidedbyauser.For
visualquestionanswering,boththetext-basedquestionandimage
contextarevariablesrequestedbyauser.

|image0|

ThisnotebookdoesnotfocusonTexttoImageretrieval.Instead,it
considersImageCaptioningandVisualQuestionAnswering.

ImageCaptioning
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ImageCaptioningisthetaskofdescribingthecontentofanimagein
words.Thistaskliesattheintersectionofcomputervisionandnatural
languageprocessing.Mostimagecaptioningsystemsusean
encoder-decoderframework,whereaninputimageisencodedintoan
intermediaterepresentationoftheinformationintheimage,andthen
decodedintoadescriptivetextsequence.

|image1|

VisualQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

VisualQuestionAnswering(VQA)isthetaskofansweringtext-based
questionsaboutimagecontent.

|image2|

ForabetterunderstandingofhowVQAworks,letusconsidera
traditionalNLPtasklikeQuestionAnswering,whichaimstoretrievethe
answertoaquestionfromagiventextinput.Typically,aquestion
answeringpipelineconsistsofthreesteps:

|image3|

1.Questionanalysis-analysisofprovidedquestioninnaturallanguage
formtounderstandtheobjectinthequestionandadditionalcontext.
Forexample,ifyouhaveaquestionlike“Howmanybridgesin
Paris?”,questionwords*“howmany”*givesahintthattheansweris
morelikelytobeanumber,*“bridges”*isthetargetobjectofthe
questionand*"inParis"*servesasadditionalcontextforthe
search.
2.Buildqueryforsearch-useanalyzedresultstoformalizequeryfor
findingthemostrelevantinformation.
3.Performasearchintheknowledgebase-sendthequerytoa
knowledgebase,typicallyprovidedtextdocumentsordatabasesserve
asasourceofknowledge.

|image4|

Thedifferencebetweentext-basedquestionansweringandvisualquestion
answeringisthatanimageisusedascontextandtheknowledgebase.

|image5|

Answeringarbitraryquestionsaboutimagesisacomplexproblembecause
itrequiresinvolvingalotofcomputervisionsub-tasks.Inthetable
below,youcanfindanexampleofquestionsandtherequiredcomputer
visionskillstofindanswers.

+-----------------------------+----------------------------------------+
|Computervisiontask|Questionexamples|
+=============================+========================================+
|Objectrecognition|Whatisshowninthepicture?Whatis|
||it?|
+-----------------------------+----------------------------------------+
|Objectdetection|Isthereanyobject(dog,man,book)|
||intheimage?Whereis…located?|
+-----------------------------+----------------------------------------+
|Objectandimageattribute|Whatcolorisanumbrella?Doesthis|
|recognition|manwearglasses?Istherecolorin|
||theimage?|
+-----------------------------+----------------------------------------+
|Scenerecognition|Isitrainy?Whatcelebrationis|
||pictured?|
+-----------------------------+----------------------------------------+
|Objectcounting|Howmanyplayersarethereonthe|
||footballfield?Howmanystepsare|
||thereonthestairs?|
+-----------------------------+----------------------------------------+
|Activityrecognition|Isthebabycrying?Whatisthewoman|
||cooking?Whataretheydoing?|
+-----------------------------+----------------------------------------+
|Spatialrelationshipsamong|Whatislocatedbetweenthesofaand|
|objects|thearmchair?Whatisinthebottom|
||leftcorner?|
+-----------------------------+----------------------------------------+
|Commonsensereasoning|Doesshehave100%vision?Doesthis|
||personhavechildren?|
+-----------------------------+----------------------------------------+
|Knowledge-basedreasoning|Isitavegetarianpizza?|
+-----------------------------+----------------------------------------+
|Textrecognition|Whatisthetitleofthebook?Whatis|
||shownonthescreen?|
+-----------------------------+----------------------------------------+

Therearealotofapplicationsforvisualquestionanswering:

-AidVisuallyImpairedPersons:VQAmodelscanbeusedtoreduce
barriersforvisuallyimpairedpeoplebyhelpingthemgetinformation
aboutimagesfromthewebandtherealworld.
-Education:VQAmodelscanbeusedtoimprovevisitorexperiencesat
museumsbyenablingobserverstodirectlyaskquestionstheyare
interestedinortobringmoreinteractivitytoschoolbooksfor
childreninterestedinacquiringspecificknowledge.
-E-commerce:VQAmodelscanretrieveinformationaboutproductsusing
photosfromonlinestores.
-Independentexpertassessment:VQAmodelscanbeprovideobjective
assessmentsinsportscompetitions,medicaldiagnosis,andforensic
examination.

..|image0|image::https://user-images.githubusercontent.com/29454499/221755717-a5b51b7e-523c-461f-b30c-4edbfaf9a134.png
..|image1|image::https://user-images.githubusercontent.com/29454499/221640847-1868117c-aac0-4806-99a4-34f218e98bb8.png
..|image2|image::https://user-images.githubusercontent.com/29454499/221641984-3c6d8b2f-dd0d-4302-a4d8-0f8564fca772.png
..|image3|image::https://user-images.githubusercontent.com/29454499/221760881-378f1ea8-eadc-4610-aff0-69ecabf62fff.png
..|image4|image::https://user-images.githubusercontent.com/29454499/222094861-3cafdf9f-d700-4741-b6c5-fb09c1a4da9a.png
..|image5|image::https://user-images.githubusercontent.com/29454499/222095118-3d5826e4-2662-4d1c-abf2-a515f23d6d6a.png

InstantiateModel
-----------------

`backtotop⬆️<#table-of-contents>`__

TheBLIPmodelwasproposedinthe`BLIP:BootstrappingLanguage-Image
Pre-trainingforUnifiedVision-LanguageUnderstandingand
Generation<https://arxiv.org/abs/2201.12086>`__paper.

..figure::https://github.com/salesforce/BLIP/raw/main/BLIP.gif
:alt:blip.gif

blip.gif

Topre-trainaunifiedvision-languagemodelwithbothunderstandingand
generationcapabilities,BLIPintroducesamultimodalmixtureofan
encoder-decoderandamulti-taskmodelwhichcanoperateinoneofthe
threemodes:

-**Unimodalencoders**,whichseparatelyencodeimagesandtext.The
imageencoderisavisiontransformer.Thetextencoderisthesame
asBERT.
-**Image-groundedtextencoder**,whichinjectsvisualinformationby
insertingacross-attentionlayerbetweentheself-attentionlayer
andthefeed-forwardnetworkforeachtransformerblockofthetext
encoder.
-**Image-groundedtextdecoder**,whichreplacesthebi-directional
self-attentionlayersinthetextencoderwithcausalself-attention
layers.

Moredetailsaboutthemodelcanbefoundinthe`research
paper<https://arxiv.org/abs/2201.12086>`__,`Salesforce
blog<https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/>`__,
`GitHubrepo<https://github.com/salesforce/BLIP>`__and`HuggingFace
model
documentation<https://huggingface.co/docs/transformers/model_doc/blip>`__.

Inthistutorial,youwillusethe
`blip-vqa-base<https://huggingface.co/Salesforce/blip-vqa-base>`__
modelavailablefordownloadfrom`Hugging
Face<https://huggingface.co/>`__.Thesameactionsarealsoapplicable
toothersimilarmodelsfromtheBLIPfamily.Althoughthismodelclass
isdesignedtoperformquestionanswering,itscomponentscanalsobe
reusedforimagecaptioning.

Tostartworkingwiththemodel,youneedtoinstantiatethe
``BlipForQuestionAnswering``class,using``from_pretrained``method.
``BlipProcessor``isahelperclassforpreparinginputdataforboth
textandvisionmodalitiesandpostprocessingofgenerationresults.

..code::ipython3

importplatform

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"torch>=2.1.0"torchvision"transformers>=4.26.0""gradio>=4.19""openvino>=2023.3.0""datasets>=2.14.6""nncf>=2.8.1""tqdm"
ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

..code::ipython3

importtime
fromPILimportImage
fromtransformersimportBlipProcessor,BlipForQuestionAnswering

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

#getmodelandprocessor
processor=BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

#setuptestinput:downloadandreadimage,preparequestion
img_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
download_file(img_url,"demo.jpg")
raw_image=Image.open("demo.jpg").convert("RGB")
question="howmanydogsareinthepicture?"
#preprocessinputdata
inputs=processor(raw_image,question,return_tensors="pt")

start=time.perf_counter()
#performgeneration
out=model.generate(**inputs)
end=time.perf_counter()-start

#postprocessresult
answer=processor.decode(out[0],skip_special_tokens=True)

..code::ipython3

print(f"Processingtime:{end:.4f}s")


..parsed-literal::

Processingtime:0.3707s


..code::ipython3

frompathlibimportPath

ifnotPath("./utils.py").exists():
download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/blip-visual-language-processing/utils.py")
fromutilsimportvisualize_results

fig=visualize_results(raw_image,answer,question)



..image::blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_7_0.png


ConvertModelstoOpenVINOIR
-----------------------------

`backtotop⬆️<#table-of-contents>`__

StartingfromOpenVINO2023.0release,OpenVINOsupportsdirectPyTorch
modelsconversiontoOpenVINOIntermediateRepresentation(IR)formatto
taketheadvantageofadvancedOpenVINOoptimizationtoolsandfeatures.
Youneedtoprovideamodelobject,inputdataformodeltracingto
OpenVINOModelConversionAPI.``ov.convert_model``functionconvert
PyTorchmodelinstanceto``ov.Model``objectthatcanbeusedfor
compilationondeviceorsavedondiskusing``ov.save_model``in
compressedtoFP16format.

Themodelconsistsofthreeparts:

-vision_model-anencoderforimagerepresentation.
-text_encoder-anencoderforinputquery,usedforquestion
answeringandtext-to-imageretrievalonly.
-text_decoder-adecoderforoutputanswer.

Tobeabletoperformmultipletasks,usingthesamemodelcomponents,
youshouldconverteachpartindependently.

VisionModel
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thevisionmodelacceptsfloatinputtensorswiththe[1,3,384,384]
shape,containingRGBimagepixelvaluesnormalizedinthe[0,1]range.

..code::ipython3

importtorch
frompathlibimportPath
importopenvinoasov

VISION_MODEL_OV=Path("blip_vision_model.xml")
vision_model=model.vision_model
vision_model.eval()

#checkthatmodelworksandsaveitoutputsforreusageastextencoderinput
withtorch.no_grad():
vision_outputs=vision_model(inputs["pixel_values"])

#ifopenvinomodeldoesnotexist,convertittoIR
ifnotVISION_MODEL_OV.exists():
#exportpytorchmodeltoov.Model
withtorch.no_grad():
ov_vision_model=ov.convert_model(vision_model,example_input=inputs["pixel_values"])
#savemodelondiskfornextusages
ov.save_model(ov_vision_model,VISION_MODEL_OV)
print(f"Visionmodelsuccessfulyconvertedandsavedto{VISION_MODEL_OV}")
else:
print(f"Visionmodelwillbeloadedfrom{VISION_MODEL_OV}")


..parsed-literal::

/home/ltalamanova/tmp_venv/lib/python3.11/site-packages/transformers/modeling_utils.py:4225:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(


..parsed-literal::

Visionmodelsuccessfulyconvertedandsavedtoblip_vision_model.xml


TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetextencoderisusedbyvisualquestionansweringtaskstobuilda
questionembeddingrepresentation.Ittakes``input_ids``witha
tokenizedquestionandoutputimageembeddingsobtainedfromthevision
modelandattentionmasksforthem.

..code::ipython3

TEXT_ENCODER_OV=Path("blip_text_encoder.xml")


text_encoder=model.text_encoder
text_encoder.eval()

#ifopenvinomodeldoesnotexist,convertittoIR
ifnotTEXT_ENCODER_OV.exists():
#prepareexampleinputs
image_embeds=vision_outputs[0]
image_attention_mask=torch.ones(image_embeds.size()[:-1],dtype=torch.long)
input_dict={
"input_ids":inputs["input_ids"],
"attention_mask":inputs["attention_mask"],
"encoder_hidden_states":image_embeds,
"encoder_attention_mask":image_attention_mask,
}
#exportPyTorchmodel
withtorch.no_grad():
ov_text_encoder=ov.convert_model(text_encoder,example_input=input_dict)
#savemodelondiskfornextusages
ov.save_model(ov_text_encoder,TEXT_ENCODER_OV)
print(f"Textencodersuccessfulyconvertedandsavedto{TEXT_ENCODER_OV}")
else:
print(f"Textencoderwillbeloadedfrom{TEXT_ENCODER_OV}")


..parsed-literal::

Textencodersuccessfulyconvertedandsavedtoblip_text_encoder.xml


TextDecoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetextdecoderisresponsibleforgeneratingthesequenceoftokensto
representmodeloutput(answertoquestionorcaption),usinganimage
(andquestion,ifrequired)representation.Thegenerationapproachis
basedontheassumptionthattheprobabilitydistributionofaword
sequencecanbedecomposedintotheproductofconditionalnextword
distributions.Inotherwords,modelpredictsthenexttokenintheloop
guidedbypreviouslygeneratedtokensuntilthestop-conditionwillbe
notreached(generatedsequenceofmaximumlengthorendofstringtoken
obtained).Thewaythenexttokenwillbeselectedoverpredicted
probabilitiesisdrivenbytheselecteddecodingmethodology.Youcan
findmoreinformationaboutthemostpopulardecodingmethodsinthis
`blog<https://huggingface.co/blog/how-to-generate>`__.Theentrypoint
forthegenerationprocessformodelsfromtheHuggingFaceTransformers
libraryisthe``generate``method.Youcanfindmoreinformationabout
itsparametersandconfigurationinthe
`documentation<https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__.
Topreserveflexibilityintheselectiondecodingmethodology,youwill
convertonlymodelinferenceforonestep.

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
Withthisoption,themodelgetsthepreviousstep’shiddenstatesas
inputandadditionallyprovideshiddenstatesforthecurrentstepas
output.Initially,youhavenopreviousstephiddenstates,sothefirst
stepdoesnotrequireyoutoprovidethem,butweshouldinitializethem
bydefaultvalues.InPyTorch,pasthiddenstateoutputsarerepresented
asalistofpairs(hiddenstateforkey,hiddenstateforvalue]for
eachtransformerlayerinthemodel.OpenVINOmodeldoesnotsupport
nestedoutputs,theywillbeflattened.

Similarto``text_encoder``,``text_decoder``canworkwithinput
sequencesofdifferentlengthsandrequirespreservingdynamicinput
shapes.

..code::ipython3

text_decoder=model.text_decoder
text_decoder.eval()

TEXT_DECODER_OV=Path("blip_text_decoder_with_past.xml")

#prepareexampleinputs
input_ids=torch.tensor([[30522]])#beginofsequencetokenid
attention_mask=torch.tensor([[1]])#attentionmaskforinput_ids
encoder_hidden_states=torch.rand((1,10,768))#encoderlasthiddenstatefromtext_encoder
encoder_attention_mask=torch.ones((1,10),dtype=torch.long)#attentionmaskforencoderhiddenstates

input_dict={
"input_ids":input_ids,
"attention_mask":attention_mask,
"encoder_hidden_states":encoder_hidden_states,
"encoder_attention_mask":encoder_attention_mask,
}
text_decoder_outs=text_decoder(**input_dict)
#extendinputdictionarywithhiddenstatesfrompreviousstep
input_dict["past_key_values"]=text_decoder_outs["past_key_values"]

text_decoder.config.torchscript=True
ifnotTEXT_DECODER_OV.exists():
#exportPyTorchmodel
withtorch.no_grad():
ov_text_decoder=ov.convert_model(text_decoder,example_input=input_dict)
#savemodelondiskfornextusages
ov.save_model(ov_text_decoder,TEXT_DECODER_OV)
print(f"Textdecodersuccessfulyconvertedandsavedto{TEXT_DECODER_OV}")
else:
print(f"Textdecoderwillbeloadedfrom{TEXT_DECODER_OV}")


..parsed-literal::

/home/ltalamanova/tmp_venv/lib/python3.11/site-packages/transformers/models/blip/modeling_blip_text.py:635:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_mask.shape[1]<attention_mask.shape[1]:
/home/ltalamanova/tmp_venv/lib/python3.11/site-packages/torch/jit/_trace.py:165:UserWarning:The.gradattributeofaTensorthatisnotaleafTensorisbeingaccessed.Its.gradattributewon'tbepopulatedduringautograd.backward().Ifyouindeedwantthe.gradfieldtobepopulatedforanon-leafTensor,use.retain_grad()onthenon-leafTensor.Ifyouaccessthenon-leafTensorbymistake,makesureyouaccesstheleafTensorinstead.Seegithub.com/pytorch/pytorch/pull/30531formoreinformations.(Triggeredinternallyataten/src/ATen/core/TensorBody.h:489.)
ifa.gradisnotNone:


..parsed-literal::

Textdecodersuccessfulyconvertedandsavedtoblip_text_decoder_with_past.xml


RunOpenVINOModel
------------------

`backtotop⬆️<#table-of-contents>`__

PrepareInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Asdiscussedbefore,themodelconsistsofseveralblockswhichcanbe
reusedforbuildingpipelinesfordifferenttasks.Inthediagrambelow,
youcanseehowimagecaptioningworks:

|image0|

Thevisualmodelacceptstheimagepreprocessedby``BlipProcessor``as
inputandproducesimageembeddings,whicharedirectlypassedtothe
textdecoderforgenerationcaptiontokens.Whengenerationisfinished,
outputsequenceoftokensisprovidedto``BlipProcessor``fordecoding
totextusingatokenizer.

Thepipelineforquestionansweringlookssimilar,butwithadditional
questionprocessing.Inthiscase,imageembeddingsandquestion
tokenizedby``BlipProcessor``areprovidedtothetextencoderandthen
multimodalquestionembeddingispassedtothetextdecoderfor
performinggenerationofanswers.

|image1|

ThenextstepisimplementingbothpipelinesusingOpenVINOmodels.

..|image0|image::https://user-images.githubusercontent.com/29454499/221865836-a56da06e-196d-449c-a5dc-4136da6ab5d5.png
..|image1|image::https://user-images.githubusercontent.com/29454499/221868167-d0081add-d9f3-4591-80e7-4753c88c1d0a.png

..code::ipython3

#createOpenVINOCoreobjectinstance
core=ov.Core()

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

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)




..parsed-literal::

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



..code::ipython3

#loadmodelsondevice
ov_vision_model=core.compile_model(VISION_MODEL_OV,device.value)
ov_text_encoder=core.compile_model(TEXT_ENCODER_OV,device.value)
ov_text_decoder_with_past=core.compile_model(TEXT_DECODER_OV,device.value)

..code::ipython3

fromfunctoolsimportpartial
fromblip_modelimporttext_decoder_forward

text_decoder.forward=partial(text_decoder_forward,ov_text_decoder_with_past=ov_text_decoder_with_past)

Themodelhelperclasshastwomethodsforgeneration:
**generate_answer**-usedforvisualquestionanswering,
**generate_caption**-usedforcaptiongeneration.Forinitialization,
modelclassacceptscompiledOpenVINOmodelsforthetextencoder,
visionmodelandtextdecoder,andalsoconfigurationforgenerationand
initialtokenfordecoderwork.

..code::ipython3

ifnotPath("./blip_model.py").exists():
download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/blip-visual-language-processing/blip_model.py")
fromblip_modelimportOVBlipModel

ov_model=OVBlipModel(model.config,model.decoder_start_token_id,ov_vision_model,ov_text_encoder,text_decoder)
out=ov_model.generate_answer(**inputs,max_length=20)

Now,themodelisreadyforgeneration.

ImageCaptioning
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

out=ov_model.generate_caption(inputs["pixel_values"],max_length=20)
caption=processor.decode(out[0],skip_special_tokens=True)
fig=visualize_results(raw_image,caption)



..image::blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_25_0.png


QuestionAnswering
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

start=time.perf_counter()
out=ov_model.generate_answer(**inputs,max_length=20)
end=time.perf_counter()-start
answer=processor.decode(out[0],skip_special_tokens=True)
fig=visualize_results(raw_image,answer,question)



..image::blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_27_0.png


..code::ipython3

print(f"Processingtime:{end:.4f}")


..parsed-literal::

Processingtime:0.1186


OptimizemodelusingNNCF
-------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingthequantizationlayersintothe
modelgraphandthenusingasubsetofthetrainingdatasetto
initializetheparametersoftheseadditionalquantizationlayers.The
frameworkisdesignedsothatmodificationstoyouroriginaltraining
codeareminor.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createadatasetforquantization.
2.Run``nncf.quantize``togetaquantizedmodelfromthepre-trained
``FP16``model.
3.Serializethe``INT8``modelusing``openvino.save_model``function.

..

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.Youcandisable
itusingwidgetbelow:

..code::ipython3

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



..code::ipython3

VISION_MODEL_OV_INT8=Path(str(VISION_MODEL_OV).replace(".xml","_int8.xml"))
TEXT_ENCODER_OV_INT8=Path(str(TEXT_ENCODER_OV).replace(".xml","_int8.xml"))
TEXT_DECODER_OV_INT8=Path(str(TEXT_DECODER_OV).replace(".xml","_int8.xml"))
int8_model=None

#Fetchskip_kernel_extensionmodule
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparedataset
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The`VQAv2<https://visualqa.org/>`__isadatasetcontaining
open-endedquestionsaboutimages.Thesequestionsrequirean
understandingofvision,languageandcommonsenseknowledgetoanswer.

..code::ipython3

%%skipnot$to_quantize.value

importnumpyasnp
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm

defpreprocess_batch(batch,vision_model,inputs_info):
"""
Preprocessesadatasetbatchbyloadingandtransformingimageandtextdata.
VQAv2datasetcontainsmultiplequestionstoimage.
Toreducedatasetpreparationtimewewillstorepreprocessedimagesin`inputs_info`.
"""
image_id=batch["image_id"]
ifimage_idininputs_info:
inputs=processor(text=batch['question'],return_tensors="np")
pixel_values=inputs_info[image_id]["pixel_values"]
encoder_hidden_states=inputs_info[image_id]["encoder_hidden_states"]
else:
inputs=processor(images=batch["image"],text=batch["question"],return_tensors="np")
pixel_values=inputs["pixel_values"]
encoder_hidden_states=vision_model(pixel_values)[vision_model.output(0)]
inputs_info[image_id]={
"pixel_values":pixel_values,
"encoder_hidden_states":encoder_hidden_states,
"text_encoder_inputs":[]
}

text_encoder_inputs={
"input_ids":inputs["input_ids"],
"attention_mask":inputs["attention_mask"]
}
inputs_info[image_id]["text_encoder_inputs"].append(text_encoder_inputs)


defprepare_input_data(dataloader,vision_model,opt_init_steps):
"""
StorecalibrationsubsetinListtoreducequantizationtime.
"""
inputs_info={}
foridx,batchinenumerate(tqdm(dataloader,total=opt_init_steps,desc="Preparecalibrationdata")):
preprocess_batch(batch,vision_model,inputs_info)

calibration_subset=[]
forimage_idininputs_info:
pixel_values=inputs_info[image_id]["pixel_values"]
encoder_hidden_states=inputs_info[image_id]["encoder_hidden_states"]
encoder_attention_mask=np.ones(encoder_hidden_states.shape[:-1],dtype=int)
fortext_encoder_inputsininputs_info[image_id]["text_encoder_inputs"]:
text_encoder_inputs["encoder_hidden_states"]=encoder_hidden_states
text_encoder_inputs["encoder_attention_mask"]=encoder_attention_mask
blip_inputs={
"vision_model_inputs":{"pixel_values":pixel_values},
"text_encoder_inputs":text_encoder_inputs,
}
calibration_subset.append(blip_inputs)
returncalibration_subset


defprepare_dataset(vision_model,opt_init_steps=300,streaming=False):
"""
Preparesavision-textdatasetforquantization.
"""
split=f"train[:{opt_init_steps}]"ifnotstreamingelse"train"
dataset=load_dataset("HuggingFaceM4/VQAv2",split=split,streaming=streaming,trust_remote_code=True)
dataset=dataset.shuffle(seed=42)
ifstreaming:
dataset=dataset.take(opt_init_steps)
calibration_subset=prepare_input_data(dataset,vision_model,opt_init_steps)
returncalibration_subset

Loadingandprocessingthedatasetinstreamingmodemaytakealong
timeanddependsonyourinternetconnection.

..code::ipython3

%%skipnot$to_quantize.value

importnncf

comp_vision_model=core.compile_model(VISION_MODEL_OV,device.value)
calibration_data=prepare_dataset(comp_vision_model)

Quantizevisionmodel
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

vision_dataset=nncf.Dataset(calibration_data,lambdax:x["vision_model_inputs"])
vision_model=core.read_model(VISION_MODEL_OV)

quantized_model=nncf.quantize(
model=vision_model,
calibration_dataset=vision_dataset,
model_type=nncf.ModelType.TRANSFORMER
)

ov.save_model(quantized_model,VISION_MODEL_OV_INT8)



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

INFO:nncf:36ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:48ignorednodeswerefoundbynameintheNNCFGraph



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



Quantizetextencoder
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

text_encoder_dataset=nncf.Dataset(calibration_data,lambdax:x["text_encoder_inputs"])
text_encoder_model=core.read_model(TEXT_ENCODER_OV)

quantized_model=nncf.quantize(
model=text_encoder_model,
calibration_dataset=text_encoder_dataset,
model_type=nncf.ModelType.TRANSFORMER
)
ov.save_model(quantized_model,TEXT_ENCODER_OV_INT8)



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
INFO:nncf:73ignorednodeswerefoundbynameintheNNCFGraph



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



Compressweightsoftextdecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thequantizationofthetextdecoderleadstosignificantaccuracyloss.
Insteadofpost-trainingquantization,wecanusedatafreeweights
compressiontoreducethemodelfootprint.

Theoptimizationprocesscontainsthefollowingsteps:

1.Run``nncf.compress_weights``togetamodelwithcompressedweights.
2.Serializethe``OpenVINO``modelusing``openvino.save_model``
function.

..code::ipython3

%%skipnot$to_quantize.value

text_decoder_model=core.read_model(TEXT_DECODER_OV)
compressed_text_decoder=nncf.compress_weights(text_decoder_model)
ov.save_model(compressed_text_decoder,str(TEXT_DECODER_OV_INT8))


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(124/124)|100%(124/124)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



RunoptimizedOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThestepsformakingpredictionswiththeoptimizedOpenVINOBLIPmodel
aresimilartothePyTorchmodel.Letuscheckthemodelresultusing
thesameinputdatalikeformodelbeforequantization

..code::ipython3

%%skipnot$to_quantize.value

q_ov_vision_model=core.compile_model(VISION_MODEL_OV_INT8,device.value)
q_ov_text_encoder=core.compile_model(TEXT_ENCODER_OV_INT8,device.value)
q_ov_text_decoder_with_past=core.compile_model(TEXT_DECODER_OV_INT8,device.value)

..code::ipython3

%%skipnot$to_quantize.value

fromfunctoolsimportpartial
fromtransformersimportBlipForQuestionAnswering
fromblip_modelimportOVBlipModel,text_decoder_forward

model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
text_decoder=model.text_decoder
text_decoder.eval()

text_decoder.forward=partial(text_decoder_forward,ov_text_decoder_with_past=q_ov_text_decoder_with_past)
int8_model=OVBlipModel(model.config,model.decoder_start_token_id,q_ov_vision_model,q_ov_text_encoder,text_decoder)

..code::ipython3

%%skipnot$to_quantize.value

raw_image=Image.open("demo.jpg").convert('RGB')
question="howmanydogsareinthepicture?"
#preprocessinputdata
inputs=processor(raw_image,question,return_tensors="pt")

Imagecaptioning
^^^^^^^^^^^^^^^^

..code::ipython3

%%skipnot$to_quantize.value

out=int8_model.generate_caption(inputs["pixel_values"],max_length=20)
caption=processor.decode(out[0],skip_special_tokens=True)
fig=visualize_results(raw_image,caption)



..image::blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_47_0.png


Questionanswering
^^^^^^^^^^^^^^^^^^

..code::ipython3

%%skipnot$to_quantize.value

out=int8_model.generate_answer(**inputs,max_length=20)
answer=processor.decode(out[0],skip_special_tokens=True)
fig=visualize_results(raw_image,answer,question)



..image::blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_49_0.png


Comparefilesizes
~~~~~~~~~~~~~~~~~~

..code::ipython3

%%skipnot$to_quantize.value

defcalculate_compression_rate(ov_model_path):
fp16_ir_model_size=Path(ov_model_path).with_suffix(".bin").stat().st_size/1024
int8_model_path=str(ov_model_path).replace(".xml","_int8.xml")
quantized_model_size=Path(int8_model_path).with_suffix(".bin").stat().st_size/1024
print(f'{ov_model_path.as_posix().split(".")[0]}')
print(f"*FP16IRmodelsize:{fp16_ir_model_size:.2f}KB")
print(f"*INT8modelsize:{quantized_model_size:.2f}KB")
print(f"*Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")

..code::ipython3

%%skipnot$to_quantize.value

forfp16_pathin[VISION_MODEL_OV,TEXT_ENCODER_OV,TEXT_DECODER_OV]:
calculate_compression_rate(fp16_path)


..parsed-literal::

blip_vision_model
*FP16IRmodelsize:168145.70KB
*INT8modelsize:84915.48KB
*Modelcompressionrate:1.980
blip_text_encoder
*FP16IRmodelsize:268087.16KB
*INT8modelsize:134676.75KB
*Modelcompressionrate:1.991
blip_text_decoder_with_past
*FP16IRmodelsize:269303.35KB
*INT8modelsize:135302.53KB
*Modelcompressionrate:1.990


CompareinferencetimeoftheFP16andoptimizedmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
models,weusemedianinferencetimeon100samplesofthecalibration
dataset.Sowecanapproximatelyestimatethespeedupofthedynamic
quantizedmodels.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplicationswithstaticshapes.

..code::ipython3

%%skipnot$to_quantize.value

importtime
importtorch

defcalculate_inference_time(blip_model,calibration_data,generate_caption):
inference_time=[]
forinputsincalibration_data:
pixel_values=torch.from_numpy(inputs["vision_model_inputs"]["pixel_values"])
input_ids=torch.from_numpy(inputs["text_encoder_inputs"]["input_ids"])
attention_mask=torch.from_numpy(inputs["text_encoder_inputs"]["attention_mask"])

start=time.perf_counter()
ifgenerate_caption:
_=blip_model.generate_caption(pixel_values,max_length=20)
else:
_=blip_model.generate_answer(pixel_values=pixel_values,input_ids=input_ids,attention_mask=attention_mask,max_length=20)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_original_model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
fp_text_decoder=fp_original_model.text_decoder
fp_text_decoder.eval()

comp_text_encoder=core.compile_model(TEXT_ENCODER_OV,device.value)
comp_text_decoder_with_past=core.compile_model(TEXT_DECODER_OV,device.value)
fp_text_decoder.forward=partial(text_decoder_forward,ov_text_decoder_with_past=comp_text_decoder_with_past)
fp16_model=OVBlipModel(model.config,model.decoder_start_token_id,comp_vision_model,comp_text_encoder,fp_text_decoder)

..code::ipython3

%%skipnot$to_quantize.value

validation_data=calibration_data[:100]

int8_caption_latency=calculate_inference_time(int8_model,validation_data,generate_caption=True)
fp16_caption_latency=calculate_inference_time(fp16_model,validation_data,generate_caption=True)

print(f"ImageCaptioningspeedup:{fp16_caption_latency/int8_caption_latency:.3f}")


..parsed-literal::

ImageCaptioningspeedup:1.254


..code::ipython3

%%skipnot$to_quantize.value

int8_generate_answer_latency=calculate_inference_time(int8_model,validation_data,generate_caption=False)
fp16_generate_answer_latency=calculate_inference_time(fp16_model,validation_data,generate_caption=False)
print(f"QuestionAnsweringspeedup:{fp16_generate_answer_latency/int8_generate_answer_latency:.3f}")


..parsed-literal::

QuestionAnsweringspeedup:1.715


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

use_quantized_model=widgets.Checkbox(
description="Usequantizedmodel",
value=int8_modelisnotNone,
disabled=int8_modelisNone,
)

use_quantized_model




..parsed-literal::

Checkbox(value=True,description='Usequantizedmodel')



..code::ipython3

importgradioasgr

ov_model=int8_modelifuse_quantized_model.valueelseov_model


defgenerate_answer(img,question):
ifimgisNone:
raisegr.Error("Pleaseuploadanimageorchooseonefromtheexampleslist")
start=time.perf_counter()
inputs=processor(img,question,return_tensors="pt")
output=ov_model.generate_answer(**inputs,max_length=20)iflen(question)elseov_model.generate_caption(inputs["pixel_values"],max_length=20)
answer=processor.decode(output[0],skip_special_tokens=True)
elapsed=time.perf_counter()-start
html=f"<p>Processingtime:{elapsed:.4f}</p>"
returnanswer,html


demo=gr.Interface(
generate_answer,
[
gr.Image(label="Image"),
gr.Textbox(
label="Question",
info="Ifthisfieldisempty,animagecaptionwillbegenerated",
),
],
[gr.Text(label="Answer"),gr.HTML()],
examples=[["demo.jpg",""],["demo.jpg",question]],
allow_flagging="never",
)
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

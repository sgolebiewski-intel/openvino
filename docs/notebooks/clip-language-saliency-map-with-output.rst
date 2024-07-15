Language-VisualSaliencywithCLIPandOpenVINO™
================================================

Thenotebookwillcoverthefollowingtopics:

-Explanationofa*saliencymap*andhowitcanbeused.
-OverviewoftheCLIPneuralnetworkanditsusageingenerating
saliencymaps.
-Howtosplitaneuralnetworkintopartsforseparateinference.
-HowtospeedupinferencewithOpenVINO™andasynchronousexecution.

SaliencyMap
------------

Asaliencymapisavisualizationtechniquethathighlightsregionsof
interestinanimage.Forexample,itcanbeusedto`explainimage
classification
predictions<https://academic.oup.com/mnras/article/511/4/5032/6529251#389668570>`__
foraparticularlabel.Hereisanexampleofasaliencymapthatyou
willgetinthisnotebook:

|image0|

CLIP
----

WhatIsCLIP?
~~~~~~~~~~~~~

CLIP(ContrastiveLanguage–ImagePre-training)isaneuralnetworkthat
canworkwithbothimagesandtexts.Ithasbeentrainedtopredict
whichrandomlysampledtextsnippetsareclosetoagivenimage,meaning
thatatextbetterdescribestheimage.Hereisavisualizationofthe
pre-trainingprocess:

|image1|`image_source<https://openai.com/blog/clip/>`__

Tosolvethetask,CLIPusestwoparts:``ImageEncoder``and
``TextEncoder``.Bothpartsareusedtoproduceembeddings,whichare
vectorsoffloating-pointnumbers,forimagesandtexts,respectively.
Giventwovectors,onecandefineandmeasurethesimilaritybetween
them.Apopularmethodtodosoisthe``cosine_similarity``,whichis
definedasthedotproductofthetwovectorsdividedbytheproductof
theirnorms:

..figure::https://user-images.githubusercontent.com/29454499/218972165-f61a82f2-9711-4ce6-84b5-58fdd1d80d10.png
:alt:cs

cs

Theresultcanrangefrom:math:`-1`to:math:`1`.Avalue:math:`1`
meansthatthevectorsaresimilar,:math:`0`meansthatthevectorsare
not“connected”atall,and:math:`-1`isforvectorswithsomehow
opposite“meaning”.TotrainCLIP,OpenAIusessamplesoftextsand
imagesandorganizesthemsothatthefirsttextcorrespondstothe
firstimageinthebatch,thesecondtexttothesecondimage,andso
on.Then,cosinesimilaritiesaremeasuredbetweenalltextsandall
images,andtheresultsareputinamatrix.Ifthematrixhasnumbers
closeto:math:`1`onadiagonalandcloseto:math:`0`elsewhere,it
indicatesthatthenetworkisappropriatelytrained.

HowtoBuildaSaliencyMapwithCLIP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ProvidinganimageandatexttoCLIPreturnstwovectors.Thecosine
similaritybetweenthesevectorsiscalculated,resultinginanumber
between:math:`-1`and:math:`1`thatindicateswhetherthetext
describestheimageornot.Theideaisthat*someregionsoftheimage
areclosertothetextquery*thanothers,andthisdifferencecanbe
usedtobuildthesaliencymap.Hereishowitcanbedone:

1.Compute``query``and``image``similarity.Thiswillrepresentthe
*neutralvalue*:math:`s_0`onthe``saliencymap``.
2.Getarandom``crop``oftheimage.
3.Compute``crop``and``query``similarity.
4.Subtractthe:math:`s_0`fromit.Ifthevalueispositive,the
``crop``isclosertothe``query``,anditshouldbearedregionon
thesaliencymap.Ifnegative,itshouldbeblue.
5.Updatethecorrespondingregiononthe``saliencymap``.
6.Repeatsteps2-5multipletimes(``n_iters``).

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`InitialImplementationwithTransformersand
Pytorch<#initial-implementation-with-transformers-and-pytorch>`__
-`SeparateTextandVisual
Processing<#separate-text-and-visual-processing>`__
-`ConverttoOpenVINO™IntermediateRepresentation(IR)
Format<#convert-to-openvino-intermediate-representation-ir-format>`__
-`InferencewithOpenVINO™<#inference-with-openvino>`__

-`Selectinferencedevice<#select-inference-device>`__

-`AccelerateInferencewith
AsyncInferQueue<#accelerate-inference-with-asyncinferqueue>`__
-`PackthePipelineintoa
Function<#pack-the-pipeline-into-a-function>`__
-`InteractivedemowithGradio<#interactive-demo-with-gradio>`__
-`WhatToDoNext<#what-to-do-next>`__

..|image0|image::https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG
..|image1|image::https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg

InitialImplementationwithTransformersandPytorch
----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Installrequirements
%pipinstall-q"openvino>=2023.1.0"
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cputransformers"torch>=2.1""gradio>=4.19"

..code::ipython3

frompathlibimportPath
fromtypingimportTuple,Union,Optional
importrequests

frommatplotlibimportcolors
importmatplotlib.pyplotasplt
importnumpyasnp
importtorch
importtqdm
fromPILimportImage
fromtransformersimportCLIPModel,CLIPProcessor


..parsed-literal::

2023-09-1214:10:49.435909:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-09-1214:10:49.470573:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-09-1214:10:50.130215:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


TogettheCLIPmodel,youwillusethe``transformers``libraryandthe
official``openai/clip-vit-base-patch16``fromOpenAI.Youcanuseany
CLIPmodelfromtheHuggingFaceHubbysimplyreplacingamodel
checkpointinthecellbelow.

Thereareseveralpreprocessingstepsrequiredtogettextandimage
datatothemodel.Imageshavetoberesized,cropped,andnormalized,
andtextmustbesplitintotokensandswappedbytokenIDs.Todothat,
youwilluse``CLIPProcessor``,whichencapsulatesallthepreprocessing
steps.

..code::ipython3

model_checkpoint="openai/clip-vit-base-patch16"

model=CLIPModel.from_pretrained(model_checkpoint).eval()
processor=CLIPProcessor.from_pretrained(model_checkpoint)

Letuswritehelperfunctionsfirst.Youwillgeneratecropcoordinates
andsizewith``get_random_crop_params``,andgettheactualcropwith
``get_crop_image``.Toupdatethesaliencymapwiththecalculated
similarity,youwilluse``update_saliency_map``.A
``cosine_similarity``functionisjustacoderepresentationofthe
formulaabove.

..code::ipython3

defget_random_crop_params(image_height:int,image_width:int,min_crop_size:int)->Tuple[int,int,int,int]:
crop_size=np.random.randint(min_crop_size,min(image_height,image_width))
x=np.random.randint(image_width-crop_size+1)
y=np.random.randint(image_height-crop_size+1)
returnx,y,crop_size


defget_cropped_image(im_tensor:np.array,x:int,y:int,crop_size:int)->np.array:
returnim_tensor[y:y+crop_size,x:x+crop_size,...]


defupdate_saliency_map(saliency_map:np.array,similarity:float,x:int,y:int,crop_size:int)->None:
saliency_map[
y:y+crop_size,
x:x+crop_size,
]+=similarity


defcosine_similarity(one:Union[np.ndarray,torch.Tensor],other:Union[np.ndarray,torch.Tensor])->Union[np.ndarray,torch.Tensor]:
returnone@other.T/(np.linalg.norm(one)*np.linalg.norm(other))

Parameterstobedefined:

-``n_iters``-numberoftimestheprocedurewillberepeated.Larger
isbetter,butwillrequiremoretimetoinference
-``min_crop_size``-minimumsizeofthecropwindow.Asmallersize
willincreasetheresolutionofthesaliencymapbutmayrequiremore
iterations
-``query``-textthatwillbeusedtoquerytheimage
-``image``-theactualimagethatwillbequeried.Youwilldownload
theimagefromalink

Theimageatthebeginningwasacquiredwith``n_iters=2000``and
``min_crop_size=50``.Youwillstartwiththelowernumberofinferences
togettheresultfaster.Itisrecommendedtoexperimentwiththe
parametersattheend,whenyougetanoptimizedmodel.

..code::ipython3

n_iters=300
min_crop_size=50

query="WhodevelopedtheTheoryofGeneralRelativity?"
image_path=Path("example.jpg")

r=requests.get("https://www.storypick.com/wp-content/uploads/2016/01/AE-2.jpg")

withimage_path.open("wb")asf:
f.write(r.content)
image=Image.open(image_path)
im_tensor=np.array(image)

x_dim,y_dim=image.size

Giventhe``model``and``processor``,theactualinferenceissimple:
transformthetextandimageintocombined``inputs``andpassittothe
model:

..code::ipython3

inputs=processor(text=[query],images=[im_tensor],return_tensors="pt")
withtorch.no_grad():
results=model(**inputs)
results.keys()




..parsed-literal::

odict_keys(['logits_per_image','logits_per_text','text_embeds','image_embeds','text_model_output','vision_model_output'])



Themodelproducesseveraloutputs,butforyourapplication,youare
interestedin``text_embeds``and``image_embeds``,whicharethe
vectorsfortextandimage,respectively.Now,youcancalculate
``initial_similarity``betweenthe``query``andthe``image``.Youalso
initializeasaliencymap.Numbersinthecommentscorrespondtothe
itemsinthe“HowToBuildaSaliencyMapWithCLIP?”listabove.

..code::ipython3

initial_similarity=cosine_similarity(results.text_embeds,results.image_embeds).item()#1.Computingqueryandimagesimilarity
saliency_map=np.zeros((y_dim,x_dim))

for_intqdm.notebook.tqdm(range(n_iters)):#6.Settingnumberoftheprocedureiterations
x,y,crop_size=get_random_crop_params(y_dim,x_dim,min_crop_size)
im_crop=get_cropped_image(im_tensor,x,y,crop_size)#2.Gettingarandomcropoftheimage

inputs=processor(text=[query],images=[im_crop],return_tensors="pt")
withtorch.no_grad():
results=model(**inputs)#3.Computingcropandquerysimilarity

similarity=(
cosine_similarity(results.text_embeds,results.image_embeds).item()-initial_similarity
)#4.Subtractingqueryandimagesimilarityfromcropandquerysimilarity
update_saliency_map(saliency_map,similarity,x,y,crop_size)#5.Updatingtheregiononthesaliencymap



..parsed-literal::

0%||0/300[00:00<?,?it/s]


Tovisualizetheresultingsaliencymap,youcanuse``matplotlib``:

..code::ipython3

plt.figure(dpi=150)
plt.imshow(saliency_map,norm=colors.TwoSlopeNorm(vcenter=0),cmap="jet")
plt.colorbar(location="bottom")
plt.title(f'Query:"{query}"')
plt.axis("off")
plt.show()



..image::clip-language-saliency-map-with-output_files/clip-language-saliency-map-with-output_15_0.png


Theresultmapisnotassmoothasintheexamplepicturebecauseofthe
lowernumberofiterations.However,thesameredandblueareasare
clearlyvisible.

Letusoverlaythesaliencymapontheimage:

..code::ipython3

defplot_saliency_map(image_tensor:np.ndarray,saliency_map:np.ndarray,query:Optional[str])->None:
fig=plt.figure(dpi=150)
plt.imshow(image_tensor)
plt.imshow(
saliency_map,
norm=colors.TwoSlopeNorm(vcenter=0),
cmap="jet",
alpha=0.5,#makesaliencymaptrasparenttoseeoriginalpicture
)
ifquery:
plt.title(f'Query:"{query}"')
plt.axis("off")
returnfig


plot_saliency_map(im_tensor,saliency_map,query);



..image::clip-language-saliency-map-with-output_files/clip-language-saliency-map-with-output_17_0.png


SeparateTextandVisualProcessing
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

Thecodeaboveisfunctional,buttherearesomerepeatedcomputations
thatcanbeavoided.Thetextembeddingcanbecomputedoncebecauseit
doesnotdependontheinputimage.Thisseparationwillalsobeuseful
inthefuture.Theinitialpreparationwillremainthesamesinceyou
stillneedtocomputethesimilaritybetweenthetextandthefull
image.Afterthat,the``get_image_features``methodcouldbeusedto
obtainembeddingsforthecroppedimages.

..code::ipython3

inputs=processor(text=[query],images=[im_tensor],return_tensors="pt")
withtorch.no_grad():
results=model(**inputs)
text_embeds=results.text_embeds#savetextembeddingstousethemlater

initial_similarity=cosine_similarity(text_embeds,results.image_embeds).item()
saliency_map=np.zeros((y_dim,x_dim))

for_intqdm.notebook.tqdm(range(n_iters)):
x,y,crop_size=get_random_crop_params(y_dim,x_dim,min_crop_size)
im_crop=get_cropped_image(im_tensor,x,y,crop_size)

image_inputs=processor(images=[im_crop],return_tensors="pt")#croppreprocessing
withtorch.no_grad():
image_embeds=model.get_image_features(**image_inputs)#calculateimageembeddingsonly

similarity=cosine_similarity(text_embeds,image_embeds).item()-initial_similarity
update_saliency_map(saliency_map,similarity,x,y,crop_size)

plot_saliency_map(im_tensor,saliency_map,query);



..parsed-literal::

0%||0/300[00:00<?,?it/s]



..image::clip-language-saliency-map-with-output_files/clip-language-saliency-map-with-output_19_1.png


Theresultmightbeslightlydifferentbecauseyouuserandomcropsto
buildasaliencymap.

ConverttoOpenVINO™IntermediateRepresentation(IR)Format
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Theprocessofbuildingasaliencymapcanbequitetime-consuming.To
speeditup,youwilluseOpenVINO.OpenVINOisaninferenceframework
designedtorunpre-trainedneuralnetworksefficiently.Onewaytouse
itistoconvertamodelfromitsoriginalframeworkrepresentationto
anOpenVINOIntermediateRepresentation(IR)formatandthenloaditfor
inference.ThemodelcurrentlyusesPyTorch.TogetanIR,youneedto
useModelConversionAPI.``ov.convert_model``functionacceptsPyTorch
modelobjectandexampleinputandconvertsittoOpenVINOModel
instance,thatreadytoloadondeviceusing``ov.compile_model``orcan
besavedondiskusing``ov.save_model``.Toseparatemodelontextand
imageparts,weoverloadforwardmethodwith``get_text_features``and
``get_image_features``methodsrespectively.Internally,PyTorch
conversiontoOpenVINOinvolvesTorchScripttracing.Forachieving
betterconversionresults,weneedtoguaranteethatmodelcanbe
successfullytraced.``model.config.torchscript=True``parameters
allowstoprepareHuggingFacemodelsforTorchScripttracing.More
detailsaboutthatcanbefoundinHuggingFaceTransformers
`documentation<https://huggingface.co/docs/transformers/torchscript>`__

..code::ipython3

importopenvinoasov

model_name=model_checkpoint.split("/")[-1]

model.config.torchscript=True
model.forward=model.get_text_features
text_ov_model=ov.convert_model(
model,
example_input={
"input_ids":inputs.input_ids,
"attention_mask":inputs.attention_mask,
},
)

#getimagesizeafterpreprocessingfromtheprocessor
crops_info=processor.image_processor.crop_size.values()ifhasattr(processor,"image_processor")elseprocessor.feature_extractor.crop_size.values()
model.forward=model.get_image_features
image_ov_model=ov.convert_model(
model,
example_input={"pixel_values":inputs.pixel_values},
input=[1,3,*crops_info],
)

ov_dir=Path("ir")
ov_dir.mkdir(exist_ok=True)
text_model_path=ov_dir/f"{model_name}_text.xml"
image_model_path=ov_dir/f"{model_name}_image.xml"

#writeresultingmodelsondisk
ov.save_model(text_ov_model,text_model_path)
ov.save_model(image_ov_model,image_model_path)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)
huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)
huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

NoCUDAruntimeisfound,usingCUDA_HOME='/usr/local/cuda'
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:287:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz*self.num_heads,tgt_len,src_len):
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:295:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_attention_mask.size()!=(bsz,1,tgt_len,src_len):
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:304:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.size()!=(bsz,1,tgt_len,src_len):
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:327:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):


Now,youhavetwoseparatemodelsfortextandimages,storedondisk
andreadytobeloadedandinferredwithOpenVINO™.

InferencewithOpenVINO™
------------------------

`backtotop⬆️<#table-of-contents>`__

1.Createaninstanceofthe``Core``objectthatwillhandleany
interactionwithOpenVINOruntimeforyou.
2.Usethe``core.read_model``methodtoloadthemodelintomemory.
3.Compilethemodelwiththe``core.compile_model``methodfora
particulardevicetoapplydevice-specificoptimizations.
4.Usethecompiledmodelforinference.

..code::ipython3

core=ov.Core()

text_model=core.read_model(text_model_path)
image_model=core.read_model(image_model_path)

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

Dropdown(description='Device:',index=2,options=('CPU','GPU','AUTO'),value='AUTO')



..code::ipython3

text_model=core.compile_model(model=text_model,device_name=device.value)
image_model=core.compile_model(model=image_model,device_name=device.value)

OpenVINOsupports``numpy.ndarray``asaninputtype,soyouchangethe
``return_tensors``to``np``.Youalsoconvertatransformers’
``BatchEncoding``objecttoapythondictionarywithinputnamesaskeys
andinputtensorsforvalues.

Onceyouhaveacompiledmodel,theinferenceissimilartoPytorch-a
compiledmodeliscallable.Justpassinputdatatoit.Inference
resultsarestoredinthedictionary.Onceyouhaveacompiledmodel,
theinferenceprocessismostlysimilar.

..code::ipython3

text_inputs=dict(processor(text=[query],images=[im_tensor],return_tensors="np"))
image_inputs=text_inputs.pop("pixel_values")

text_embeds=text_model(text_inputs)[0]
image_embeds=image_model(image_inputs)[0]

initial_similarity=cosine_similarity(text_embeds,image_embeds)
saliency_map=np.zeros((y_dim,x_dim))

for_intqdm.notebook.tqdm(range(n_iters)):
x,y,crop_size=get_random_crop_params(y_dim,x_dim,min_crop_size)
im_crop=get_cropped_image(im_tensor,x,y,crop_size)

image_inputs=processor(images=[im_crop],return_tensors="np").pixel_values
image_embeds=image_model(image_inputs)[image_model.output()]

similarity=cosine_similarity(text_embeds,image_embeds)-initial_similarity
update_saliency_map(saliency_map,similarity,x,y,crop_size)

plot_saliency_map(im_tensor,saliency_map,query);



..parsed-literal::

0%||0/300[00:00<?,?it/s]



..image::clip-language-saliency-map-with-output_files/clip-language-saliency-map-with-output_29_1.png


AccelerateInferencewith``AsyncInferQueue``
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Upuntilnow,thepipelinewassynchronous,whichmeansthatthedata
preparation,modelinputpopulation,modelinference,andoutput
processingissequential.Thatisasimple,butnotthemosteffective
waytoorganizeaninferencepipelineinyourcase.Toutilizethe
availableresourcesmoreefficiently,youwilluse``AsyncInferQueue``.
Itcanbeinstantiatedwithcompiledmodelandanumberofjobs-
parallelexecutionthreads.Ifyoudonotpassanumberofjobsorpass
``0``,thenOpenVINOwillpicktheoptimalnumberbasedonyourdevice
andheuristics.Afteracquiringtheinferencequeue,youhavetwojobs
todo:

-Preprocessthedataandpushittotheinferencequeue.The
preprocessingstepswillremainthesame
-Telltheinferencequeuewhattodowiththemodeloutputafterthe
inferenceisfinished.Itisrepresentedbyapythonfunctioncalled
``callback``thattakesaninferenceresultanddatathatyoupassed
totheinferencequeuealongwiththepreparedinputdata

Everythingelsewillbehandledbythe``AsyncInferQueue``instance.

Thereisanotherlow-hangingbitofoptimization.Youareexpectingmany
inferencerequestsforyourimagemodelatonceandwantthemodelto
processthemasfastaspossible.Inotherwords-maximizethe
**throughput**.Todothat,youcanrecompilethemodelgivingitthe
performancehint.

..code::ipython3

fromtypingimportDict,Any


image_model=core.read_model(image_model_path)

image_model=core.compile_model(
model=image_model,
device_name=device.value,
config={"PERFORMANCE_HINT":"THROUGHPUT"},
)

..code::ipython3

text_inputs=dict(processor(text=[query],images=[im_tensor],return_tensors="np"))
image_inputs=text_inputs.pop("pixel_values")

text_embeds=text_model(text_inputs)[text_model.output()]
image_embeds=image_model(image_inputs)[image_model.output()]

initial_similarity=cosine_similarity(text_embeds,image_embeds)
saliency_map=np.zeros((y_dim,x_dim))

Yourcallbackshoulddothesamethingthatyoudidafterinferencein
thesyncmode:

-Pulltheimageembeddingsfromaninferencerequest.
-Computecosinesimilaritybetweentextandimageembeddings.
-Updatesaliencymapbased.

Ifyoudonotchangetheprogressbar,itwillshowtheprogressof
pushingdatatotheinferencequeue.Totracktheactualprogress,you
shouldpassaprogressbarobjectandcall``update``methodafter
``update_saliency_map``call.

..code::ipython3

defcompletion_callback(
infer_request:ov.InferRequest,#inferenteresult
user_data:Dict[str,Any],#datathatyoupassedalongwithinputpixelvalues
)->None:
pbar=user_data.pop("pbar")

image_embeds=infer_request.get_output_tensor().data
similarity=cosine_similarity(user_data.pop("text_embeds"),image_embeds)-user_data.pop("initial_similarity")
update_saliency_map(**user_data,similarity=similarity)

pbar.update(1)#updatetheprogressbar


infer_queue=ov.AsyncInferQueue(image_model)
infer_queue.set_callback(completion_callback)

..code::ipython3

definfer(
im_tensor,
x_dim,
y_dim,
text_embeds,
image_embeds,
initial_similarity,
saliency_map,
query,
n_iters,
min_crop_size,
_tqdm=tqdm.notebook.tqdm,
include_query=True,
):
with_tqdm(total=n_iters)aspbar:
for_inrange(n_iters):
x,y,crop_size=get_random_crop_params(y_dim,x_dim,min_crop_size)
im_crop=get_cropped_image(im_tensor,x,y,crop_size)

image_inputs=processor(images=[im_crop],return_tensors="np")

#pushdatatothequeue
infer_queue.start_async(
#passinferencedataasusual
image_inputs.pixel_values,
#thedatathatwillbepassedtothecallbackaftertheinferencecomplete
{
"text_embeds":text_embeds,
"saliency_map":saliency_map,
"initial_similarity":initial_similarity,
"x":x,
"y":y,
"crop_size":crop_size,
"pbar":pbar,
},
)

#afteryoupushedalldatatothequeueyouwaituntilallcallbacksfinished
infer_queue.wait_all()

returnplot_saliency_map(im_tensor,saliency_map,queryifinclude_queryelseNone)


infer(
im_tensor,
x_dim,
y_dim,
text_embeds,
image_embeds,
initial_similarity,
saliency_map,
query,
n_iters,
min_crop_size,
_tqdm=tqdm.notebook.tqdm,
include_query=True,
);



..parsed-literal::

0%||0/300[00:00<?,?it/s]



..image::clip-language-saliency-map-with-output_files/clip-language-saliency-map-with-output_35_1.png


PackthePipelineintoaFunction
---------------------------------

`backtotop⬆️<#table-of-contents>`__

Letuswrapallcodeinthefunctionandaddauserinterfacetoit.

..code::ipython3

importipywidgetsaswidgets


defbuild_saliency_map(
image:Image,
query:str,
n_iters:int=n_iters,
min_crop_size=min_crop_size,
_tqdm=tqdm.notebook.tqdm,
include_query=True,
):
x_dim,y_dim=image.size
im_tensor=np.array(image)

text_inputs=dict(processor(text=[query],images=[im_tensor],return_tensors="np"))
image_inputs=text_inputs.pop("pixel_values")

text_embeds=text_model(text_inputs)[text_model.output()]
image_embeds=image_model(image_inputs)[image_model.output()]

initial_similarity=cosine_similarity(text_embeds,image_embeds)
saliency_map=np.zeros((y_dim,x_dim))

returninfer(
im_tensor,
x_dim,
y_dim,
text_embeds,
image_embeds,
initial_similarity,
saliency_map,
query,
n_iters,
min_crop_size,
_tqdm=_tqdm,
include_query=include_query,
)

Thefirstversionwillenablepassingalinktotheimage,asyouhave
donesofarinthenotebook.

..code::ipython3

n_iters_widget=widgets.BoundedIntText(
value=n_iters,
min=1,
max=10000,
description="n_iters",
)
min_crop_size_widget=widgets.IntSlider(
value=min_crop_size,
min=1,
max=200,
description="min_crop_size",
)


@widgets.interact_manual(image_link="",query="",n_iters=n_iters_widget,min_crop_size=min_crop_size_widget)
defbuild_saliency_map_from_image_link(
image_link:str,
query:str,
n_iters:int,
min_crop_size:int,
)->None:
try:
image_bytes=requests.get(image_link,stream=True).raw
exceptrequests.RequestExceptionase:
print(f"Cannotloadimagefromlink:{image_link}\nException:{e}")
return

image=Image.open(image_bytes)
image=image.convert("RGB")#removetransparencychannelorconvertgrayscale1channelto3channels

build_saliency_map(image,query,n_iters,min_crop_size)



..parsed-literal::

interactive(children=(Text(value='',continuous_update=False,description='image_link'),Text(value='',contin…


Thesecondversionwillenableloadingtheimagefromyourcomputer.

..code::ipython3

importio


load_file_widget=widgets.FileUpload(
accept="image/*",
multiple=False,
description="Imagefile",
)


@widgets.interact_manual(
file=load_file_widget,
query="",
n_iters=n_iters_widget,
min_crop_size=min_crop_size_widget,
)
defbuild_saliency_map_from_file(
file:Path,
query:str="",
n_iters:int=2000,
min_crop_size:int=50,
)->None:
image_bytes=io.BytesIO(file[0]["content"])
try:
image=Image.open(image_bytes)
exceptExceptionase:
print(f"Cannotloadtheimage:{e}")
return

image=image.convert("RGB")

build_saliency_map(image,query,n_iters,min_crop_size)



..parsed-literal::

interactive(children=(FileUpload(value=(),accept='image/*',description='Imagefile'),Text(value='',continu…


InteractivedemowithGradio
----------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


def_process(image,query,n_iters,min_crop_size,_=gr.Progress(track_tqdm=True)):
saliency_map=build_saliency_map(image,query,n_iters,min_crop_size,_tqdm=tqdm.tqdm,include_query=False)

returnsaliency_map


demo=gr.Interface(
_process,
[
gr.Image(label="Image",type="pil"),
gr.Textbox(label="Query"),
gr.Slider(1,10000,n_iters,label="Numberofiterations"),
gr.Slider(1,200,min_crop_size,label="Minimumcropsize"),
],
gr.Plot(label="Result"),
examples=[[image_path,query]],
)
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


WhatToDoNext
---------------

`backtotop⬆️<#table-of-contents>`__

Nowthatyouhaveaconvenientinterfaceandacceleratedinference,you
canexploretheCLIPcapabilitiesfurther.Forexample:

-CanCLIPread?Canitdetecttextregionsingeneralandspecific
wordsontheimage?
-WhichfamouspeopleandplacesdoesCLIPknow?
-CanCLIPidentifyplacesonamap?Orplanets,stars,and
constellations?
-ExploredifferentCLIPmodelsfromHuggingFaceHub:justchangethe
``model_checkpoint``atthebeginningofthenotebook.
-Addbatchprocessingtothepipeline:modify
``get_random_crop_params``,``get_cropped_image``and
``update_saliency_map``functionstoprocessmultiplecropimagesat
onceandacceleratethepipelineevenmore.
-Optimizemodelswith
`NNCF<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__
togetfurtheracceleration.Youcanfindexamplehowtoquantize
CLIPmodelin`this
notebook<../clip-zero-shot-image-classification>`__

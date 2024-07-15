Visual-languageassistantwithLLaVANextandOpenVINO
======================================================

`LLaVA-NeXT<https://llava-vl.github.io/blog/2024-01-30-llava-next/>`__
isnewgenerationofLLaVAmodelfamilythatmarksbreakthroughin
advancedlanguagereasoningoverimages,introducingimprovedOCRand
expandedworldknowledge.`LLaVA<https://llava-vl.github.io>`__(Large
LanguageandVisionAssistant)islargemultimodalmodelthataimsto
developageneral-purposevisualassistantthatcanfollowbothlanguage
andimageinstructionstocompletevariousreal-worldtasks.Theideais
tocombinethepoweroflargelanguagemodels(LLMs)withvision
encoderslikeCLIPtocreateanend-to-endtrainedneuralassistantthat
understandsandactsuponmultimodalinstructions.

InthistutorialweconsiderhowtoconvertandoptimizeLLaVA-NeXT
modelfromTransformerslibraryforcreatingmultimodalchatbot.Wewill
utilizethepowerof
`llava-v1.6-mistral-7b<https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf>`__
modelforcreatingmultimodalchatbot,butthesimilaractionsarealso
applicabletoothermodelsofLLaVAfamilycompatiblewithHuggingFace
transformersimplementation.Additionally,wedemonstratehowtoapply
statefultransformationonLLMpartandmodeloptimizationtechniques
likeweightscompressionandquantizationusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`DownloadPyTorchmodel<#download-pytorch-model>`__
-`ConvertmodeltoOpenVINOIntermediate
Representation<#convert-model-to-openvino-intermediate-representation>`__

-`ImageEncoder<#image-encoder>`__
-`TextEmbedding<#text-embedding>`__
-`LanguageModel<#language-model>`__

-`CompressLanguageModelWeightsto4
bits<#compress-language-model-weights-to-4-bits>`__
-`QuantizeImageEncoderto8
bits<#quantize-image-encoder-to-8-bits>`__

-`Preparedatasets<#prepare-datasets>`__
-`Performquantization<#perform-quantization>`__

-`Preparemodelinference
pipeline<#prepare-model-inference-pipeline>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectdevice<#select-device>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2024.0.0""nncf>=2.9.0""torch>=2.1""transformers>=4.39.1""accelerate""pillow""gradio>=4.26""datasets>=2.14.6""tqdm"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

MODEL_DIR=Path("model")
IMAGE_ENCODER_PATH=MODEL_DIR/"image_encoder.xml"
INPUT_EMBEDDING_PATH=MODEL_DIR/"input_embeddings.xml"
LANGUAGE_MODEL_PATH=MODEL_DIR/"language_model.xml"

requires_pt_model_loading=notall([p.exists()forpin[IMAGE_ENCODER_PATH,INPUT_EMBEDDING_PATH,LANGUAGE_MODEL_PATH]])

DownloadPyTorchmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtransformersimportLlavaNextProcessor,LlavaNextForConditionalGeneration
importtorch
importgc

processor=LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
image_encoder_model,input_embedding_model,language_model=None,None,None


classImageEncoder(torch.nn.Module):
def__init__(self,config,vision_tower,multi_modal_projector):
super().__init__()
self.config=config
self.vision_tower=vision_tower
self.multi_modal_projector=multi_modal_projector

defforward(self,pixel_values):
batch_size,num_patches,num_channels,height,width=pixel_values.shape
reshaped_pixel_values=pixel_values.view(batch_size*num_patches,num_channels,height,width)
image_features=self.vision_tower(reshaped_pixel_values,output_hidden_states=True)
selected_image_feature=image_features.hidden_states[self.config.vision_feature_layer]
ifself.config.vision_feature_select_strategy=="default":
selected_image_feature=selected_image_feature[:,1:]
elifself.config.vision_feature_select_strategy=="full":
selected_image_feature=selected_image_feature
image_features=self.multi_modal_projector(selected_image_feature)
returnimage_features


ifrequires_pt_model_loading:
model=LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",low_cpu_mem_usage=True)
model.config.save_pretrained(MODEL_DIR)
image_encoder_model=ImageEncoder(model.config,model.vision_tower,model.multi_modal_projector)
input_embedding_model=input_embedding_model=model.get_input_embeddings()
language_model=model.language_model
delmodel
gc.collect()


..parsed-literal::

2024-04-0412:27:23.875042:Itensorflow/core/util/port.cc:111]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-04-0412:27:23.877406:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-0412:27:23.907479:Etensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342]UnabletoregistercuDNNfactory:AttemptingtoregisterfactoryforplugincuDNNwhenonehasalreadybeenregistered
2024-04-0412:27:23.907505:Etensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609]UnabletoregistercuFFTfactory:AttemptingtoregisterfactoryforplugincuFFTwhenonehasalreadybeenregistered
2024-04-0412:27:23.907525:Etensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518]UnabletoregistercuBLASfactory:AttemptingtoregisterfactoryforplugincuBLASwhenonehasalreadybeenregistered
2024-04-0412:27:23.913713:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-0412:27:23.914384:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-04-0412:27:24.847675:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


OpenVINO##ConvertmodeltoOpenVINOIntermediateRepresentation`back
totop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversiontoOpenVINOIntermediate
Representation(IR).`OpenVINOmodelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
shouldbeusedforthesepurposes.``ov.convert_model``functionaccepts
originalPyTorchmodelinstanceandexampleinputfortracingand
returns``ov.Model``representingthismodelinOpenVINOframework.
Convertedmodelcanbeusedforsavingondiskusing``ov.save_model``
functionordirectlyloadingondeviceusing``core.complie_model``.

LLaVA-NeXTisautoregressivetransformergenerativemodel,itmeansthat
eachnextmodelstepdependsfrommodeloutputfrompreviousstep.The
generationapproachisbasedontheassumptionthattheprobability
distributionofawordsequencecanbedecomposedintotheproductof
conditionalnextworddistributions.Inotherwords,modelpredictsthe
nexttokenintheloopguidedbypreviouslygeneratedtokensuntilthe
stop-conditionwillbenotreached(generatedsequenceofmaximumlength
orendofstringtokenobtained).Thewaythenexttokenwillbe
selectedoverpredictedprobabilitiesisdrivenbytheselecteddecoding
methodology.Youcanfindmoreinformationaboutthemostpopular
decodingmethodsinthis
`blog<https://huggingface.co/blog/how-to-generate>`__.Theentrypoint
forthegenerationprocessformodelsfromtheHuggingFaceTransformers
libraryisthe``generate``method.Youcanfindmoreinformationabout
itsparametersandconfigurationinthe
`documentation<https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__.
Topreserveflexibilityintheselectiondecodingmethodology,wewill
convertonlymodelinferenceforonestep.

Theinferenceflowhasdifferenceonfirststepandforthenext.Onthe
firststep,modelacceptpreprocessedinputinstructionandimage,that
transformedtotheunifiedembeddingspaceusing``input_embedding``and
``image_encoder``models,afterthat``languagemodel``,LLM-basedpart
ofmodel,runsoninputembeddingstopredictprobabilityofnext
generatedtokens.Onthenextstep,``language_model``acceptsonlynext
tokenidselectedbasedonsamplingstrategyandprocessedby
``input_embedding``modelandcachedattentionkeyandvalues.Sincethe
outputsideisauto-regressive,anoutputtokenhiddenstateremainsthe
sameoncecomputedforeveryfurthergenerationstep.Therefore,
recomputingiteverytimeyouwanttogenerateanewtokenseems
wasteful.Withthecache,themodelsavesthehiddenstateonceithas
beencomputed.Themodelonlycomputestheoneforthemostrecently
generatedoutputtokenateachtimestep,re-usingthesavedonesfor
hiddentokens.Thisreducesthegenerationcomplexityfrom
:math:`O(n^3)`to:math:`O(n^2)`foratransformermodel.Moredetails
abouthowitworkscanbefoundinthis
`article<https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.

Tosumupabove,modelconsistsof3parts:

-**ImageEncoder**forencodinginputimagesintoembeddingspace
-**InputEmbedding**forconversioninputtexttokensintoembedding
space
-**LanguageModel**forgenerationanswerbasedoninputembeddings
providedbyImageEncoderandInputEmbeddingmodels.

Let’sconverteachmodelpart.

ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ImageEncoderisrepresentedinLLaVAbypretrainedCLIPmodel.

..code::ipython3

importtorch
importopenvinoasov
importgc


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


ifnotIMAGE_ENCODER_PATH.exists():
ov_image_encoder=ov.convert_model(image_encoder_model,example_input=torch.zeros((1,5,3,336,336)))
ov.save_model(ov_image_encoder,IMAGE_ENCODER_PATH)
delov_image_encoder
cleanup_torchscript_cache()

delimage_encoder_model
gc.collect();

TextEmbedding
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

InLLMs,inputembeddingisapartoflanguagemodel,butforLLaVAthe
firststephiddenstateproducedbythismodelpartshouldbeintegrated
withimageembeddingsintocommonembeddingspace.Forabilitytoreuse
thismodelpartandavoidintroductionofllmmodelinstance,wewill
useitseparately.

..code::ipython3

llm_input=None

ifnotLANGUAGE_MODEL_PATH.exists():
llm_input=input_embedding_model(torch.ones((2,2),dtype=torch.int64))

ifnotINPUT_EMBEDDING_PATH.exists():
ov_input_embeddings_model=ov.convert_model(input_embedding_model,example_input=torch.ones((2,2),dtype=torch.int64))
ov.save_model(ov_input_embeddings_model,INPUT_EMBEDDING_PATH)
delov_input_embeddings_model
cleanup_torchscript_cache()

delinput_embedding_model
gc.collect();

LanguageModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LanguageModelisresponsibleforgenerationanswerinLLaVA.Thispart
isverysimilartostandardLLMfortextgeneration.Ourmodeluses
`mistralai/Mistral-7B-Instruct-v0.2<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__
asbaseLLM.Tooptimizethegenerationprocessandusememorymore
efficiently,HuggingFacetransformersAPIprovidesamechanismfor
cachingmodelstateexternallyusing``use_cache=True``parameterand
``past_key_values``argumentininputsandoutputs.Withthecache,the
modelsavesthehiddenstateonceithasbeencomputed.Themodelonly
computestheoneforthemostrecentlygeneratedoutputtokenateach
timestep,re-usingthesavedonesforhiddentokens.Thisreducesthe
generationcomplexityfrom:math:`O(n^3)`to:math:`O(n^2)`fora
transformermodel.Withthisoption,themodelgetsthepreviousstep’s
hiddenstates(cachedattentionkeysandvalues)asinputand
additionallyprovideshiddenstatesforthecurrentstepasoutput.It
meansforallnextiterations,itisenoughtoprovideonlyanewtoken
obtainedfromthepreviousstepandcachedkeyvaluestogetthenext
tokenprediction.

WithincreasingmodelsizelikeinmodernLLMs,wealsocannotean
increaseinthenumberofattentionblocksandsizepastkeyvalues
tensorsrespectively.Thestrategyforhandlingcachestateasmodel
inputsandoutputsintheinferencecyclemaybecomeabottleneckfor
memory-boundedsystems,especiallywithprocessinglonginputsequences,
forexampleinachatbotscenario.OpenVINOsuggestsatransformation
thatremovesinputsandcorrespondingoutputswithcachetensorsfrom
themodelkeepingcachehandlinglogicinsidethemodel.Suchmodelsare
alsocalledstateful.Astatefulmodelisamodelthatimplicitly
preservesdatabetweentwoconsecutiveinferencecalls.Thetensors
savedfromonerunarekeptinaninternalmemorybuffercalleda
``state``ora``variable``andmaybepassedtothenextrun,while
neverbeingexposedasmodeloutput.Hidingthecacheenablesstoring
andupdatingthecachevaluesinamoredevice-friendlyrepresentation.
Ithelpstoreducememoryconsumptionandadditionallyoptimizemodel
performance.Moredetailsaboutstatefulmodelsandworkingwithstate
canbefoundin`OpenVINO
documentation<https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html>`__.

..code::ipython3

fromtypingimportOptional,Tuple,List
fromopenvino.runtimeimportopset13
importnumpyasnp


defmodel_has_state(ov_model:ov.Model):
#TODO:Provideabetterwaybasedonthevariablesavailability,butOVPythonAPIdoesn'texposerequiredmethods
returnlen(ov_model.get_sinks())>0


defmodel_has_input_output_name(ov_model:ov.Model,name:str):
"""
Helperfunctionforcheckingthatmodelhasspecifiedinputoroutputname

Parameters:
ov_model(ov.Model):#TODO:Canwederivethedimensionsfromthemodeltopology?
name(str):
nameofinputoroutput

Returns:
TrueifinputoroutputwithrequestednameexistselseFalse
"""
returnnameinsum([list(t.get_names())fortinov_model.inputs+ov_model.outputs],[])


deffuse_cache_reorder(
ov_model:ov.Model,
not_kv_inputs:List[str],
key_value_input_names:List[str],
gather_dim:int,
):
"""
Fusesreored_cacheduringgeneratecycleintoov.Model.Usedwithstatefulmodels,becausewecannotmodifymodelstatedirectly.

Addsanewbeam_idxparameterandGatheroppereachkv-cacheinputinagivenmodel.
Shouldberunbeforemake_stateful.Implementsoptimumum's_reorder_cache
insidethemodelinthebeginningofeachiteration.
Gatherworksalonggivengather_dimdimensionthatmayvaryfrommodeltomodel.
KV-cacheinputsareidentifiedbasedonnamesinkey_value_input_names.
Appendthenewbeam_idxparametertonot_kv_inputs.

Parameters:
ov_model(`ov.Model`):
openvinomodelforprocessing
not_kv_inputs(`List[str]`):
listofinputnodesinmodelthatnotrelatedtopastkeyvalues
key_value_input_names(`List[str]`):
listofnamesforkeyvalueinputlayers
gather_dim(int):
dimensionforgatheringcacheduringreorderpass
"""

ifmodel_has_input_output_name(ov_model,"beam_idx"):
raiseValueError("Modelalreadyhasfusedcache")
input_batch=ov_model.input("inputs_embeds").get_partial_shape()[0]
beam_idx=opset13.parameter(name="beam_idx",dtype=ov.Type.i32,shape=ov.PartialShape([input_batch]))
beam_idx.output(0).get_tensor().add_names({"beam_idx"})#whylistisnotaccepted?
ov_model.add_parameters([beam_idx])
not_kv_inputs.append(ov_model.inputs[-1])
#Gooverallcacheparametersandfuse_reorder_cachewithindicesprovidedbythenewparameterbeam_idx
forinput_nameinkey_value_input_names:
parameter_output_port=ov_model.input(input_name)
consumers=parameter_output_port.get_target_inputs()
gather=opset13.gather(parameter_output_port,beam_idx,opset13.constant(gather_dim))
forconsumerinconsumers:
consumer.replace_source_output(gather.output(0))
ov_model.validate_nodes_and_infer_types()


defbuild_state_initializer(ov_model:ov.Model,batch_dim:int):
"""
BuildinitializationShapeOfExpressionforallReadValueops

Parameters:
ov_model(ov.Model):
openvinomodel
batch_dim(int):
indexofdimensioncorrespondingtobatchsize
"""
input_ids=ov_model.input("inputs_embeds")
batch=opset13.gather(
opset13.shape_of(input_ids,output_type="i64"),
opset13.constant([0]),
opset13.constant(0),
)
foropinov_model.get_ops():
ifop.get_type_name()=="ReadValue":
dims=[dim.min_lengthfordiminlist(op.get_output_partial_shape(0))]
dims[batch_dim]=batch
dims=[(opset13.constant(np.array([dim],dtype=np.int64))ifisinstance(dim,int)elsedim)fordimindims]
shape=opset13.concat(dims,axis=0)
broadcast=opset13.broadcast(opset13.constant(0.0,dtype=op.get_output_element_type(0)),shape)
op.set_arguments([broadcast])
ov_model.validate_nodes_and_infer_types()


defmake_stateful(
ov_model:ov.Model,
not_kv_inputs:List[str],
key_value_input_names:List[str],
key_value_output_names:List[str],
batch_dim:int,
num_attention_heads:int,
num_beams_and_batch:int=None,
):
"""
Hideskv-cacheinputsandoutputsinsidethemodelasvariables.

Parameters:
ov_model(ov.Model):
openvinomodel
not_kv_inputs(`List[str]`):
listofinputnodesinmodelthatnotrelatedtopastkeyvalues
key_value_input_names(`List[str]`):
listofnamesforkeyvalueinputlayers
key_value_output_names(`List[str]`):
listofnamesforkeyvalueinputlayers
batch_dim(int):
indexofbatchdimensioninkeyvaluelayers
num_attention_heads(int):
numberofattentionheadsforbatchdimensioninitialization
num_beams_an_batch(int):
precalculatednumberofbeamsandbatchforshapesinitialization
"""
fromopenvino._offline_transformationsimportapply_make_stateful_transformation

input_output_map={}

ifnum_beams_and_batchisnotNone:
#Setbatchsizeforinput_idsandattentionmasktoavoiddynamicdimensiongotpropagatedfromtheendofthemodelbacktoReadValue
forinputinnot_kv_inputs:
shape=input.get_partial_shape()
ifshape.rank.get_length()<=2:#==1forbeam_index
shape[0]=num_beams_and_batch
input.get_node().set_partial_shape(shape)
forkv_name_pairinzip(key_value_input_names,key_value_output_names):
input_output_map[kv_name_pair[0]]=kv_name_pair[1]
ifnum_beams_and_batchisnotNone:
input=ov_model.input(kv_name_pair[0])
shape=input.get_partial_shape()
shape[batch_dim]=num_beams_and_batch*num_attention_heads
input.get_node().set_partial_shape(shape)

ifnum_beams_and_batchisnotNone:
#Re-validationmodelifshapesarealteredabove
ov_model.validate_nodes_and_infer_types()

apply_make_stateful_transformation(ov_model,input_output_map)
ifnum_beams_and_batchisNone:
build_state_initializer(ov_model,batch_dim)


defpatch_stateful(ov_model):
key_value_input_names=[key.get_any_name()forkeyinov_model.inputs[2:-1]]
key_value_output_names=[key.get_any_name()forkeyinov_model.outputs[1:]]
not_kv_inputs=[inputforinputinov_model.inputsifnotany(nameinkey_value_input_namesfornameininput.get_names())]
ifnotkey_value_input_namesornotkey_value_output_names:
return
batch_dim=0
num_attention_heads=1

fuse_cache_reorder(ov_model,not_kv_inputs,key_value_input_names,batch_dim)
make_stateful(
ov_model,
not_kv_inputs,
key_value_input_names,
key_value_output_names,
batch_dim,
num_attention_heads,
None,
)

..code::ipython3

make_stateful_model=True
core=ov.Core()

ifnotLANGUAGE_MODEL_PATH.exists():
pkv=language_model(inputs_embeds=llm_input,attention_mask=torch.ones((2,2),dtype=torch.int64))[1]
model_inputs=["attention_mask","position_ids"]
model_outputs=["logits"]
foridxinrange(len(pkv)):
model_inputs.extend([f"past_key_values.{idx}.key",f"past_key_values.{idx}.value"])
model_outputs.extend([f"present.{idx}.key",f"present.{idx}.value"])
model_inputs.append("inputs_embeds")
language_model.config.torchscript=True
position_ids=torch.tensor([[2,3],[2,3]])
ov_model=ov.convert_model(
language_model,
example_input={
"inputs_embeds":llm_input,
"attention_mask":torch.ones((2,4)),
"past_key_values":pkv,
"position_ids":position_ids,
},
)

forinput,input_nameinzip(ov_model.inputs,model_inputs):
input.get_tensor().set_names({input_name})

foroutput,output_nameinzip(ov_model.outputs,model_outputs):
output.get_tensor().set_names({output_name})
ifmake_stateful_model:
patch_stateful(ov_model)
ov.save_model(ov_model,LANGUAGE_MODEL_PATH)
delov_model
cleanup_torchscript_cache()
dellanguage_model
gc.collect()

CompressLanguageModelWeightsto4bits
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Forreducingmemoryconsumption,weightscompressionoptimizationcanbe
appliedusing`NNCF<https://github.com/openvinotoolkit/nncf>`__.Weight
compressionaimstoreducethememoryfootprintofamodel.Itcanalso
leadtosignificantperformanceimprovementforlargememory-bound
models,suchasLargeLanguageModels(LLMs).LLMsandothermodels,
whichrequireextensivememorytostoretheweightsduringinference,
canbenefitfromweightcompressioninthefollowingways:

-enablingtheinferenceofexceptionallylargemodelsthatcannotbe
accommodatedinthememoryofthedevice;

-improvingtheinferenceperformanceofthemodelsbyreducingthe
latencyofthememoryaccesswhencomputingtheoperationswith
weights,forexample,Linearlayers.

`NeuralNetworkCompressionFramework
(NNCF)<https://github.com/openvinotoolkit/nncf>`__provides4-bit/
8-bitmixedweightquantizationasacompressionmethodprimarily
designedtooptimizeLLMs.Themaindifferencebetweenweights
compressionandfullmodelquantization(post-trainingquantization)is
thatactivationsremainfloating-pointinthecaseofweights
compressionwhichleadstoabetteraccuracy.Weightcompressionfor
LLMsprovidesasolidinferenceperformanceimprovementwhichisonpar
withtheperformanceofthefullmodelquantization.Inaddition,weight
compressionisdata-freeanddoesnotrequireacalibrationdataset,
makingiteasytouse.

``nncf.compress_weights``functioncanbeusedforperformingweights
compression.ThefunctionacceptsanOpenVINOmodelandother
compressionparameters.ComparedtoINT8compression,INT4compression
improvesperformanceevenmore,butintroducesaminordropin
predictionquality.

Moredetailsaboutweightscompression,canbefoundin`OpenVINO
documentation<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

**Note:**weightscompressionprocessmayrequireadditionaltimeand
memoryforperforming.Youcandisableitusingwidgetbelow:

..code::ipython3

importipywidgetsaswidgets

to_compress_weights=widgets.Checkbox(
value=True,
description="WeightsCompression",
disabled=False,
)

to_compress_weights




..parsed-literal::

Checkbox(value=True,description='WeightsCompression')



..code::ipython3

importnncf

compression_configuration={
"mode":nncf.CompressWeightsMode.INT4_SYM,
"group_size":64,
"ratio":0.6,
}

LANGUAGE_MODEL_PATH_INT4=LANGUAGE_MODEL_PATH.parent/LANGUAGE_MODEL_PATH.name.replace(".xml","-int4.xml")
ifto_compress_weights.valueandnotLANGUAGE_MODEL_PATH_INT4.exists():
ov_model=core.read_model(LANGUAGE_MODEL_PATH)
ov_compressed_model=nncf.compress_weights(ov_model,**compression_configuration)
ov.save_model(ov_compressed_model,LANGUAGE_MODEL_PATH_INT4)
delov_compressed_model
delov_model
gc.collect()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


QuantizeImageEncoderto8bits
--------------------------------

`backtotop⬆️<#table-of-contents>`__

Thegoalofthispartoftutorialistodemonstratehowtospeedupthe
imageencoderbyapplying8-bitpost-trainingquantizationfrom
`NNCF<https://github.com/openvinotoolkit/nncf/>`__(NeuralNetwork
CompressionFramework)andinferquantizedmodelviaOpenVINO™Toolkit.
`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.Theoptimizationprocesscontainsthefollowingsteps:

1.Preparequantizationdataset
2.QuantizetheconvertedOpenVINOmodelwithNNCF.
3.Savequantizedmodelondiskfornextusage.

..

**Note:**quantizationprocessmayrequireadditionaltimeandmemory
forperforming.Youcandisableitusingwidgetbelow:

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

IMAGE_ENCODER_PATH_INT8=IMAGE_ENCODER_PATH.parent/IMAGE_ENCODER_PATH.name.replace(".xml","-int8.xml")


importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparedatasets
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The`Conceptual
Captions<https://ai.google.com/research/ConceptualCaptions/>`__dataset
consistingof~3.3Mimagesannotatedwithcaptionsisusedtoquantize
model.

..code::ipython3

%%skipnot$to_quantize.value

importrequests
fromioimportBytesIO
importnumpyasnp
fromPILimportImage
fromrequests.packages.urllib3.exceptionsimportInsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


defget_pil_from_url(url):
"""
DownloadsandconvertsanimagefromaURLtoaPILImageobject.
"""
response=requests.get(url,verify=False,timeout=20)
image=Image.open(BytesIO(response.content))
returnimage.convert("RGB")

defcollate_fn(example,image_column="image_url"):
"""
Preprocessesanexamplebyloadingandtransformingimageandtextdata.
Checksifthetextdataintheexampleisvalidbycallingthe`check_text_data`function.
DownloadstheimagespecifiedbytheURLintheimage_columnbycallingthe`get_pil_from_url`function.
Ifthereisanyerrorduringthedownloadprocess,returnsNone.
Returnsthepreprocessedinputswithtransformedimageandtextdata.
"""
assertlen(example)==1
example=example[0]
url=example[image_column]
try:
image=get_pil_from_url(url)
h,w=image.size
ifh==1orw==1:
returnNone
exceptException:
returnNone

inputs=processor.image_processor(images=[image],return_tensors="pt")
returninputs

..code::ipython3

%%skipnot$to_quantize.value

importtorch
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm

defprepare_calibration_data(dataloader,init_steps):
"""
Thisfunctionpreparescalibrationdatafromadataloaderforaspecifiednumberofinitializationsteps.
Ititeratesoverthedataloader,fetchingbatchesandstoringtherelevantdata.
"""
data=[]
print(f"Fetching{init_steps}samplesfortheinitialization...")
withtqdm(total=init_steps)aspbar:
forbatchindataloader:
iflen(data)==init_steps:
break
ifbatch:
pbar.update(1)
withtorch.no_grad():
data.append(
{
"pixel_values":batch["pixel_values"].to("cpu")
}
)
returndata


defprepare_dataset(opt_init_steps=50,max_train_samples=1000):
"""
Preparesavision-textdatasetforquantization.
"""
dataset=load_dataset("google-research-datasets/conceptual_captions",trust_remote_code=True)
train_dataset=dataset["train"].shuffle(seed=42)
dataloader=torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn,batch_size=1)
calibration_data=prepare_calibration_data(dataloader,opt_init_steps)
returncalibration_data

..code::ipython3

%%skipnot$to_quantize.value

vcalibration_data=[]
ifnotIMAGE_ENCODER_PATH_INT8.exists():
calibration_data=prepare_dataset()

Performquantization
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedmodel.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value


ifnotIMAGE_ENCODER_PATH_INT8.exists():
iflen(calibration_data)==0:
raiseRuntimeError(
'Calibrationdatasetisempty.Pleasecheckinternetconnectionandtrytodownloadimagesmanually.'
)

ov_model=core.read_model(IMAGE_ENCODER_PATH)
calibration_dataset=nncf.Dataset(calibration_data)
quantized_model=nncf.quantize(
model=ov_model,
calibration_dataset=calibration_dataset,
model_type=nncf.ModelType.TRANSFORMER,
subset_size=len(calibration_data),
#SmoothQuantalgorithmreducesactivationquantizationerror;optimalalphavaluewasobtainedthroughgridsearch
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
)
ov.save_model(quantized_model,IMAGE_ENCODER_PATH_INT8)
delov_model
delquantized_model
gc.collect()

Preparemodelinferencepipeline
--------------------------------

`backtotop⬆️<#table-of-contents>`__

|image0|

``OVLlavaForCausalLM``classprovidesease-to-useinterfaceforusing
modelingenerationscenario.Itisbasedon
``transformers.generation.GenerationMixin``thatgivesusopportunityto
reuseallreachcapabilitiesforgenerationimplementedinHuggingFace
Transformerslibrary.Moredetailsaboutthisinterfacecanbefoundin
`HuggingFace
documentation<https://huggingface.co/docs/transformers/main_classes/text_generation>`__.

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a562e9de-5b94-4e24-ac52-532019fc92d3

..code::ipython3

importtorch
fromtransformers.generationimportGenerationConfig,GenerationMixin
fromtransformers.modeling_outputsimportCausalLMOutputWithPast
fromtransformersimportAutoConfig
fromtransformers.models.llava_next.modeling_llava_nextimport(
get_anyres_image_grid_shape,
unpad_image,
)
importopenvinoasov


classOVLlavaForCausalLM(GenerationMixin):
def__init__(
self,
core,
image_encoder_path,
input_embedding_path,
language_model_path,
device,
):
self.image_encoder=core.compile_model(core.read_model(image_encoder_path),device)
self.input_embeddings=core.compile_model(core.read_model(input_embedding_path),device)
self.model=core.read_model(language_model_path)
self.input_names={key.get_any_name():idxforidx,keyinenumerate(self.model.inputs)}
self.output_names={idx:keyforidx,keyinenumerate(self.model.outputs)}
self.key_value_input_names=[keyforkeyinlist(self.input_names)ifkeynotin["beam_idx","inputs_embeds","attention_mask","position_ids"]]
self.key_value_output_names=[keyforkeyinlist(self.output_names)[1:]]
self.stateful=len(self.key_value_input_names)==0
compiled_model=core.compile_model(self.model,device)
self.request=compiled_model.create_infer_request()
self.config=AutoConfig.from_pretrained(Path(language_model_path).parent)
self.generation_config=GenerationConfig.from_model_config(self.config)
self.main_input_name="input_ids"
self.device=torch.device("cpu")
self.num_pkv=2
self.next_beam_idx=None
self.image_newline=torch.zeros(self.config.text_config.hidden_size,dtype=torch.float32)
self.pad_token_id=self.config.pad_token_idifself.config.pad_token_idisnotNoneelse-1
self.past_len=0
self._supports_cache_class=False

defcan_generate(self):
"""ReturnsTruetovalidatethecheckthatthemodelusing`GenerationMixin.generate()`canindeedgenerate."""
returnTrue

def__call__(
self,
input_ids:torch.LongTensor,
pixel_values:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
position_ids:Optional[torch.LongTensor]=None,
image_sizes=None,
**kwargs,
)->CausalLMOutputWithPast:
returnself.forward(
input_ids,
pixel_values,
attention_mask,
past_key_values,
position_ids,
image_sizes,
**kwargs,
)

defforward(
self,
input_ids:torch.LongTensor,
pixel_values:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
position_ids:Optional[torch.LongTensor]=None,
image_sizes=None,
**kwargs,
)->CausalLMOutputWithPast:
"""Generalinferencemethod"""
inputs={}
ifpast_key_valuesisnotNone:
inputs={}
ifnotself.stateful:
past_key_values=tuple(past_key_valueforpkv_per_layerinpast_key_valuesforpast_key_valueinpkv_per_layer)
#Addthepast_key_valuestothedecoderinputs
inputs=dict(zip(self.key_value_input_names,past_key_values))
#input_ids=np.array(input_ids)[:,-1:]
inputs_embeds=self.input_embeddings(input_ids)[0]
inputs["inputs_embeds"]=inputs_embeds
#inputs["attention_mask"]=attention_mask
if"beam_idx"inself.input_names:
inputs["beam_idx"]=self.next_beam_idxifself.next_beam_idxisnotNoneelsenp.arange(batch_size,dtype=int)

ifnotself.stateful:
first_layer_past_key_value=torch.from_numpy(past_key_values[0][0][:,:,:,0])
else:
first_layer_past_key_value=torch.from_numpy(self.request.query_state()[0].state.data[:,:,:,0])

#Sumalldimensionsofhead_dim(-2)toavoidrandomerrorssuchas:https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
batch_index,non_attended_tokens=torch.where(first_layer_past_key_value.float().sum(-2)==0)

#Getthetargetlength
target_length=input_ids.shape[1]
past_length=first_layer_past_key_value.shape[-1]

extended_attention_mask=torch.ones(
(attention_mask.shape[0],past_length),
dtype=attention_mask.dtype,
device=attention_mask.device,
)

#Filteroutonlythetokensthatcanbeun-attended,thiscanhappen
#ifoneusesLlava+Fusedmoduleswherethecacheonthe
#firstiterationisalreadybigenough,orifonepassescustomcache
valid_indices=non_attended_tokens<extended_attention_mask.size(-1)
new_batch_index=batch_index[valid_indices]
new_non_attended_tokens=non_attended_tokens[valid_indices]

#Zero-outtheplaceswherewedon'tneedtoattend
extended_attention_mask[new_batch_index,new_non_attended_tokens]=0

attention_mask=torch.cat((extended_attention_mask,attention_mask[:,-target_length:]),dim=1)
position_ids=torch.sum(attention_mask,dim=1).unsqueeze(-1)-1
inputs["attention_mask"]=attention_mask
inputs["position_ids"]=position_ids

else:
inputs=self.prepare_multimodal_input(input_ids,pixel_values,attention_mask,position_ids,image_sizes)

#Runinference
self.request.start_async(inputs,share_inputs=True)
self.request.wait()

logits=torch.from_numpy(self.request.get_tensor(self.output_names[0]).data)

ifnotself.stateful:
#Tupleoflengthequalto:numberoflayer*numberofpast_key_valueperdecoderlayer(2correspondstotheself-attentionlayer)
past_key_values=tuple(self.request.get_tensor(key).dataforkeyinself.key_value_output_names)
#Tupleoftupleoflength`n_layers`,witheachtupleoflengthequalto2(k/vofself-attention)
past_key_values=tuple(past_key_values[i:i+self.num_pkv]foriinrange(0,len(past_key_values),self.num_pkv))
else:
past_key_values=((),)
self.past_len+=inputs["inputs_embeds"].shape[1]
returnCausalLMOutputWithPast(logits=logits,past_key_values=past_key_values)

defprepare_multimodal_input(self,input_ids,pixel_values,attention_mask,position_ids,image_sizes=None):
"""Preprocessingfunctionforembeddingmultimodaldata"""
inputs={}
inputs_embeds=torch.from_numpy(self.input_embeddings(input_ids)[0])
batch_size=input_ids.shape[0]
ifnotself.stateful:
forinput_nameinself.key_value_input_names:
model_inputs=self.modeget_anyres_image_grid_shapel.input(input_name)
shape=model_inputs.get_partial_shape()
shape[0]=batch_size
ifshape[2].is_dynamic:
shape[2]=0
else:
shape[1]=0
inputs[input_name]=ov.Tensor(model_inputs.get_element_type(),shape.get_shape())
else:
self.past_len=0
self.request.reset_state()
#Setinitialvalueforthenextbeam_idxinputthatwillbeusedatthecurrentiteration
#andwillbeoptionallyupdatedby_reorder_cacheatthenextiterationsifbeam_searchisused
self.next_beam_idx=np.arange(batch_size,dtype=int)

if"beam_idx"inself.input_names:
inputs["beam_idx"]=self.next_beam_idxifself.next_beam_idxisnotNoneelsenp.arange(batch_size,dtype=int)
ifpixel_valuesisNone:
inputs["inputs_embeds"]=inputs_embeds
inputs["attention_mask"]=attention_mask
ifposition_idsisNone:
position_ids=torch.cumsum(attention_mask,axis=1)-1
position_ids[attention_mask==0]=1
inputs["position_ids"]=position_ids
res=self.image_encoder(pixel_values)
image_features=torch.from_numpy(res[0])
split_sizes=[image.shape[0]forimageinpixel_values]
image_features=torch.split(image_features,split_sizes,dim=0)

#NOTEweonlysupportmultimodal_patch_merge_type=="spatial_unpad"
height=width=self.config.vision_config.image_size//self.config.vision_config.patch_size

new_image_features=[]
forimage_idx,image_featureinenumerate(image_features):
ifimage_feature.shape[0]>1:
base_image_feature=image_feature[0]
image_feature=image_feature[1:]

ifheight*width!=base_image_feature.shape[0]:
raiseValueError("Thenumberofpatchesisnotconsistentwiththeimagesize.")
num_patch_height,num_patch_width=get_anyres_image_grid_shape(
image_sizes[image_idx],
self.config.image_grid_pinpoints,
self.config.vision_config.image_size,
)
image_feature=image_feature.view(num_patch_height,num_patch_width,height,width,-1)
image_feature=image_feature.permute(4,0,2,1,3).contiguous()
image_feature=image_feature.flatten(1,2).flatten(2,3)
image_feature=unpad_image(image_feature,image_sizes[image_idx])
image_feature=torch.cat(
(
image_feature,
self.image_newline[:,None,None].expand(*image_feature.shape[:-1],1),
),
dim=-1,
)
image_feature=image_feature.flatten(1,2).transpose(0,1)
image_feature=torch.cat((base_image_feature,image_feature),dim=0)
else:
image_feature=image_feature[0]
image_feature=torch.cat((image_feature,self.image_newline[None]),dim=0)
new_image_features.append(image_feature)
image_features=torch.stack(new_image_features,dim=0)

(
inputs_embeds,
attention_mask,
position_ids,
)=self._merge_input_ids_with_image_features(image_features,inputs_embeds,input_ids,attention_mask,None)
inputs["inputs_embeds"]=inputs_embeds
inputs["attention_mask"]=attention_mask
inputs["position_ids"]=position_ids

returninputs

def_merge_input_ids_with_image_features(self,image_features,inputs_embeds,input_ids,attention_mask,labels):
num_images,num_image_patches,embed_dim=image_features.shape
batch_size,sequence_length=input_ids.shape
left_padding=nottorch.sum(input_ids[:,-1]==torch.tensor(self.pad_token_id))
#1.Createamasktoknowwherespecialimagetokensare
special_image_token_mask=input_ids==self.config.image_token_index
num_special_image_tokens=torch.sum(special_image_token_mask,dim=-1)
#Computethemaximumembeddimension
max_embed_dim=(num_special_image_tokens.max()*(num_image_patches-1))+sequence_length
batch_indices,non_image_indices=torch.where(input_ids!=self.config.image_token_index)

#2.Computethepositionswheretextshouldbewritten
#Calculatenewpositionsfortexttokensinmergedimage-textsequence.
#`special_image_token_mask`identifiesimagetokens.Eachimagetokenwillbereplacedby`nb_text_tokens_per_images-1`texttokens.
#`torch.cumsum`computeshoweachimagetokenshiftssubsequenttexttokenpositions.
#-1toadjustforzero-basedindexing,as`cumsum`inherentlyincreasesindicesbyone.
new_token_positions=torch.cumsum((special_image_token_mask*(num_image_patches-1)+1),-1)-1
nb_image_pad=max_embed_dim-1-new_token_positions[:,-1]
ifleft_padding:
new_token_positions+=nb_image_pad[:,None]#offsetforleftpadding
text_to_overwrite=new_token_positions[batch_indices,non_image_indices]

#3.Createthefullembedding,alreadypaddedtothemaximumposition
final_embedding=torch.zeros(
batch_size,
max_embed_dim,
embed_dim,
dtype=inputs_embeds.dtype,
device=inputs_embeds.device,
)
final_attention_mask=torch.zeros(
batch_size,
max_embed_dim,
dtype=attention_mask.dtype,
device=inputs_embeds.device,
)
#IncasetheVisionmodelortheLanguagemodelhasbeenoffloadedtoCPU,weneedtomanually
#setthecorrespondingtensorsintotheircorrecttargetdevice.
target_device=inputs_embeds.device
batch_indices,non_image_indices,text_to_overwrite=(
batch_indices.to(target_device),
non_image_indices.to(target_device),
text_to_overwrite.to(target_device),
)
attention_mask=attention_mask.to(target_device)

#4.Filltheembeddingsbasedonthemask.Ifwehave["hey""<image>","how","are"]
#weneedtoindexcopyon[0,577,578,579]forthetextand[1:576]fortheimagefeatures
final_embedding[batch_indices,text_to_overwrite]=inputs_embeds[batch_indices,non_image_indices]
final_attention_mask[batch_indices,text_to_overwrite]=attention_mask[batch_indices,non_image_indices]
iflabelsisnotNone:
final_labels[batch_indices,text_to_overwrite]=labels[batch_indices,non_image_indices]

#5.Filltheembeddingscorrespondingtotheimages.Anythingthatisstillzerosneedsfilling
image_to_overwrite=torch.all(final_embedding==0,dim=-1)
image_to_overwrite&=image_to_overwrite.cumsum(-1)-1>=nb_image_pad[:,None].to(target_device)
ifimage_to_overwrite.sum()!=image_features.shape[:-1].numel():
raiseValueError(
f"Theinputprovidedtothemodelarewrong.Thenumberofimagetokensis{torch.sum(special_image_token_mask)}while"
f"thenumberofimagegiventothemodelis{num_images}.Thispreventscorrectindexingandbreaksbatchgeneration."
)

final_embedding[image_to_overwrite]=image_features.contiguous().reshape(-1,embed_dim).to(target_device)
final_attention_mask|=image_to_overwrite
position_ids=(final_attention_mask.cumsum(-1)-1).masked_fill_((final_attention_mask==0),1)

#6.Maskouttheembeddingatpaddingpositions,aswelaterusethepast_key_valuevaluetodeterminethenon-attendedtokens.
batch_indices,pad_indices=torch.where(input_ids==self.pad_token_id)
indices_to_mask=new_token_positions[batch_indices,pad_indices]

final_embedding[batch_indices,indices_to_mask]=0

returnfinal_embedding,final_attention_mask,position_ids

defprepare_inputs_for_generation(
self,
input_ids,
past_key_values=None,
inputs_embeds=None,
pixel_values=None,
image_sizes=None,
attention_mask=None,
**kwargs,
):
ifpast_key_valuesisnotNone:
ifnotself.stateful:
cache_length=past_length=past_key_values[0][0].shape[2]
else:
cache_length=past_length=self.past_len

#Keeponlytheunprocessedtokens:
#1-Ifthelengthoftheattention_maskexceedsthelengthofinput_ids,thenweareinasettingwhere
#someoftheinputsareexclusivelypassedaspartofthecache(e.g.whenpassinginput_embedsas
#input)
ifattention_maskisnotNoneandattention_mask.shape[1]>input_ids.shape[1]:
input_ids=input_ids[:,-(attention_mask.shape[1]-past_length):]
#2-Ifthepast_lengthissmallerthaninput_ids',theninput_idsholdsallinputtokens.Wecandiscard
#input_idsbasedonthepast_length.llava
elifpast_length<input_ids.shape[1]:
input_ids=input_ids[:,past_length:]
#3-Otherwise(past_length>=input_ids.shape[1]),let'sassumeinput_idsonlyhasunprocessedtokens.
elifself.config.image_token_indexininput_ids:
input_ids=input_ids[:,input_ids.shape[1]-1:]
#Ifthecachehasseenmoretokensthanitcanhold,thenthecachehasasizelimit.Let'sdiscardthe
#olderattentionvalues,astheircorrespondingvaluesarenotpartoftheinput.
ifcache_length<past_lengthandattention_maskisnotNone:
attention_mask=attention_mask[:,-(cache_length+input_ids.shape[1]):]

position_ids=kwargs.get("position_ids",None)
ifattention_maskisnotNoneandposition_idsisNone:
#createposition_idsontheflyforbatchgllavaenerationsubset_siz
position_ids=attention_mask.long().cumsum(-1)-1
position_ids.masked_fill_(attention_mask==0,1)
ifpast_key_values:
position_ids=position_ids[:,-input_ids.shape[1]:]

#if`inputs_embeds`arepassed,weonlywanttousetheminthe1stgenerationstep
ifinputs_embedsisnotNoneandpast_key_valuesisNone:
model_inputs={"inputs_embeds":inputs_embeds}
else:
model_inputs={"input_ids":input_ids}

model_inputs.update(
{
"position_ids":position_ids,
"past_key_values":past_key_values,
"use_cache":kwargs.get("use_cache"),
"attention_mask":attention_mask,
"pixel_values":pixel_values,
"image_sizes":image_sizes,
}
)
returnmodel_inputs

RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Selectdevice
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

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

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1'),value='CPU')



..code::ipython3

use_int4_lang_model=widgets.Checkbox(
value=LANGUAGE_MODEL_PATH_INT4.exists(),
description="INT4languagemodel",
disabled=notLANGUAGE_MODEL_PATH_INT4.exists(),
)

use_int4_lang_model




..parsed-literal::

Checkbox(value=True,description='INT4languagemodel')



..code::ipython3

use_int8_image_encoder=widgets.Checkbox(
value=IMAGE_ENCODER_PATH_INT8.exists(),
description="INT8imageencoder",
disabled=notIMAGE_ENCODER_PATH_INT8.exists(),
)

use_int8_image_encoder




..parsed-literal::

Checkbox(value=True,description='INT4languagemodel')



..code::ipython3

lang_model_path=LANGUAGE_MODEL_PATH_INT4ifuse_int4_lang_model.valueelseLANGUAGE_MODEL_PATH
image_encoder_path=IMAGE_ENCODER_PATH_INT8ifuse_int8_image_encoder.valueelseIMAGE_ENCODER_PATH

ov_llava_model=OVLlavaForCausalLM(core,image_encoder_path,INPUT_EMBEDDING_PATH,lang_model_path,device.value)

..code::ipython3

fromPILimportImage
importrequests


fromtransformersimportTextStreamer

url="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image=Image.open(requests.get(url,stream=True).raw)
question="Whatisunusualonthisimage?"
prompt=f"[INST]<image>\n{question}[/INST]"
streamer=TextStreamer(processor,skip_special_tokens=True,skip_prompt=True)

inputs=processor(prompt,image,return_tensors="pt")
print(f"Question:\n{question}")
image


..parsed-literal::

Question:
Whatisunusualonthisimage?




..image::llava-next-multimodal-chatbot-with-output_files/llava-next-multimodal-chatbot-with-output_32_1.png



..code::ipython3

print("Answer:")
streamer=TextStreamer(processor,skip_special_tokens=True,skip_prompt=True)
output=ov_llava_model.generate(**inputs,max_new_tokens=49,streamer=streamer)


..parsed-literal::

Setting`pad_token_id`to`eos_token_id`:2foropen-endgeneration.


..parsed-literal::

Answer:
Theimageshowsacatlyingonitsbackinsideacardboardbox.What'sunusualisthatthecatappearstobeinarelaxedandsomewhathuman-likepose,withitspawsupintheairanditsbellyexposed.


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
fromtransformersimportTextIteratorStreamer
fromthreadingimportThread
fromPILimportImage
importtorch

example_image_urls=[
(
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
"bee.jpg",
),
(
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
"baklava.png",
),
]
forurl,file_nameinexample_image_urls:
Image.open(requests.get(url,stream=True).raw).save(file_name)


defbot_streaming(message,history):
print(message)
ifmessage["files"]:
image=message["files"][-1]["path"]ifisinstance(message["files"][-1],dict)elsemessage["files"][-1]
else:
#ifthere'snoimageuploadedforthisturn,lookforimagesinthepastturns
#keptinsidetuples,takethelastone
forhistinhistory:
ifisinstance(hist[0],tuple):
image=hist[0][0]

ifimageisNone:
gr.Error("YouneedtouploadanimageforLLaVAtowork.")
prompt=f"[INST]<image>\n{message['text']}[/INST]"
image=Image.open(image).convert("RGB")
inputs=processor(prompt,image,return_tensors="pt")

streamer=TextIteratorStreamer(processor,**{"skip_special_tokens":True})
generation_kwargs=dict(inputs,streamer=streamer,max_new_tokens=100)

thread=Thread(target=ov_llava_model.generate,kwargs=generation_kwargs)
thread.start()

text_prompt=f"[INST]\n{message['text']}[/INST]"

buffer=""
fornew_textinstreamer:
buffer+=new_text
generated_text_without_prompt=buffer[len(text_prompt):]
yieldgenerated_text_without_prompt


demo=gr.ChatInterface(
fn=bot_streaming,
title="LLaVANeXT",
examples=[
{"text":"Whatisontheflower?","files":["./bee.jpg"]},
{"text":"Howtomakethispastry?","files":["./baklava.png"]},
],
description="Try[LLaVANeXT](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next)inthisdemousingOpenVINO.Uploadanimageandstartchattingaboutit,orsimplytryoneoftheexamplesbelow.Ifyoudon'tuploadanimage,youwillreceiveanerror.",
stop_btn="StopGeneration",
multimodal=True,
)

try:
demo.launch(debug=False)
exceptException:
demo.launch(debug=False,share=True)

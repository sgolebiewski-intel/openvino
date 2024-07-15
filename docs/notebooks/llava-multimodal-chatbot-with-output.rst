Visual-languageassistantwithLLaVAandOpenVINO
=================================================

`LLaVA<https://llava-vl.github.io>`__(LargeLanguageandVision
Assistant)islargemultimodalmodelthataimstodevelopa
general-purposevisualassistantthatcanfollowbothlanguageandimage
instructionstocompletevariousreal-worldtasks.Theideaisto
combinethepoweroflargelanguagemodels(LLMs)withvisionencoders
likeCLIPtocreateanend-to-endtrainedneuralassistantthat
understandsandactsuponmultimodalinstructions.

Inthefieldofartificialintelligence,thegoalistocreatea
versatileassistantcapableofunderstandingandexecutingtasksbased
onbothvisualandlanguageinputs.Currentapproachesoftenrelyon
largevisionmodelsthatsolvetasksindependently,withlanguageonly
usedtodescribeimagecontent.Whileeffective,thesemodelshavefixed
interfaceswithlimitedinteractivityandadaptabilitytouser
instructions.Ontheotherhand,largelanguagemodels(LLMs)haveshown
promiseasauniversalinterfaceforgeneral-purposeassistants.By
explicitlyrepresentingvarioustaskinstructionsinlanguage,these
modelscanbeguidedtoswitchandsolvedifferenttasks.Toextendthis
capabilitytothemultimodaldomain,the`LLaVA
paper<https://arxiv.org/abs/2304.08485>`__introduces\`visual
instruction-tuning,anovelapproachtobuildingageneral-purpose
visualassistant.

InthistutorialweconsiderhowtouseLLaVAmodeltobuildmultimodal
chatbot.Fordemonstrationpurposeswewilluse
`LLaVA-Lightning-MPT-7B-preview<https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview>`__
modelforconversion,similarstepsrequiredtorunothermodelsfrom
`LLaVAModel
Zoo<https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md>`__.

Thetutorialconsistsfromfollowingsteps:

-Installprerequisites
-Prepareinputprocessorandtokenizer
-Downloadoriginalmodel
-Compressmodelweightsto4and8bitsusingNNCF
-ConvertmodeltoOpenVINOIntermediateRepresentation(IR)format
-PrepareOpenVINO-basedinferencepipeline
-RunOpenVINOmodel

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Aboutmodel<#about-model>`__
-`Prerequisites<#prerequisites>`__
-`Buildmodeltokenizerandimage
processor<#build-model-tokenizer-and-image-processor>`__
-`BuildmodelandconvertittoOpenVINOIR
format<#build-model-and-convert-it-to-openvino-ir-format>`__

-`Preparehelpersformodel
conversion<#prepare-helpers-for-model-conversion>`__
-`ConvertandOptimizeModel<#convert-and-optimize-model>`__

-`InstantiatePyTorchmodel
:math:`\Uparrow`\(#Table-of-content:)<#instantiate-pytorch-model>`__
-`CompressModelweightsto4and8bitsusingNNCF
:math:`\Uparrow`\(#Table-of-content:)<#compress-model-weights-to-4-and-8-bits-using-nncf>`__
-`ConvertmodeltoOpenVINOIRformat
:math:`\Uparrow`\(#Table-of-content:)<#convert-model-to-openvino-ir-format>`__

-`PrepareOpenVINObasedinference
pipeline<#prepare-openvino-based-inference-pipeline>`__
-`Runmodelinference<#run-model-inference>`__

-`Selectinferencedevice<#select-inference-device>`__
-`LoadOpenVINOmodel<#load-openvino-model>`__
-`Prepareinputdata<#prepare-input-data>`__
-`Testmodelinference<#test-model-inference>`__

-`Interactivedemo<#interactive-demo>`__

Aboutmodel
-----------

`backtotop⬆️<#table-of-contents>`__

LLaVAconnectspre-trained`CLIP
ViT-L/14<https://openai.com/research/clip>`__visualencoderandlarge
languagemodellikeVicuna,LLaMav2orMPT,usingasimpleprojection
matrix

..figure::https://llava-vl.github.io/images/llava_arch.png
:alt:vlp_matrix.png

vlp_matrix.png

Modeltrainingprocedureconsistsof2stages:

-Stage1:Pre-trainingforFeatureAlignment.Onlytheprojection
matrixisupdated,basedonasubsetofCC3M.
-Stage2:Fine-tuningEnd-to-End..BoththeprojectionmatrixandLLM
areupdatedfortwodifferentusescenarios:

-VisualChat:LLaVAisfine-tunedonourgeneratedmultimodal
instruction-followingdatafordailyuser-orientedapplications.
-ScienceQA:LLaVAisfine-tunedonthismultimodalreasoning
datasetforthesciencedomain.

Moredetailsaboutmodelcanbefoundinoriginal`project
web-page<https://llava-vl.github.io/>`__,
`paper<https://arxiv.org/abs/2304.08485>`__and
`repo<https://github.com/haotian-liu/LLaVA>`__.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequireddependencies

..code::ipython3

importsys

%pipinstall-q"torch>=2.1.0""torchvision""torchaudio"--index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.2.0""nncf>=2.7.0""sentencepiece""tokenizers>=0.12.1""transformers>=4.37.2""gradio>=4.19""einops"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.

[notice]Anewreleaseofpipisavailable:23.3.2->24.0
[notice]Toupdate,run:pipinstall--upgradepip
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

repo_dir=Path("LLaVA")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/haotian-liu/LLaVA.git

sys.path.insert(0,str(repo_dir.resolve()))

Buildmodeltokenizerandimageprocessor
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Forstartingworkwithmodel,weneedunderstandhowtoprepareinput
datafirst.Asitisalreadydiscussedbefore,LLaVAismultimodalmodel
thatacceptsinputuserinstructionsintextformatandimagefor
analysis.Inthesametime,LLaVAiscombinationof2fundamental
pretrainedmodelsfortextandimageprocessing,CLIPandMPT,eachof
themhasownapproachforpreparingdata-tokenizationforinputtext
andpreprocessingforinputimage.LLaVAreusesthesestepswithsmall
adoption:introducedspecialtokensthatservesforspecificationof
imagelocationinthetextthatshouldbeinjectedinprovideduser
instruction.

..code::ipython3

fromtransformersimportAutoTokenizer,AutoConfig,CLIPImageProcessor
fromllava.model.language_model.llava_mptimportLlavaMptForCausalLM

model_id="liuhaotian/LLaVA-Lightning-MPT-7B-preview"

config=AutoConfig.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)
image_processor=CLIPImageProcessor.from_pretrained(config.mm_vision_tower)


..parsed-literal::

Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


..code::ipython3

fromllava.constantsimport(
DEFAULT_IMAGE_PATCH_TOKEN,
DEFAULT_IM_START_TOKEN,
DEFAULT_IM_END_TOKEN,
DEFAULT_IMAGE_TOKEN,
)

mm_use_im_start_end=getattr(config,"mm_use_im_start_end",False)
mm_use_im_patch_token=getattr(config,"mm_use_im_patch_token",True)
ifmm_use_im_patch_token:
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN],special_tokens=True)
ifmm_use_im_start_end:
tokenizer.add_tokens([DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN],special_tokens=True)

ifhasattr(config,"max_sequence_length"):
context_len=config.max_sequence_length
else:
context_len=2048

BuildmodelandconvertittoOpenVINOIRformat
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

LLaVAisautoregressivetransformergenerativemodel,itmeansthateach
nextmodelstepdependsfrommodeloutputfrompreviousstep.The
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
transformedtotheunifiedembeddingspaceusing``token_embedding``and
``image_encoder``models,afterthatLLM-basedpartofmodelrunson
inputembeddingstopredictprobabilityofnextgeneratedtokens.Onthe
nextstep,modelacceptsonlynexttokenidselectedbasedonsampling
strategyandcachedattentionkeyandvalues.Sincetheoutputsideis
auto-regressive,anoutputtokenhiddenstateremainsthesameonce
computedforeveryfurthergenerationstep.Therefore,recomputingit
everytimeyouwanttogenerateanewtokenseemswasteful.Withthe
cache,themodelsavesthehiddenstateonceithasbeencomputed.The
modelonlycomputestheoneforthemostrecentlygeneratedoutputtoken
ateachtimestep,re-usingthesavedonesforhiddentokens.This
reducesthegenerationcomplexityfrom:math:`O(n^3)`to:math:`O(n^2)`
foratransformermodel.Moredetailsabouthowitworkscanbefoundin
this
`article<https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.

Preparehelpersformodelconversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThecodebelowpreparesfunctionforconvertingLLaVAmodeltoOpenVINO
IntermediateRepresentationformat.Itsplitsmodelonpartsdescribed
above,prepareexampleinputsforeachpartandconverteachpartusing
`OpenVINOModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__.
``ov.convert_model``functionacceptsPyTorchmodelinstanceandreturns
``ov.Model``objectthatrepresentmodelinOpenVINOformat.Itisready
touseforloadingondeviceusing``ov.compile_model``orcanbesaved
ondiskusing``ov.save_model``.

..code::ipython3

fromfunctoolsimportwraps
importgc
importwarnings
importtorch
importopenvinoasov
importnncf
fromtypingimportOptional,Tuple,List
importtorch.nn.functionalasF

warnings.filterwarnings("ignore")


classModelWrapper(torch.nn.Module):
"""
Modelwrapperclassforexportforsplitingoriginalforwardlogiconpreparingmultimodaldataandinferenceusingit.
Thatallowsustosperateimageencoderandtokenembeddingsmodelfromgeneralflow.
"""

def__init__(self,model):
super().__init__()
self.model=model

defforward(
self,
input_ids:torch.LongTensor=None,
past_key_values:Optional[List[torch.FloatTensor]]=None,
inputs_embeds:Optional[torch.FloatTensor]=None,
attention_mask:Optional[torch.Tensor]=None,
):
outputs=self.model.transformer(
input_ids=input_ids,
inputs_embeds=inputs_embeds,
past_key_values=past_key_values,
attention_mask=attention_mask,
return_dict=True,
output_attentions=False,
output_hidden_states=False,
use_cache=True,
)
logits=F.linear(
outputs.last_hidden_state.to(self.model.transformer.wte.weight.device),
self.model.transformer.wte.weight.to(outputs.last_hidden_state.dtype),
)

return(logits,tuple(outputs.past_key_values))


defpatch_model_forward(model):
"""
Helperfunctionforpatchingmodelforwardformodelwithpast.
ItmakesmodelmoreconvinientforexporttoTorchScriptformatavoidinglimitation
thatlistoftensorscannotbecorrectlytracedasmodelinput
"""

orig_forward=model.forward

@wraps(orig_forward)
defts_patched_forward(
input_ids:torch.Tensor,
past_key_values:Tuple[Tuple[torch.Tensor]],
attention_mask:torch.LongTensor,
):
pkv_list=list(past_key_values)
outs=orig_forward(
input_ids=input_ids,
past_key_values=pkv_list,
attention_mask=attention_mask,
)
returnouts

model.forward=ts_patched_forward
returnmodel


defflattenize_inputs(inputs):
"""
Helperfunctionformakingnestedinputsflattens
"""
flatten_inputs=[]
forinput_dataininputs:
ifinput_dataisNone:
continue
ifisinstance(input_data,(list,tuple)):
flatten_inputs.extend(flattenize_inputs(input_data))
else:
flatten_inputs.append(input_data)
returnflatten_inputs


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


defpostprocess_converted_model(
ov_model,
example_input=None,
input_names=None,
output_names=None,
dynamic_shapes=None,
):
"""
Helperfunctionforapplingpostprocessingonconvertedmodelwithupdatinginputnames,shapesandoutputnames
acordingtorequestedspecification
"""
flatten_example_inputs=flattenize_inputs(example_input)ifexample_inputelse[]

ifinput_names:
forinp_name,m_input,input_datainzip(input_names,ov_model.inputs,flatten_example_inputs):
input_node=m_input.get_node()
ifinput_node.element_type==ov.Type.dynamic:
m_input.get_node().set_element_type(ov.Type.f32)
shape=list(input_data.shape)
ifdynamic_shapesisnotNoneandinp_nameindynamic_shapes:
forkindynamic_shapes[inp_name]:
shape[k]=-1
input_node.set_partial_shape(ov.PartialShape(shape))
m_input.get_tensor().set_names({inp_name})

ifoutput_names:
forout,out_nameinzip(ov_model.outputs,output_names):
out.get_tensor().set_names({out_name})
ov_model.validate_nodes_and_infer_types()
returnov_model


defconvert_llava_mpt(
pt_model:torch.nn.Module,
model_path:Path,
image_encoder_wc_parameters:Optional[dict]=None,
llava_wc_parameters:Optional[dict]=None,
):
"""
LLaVAMPTmodelconversionfunction

Params:
pt_model:PyTorchmodel
model_path:pathforsavingmodel
Returns:
None
"""
ov_out_path=Path(model_path)
pt_model.config.save_pretrained(ov_out_path)
pt_model.config.use_cache=True
pt_model.config.torchscript=True
first_stage_model_path=ov_out_path/"llava_input_embed.xml"
image_encoder_path=ov_out_path/"image_encoder.xml"
token_embedding_model_path=ov_out_path/"token_embed.xml"
second_stage_model_path=ov_out_path/"llava_with_past.xml"
ifnotimage_encoder_path.exists():
model.forward=model.encode_images
ov_model=ov.convert_model(
model,
example_input=torch.zeros((1,3,224,224)),
input=[(-1,3,224,224)],
)
ifimage_encoder_wc_parametersisnotNone:
print("Applyingweightcompressiontoimageencoder")
ov_model=nncf.compress_weights(ov_model,**image_encoder_wc_parameters)
ov.save_model(ov_model,image_encoder_path)
cleanup_torchscript_cache()
delov_model
gc.collect()
print("ImageEncodermodelsuccessfullyconverted")

ifnottoken_embedding_model_path.exists():
model.forward=model.get_model().embed_tokens
ov_model=ov.convert_model(model,example_input=torch.ones((1,10),dtype=torch.long))
ov.save_model(ov_model,token_embedding_model_path)
cleanup_torchscript_cache()
delov_model
gc.collect()
print("TokenEmbeddingmodelsuccessfullyconverted")

iffirst_stage_model_path.exists()andsecond_stage_model_path.exists():
print("LLaVAmodelsuccessfullyconverted")
delpt_model
return
model_wrap=ModelWrapper(model)
example_input_first_stage={
"inputs_embeds":torch.zeros((1,307,4096)),
"attention_mask":torch.ones((1,307),dtype=torch.long),
}
outs=model_wrap(**example_input_first_stage)
inputs=["input_ids"]
outputs=["logits"]
dynamic_shapes={"input_ids":{1:"seq_len"},"attention_mask":{1:"seq_len"}}
foridxinrange(len(outs[1])):
inputs.extend([f"past_key_values.{idx}.key",f"past_key_values.{idx}.value"])
dynamic_shapes[inputs[-1]]={2:"past_sequence+sequence"}
dynamic_shapes[inputs[-2]]={2:"past_sequence+sequence"}
outputs.extend([f"present.{idx}.key",f"present.{idx}.value"])

inputs.extend(["attention_mask"])
ifnotfirst_stage_model_path.exists():
ov_model=ov.convert_model(model_wrap,example_input=example_input_first_stage)
ov_model=postprocess_converted_model(ov_model,output_names=outputs)
ifllava_wc_parametersisnotNone:
print("ApplyingweightcompressiontofirststageLLavamodel")
ov_model=nncf.compress_weights(ov_model,**llava_wc_parameters)
ov.save_model(ov_model,first_stage_model_path)
cleanup_torchscript_cache()
delov_model
gc.collect()

ifnotsecond_stage_model_path.exists():
model_wrap=patch_model_forward(model_wrap)
example_input_second_stage={
"input_ids":torch.ones((1,1),dtype=torch.long),
"past_key_values":outs[1],
"attention_mask":torch.ones((1,outs[1][-1][-1].shape[-2]+1),dtype=torch.long),
}
ov_model=ov.convert_model(model_wrap,example_input=example_input_second_stage)
ov_model=postprocess_converted_model(
ov_model,
example_input=example_input_second_stage.values(),
input_names=inputs,
output_names=outputs,
dynamic_shapes=dynamic_shapes,
)
ifllava_wc_parametersisnotNone:
print("ApplyingweightcompressiontosecondstageLLavamodel")
ov_model=nncf.compress_weights(ov_model,**llava_wc_parameters)
ov.save_model(ov_model,second_stage_model_path)
cleanup_torchscript_cache()
delov_model
gc.collect()
print("LLaVAmodelsuccessfullyconverted")
delmodel_wrap
delpt_model


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


ConvertandOptimizeModel
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Ourmodelconversionandoptimizationconsistoffollowingsteps:1.
DownloadoriginalPyTorchmodel.2.CompressmodelweightsusingNNCF3.
ConvertmodeltoOpenVINOformatandsaveitondisk.

Let’sconsidereachstepmoredeeply.

InstantiatePyTorchmodel`:math:`\Uparrow`<#table-of-content>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ForcreatingPyTorchmodelweshoulduse``from_pretrained``methodof
``LlavaMPTForCausalLM``modelclass.Modelweightswillbedownloaded
from`HuggingFacehub<https://huggingface.co/models>`__duringfirst
run.Itmaytakessometimeandrequiresatleast13Gbfreespaceon
disk.

CompressModelweightsto4and8bitsusingNNCF`:math:`\Uparrow`<#table-of-content>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

**Note**:ThereisnospeedupforINT4compressedmodelsondGPU.

ConvertmodeltoOpenVINOIRformat`:math:`\Uparrow`<#table-of-content>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ConvertmodeltoOpenVINOformatusingconversionhelperfunction
definedabove.

PleaseselectbelowwhetheryouwouldliketorunINT4weight
compressioninsteadofINT8weightcompression.

..code::ipython3

importipywidgetsaswidgets

compression_mode=widgets.Dropdown(
options=["INT4","INT8"],
value="INT4",
description="Compressionmode:",
disabled=False,
)

compression_mode




..parsed-literal::

Dropdown(description='Compressionmode:',options=('INT4','INT8'),value='INT4')



..code::ipython3

ifcompression_mode.value=="INT4":
compressed_model_dir=Path("llava-mpt/INT4_compressed_weights")
llava_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT4_ASYM,group_size=128,ratio=0.8)
else:
compressed_model_dir=Path("llava-mpt/INT8_compressed_weights")
llava_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT8)

ifnotcompressed_model_dir.exists():
compressed_model_dir.mkdir(exist_ok=True,parents=True)
config.save_pretrained(compressed_model_dir)
model=LlavaMptForCausalLM.from_pretrained(model_id)
vision_tower=model.get_vision_tower()
ifnotvision_tower.is_loaded:
vision_tower.load_model()

ifmm_use_im_start_end:
model.resize_token_embeddings(len(tokenizer))

model.eval()
withtorch.no_grad():
convert_llava_mpt(
model,
compressed_model_dir,
image_encoder_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT8),
llava_wc_parameters=llava_wc_parameters,
)
delmodel
gc.collect();



..parsed-literal::

Loadingcheckpointshards:0%||0/2[00:00<?,?it/s]


..parsed-literal::

Applyingweightcompressiontoimageencoder
INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(139/139)|100%(139/139)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

ImageEncodermodelsuccessfullyconverted
TokenEmbeddingmodelsuccessfullyconverted
ApplyingweightcompressiontofirststageLLavamodel



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|23%(38/129)|21%(37/128)|
+--------------+---------------------------+-----------------------------------+
|4|77%(91/129)|79%(91/128)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

ApplyingweightcompressiontosecondstageLLavamodel



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|26%(39/130)|21%(37/128)|
+--------------+---------------------------+-----------------------------------+
|4|74%(91/130)|79%(91/128)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

LLaVAmodelsuccessfullyconverted


PrepareOpenVINObasedinferencepipeline
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

``OVLlavaMPTForCausalLM``classprovidesease-to-useinterfaceforusing
modelingenerationscenario.Itisbasedon
``transformers.generation.GenerationMixin``thatgivesusopportunityto
reuseallreachcapabilitiesforgenerationimplementedinHuggingFace
Transformerslibrary.Moredetailsaboutthisinterfacecanbefoundin
`HuggingFace
documentation<https://huggingface.co/docs/transformers/main_classes/text_generation>`__.

..code::ipython3

fromtransformers.generationimportGenerationConfig,GenerationMixin
fromtransformers.modeling_outputsimportCausalLMOutputWithPast
fromtransformersimportAutoConfig
importnumpyasnp
importtorch


classOVLlavaMPTForCausalLM(GenerationMixin):
def__init__(self,core,model_dir,device):
self.image_encoder=core.compile_model(model_dir/"image_encoder.xml",device)
self.token_embed=core.compile_model(model_dir/"token_embed.xml",device)
self.model=core.read_model(model_dir/"llava_with_past.xml")
self.model_input_embed=core.compile_model(model_dir/"llava_input_embed.xml",device)
self.input_names={key.get_any_name():idxforidx,keyinenumerate(self.model.inputs)}
self.output_names={key.get_any_name():idxforidx,keyinenumerate(self.model.outputs)}
self.key_value_input_names=[keyforkeyinself.input_namesif"key_values"inkey]
self.key_value_output_names=[keyforkeyinself.output_namesif"present"inkey]
compiled_model=core.compile_model(self.model,device)
self.request=compiled_model.create_infer_request()
self.config=AutoConfig.from_pretrained(model_dir)
self.generation_config=GenerationConfig.from_model_config(config)
self.main_input_name="input_ids"
self.device=torch.device("cpu")
self.num_pkv=2
self._supports_cache_class=False

defcan_generate(self):
"""ReturnsTruetovalidatethecheckthatthemodelusing`GenerationMixin.generate()`canindeedgenerate."""
returnTrue

def__call__(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
prefix_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->CausalLMOutputWithPast:
returnself.forward(input_ids,images,attention_mask,prefix_mask,past_key_values)

defforward(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
prefix_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->CausalLMOutputWithPast:
"""Generalinferencemethod"""
inputs={}
ifpast_key_valuesisnotNone:
#Flattenthepast_key_values
attention_mask=torch.ones(
(input_ids.shape[0],past_key_values[-1][-1].shape[-2]+1),
dtype=input_ids.dtype,
)
past_key_values=tuple(past_key_valueforpkv_per_layerinpast_key_valuesforpast_key_valueinpkv_per_layer)
#Addthepast_key_valuestothedecoderinputs
inputs=dict(zip(self.key_value_input_names,past_key_values))

else:
returnself.forward_with_image(input_ids,images,attention_mask)
inputs["input_ids"]=np.array(input_ids)

if"attention_mask"inself.input_names:
inputs["attention_mask"]=np.array(attention_mask)

#Runinference
self.request.start_async(inputs,share_inputs=True)
self.request.wait()

logits=torch.from_numpy(self.request.get_tensor("logits").data)

#Tupleoflengthequalto:numberoflayer*numberofpast_key_valueperdecoderlayer(2correspondstotheself-attentionlayer)
past_key_values=tuple(self.request.get_tensor(key).dataforkeyinself.key_value_output_names)
#Tupleoftupleoflength`n_layers`,witheachtupleoflengthequalto2(k/vofself-attention)

past_key_values=tuple(past_key_values[i:i+self.num_pkv]foriinrange(0,len(past_key_values),self.num_pkv))
returnCausalLMOutputWithPast(logits=logits,past_key_values=past_key_values)

defforward_with_image(self,input_ids,images,attention_mask):
"""Firststepinferencemethod,thatresolvesmultimodaldata"""
input_embed,attention_mask=self.prepare_multimodal_input(input_ids,images,attention_mask)
outs=self.model_input_embed([input_embed,attention_mask])
logits=outs[0]
pkv=list(outs.values())[1:]
pkv=tuple(pkv[i:i+self.num_pkv]foriinrange(0,len(pkv),self.num_pkv))
returnCausalLMOutputWithPast(logits=torch.from_numpy(logits),past_key_values=pkv)

defprepare_multimodal_input(self,input_ids,images,attention_mask):
"""Preprocessingfunctionforembeddingmultimodaldata"""
image_features=[]
ifimagesisnotNone:
image_features=self.image_encoder(images)[0]

new_input_embeds=[]
cur_image_idx=0
forbatch_idx,cur_input_idsinenumerate(input_ids):
if(cur_input_ids==IMAGE_TOKEN_INDEX).sum()==0:
#multimodalLLM,butthecurrentsampleisnotmultimodal
cur_input_embeds=torch.from_numpy(self.token_embed(cur_input_ids.unsqueeze(0))[0][0])
new_input_embeds.append(cur_input_embeds)
cur_image_idx+=1
continue
image_token_indices=torch.where(cur_input_ids==IMAGE_TOKEN_INDEX)[0]
cur_new_input_embeds=[]
whileimage_token_indices.numel()>0:
cur_image_features=image_features[cur_image_idx]
image_token_start=image_token_indices[0]
ifgetattr(self.config,"tune_mm_mlp_adapter",False)andgetattr(self.config,"mm_use_im_start_end",False):
embd=self.token_embed(cur_input_ids[:image_token_start-1].unsqueeze(0))[0][0]
cur_new_input_embeds.append(embd)
embd=self.token_embed(cur_input_ids[image_token_start-1:image_token_start].unsqueeze(0))[0][0]
cur_new_input_embeds.append(embd)
cur_new_input_embeds.append(cur_image_features)
embd=self.token_embed(cur_input_ids[image_token_start+1:image_token_start+2].unsqueeze(0))[0][0]
cur_new_input_embeds.append(embd)
else:
cur_new_input_embeds.append(self.token_embed(cur_input_ids[:image_token_start].unsqueeze(0))[0][0])
cur_new_input_embeds.append(cur_image_features)
cur_image_idx+=1
ifgetattr(self.config,"tune_mm_mlp_adapter",False)andgetattr(self.config,"mm_use_im_start_end",False):
cur_input_ids=cur_input_ids[image_token_start+2:]
else:
cur_input_ids=cur_input_ids[image_token_start+1:]
image_token_indices=torch.where(cur_input_ids==IMAGE_TOKEN_INDEX)[0]
ifcur_input_ids.numel()>0:
ifgetattr(self.config,"tune_mm_mlp_adapter",False)andgetattr(self.config,"mm_use_im_start_end",False):
cur_new_input_embeds.append(self.token_embed(cur_input_ids.unsqueeze(0))[0][0])
else:
cur_new_input_embeds.append(self.token_embed(cur_input_ids.unsqueeze(0))[0][0])
cur_new_input_embeds=[torch.from_numpy(x)forxincur_new_input_embeds]
cur_new_input_embeds=torch.cat(cur_new_input_embeds,dim=0)
new_input_embeds.append(cur_new_input_embeds)

ifany(x.shape!=new_input_embeds[0].shapeforxinnew_input_embeds):
max_len=max(x.shape[0]forxinnew_input_embeds)

new_input_embeds_align=[]
forcur_new_embedinnew_input_embeds:
cur_new_embed=torch.cat(
(
cur_new_embed,
torch.zeros(
(max_len-cur_new_embed.shape[0],cur_new_embed.shape[1]),
dtype=cur_new_embed.dtype,
),
),
dim=0,
)
new_input_embeds_align.append(cur_new_embed)
new_input_embeds=torch.stack(new_input_embeds_align,dim=0)

ifattention_maskisnotNone:
new_attention_mask=[]
forcur_attention_mask,cur_new_labels,cur_new_labels_aligninzip(attention_mask,_new_labels,new_labels):
new_attn_mask_pad_left=torch.full(
(cur_new_labels.shape[0]-labels.shape[1],),
True,
dtype=attention_mask.dtype,
)
new_attn_mask_pad_right=torch.full(
(cur_new_labels_align.shape[0]-cur_new_labels.shape[0],),
False,
dtype=attention_mask.dtype,
)
cur_new_attention_mask=torch.cat(
(
new_attn_mask_pad_left,
cur_attention_mask,
new_attn_mask_pad_right,
),
dim=0,
)
new_attention_mask.append(cur_new_attention_mask)
attention_mask=torch.stack(new_attention_mask,dim=0)
assertattention_mask.shape==new_labels.shape
else:
new_input_embeds=torch.stack(new_input_embeds,dim=0)

ifattention_maskisnotNone:
new_attn_mask_pad_left=torch.full(
(
attention_mask.shape[0],
new_input_embeds.shape[1]-input_ids.shape[1],
),
True,
dtype=attention_mask.dtype,
)
attention_mask=torch.cat((new_attn_mask_pad_left,attention_mask),dim=1)
assertattention_mask.shape==new_input_embeds.shape[:2]

returnnew_input_embeds,attention_mask

defprepare_inputs_for_generation(self,input_ids,past_key_values=None,**kwargs):
"""
ThisfunctionisusedduringrunningGenerationMixin.generateforpreparingmodelspecificinputsfor
eachgenerationstep
"""
past_len=0
ifpast_key_valuesisnotNone:
input_ids=input_ids[:,-1].unsqueeze(-1)
past_len=past_key_values[-1][-1].shape[-2]
attention_mask=kwargs.get(
"attention_mask",
torch.ones(input_ids.shape[0],input_ids.shape[1]+past_len),
)
ifnotkwargs.get("use_cache",True):
raiseNotImplementedError("MPTwithprefix_lm=Truedoesnotsupportuse_cache=False.")
else:
prefix_mask=None
return{
"input_ids":input_ids,
"attention_mask":attention_mask,
"prefix_mask":prefix_mask,
"past_key_values":past_key_values,
"images":kwargs.get("images",None),
}

def_reorder_cache(self,past_key_values:Tuple[Tuple[torch.Tensor]],beam_idx:torch.Tensor)->Tuple[Tuple[torch.Tensor]]:
"""
Thisfunctionisusedtore-orderthe`past_key_values`cacheif[`~PreTrainedModel.beam_search`]or
[`~PreTrainedModel.beam_sample`]iscalled.
Thisisrequiredtomatch`past_key_values`withthecorrectbeam_idxateverygenerationstep.
"""

#fromtransformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
returntuple(tuple(np.take(past_state,beam_idx,0)forpast_stateinlayer_past)forlayer_pastinpast_key_values)

Runmodelinference
-------------------

`backtotop⬆️<#table-of-contents>`__

Now,whenwehavemodelanddefinedgenerationpipeline,wecanrun
modelinference.

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

**Note**:ThereisnospeedupforINT4compressedmodelsondGPU.

..code::ipython3

importipywidgetsaswidgets

core=ov.Core()

support_devices=core.available_devices
if"NPU"insupport_devices:
support_devices.remove("NPU")

device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



LoadOpenVINOmodel
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_model=OVLlavaMPTForCausalLM(core,compressed_model_dir,device.value)

Prepareinputdata
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forpreparinginputdata,wewillusetokenizerandimageprocessor
definedinthebeggingofourtutorial.Foralignmentwithoriginal
PyTorchimplementationwewillusePyTorchtensorsasinput.

..code::ipython3

importrequests
fromPILimportImage
fromioimportBytesIO


defload_image(image_file):
ifimage_file.startswith("http")orimage_file.startswith("https"):
response=requests.get(image_file)
image=Image.open(BytesIO(response.content)).convert("RGB")
else:
image=Image.open(image_file).convert("RGB")
returnimage


image_file="https://llava-vl.github.io/static/images/view.jpg"

image=load_image(image_file)
image_tensor=image_processor.preprocess(image,return_tensors="pt")["pixel_values"]

text_message="WhatarethethingsIshouldbecautiousaboutwhenIvisithere?"
print(f"Question:{text_message}")
image


..parsed-literal::

Question:WhatarethethingsIshouldbecautiousaboutwhenIvisithere?




..image::llava-multimodal-chatbot-with-output_files/llava-multimodal-chatbot-with-output_20_1.png



Testmodelinference
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Generationprocessforlongresponsemaybetimeconsuming,foraccessing
partialresultassoonasitisgeneratedwithoutwaitingwhenwhole
processfinished,StreamingAPIcanbeused.Tokenstreamingisthemode
inwhichthegenerativesystemreturnsthetokensonebyoneasthe
modelgeneratesthem.Thisenablesshowingprogressivegenerationsto
theuserratherthanwaitingforthewholegeneration.Streamingisan
essentialaspectoftheend-userexperienceasitreduceslatency,one
ofthemostcriticalaspectsofasmoothexperience.Youcanfindmore
detailsabouthowstreamingworkin`HuggingFace
documentation<https://huggingface.co/docs/text-generation-inference/conceptual/streaming>`__.

Alsoforsimplificationofpreparinginputinconversationalmode,we
willuseConversationTemplatehelperprovidedbymodelauthorsfor
accumulatinghistoryofprovidedmessagesandimages.

..code::ipython3

fromllava.mm_utilsimporttokenizer_image_token,KeywordsStoppingCriteria
fromllava.constantsimportIMAGE_TOKEN_INDEX
fromtransformersimportTextStreamer
fromllava.conversationimportconv_templates,SeparatorStyle

#Prepare
streamer=TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
conv_mode="mpt"

conv=conv_templates[conv_mode].copy()
roles=("user","assistant")

ifmm_use_im_start_end:
inp=DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN+"\n"+text_message
else:
inp=DEFAULT_IMAGE_TOKEN+"\n"+text_message
conv.append_message(conv.roles[0],inp)
conv.append_message(conv.roles[1],None)

prompt=conv.get_prompt()
input_ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0)
stop_str=conv.sepifconv.sep_style!=SeparatorStyle.TWOelseconv.sep2
keywords=[stop_str]
stopping_criteria=KeywordsStoppingCriteria(keywords,tokenizer,input_ids)
streamer=TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
print("Answer:")

output_ids=ov_model.generate(
input_ids,
images=image_tensor,
do_sample=True,
temperature=0.2,
max_new_tokens=1024,
streamer=streamer,
use_cache=True,
stopping_criteria=[stopping_criteria],
)


..parsed-literal::

Answer:
Whenvisitingthislocation,Ishouldbecautiousaboutthewaterlevelandthepresenceofboats.Theimageshowsadockwithaboatinthewater,andthewaterappearstoberelativelyshallow.Itisessentialtobemindfulofthewaterdepthwhenapproachingthedock,asitcouldbedangeroustostepintothewaterwithoutcheckingthewaterlevel.Additionally,Ishouldbeawareoftheboatsinthewater,astheycouldposeariskiftheyarenotproperlysecuredoriftheyarenotbeingusedasintended.Itiscrucialtomaintainasafedistancefromtheboatsandfollowanypostedsignsorguidelinestoensureasafeandenjoyableexperience.


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
fromthreadingimportEvent,Thread
fromtransformersimportTextIteratorStreamer

title_markdown="""
#🌋LLaVA:LargeLanguageandVisionAssistant
"""

tos_markdown="""
###Termsofuse
Byusingthisservice,usersarerequiredtoagreetothefollowingterms:
Theserviceisaresearchpreviewintendedfornon-commercialuseonly.Itonlyprovideslimitedsafetymeasuresandmaygenerateoffensivecontent.Itmustnotbeusedforanyillegal,harmful,violent,racist,orsexualpurposes.Theservicemaycollectuserdialoguedataforfutureresearch.
"""

conv=conv_templates[conv_mode].copy()
conv.messages=[]


defclear_history(textbox,imagebox,chatbot):
"""
callbackfunctionforclearingchatwindowsininterfaceonclearbuttonclick

Params:
textbox:currenttextboxforusermessagesstate
imagebox:currentimageboxstate
chatbot:currentchatbotstate
Returns:
emptytextbox,imageboxandchatbotstates
"""
conv.messages=[]

returnNone,None,None


defuser(message,history):
"""
callbackfunctionforupdatingusermessagesininterfaceonsubmitbuttonclick

Params:
message:currentmessage
history:conversationhistory
Returns:
updatedmessageandconversationhistory
"""
#Appendtheuser'smessagetotheconversationhistory
return"",history+[[message,""]]


defbot(image,history,temperature=0.2,top_p=0.7,max_new_tokens=1024):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory
temperature:parameterforcontrolthelevelofcreativityinAI-generatedtext.
Byadjustingthe`temperature`,youcaninfluencetheAImodel'sprobabilitydistribution,makingthetextmorefocusedordiverse.
top_p:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability.

"""

text=history[-1][0]
iflen(text)<=0andimageisNone:
conv.skip_next=True
yieldhistory
text=text[:1536]#Hardcut-off
ifimageisnotNone:
text=text[:1200]#Hardcut-offforimages
if"<image>"notintext:
text=text+"\n<image>"
text=(text,image,"Resize")
conv.append_message(conv.roles[0],text)
conv.append_message(conv.roles[1],None)
conv.skip_next=False

#Constructtheinputmessagestringforthemodelbyconcatenatingthecurrentsystemmessageandconversationhistory
prompt=conv.get_prompt()
image=conv.get_images(return_pil=True)
ifnotimage:
image_tensor=None
else:
image_tensor=image_processor.preprocess(image,return_tensors="pt")["pixel_values"]
input_ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0)
stop_str=conv.sepifconv.sep_style!=SeparatorStyle.TWOelseconv.sep2
keywords=[stop_str]
stopping_criteria=KeywordsStoppingCriteria(keywords,tokenizer,input_ids)
#Tokenizethemessagesstring
streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
generate_kwargs=dict(
input_ids=input_ids,
images=image_tensor,
max_new_tokens=max_new_tokens,
temperature=temperature,
do_sample=temperature>0.001,
top_p=top_p,
streamer=streamer,
use_cache=True,
stopping_criteria=[stopping_criteria],
)

stream_complete=Event()

defgenerate_and_signal_complete():
"""
genrationfunctionforsinglethread
"""
ov_model.generate(**generate_kwargs)
stream_complete.set()

t1=Thread(target=generate_and_signal_complete)
t1.start()

#Initializeanemptystringtostorethegeneratedtext
partial_text=""
fornew_textinstreamer:
ifnotnew_text:
continue
partial_text+=new_text
conv.messages[-1][-1]=partial_text
history[-1][1]=partial_text
yieldhistory


withgr.Blocks(title="LLaVA")asdemo:
gr.Markdown(title_markdown)

withgr.Row():
withgr.Column():
imagebox=gr.Image(type="pil")
withgr.Accordion("Parameters",open=False,visible=True)asparameter_row:
temperature=gr.Slider(
minimum=0.0,
maximum=1.0,
value=0.2,
step=0.1,
interactive=True,
label="Temperature",
)
top_p=gr.Slider(
minimum=0.0,
maximum=1.0,
value=0.7,
step=0.1,
interactive=True,
label="TopP",
)
max_output_tokens=gr.Slider(
minimum=0,
maximum=1024,
value=512,
step=64,
interactive=True,
label="Maxoutputtokens",
)

withgr.Column(scale=3):
withgr.Column(scale=6):
chatbot=gr.Chatbot(height=400)
withgr.Row():
withgr.Column(scale=8):
textbox=gr.Textbox(
show_label=False,
placeholder="EntertextandpressENTER",
visible=True,
container=False,
)
withgr.Column(scale=1,min_width=60):
submit_btn=gr.Button(value="Submit",visible=True)
withgr.Row(visible=True)asbutton_row:
clear_btn=gr.Button(value="🗑️Clearhistory",interactive=True)

gr.Markdown(tos_markdown)

submit_event=textbox.submit(
fn=user,
inputs=[textbox,chatbot],
outputs=[textbox,chatbot],
queue=False,
).then(
bot,
[imagebox,chatbot,temperature,top_p,max_output_tokens],
chatbot,
queue=True,
)
#Registerlisteners
clear_btn.click(clear_history,[textbox,imagebox,chatbot],[chatbot,textbox,imagebox])
submit_click_event=submit_btn.click(
fn=user,
inputs=[textbox,chatbot],
outputs=[textbox,chatbot],
queue=False,
).then(
bot,
[imagebox,chatbot,temperature,top_p,max_output_tokens],
chatbot,
queue=True,
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
try:
demo.queue(max_size=2).launch(debug=False)
exceptException:
demo.queue(max_size=2).launch(share=True,debug=False)

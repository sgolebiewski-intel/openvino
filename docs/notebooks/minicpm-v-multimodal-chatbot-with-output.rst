Visual-languageassistantwithMiniCPM-V2andOpenVINO
======================================================

MiniCPM-V2isastrongmultimodallargelanguagemodelforefficient
end-sidedeployment.ThemodelisbuiltbasedonSigLip-400Mand
MiniCPM-2.4B,connectedbyaperceiverresampler.MiniCPM-V2.0has
severalnotablefeatures:\***Outperformingmanypopularmodelsonmany
benchmarks**(includingOCRBench,TextVQA,MME,MMB,MathVista,etc).
StrongOCRcapability,achievingcomparableperformancetoGeminiProin
scene-textunderstanding.\***TrustworthyBehavior**.LLMsareknown
forsufferingfromhallucination,oftengeneratingtextnotfactually
groundedinimages.MiniCPM-V2.0isthefirstend-sideLLMalignedvia
multimodalRLHFfortrustworthybehavior(usingtherecent
`RLHF-V<https://rlhf-v.github.io/>`__[CVPR’24]seriestechnique).This
allowsthemodeltomatchGPT-4VinpreventinghallucinationsonObject
HalBench.\***High-ResolutionImagesatAnyAspectRaito.**MiniCPM-V
2.0canaccept1.8millionpixels(e.g.,1344x1344)imagesatanyaspect
ratio.Thisenablesbetterperceptionoffine-grainedvisualinformation
suchassmallobjectsandopticalcharacters,whichisachievedviaa
recenttechniquefrom`LLaVA-UHD<https://arxiv.org/pdf/2403.11703>`__.
\***HighEfficiency.**Forvisualencoding,modelcompressestheimage
representationsintomuchfewertokensviaaperceiverresampler.This
allowsMiniCPM-V2.0tooperatewithfavorablememorycostandspeed
duringinferenceevenwhendealingwithhigh-resolutionimages.\*
**BilingualSupport.**MiniCPM-V2.0supportsstrongbilingual
multimodalcapabilitiesinbothEnglishandChinese.Thisisenabledby
generalizingmultimodalcapabilitiesacrosslanguages,atechniquefrom
`VisCPM<https://arxiv.org/abs/2308.12038>`__\[ICLR’24].

InthistutorialweconsiderhowtoconvertandoptimizeMiniCPM-V2
modelforcreatingmultimodalchatbot.Additionally,wedemonstratehow
toapplystatefultransformationonLLMpartandmodeloptimization
techniqueslikeweightscompressionusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`DownloadPyTorchmodel<#download-pytorch-model>`__
-`ConvertmodeltoOpenVINOIntermediate
Representation<#convert-model-to-openvino-intermediate-representation>`__

-`Textembeddings<#text-embeddings>`__
-`LanguageModel<#language-model>`__
-`CompressLanguageModelWeightsto4
bits<#compress-language-model-weights-to-4-bits>`__
-`ImageEncoder<#image-encoder>`__

-`Preparemodelinference
pipeline<#prepare-model-inference-pipeline>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectdevice<#select-device>`__
-`Selectmodelvariant<#select-model-variant>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.1""torchvision""timm""transformers>=4.40""Pillow""gradio>=4.19""tqdm""sentencepiece"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2024.2.0""nncf>=2.11.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


DownloadPyTorchmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtransformersimportAutoModel,AutoTokenizer
frompathlibimportPath

model_dir=Path("model")
text_emb_path=model_dir/"language_model/embed_tokens.xml"
image_encoder_path=model_dir/"image_encoder.xml"
llm_path=model_dir/"language_model/language_model.xml"

model=None

ifnotall([text_emb_path.exists(),image_encoder_path.exists(),llm_path.exists()]):
model=AutoModel.from_pretrained("openbmb/MiniCPM-V-2",trust_remote_code=True)
model.eval()
model.config.save_pretrained(model_dir)
tokenizer=AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2",trust_remote_code=True)
tokenizer.save_pretrained(model_dir)



..parsed-literal::

Loadingcheckpointshards:0%||0/2[00:00<?,?it/s]


ConvertmodeltoOpenVINOIntermediateRepresentation
-----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversiontoOpenVINOIntermediate
Representation(IR).`OpenVINOmodelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
shouldbeusedforthesepurposes.``ov.convert_model``functionaccepts
originalPyTorchmodelinstanceandexampleinputfortracingand
returns``ov.Model``representingthismodelinOpenVINOframework.
Convertedmodelcanbeusedforsavingondiskusing``ov.save_model``
functionordirectlyloadingondeviceusing``core.complie_model``.

MiniCPM-V2isautoregressivetransformergenerativemodel,itmeansthat
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

-**ImageEncoder**forencodinginputimagesintoembeddingspace.It
includesSigLIPmodelandResampler.
-**InputEmbedding**forconversioninputtexttokensintoembedding
space
-**LanguageModel**forgenerationanswerbasedoninputembeddings
providedbyImageEncoderandInputEmbeddingmodels.

Let’sconverteachmodelpart.

Textembeddings
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

InLLMs,inputembeddingisapartoflanguagemodel,butformultimodal
case,thefirststephiddenstateproducedbythismodelpartshouldbe
integratedwithimageembeddingsintocommonembeddingspace.For
abilitytoreusethismodelpartandavoidintroductionofllmmodel
instance,wewilluseitseparately.

..code::ipython3

importopenvinoasov
importtorch
importgc


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


ifnottext_emb_path.exists():
ov_model=ov.convert_model(model.llm.model.embed_tokens,example_input=torch.ones([1,10],dtype=torch.long))

ov.save_model(ov_model,text_emb_path)
delov_model
cleanup_torchscript_cache()
gc.collect()


..parsed-literal::

['input']


LanguageModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LanguageModelisresponsibleforgenerationanswerinMiniCPM-V.This
partisverysimilartostandardLLMfortextgeneration.Ourmodeluses
`MiniCPM-2.4B<https://github.com/OpenBMB/MiniCPM/>`__asbaseLLM.To
optimizethegenerationprocessandusememorymoreefficiently,
HuggingFacetransformersAPIprovidesamechanismforcachingmodel
stateexternallyusing``use_cache=True``parameterand
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

importtypes
fromtransformers.cache_utilsimportCache,DynamicCache
fromtransformers.modeling_attn_mask_utilsimport_prepare_4d_causal_attention_mask
fromtransformers.modeling_outputsimportBaseModelOutputWithPast,CausalLMOutputWithPast
fromtypingimportUnion


defforward_wrap(self,attention_mask,position_ids,past_key_values,inputs_embeds):
result=self._orig_forward(
input_ids=None,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds
)
returntuple(result.values())


def_update_causal_mask(
self,
attention_mask:torch.Tensor,
input_tensor:torch.Tensor,
cache_position:torch.Tensor,
past_key_values:Cache,
output_attentions:bool,
):
past_seen_tokens=past_key_values.get_seq_length()ifpast_key_valuesisnotNoneelse0

dtype,device=input_tensor.dtype,input_tensor.device
min_dtype=torch.finfo(dtype).min
sequence_length=input_tensor.shape[1]

target_length=attention_mask.shape[-1]ifisinstance(attention_mask,torch.Tensor)elsepast_seen_tokens+sequence_length+1

ifattention_maskisnotNoneandattention_mask.dim()==4:
#inthiscaseweassumethatthemaskcomesalreadyininvertedformandrequiresnoinversionorslicing
ifattention_mask.max()!=0:
raiseValueError("Custom4Dattentionmaskshouldbepassedininvertedformwithmax==0`")
causal_mask=attention_mask
else:
causal_mask=torch.full((sequence_length,target_length),fill_value=min_dtype,dtype=dtype,device=device)
ifsequence_length!=1:
causal_mask=torch.triu(causal_mask,diagonal=1)
causal_mask*=torch.arange(target_length,device=device)>cache_position.reshape(-1,1)
causal_mask=causal_mask[None,None,:,:].expand(input_tensor.shape[0],1,-1,-1)
ifattention_maskisnotNone:
causal_mask=causal_mask.clone()#copytocontiguousmemoryforin-placeedit
mask_length=attention_mask.shape[-1]
padding_mask=causal_mask[:,:,:,:mask_length]+attention_mask[:,None,None,:]
padding_mask=padding_mask==0
causal_mask[:,:,:,:mask_length]=causal_mask[:,:,:,:mask_length].masked_fill(padding_mask,min_dtype)

returncausal_mask


def_model_forward(
self,
input_ids:torch.LongTensor=None,
attention_mask:Optional[torch.Tensor]=None,
position_ids:Optional[torch.LongTensor]=None,
past_key_values:Optional[List[torch.FloatTensor]]=None,
inputs_embeds:Optional[torch.FloatTensor]=None,
use_cache:Optional[bool]=None,
output_attentions:Optional[bool]=None,
output_hidden_states:Optional[bool]=None,
return_dict:Optional[bool]=None,
)->Union[Tuple,BaseModelOutputWithPast]:
output_attentions=output_attentionsifoutput_attentionsisnotNoneelseself.config.output_attentions
output_hidden_states=output_hidden_statesifoutput_hidden_statesisnotNoneelseself.config.output_hidden_states
use_cache=use_cacheifuse_cacheisnotNoneelseself.config.use_cache

return_dict=return_dictifreturn_dictisnotNoneelseself.config.use_return_dict

#retrieveinput_idsandinputs_embeds
ifinput_idsisnotNoneandinputs_embedsisnotNone:
raiseValueError("Youcannotspecifybothinput_idsandinputs_embedsatthesametime")
elifinput_idsisnotNone:
batch_size,seq_length=input_ids.shape[:2]
elifinputs_embedsisnotNone:
batch_size,seq_length=inputs_embeds.shape[:2]
else:
raiseValueError("Youhavetospecifyeitherinput_idsorinputs_embeds")

past_key_values_length=0
ifuse_cache:
use_legacy_cache=notisinstance(past_key_values,Cache)
ifuse_legacy_cache:
past_key_values=DynamicCache.from_legacy_cache(past_key_values)
past_key_values_length=past_key_values.get_usable_length(seq_length)

ifposition_idsisNone:
device=input_ids.deviceifinput_idsisnotNoneelseinputs_embeds.device
position_ids=torch.arange(
past_key_values_length,
seq_length+past_key_values_length,
dtype=torch.long,
device=device,
)
position_ids=position_ids.unsqueeze(0)

ifinputs_embedsisNone:
inputs_embeds=self.embed_tokens(input_ids)*self.config.scale_emb
ifself._use_sdpaandnotoutput_attentions:
#output_attentions=TruecannotbesupportedwhenusingSDPA,andwefallbackon
#themanualimplementationthatrequiresa4Dcausalmaskinallcases.
past_seen_tokens=past_key_values.get_seq_length()ifpast_key_valuesisnotNoneelse0
cache_position=torch.arange(past_seen_tokens,past_seen_tokens+inputs_embeds.shape[1],device=inputs_embeds.device)
attention_mask=self._update_causal_mask(attention_mask,inputs_embeds,cache_position,past_key_values,output_attentions)
else:
#4dmaskispassedthroughthelayers
attention_mask=_prepare_4d_causal_attention_mask(
attention_mask,
(batch_size,seq_length),
inputs_embeds,
past_key_values_length,
)

#embedpositions
hidden_states=inputs_embeds

#decoderlayers
all_hidden_states=()ifoutput_hidden_stateselseNone
all_self_attns=()ifoutput_attentionselseNone
next_decoder_cache=None

fordecoder_layerinself.layers:
ifoutput_hidden_states:
all_hidden_states+=(hidden_states,)

layer_outputs=decoder_layer(
hidden_states,
attention_mask=attention_mask,
position_ids=position_ids,
past_key_value=past_key_values,
output_attentions=output_attentions,
use_cache=use_cache,
)

hidden_states=layer_outputs[0]

ifuse_cache:
next_decoder_cache=layer_outputs[2ifoutput_attentionselse1]

ifoutput_attentions:
all_self_attns+=(layer_outputs[1],)

hidden_states=self.norm(hidden_states)

#addhiddenstatesfromthelastdecoderlayer
ifoutput_hidden_states:
all_hidden_states+=(hidden_states,)

next_cache=None
ifuse_cache:
next_cache=next_decoder_cache.to_legacy_cache()ifuse_legacy_cacheelsenext_decoder_cache
ifnotreturn_dict:
returntuple(vforvin[hidden_states,next_cache,all_hidden_states,all_self_attns]ifvisnotNone)
returnBaseModelOutputWithPast(
last_hidden_state=hidden_states,
past_key_values=next_cache,
hidden_states=all_hidden_states,
attentions=all_self_attns,
)


ifnotllm_path.exists():
model.llm.model.forward=types.MethodType(_model_forward,model.llm.model)
model.llm.model._update_causal_mask=types.MethodType(_update_causal_mask,model.llm.model)
llm_input=torch.zeros([2,2,2304])
pkv=model.llm(inputs_embeds=llm_input,attention_mask=torch.ones((2,2),dtype=torch.int64))[1]
model_inputs=["attention_mask","position_ids"]
model_outputs=["logits"]
foridxinrange(len(pkv)):
model_inputs.extend([f"past_key_values.{idx}.key",f"past_key_values.{idx}.value"])
model_outputs.extend([f"present.{idx}.key",f"present.{idx}.value"])
model_inputs.append("inputs_embeds")
model.llm._orig_forward=model.llm.forward

model.llm.forward=types.MethodType(forward_wrap,model.llm)
position_ids=torch.tensor([[2,3],[2,3]])
ov_model=ov.convert_model(
model.llm,
example_input={
"inputs_embeds":llm_input,
"attention_mask":torch.ones([2,4],dtype=torch.int64),
"past_key_values":pkv,
"position_ids":position_ids,
},
)

forinput,input_nameinzip(ov_model.inputs,model_inputs):
input.get_tensor().set_names({input_name})

foroutput,output_nameinzip(ov_model.outputs,model_outputs):
output.get_tensor().set_names({output_name})
patch_stateful(ov_model)

ov.save_model(ov_model,llm_path)
model.llm.config.save_pretrained(llm_path.parent)
delov_model
cleanup_torchscript_cache()
delmodel.llm
gc.collect()


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/tmp/ipykernel_150470/514161198.py:38:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifsequence_length!=1:
/opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V-2/187851962daa9b63072d40ec802f597b71bff532/modeling_minicpm.py:176:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifseq_len>self.max_seq_len_cached:
/opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V-2/187851962daa9b63072d40ec802f597b71bff532/modeling_minicpm.py:883:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.size()!=(bsz,1,q_len,kv_seq_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:165:UserWarning:The.gradattributeofaTensorthatisnotaleafTensorisbeingaccessed.Its.gradattributewon'tbepopulatedduringautograd.backward().Ifyouindeedwantthe.gradfieldtobepopulatedforanon-leafTensor,use.retain_grad()onthenon-leafTensor.Ifyouaccessthenon-leafTensorbymistake,makesureyouaccesstheleafTensorinstead.Seegithub.com/pytorch/pytorch/pull/30531formoreinformations.(Triggeredinternallyataten/src/ATen/core/TensorBody.h:489.)
ifa.gradisnotNone:


..parsed-literal::

['attention_mask','position_ids','past_key_values','inputs_embeds']


CompressLanguageModelWeightsto4bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
importshutil

compression_configuration={
"mode":nncf.CompressWeightsMode.INT4_SYM,
"group_size":64,
"ratio":0.6,
}


core=ov.Core()
llm_int4_path=llm_path.parent.parent/"language_model_int4"/llm_path.name
ifto_compress_weights.valueandnotllm_int4_path.exists():
ov_model=core.read_model(llm_path)
ov_compressed_model=nncf.compress_weights(ov_model,**compression_configuration)
ov.save_model(ov_compressed_model,llm_int4_path)
delov_compressed_model
delov_model
gc.collect()
shutil.copy(text_emb_path,llm_int4_path.parent/text_emb_path.name)
shutil.copy(text_emb_path.with_suffix(".bin"),llm_int4_path.parent/text_emb_path.with_suffix(".bin").name)
shutil.copy(llm_path.parent/"config.json",llm_int4_path.parent/"config.json")


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

2024-07-1301:04:47.035322:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:04:47.077265:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:04:47.647632:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│46%(123/281)│40%(122/280)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│54%(158/281)│60%(158/280)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ImageEncoderisrepresentedinMiniCPM-Vbypretrained
`SigLIP<https://huggingface.co/google/siglip-so400m-patch14-384>`__
model.Additionally,MiniCPMusesperceiverresamplerthatcompresses
theimagerepresentations.Wewillcombinethemtogetherintoonemodel.

..code::ipython3

classImageEncoder(torch.nn.Module):
def__init__(self,vpm,resampler):
super().__init__()
self.vpm=vpm
self.resampler=resampler

defforward(self,pixel_values,tgt_size):
vision_embedding=self.vpm.forward_features(pixel_values)
ifhasattr(self.vpm,"num_prefix_tokens")andself.vpm.num_prefix_tokens>0:
vision_embedding=vision_embedding[:,self.vpm.num_prefix_tokens:]
ifself.resampler.adaptive:
pos_embed=(
self.get_2d_sincos_pos_embed(self.resampler.embed_dim,tgt_size).float().to(device=vision_embedding.device,dtype=vision_embedding.dtype)
)
else:
pos_embed=self.get_abs_pos(self.resampler.pos_embed,tgt_size)

x=self.resampler.kv_proj(vision_embedding)
x=self.resampler.ln_kv(x).permute(1,0,2)

N=x.shape[1]
q=self.resampler.ln_q(self.resampler.query)
out=self.resampler.attn(self.resampler._repeat(q,N)+self.resampler.pos_embed.unsqueeze(1),x+pos_embed.unsqueeze(1),x,attn_mask=None)[0]
x=out.permute(1,0,2)

x=self.resampler.ln_post(x)
x=x@self.resampler.proj
returnx

defget_2d_sincos_pos_embed(self,embed_dim,grid_size,cls_token=False):
"""
grid_size:intofthegridheightandwidth
return:
pos_embed:[grid_size*grid_size,embed_dim]or[1+grid_size*grid_size,embed_dim](w/orw/ocls_token)
"""

grid_h_size,grid_w_size=grid_size[0],grid_size[1]

grid_h=torch.arange(grid_h_size,dtype=torch.float32)
grid_w=torch.arange(grid_w_size,dtype=torch.float32)
grid=torch.meshgrid(grid_w,grid_h)#herewgoesfirst
grid=torch.stack(grid,dim=0)

grid=grid.reshape([2,1,grid_h.shape[0],grid_w.shape[0]])
pos_embed=self.get_2d_sincos_pos_embed_from_grid(embed_dim,grid)
ifcls_token:
pos_embed=torch.cat([torch.zeros([1,embed_dim]),pos_embed],dim=0)
returnpos_embed

defget_2d_sincos_pos_embed_from_grid(self,embed_dim,grid):
#usehalfofdimensionstoencodegrid_h
emb_h=self.get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[0])#(H*W,D/2)
emb_w=self.get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[1])#(H*W,D/2)

emb=torch.cat([emb_h,emb_w],dim=1)#(H*W,D)
returnemb

defget_1d_sincos_pos_embed_from_grid(self,embed_dim,pos):
"""
embed_dim:outputdimensionforeachposition
pos:alistofpositionstobeencoded:size(M,)
out:(M,D)
"""
assertembed_dim%2==0
omega=torch.arange(embed_dim//2,dtype=torch.float32)
omega/=embed_dim/2.0
omega=1.0/10000**omega#(D/2,)

pos=pos.reshape(-1)#(M,)
out=torch.einsum("m,d->md",pos,omega)#(M,D/2),outerproduct

emb_sin=torch.sin(out)#(M,D/2)
emb_cos=torch.cos(out)#(M,D/2)

emb=torch.cat([emb_sin,emb_cos],axis=1)#(M,D)
returnemb


ifnotimage_encoder_path.exists():
image_encoder=ImageEncoder(model.vpm,model.resampler)
ov_model=ov.convert_model(image_encoder,example_input=[torch.ones([1,3,448,448]),torch.tensor([32,32],dtype=torch.int32)])
ov.save_model(ov_model,image_encoder_path)
delov_model
cleanup_torchscript_cache()

delmodel
gc.collect()


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/layers/pos_embed.py:29:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnum_new_tokens==num_pos_tokensandnew_size[0]==new_size[1]:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/layers/pos_embed.py:33:TracerWarning:ConvertingatensortoaPythonfloatmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
hw=int(math.sqrt(num_pos_tokens-num_prefix_tokens))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/functional.py:512:UserWarning:torch.meshgrid:inanupcomingrelease,itwillberequiredtopasstheindexingargument.(Triggeredinternallyat../aten/src/ATen/native/TensorShape.cpp:3587.)
return_VF.meshgrid(tensors,**kwargs)#type:ignore[attr-defined]


..parsed-literal::

['pixel_values','tgt_size']




..parsed-literal::

3680



Preparemodelinferencepipeline
--------------------------------

`backtotop⬆️<#table-of-contents>`__

|image0|

Asdiscussed,themodelcomprisesImageEncoderandLLM(withseparated
textembeddingpart)thatgeneratesanswer.Let’sdefineLLMinference
classthatwillrepresentgenerationcycle,Itisbasedon`HuggingFace
Transformers
GenerationMixin<https://huggingface.co/docs/transformers/main_classes/text_generation>`__
andlookssimilarto`Optimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__\``OVModelForCausalLM``\that
isusedforLLMinference.

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/2727402e-3697-442e-beca-26b149967c84

..code::ipython3

fromtransformers.generationimportGenerationMixin
fromtransformersimportAutoConfig,GenerationConfig

core=ov.Core()


classOvModelForCausalLMWithEmb(GenerationMixin):
def__init__(self,model_dir,device="CPU",ov_config=None,compile=True)->None:
self._supports_cache_class=False
self.config=AutoConfig.from_pretrained(model_dir,trust_remote_code=True)
self.config.is_decoder=True
self.config.is_encoder_decoder=False
self.generation_config=GenerationConfig.from_model_config(self.config)
model_dir=Path(model_dir)
self.model=core.read_model(model_dir/"language_model.xml")
self.token_emb=core.read_model(model_dir/"embed_tokens.xml")
self.request=None
self.token_emb_request=None
self._device=device.upper()
self.device=torch.device("cpu")
self.ov_config=ov_config
self.next_beam_idx=None
self._past_length=None
self.input_names=[input_t.get_any_name()forinput_tinself.model.inputs]
self.main_input_name="input_ids"
ifcompile:
self.compile()

defcompile(self):
ifself.requestisNone:
self.request=core.compile_model(self.model,self._device,self.ov_config).create_infer_request()
self._compile_token_emb()

def_compile_token_emb(self):
ifself.token_emb_requestisNone:
self.token_emb_request=core.compile_model(self.token_emb,self._device,self.ov_config)

defto(self,device:str):
ifisinstance(device,str):
self._device=device.upper()
self.clear_requests()

returnself

defclear_requests(self):
delself.request
delself.token_emb_request
self.request=None
self.token_emb_request=None

defembed_tokens(self,input_ids:torch.LongTensor):
self._compile_token_emb()
res=self.token_emb_request(input_ids,share_inputs=True)
returnres[0]

defprepare_inputs(
self,
input_ids:torch.LongTensor,
attention_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
position_ids:Optional[torch.LongTensor]=None,
inputs_embeds:Optional[torch.FloatTensor]=None,
**kwargs,
):
batch_size=input_ids.shape[0]ifinput_idsisnotNoneelseinputs_embeds.shape[0]

inputs={}
#past_key_valuesarenotusedexplicitly,insteadtheyarehandledinsidethemodel
ifpast_key_valuesisNone:
#Thisisthefirstiterationinasequence,resetallstates
ifself.requestisnotNone:
self.request.reset_state()
#Setinitialvalueforthenextbeam_idxinputthatwillbeusedatthecurrentiteration
#andwillbeoptionallyupdatedby_reorder_cacheatthenextiterationsifbeam_searchisused
self.next_beam_idx=np.arange(batch_size,dtype=int)
self._past_length=0
past_len=self._get_past_length(past_key_values)

ifinputs_embedsisNone:
inputs_embeds=self.embed_tokens(input_idsifpast_key_valuesisNoneelseinput_ids[:,-1:])*self.config.scale_emb
inputs["inputs_embeds"]=inputs_embeds

#Addtheattention_maskinputswhenneeded
if"attention_mask"inself.input_namesor"position_ids"inself.input_names:
ifattention_maskisnotNone:
attention_mask=np.array(attention_mask)
else:
attention_mask=np.ones((inputs_embeds.shape[0],inputs_embeds.shape[1]+past_len),dtype=int)

if"attention_mask"inself.input_names:
inputs["attention_mask"]=attention_mask

if"position_ids"inself.input_names:
ifposition_idsisnotNone:
position_ids=np.array(position_ids)
else:
position_ids=np.cumsum(attention_mask,axis=1)-1
position_ids[attention_mask==0]=1
ifpast_key_values:
position_ids=position_ids[:,-input_ids.shape[1]:]

inputs["position_ids"]=position_ids

if"beam_idx"inself.input_names:
inputs["beam_idx"]=self.next_beam_idxifself.next_beam_idxisnotNoneelsenp.arange(batch_size,dtype=int)

returninputs

defforward(
self,
input_ids:torch.LongTensor,
attention_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
position_ids:Optional[torch.LongTensor]=None,
inputs_embeds:Optional[torch.LongTensor]=None,
**kwargs,
):
self.compile()

inputs=self.prepare_inputs(
input_ids=input_ids,
attention_mask=attention_mask,
past_key_values=past_key_values,
position_ids=position_ids,
inputs_embeds=inputs_embeds,
**kwargs,
)

#Runinference
self.request.start_async(inputs,share_inputs=True)
self.request.wait()
logits=self.request.get_tensor("logits").data
logits=torch.from_numpy(logits).to(self.device)
past_key_values=((),)
self._past_length+=inputs["inputs_embeds"].shape[1]

returnCausalLMOutputWithPast(logits=logits,past_key_values=past_key_values)

#Adaptedfromtransformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
defprepare_inputs_for_generation(self,input_ids,past_key_values=None,inputs_embeds=None,**kwargs):
#ifmodelisusedasadecoderinencoder-decodermodel,thedecoderattentionmaskiscreatedonthefly
attention_mask=kwargs.get("attention_mask",None)
use_cache=kwargs.get("use_cache",None)

ifpast_key_valuesisnotNone:
past_len=self._get_past_length(past_key_values)
#Keeponlytheunprocessedtokens:
#1-Ifthelengthoftheattention_maskexceedsthelengthofinput_ids,thenweareinasettingwhere
#someoftheinputsareexclusivelypassedaspartofthecache(e.g.whenpassinginput_embedsas
#input)
ifattention_maskisnotNoneandinput_idsisnotNoneandattention_mask.shape[1]>input_ids.shape[1]:
input_ids=input_ids[:,-(attention_mask.shape[1]-past_len):]
#2-Ifthepast_lengthissmallerthaninput_ids',theninput_idsholdsallinputtokens.Wecandiscard
#input_idsbasedonthepast_length.
elifinput_idsisnotNoneandpast_len<input_ids.shape[1]:
input_ids=input_ids[:,past_len:]
#3-Otherwise(past_length>=input_ids.shape[1]),let'sassumeinput_idsonlyhasunprocessedtokens
position_ids=kwargs.get("position_ids",None)
ifattention_maskisnotNoneandposition_idsisNoneand"position_ids"inself.input_names:
#createposition_idsontheflyforbatchgeneration
position_ids=attention_mask.long().cumsum(-1)-1
position_ids.masked_fill_(attention_mask==0,1)
ifpast_key_valuesandinput_idsisnotNone:
position_ids=position_ids[:,-input_ids.shape[1]:]

model_inputs={
"input_ids":input_ids,
"past_key_values":past_key_values,
"use_cache":use_cache,
"position_ids":position_ids,
"attention_mask":attention_mask,
"inputs_embeds":inputs_embedsifpast_key_valuesisNoneelseNone,
}

returnmodel_inputs

def_get_past_length(self,past_key_values=None):
ifpast_key_valuesisNone:
return0
returnself._past_length

#Adaptedfromtransformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
def_reorder_cache(self,past_key_values:Tuple[Tuple[torch.Tensor]],beam_idx:torch.Tensor)->Tuple[Tuple[torch.Tensor]]:
"""
Thisfunctionisusedtore-orderthe`past_key_values`cacheif[`~PreTrainedModel.beam_search`]or
[`~PreTrainedModel.beam_sample`]iscalled.
Thisisrequiredtomatch`past_key_values`withthecorrectbeam_idxateverygenerationstep.
"""
self.next_beam_idx=np.array(beam_idx)#savebeam_idxtobeusedasaninputinthenextiteration
returnpast_key_values

defcan_generate(self):
"""ReturnsTruetovalidatethecheckthatthemodelusing`GenerationMixin.generate()`canindeedgenerate."""

returnTrue

def__call__(self,*args,**kwargs):
returnself.forward(*args,**kwargs)

Now,itisorderofgeneralmultimodalmodelclass``OvMiniCPMVModel``
thatwillhandlechatbotfunctionalityincludingimageprocessingand
answergenerationusingLLM.

..code::ipython3

fromtypingimportList,Optional
importmath
importjson
importtorch
fromtorchvisionimporttransforms
fromtimm.dataimportIMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD
fromPILimportImage


defpad(orig_items,key,max_length=None,padding_value=0,padding_side="left"):
items=[]
ifisinstance(orig_items[0][key],list):
assertisinstance(orig_items[0][key][0],torch.Tensor)
foritinorig_items:
fortrinit[key]:
items.append({key:tr})
else:
assertisinstance(orig_items[0][key],torch.Tensor)
items=orig_items

batch_size=len(items)
shape=items[0][key].shape
dim=len(shape)
assertdim<=3
ifmax_lengthisNone:
max_length=0
max_length=max(max_length,max(item[key].shape[-1]foriteminitems))
min_length=min(item[key].shape[-1]foriteminitems)
dtype=items[0][key].dtype

ifdim==1:
returntorch.cat([item[key]foriteminitems],dim=0)
elifdim==2:
ifmax_length==min_length:
returntorch.cat([item[key]foriteminitems],dim=0)
tensor=torch.zeros((batch_size,max_length),dtype=dtype)+padding_value
else:
tensor=torch.zeros((batch_size,max_length,shape[-1]),dtype=dtype)+padding_value

fori,iteminenumerate(items):
ifdim==2:
ifpadding_side=="left":
tensor[i,-len(item[key][0]):]=item[key][0].clone()
else:
tensor[i,:len(item[key][0])]=item[key][0].clone()
elifdim==3:
ifpadding_side=="left":
tensor[i,-len(item[key][0]):,:]=item[key][0].clone()
else:
tensor[i,:len(item[key][0]),:]=item[key][0].clone()

returntensor


defslice_image(image,max_slice_nums=9,scale_resolution=448,patch_size=14,never_split=False):
original_size=image.size
original_width,original_height=original_size
log_ratio=math.log(original_width/original_height)
ratio=original_width*original_height/(scale_resolution*scale_resolution)
multiple=min(math.ceil(ratio),max_slice_nums)

source_image=None
best_grid=None
patches=[]

ifmultiple<=1ornever_split:
#dontneedtoslice,upsample
best_size=find_best_resize(original_size,scale_resolution,patch_size,allow_upscale=True)
source_image=image.resize(best_size,Image.Resampling.BICUBIC)
else:
candidate_split_grids_nums=[]
foriin[multiple-1,multiple,multiple+1]:
ifi==1ori>max_slice_nums:
continue
candidate_split_grids_nums.append(i)

#sourceimage,down-samplingandensuredividedbypatch_size
best_resize=find_best_resize(original_size,scale_resolution,patch_size)
source_image=image.copy().resize(best_resize,Image.Resampling.BICUBIC)
candidate_grids=[]

#findbestgrid
forsplit_grids_numsincandidate_split_grids_nums:
m=1
whilem<=split_grids_nums:
ifsplit_grids_nums%m==0:
candidate_grids.append([m,split_grids_nums//m])
m+=1

best_grid=[1,1]
min_error=float("inf")
forgridincandidate_grids:
error=abs(log_ratio-math.log(grid[0]/grid[1]))
iferror<min_error:
best_grid=grid
min_error=error

refine_size=get_refine_size(original_size,best_grid,scale_resolution,patch_size,allow_upscale=True)

refine_image=image.resize(refine_size,Image.Resampling.BICUBIC)
patches=split_to_patches(refine_image,best_grid)

returnsource_image,patches,best_grid


defensure_divide(length,patch_size):
returnmax(round(length/patch_size)*patch_size,patch_size)


deffind_best_resize(original_size,scale_resolution,patch_size,allow_upscale=False):
width,height=original_size
if(width*height>scale_resolution*scale_resolution)orallow_upscale:
r=width/height
height=int(scale_resolution/math.sqrt(r))
width=int(height*r)
best_width=ensure_divide(width,patch_size)
best_height=ensure_divide(height,patch_size)
return(best_width,best_height)


defget_refine_size(original_size,grid,scale_resolution,patch_size,allow_upscale=False):
width,height=original_size
grid_x,grid_y=grid

refine_width=ensure_divide(width,grid_x)
refine_height=ensure_divide(height,grid_y)
grid_width=refine_width/grid_x
grid_height=refine_height/grid_y

best_grid_size=find_best_resize(
(grid_width,grid_height),
scale_resolution,
patch_size,
allow_upscale=allow_upscale,
)

refine_size=(best_grid_size[0]*grid_x,best_grid_size[1]*grid_y)

returnrefine_size


defsplit_to_patches(image,grid):
patches=[]
width,height=image.size
grid_x=int(width/grid[0])
grid_y=int(height/grid[1])

foriinrange(0,height,grid_y):
images=[]
forjinrange(0,width,grid_x):
box=(j,i,j+grid_x,i+grid_y)
patch=image.crop(box)
images.append(patch)
patches.append(images)

returnpatches


defget_grid_placeholder(tokenizer,grid,query_num):
image_placeholder=tokenizer.im_start+tokenizer.unk_token*query_num+tokenizer.im_end

cols=grid[0]
rows=grid[1]
slices=[]
foriinrange(rows):
lines=[]
forjinrange(cols):
lines.append(image_placeholder)
slices.append("".join(lines))
slice_placeholder=tokenizer.slice_start+"\n".join(slices)+tokenizer.slice_end
returnslice_placeholder


classOvMiniCPMVModel:
def__init__(self,config,vpm,llm,tokenizer)->None:
self.config=config
self.vpm=vpm
self.llm=llm
self.transform=self.init_transform()
self.tokenizer=tokenizer
self.device=torch.device("cpu")

definit_transform(self):
returntransforms.Compose(
[
transforms.ToTensor(),
transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,std=IMAGENET_INCEPTION_STD),
]
)

defget_vision_embedding(self,pixel_values):
res=[]
forpixel_valueinpixel_values:
h,w=pixel_value.shape[-2:]
tgt_size=torch.from_numpy(np.array([math.ceil(h/self.config.patch_size),math.ceil(w/self.config.patch_size)]))
vision_embedding=self.vpm([pixel_value.unsqueeze(0),tgt_size])[0]
res.append(vision_embedding)
returnnp.vstack(res)

defget_vllm_embedding(self,data):
if"vision_hidden_states"notindata:
pixel_values_list=data["pixel_values"]
vision_hidden_states=[]
forpixel_valuesinpixel_values_list:
iflen(pixel_values)>0:
vision_hidden_states.append(torch.from_numpy(self.get_vision_embedding(pixel_values)))
else:
vision_hidden_states.append([])

else:
vision_hidden_states=data["vision_hidden_states"]

vllm_embedding=torch.from_numpy(self.llm.embed_tokens(data["input_ids"]))*self.llm.config.scale_emb
bs=len(data["input_ids"])
foriinrange(bs):
cur_vs_hs=vision_hidden_states[i]
iflen(cur_vs_hs)>0:
cur_vllm_emb=vllm_embedding[i]
cur_image_bound=data["image_bound"][i]
iflen(cur_image_bound)>0:
image_indices=torch.stack([torch.arange(r[0],r[1],dtype=torch.long)forrincur_image_bound])

cur_vllm_emb.scatter_(
0,
image_indices.view(-1,1).repeat(1,cur_vllm_emb.shape[-1]),
cur_vs_hs.view(-1,cur_vs_hs.shape[-1]),
)

returnvllm_embedding

defforward(self,data,**kwargs):
vllm_embedding=self.get_vllm_embedding(data)
position_ids=data["position_ids"]
ifposition_ids.dtype!=torch.int64:
position_ids=position_ids.long()

returnself.llm(input_ids=None,position_ids=position_ids,inputs_embeds=vllm_embedding,**kwargs)

def_convert_to_tensors(self,tokenizer,input_str,max_inp_length:Optional[int]=None):
iftokenizer.add_bos_token:
input_ids=tokenizer.encode(input_str)
else:
input_ids=[tokenizer.bos_id]+tokenizer.encode(input_str)
ifmax_inp_lengthisnotNone:
input_ids=input_ids[:max_inp_length]
input_ids=torch.tensor(input_ids,dtype=torch.int32)

image_start_tokens=torch.where(input_ids==tokenizer.im_start_id)[0]
#跳过im_start
image_start_tokens+=1
image_end_tokens=torch.where(input_ids==tokenizer.im_end_id)[0]
valid_image_nums=max(len(image_start_tokens),len(image_end_tokens))
image_bound=torch.hstack(
[
image_start_tokens[:valid_image_nums].unsqueeze(-1),
image_end_tokens[:valid_image_nums].unsqueeze(-1),
]
)

model_input={}
model_input["input_ids"]=input_ids.unsqueeze(0)
model_input["image_bound"]=image_bound

returnmodel_input

def_process_list(self,tokenizer,data_list:List[str],max_inp_length:Optional[int]=None):
pad_keys=["input_ids"]
input_tensors=[]
fordataindata_list:
input_tensors.append(self._convert_to_tensors(tokenizer,data,max_inp_length))
padded={}
forkeyinpad_keys:
padded[key]=pad(input_tensors,key,padding_side="left").to(self.device)
padded["image_bound"]=[i["image_bound"]foriininput_tensors]
returnpadded

def_decode(self,inputs_embeds,tokenizer,**kwargs):
output=self.llm.generate(inputs_embeds=inputs_embeds,pad_token_id=0,eos_token_id=tokenizer.eos_token_id,**kwargs)
returnself._decode_text(output,tokenizer)

def_decode_text(self,result_ids,tokenizer):
result_text=[]
forresultinresult_ids:
result=result[result!=0]
ifresult[0]==tokenizer.bos_id:
result=result[1:]
ifresult[-1]==tokenizer.eos_id:
result=result[:-1]
result_text.append(tokenizer.decode(result).strip())
returnresult_text

defslice_image(self,image):
returnslice_image(
image,
self.config.max_slice_nums,
self.config.scale_resolution,
self.config.patch_size,
)

defget_slice_image_placeholder(self,image,tokenizer):
image_placeholder=tokenizer.im_start+tokenizer.unk_token*self.config.query_num+tokenizer.im_end

slice_images=[]

source_image,patches,best_grid=slice_image(
image,
self.config.max_slice_nums,
self.config.scale_resolution,
self.config.patch_size,
)

slice_images.append(source_image)
final_placeholder=image_placeholder

iflen(patches)>0:
foriinrange(len(patches)):
forjinrange(len(patches[0])):
slice_images.append(patches[i][j])

final_placeholder+=get_grid_placeholder(tokenizer,best_grid,self.config.query_num)

returnslice_images,final_placeholder

defgenerate(self,data_list=None,img_list=None,tokenizer=None,max_inp_length:Optional[int]=None,vision_hidden_states=None,**kwargs):
assertdata_listisnotNone
bs=len(data_list)
ifimg_listisNone:
img_list=[[]foriinrange(bs)]
assertbs==len(img_list)

model_inputs=self._process_list(tokenizer,data_list,max_inp_length)

ifvision_hidden_statesisNone:
pixel_values=[]
foriinrange(bs):
img_inps=[]
forimginimg_list[i]:
img_inps.append(self.transform(img).to(self.device))
ifimg_inps:
pixel_values.append(img_inps)
else:
pixel_values.append([])
model_inputs["pixel_values"]=pixel_values
else:
model_inputs["vision_hidden_states"]=vision_hidden_states

withtorch.inference_mode():
model_inputs["inputs_embeds"]=self.get_vllm_embedding(model_inputs)

result=self._decode(model_inputs["inputs_embeds"],tokenizer,**kwargs)

returnresult

defchat(self,image,msgs,context,tokenizer,vision_hidden_states=None,max_new_tokens=1024,sampling=True,max_inp_length=2048,**kwargs):
ifisinstance(msgs,str):
msgs=json.loads(msgs)
#msgstoprompt
prompt=""
fori,msginenumerate(msgs):
role=msg["role"]
content=msg["content"]
assertrolein["user","assistant"]
ifi==0:
ifimageisNone:
images=[]
else:
assertrole=="user","Theroleoffirstmsgshouldbeuser"
ifself.config.slice_mode:
images,final_placeholder=self.get_slice_image_placeholder(image,tokenizer)
content=final_placeholder+"\n"+content
else:
images=[image]
content=tokenizer.im_start+tokenizer.unk_token*self.config.query_num+tokenizer.im_end+"\n"+content
prompt+="<用户>"ifrole=="user"else"<AI>"
prompt+=content
prompt+="<AI>"
final_input=prompt

ifsampling:
generation_config={
"top_p":0.8,
"top_k":100,
"temperature":0.7,
"do_sample":True,
"repetition_penalty":1.05,
"streamer":None,
}
else:
generation_config={
"num_beams":3,
"repetition_penalty":1.2,
"streamer":None,
}

generation_config.update((k,kwargs[k])forkingeneration_config.keys()&kwargs.keys())

withtorch.inference_mode():
res=self.generate(
data_list=[final_input],
max_inp_length=max_inp_length,
img_list=[images],
tokenizer=tokenizer,
max_new_tokens=max_new_tokens,
vision_hidden_states=vision_hidden_states,
**generation_config
)
answer=res[0]
context=msgs.copy()
context.append({"role":"assistant","content":answer})

returnanswer,context,generation_config

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

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



Selectmodelvariant
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

use_int4_lang_model=widgets.Checkbox(
value=llm_int4_path.exists(),
description="INT4languagemodel",
disabled=notllm_int4_path.exists(),
)

use_int4_lang_model




..parsed-literal::

Checkbox(value=True,description='INT4languagemodel')



..code::ipython3

llm=OvModelForCausalLMWithEmb(llm_path.parentifnotuse_int4_lang_model.valueelsellm_int4_path.parent,device.value)

..code::ipython3

visual_encoder=core.compile_model(image_encoder_path,device.value)

..code::ipython3

config=AutoConfig.from_pretrained(model_dir,trust_remote_code=True)
tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

..code::ipython3

model=OvMiniCPMVModel(config,visual_encoder,llm,tokenizer)

..code::ipython3

importrequests

url="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image=Image.open(requests.get(url,stream=True).raw)
question="Whatisunusualonthisimage?"

print(f"Question:\n{question}")
image


..parsed-literal::

Question:
Whatisunusualonthisimage?




..image::minicpm-v-multimodal-chatbot-with-output_files/minicpm-v-multimodal-chatbot-with-output_27_1.png



..code::ipython3

fromtransformersimportTextStreamer

msgs=[{"role":"user","content":question}]

streamer=TextStreamer(tokenizer=tokenizer,skip_special_tokens=True)

print("Answer:")
res,context,_=model.chat(image=image,msgs=msgs,context=None,tokenizer=tokenizer,sampling=True,temperature=0.7,streamer=streamer)


..parsed-literal::

Answer:
Theunusualaspectofthisimageisthepresenceofacatlayinginsideanopencardboardbox.


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
importtraceback
importre
fromtransformersimportTextIteratorStreamer
fromthreadingimportThread


ERROR_MSG="Error,pleaseretry"
model_name="MiniCPM-V2.0"

form_radio={"choices":["BeamSearch","Sampling"],"value":"Sampling","interactive":True,"label":"DecodeType"}
#BeamForm
num_beams_slider={"minimum":0,"maximum":5,"value":3,"step":1,"interactive":True,"label":"NumBeams"}
repetition_penalty_slider={"minimum":0,"maximum":3,"value":1.2,"step":0.01,"interactive":True,"label":"RepetitionPenalty"}
repetition_penalty_slider2={"minimum":0,"maximum":3,"value":1.05,"step":0.01,"interactive":True,"label":"RepetitionPenalty"}
max_new_tokens_slider={"minimum":1,"maximum":4096,"value":1024,"step":1,"interactive":True,"label":"MaxNewTokens"}

top_p_slider={"minimum":0,"maximum":1,"value":0.8,"step":0.05,"interactive":True,"label":"TopP"}
top_k_slider={"minimum":0,"maximum":200,"value":100,"step":1,"interactive":True,"label":"TopK"}
temperature_slider={"minimum":0,"maximum":2,"value":0.7,"step":0.05,"interactive":True,"label":"Temperature"}


defcreate_component(params,comp="Slider"):
ifcomp=="Slider":
returngr.Slider(
minimum=params["minimum"],
maximum=params["maximum"],
value=params["value"],
step=params["step"],
interactive=params["interactive"],
label=params["label"],
)
elifcomp=="Radio":
returngr.Radio(choices=params["choices"],value=params["value"],interactive=params["interactive"],label=params["label"])
elifcomp=="Button":
returngr.Button(value=params["value"],interactive=True)


defchat(img,msgs,ctx,params=None,vision_hidden_states=None):
default_params={"num_beams":3,"repetition_penalty":1.2,"max_new_tokens":1024}
ifparamsisNone:
params=default_params
ifimgisNone:
return-1,"Error,invalidimage,pleaseuploadanewimage",None,None
try:
image=img.convert("RGB")
streamer=TextIteratorStreamer(tokenizer,**{"skip_special_tokens":True})
generation_params={"image":image,"msgs":msgs,"context":None,"tokenizer":tokenizer,"streamer":streamer,**params}
thread=Thread(target=model.chat,kwargs=generation_params)
thread.start()

buffer=""

forresinstreamer:
res=re.sub(r"(<box>.*</box>)","",res)
res=res.replace("<ref>","")
res=res.replace("</ref>","")
res=res.replace("<box>","")
new_text=res.replace("</box>","")
buffer+=new_text
yield-1,buffer,None,None
exceptExceptionaserr:
print(err)
traceback.print_exc()
return-1,ERROR_MSG,None,None


defupload_img(image,_chatbot,_app_session):
image=Image.fromarray(image)

_app_session["sts"]=None
_app_session["ctx"]=[]
_app_session["img"]=image
_chatbot.append(("","Imageuploadedsuccessfully,youcantalktomenow"))
return_chatbot,_app_session


defrespond(_question,_chat_bot,_app_cfg,params_form,num_beams,repetition_penalty,repetition_penalty_2,top_p,top_k,temperature):
if_app_cfg.get("ctx",None)isNone:
_chat_bot.append((_question,"Pleaseuploadanimagetostart"))
return"",_chat_bot,_app_cfg

_context=_app_cfg["ctx"].copy()
if_context:
_context.append({"role":"user","content":_question})
else:
_context=[{"role":"user","content":_question}]

ifparams_form=="BeamSearch":
params={"sampling":False,"num_beams":num_beams,"repetition_penalty":repetition_penalty,"max_new_tokens":896}
else:
params={
"sampling":True,
"top_p":top_p,
"top_k":top_k,
"temperature":temperature,
"repetition_penalty":repetition_penalty_2,
"max_new_tokens":896,
}

_context.append({"role":"assistant","content":""})
_chat_bot.append([_question,""])
forcode,_answer,_,stsinchat(_app_cfg["img"],_context,None,params):
_context[-1]["content"]=_answer
_chat_bot[-1][-1]=_answer

ifcode==0:
_app_cfg["ctx"]=_context
_app_cfg["sts"]=sts
yield"",_chat_bot,_app_cfg


defregenerate_button_clicked(_question,_chat_bot,_app_cfg,params_form,num_beams,repetition_penalty,repetition_penalty_2,top_p,top_k,temperature):
iflen(_chat_bot)<=1:
_chat_bot.append(("Regenerate","Noquestionforregeneration."))
return"",_chat_bot,_app_cfg
elif_chat_bot[-1][0]=="Regenerate":
return"",_chat_bot,_app_cfg
else:
_question=_chat_bot[-1][0]
_chat_bot=_chat_bot[:-1]
_app_cfg["ctx"]=_app_cfg["ctx"][:-2]
fortext,_chatbot,_app_cfginrespond(
_question,_chat_bot,_app_cfg,params_form,num_beams,repetition_penalty,repetition_penalty_2,top_p,top_k,temperature
):
yieldtext,_chatbot,_app_cfg


withgr.Blocks()asdemo:
withgr.Row():
withgr.Column(scale=1,min_width=300):
params_form=create_component(form_radio,comp="Radio")
withgr.Accordion("BeamSearch")asbeams_according:
num_beams=create_component(num_beams_slider)
repetition_penalty=create_component(repetition_penalty_slider)
withgr.Accordion("Sampling")assampling_according:
top_p=create_component(top_p_slider)
top_k=create_component(top_k_slider)
temperature=create_component(temperature_slider)
repetition_penalty_2=create_component(repetition_penalty_slider2)
regenerate=create_component({"value":"Regenerate"},comp="Button")
withgr.Column(scale=3,min_width=500):
app_session=gr.State({"sts":None,"ctx":None,"img":None})
bt_pic=gr.Image(label="Uploadanimagetostart")
chat_bot=gr.Chatbot(label=f"Chatwith{model_name}")
txt_message=gr.Textbox(label="Inputtext")

regenerate.click(
regenerate_button_clicked,
[txt_message,chat_bot,app_session,params_form,num_beams,repetition_penalty,repetition_penalty_2,top_p,top_k,temperature],
[txt_message,chat_bot,app_session],
)
txt_message.submit(
respond,
[txt_message,chat_bot,app_session,params_form,num_beams,repetition_penalty,repetition_penalty_2,top_p,top_k,temperature],
[txt_message,chat_bot,app_session],
)
bt_pic.upload(lambda:None,None,chat_bot,queue=False).then(upload_img,inputs=[bt_pic,chat_bot,app_session],outputs=[chat_bot,app_session])


try:
demo.launch(debug=False)
exceptException:
demo.launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


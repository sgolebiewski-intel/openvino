Visual-languageassistantwithnanoLLaVAandOpenVINO
=====================================================

nanoLLaVAisa“smallbutmighty”1Bvision-languagemodeldesignedto
runefficientlyonedgedevices.Ituses
`SigLIP-400m<https://huggingface.co/google/siglip-so400m-patch14-384>`__
asImageEncoderand
`Qwen1.5-0.5B<https://huggingface.co/Qwen/Qwen1.5-0.5B>`__asLLM.In
thistutorial,weconsiderhowtoconvertandrunnanoLLaVAmodelusing
OpenVINO.Additionally,wewilloptimizemodelusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchmodel<#load-pytorch-model>`__
-`RunPyTorchModelInference<#run-pytorch-model-inference>`__
-`ConvertandOptimizemodel<#convert-and-optimize-model>`__

-`ConvertmodeltoOpenVINOIR
format<#convert-model-to-openvino-ir-format>`__
-`CompressModelweightsto4and8bitsusing
NNCF<#compress-model-weights-to-4-and-8-bits-using-nncf>`__
-`ImageEncoder<#image-encoder>`__
-`TextEmbeddings<#text-embeddings>`__
-`LanguageModel<#language-model>`__

-`Preparemodelinference
pipeline<#prepare-model-inference-pipeline>`__
-`RunOpenVINOModelInference<#run-openvino-model-inference>`__

-`Selectdevice<#select-device>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.1""transformers>=4.40""accelerate""pillow""gradio>=4.26""openvino>=2024.1.0""tqdm""nncf>=2.10"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.3.1+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

fromhuggingface_hubimportsnapshot_download
frompathlibimportPath

model_local_dir=Path("nanoLLaVA")

ifnotmodel_local_dir.exists():
snapshot_download(repo_id="qnguyen3/nanoLLaVA",local_dir=model_local_dir)

modeling_file=model_local_dir/"modeling_llava_qwen2.py"
orig_modeling_file=model_local_dir/f"orig_{modeling_file.name}"


#modelcodedependsfromflash_attnpackagethatmaybeproblematictoload.Patchmodelcodeforavoidingimportofthispackage
ifnotorig_modeling_file.exists():
modeling_file.rename(orig_modeling_file)
withorig_modeling_file.open("r")asf:
content=f.read()
replacement_lines=[
("fromflash_attnimportflash_attn_func,flash_attn_varlen_func",""),
("fromflash_attn.bert_paddingimportindex_first_axis,pad_input,unpad_input",""),
('_flash_supports_window_size="window_size"inlist(inspect.signature(flash_attn_func).parameters)',"pass"),
]

forreplace_pairinreplacement_lines:
content=content.replace(*replace_pair)

withmodeling_file.open("w")asf:
f.write(content)



..parsed-literal::

Fetching14files:0%||0/14[00:00<?,?it/s]



..parsed-literal::

README.md:0%||0.00/3.47k[00:00<?,?B/s]



..parsed-literal::

generation_config.json:0%||0.00/172[00:00<?,?B/s]



..parsed-literal::

example_1.png:0%||0.00/200k[00:00<?,?B/s]



..parsed-literal::

config.json:0%||0.00/1.28k[00:00<?,?B/s]



..parsed-literal::

configuration_llava_qwen2.py:0%||0.00/8.87k[00:00<?,?B/s]



..parsed-literal::

.gitattributes:0%||0.00/1.52k[00:00<?,?B/s]



..parsed-literal::

added_tokens.json:0%||0.00/80.0[00:00<?,?B/s]



..parsed-literal::

modeling_llava_qwen2.py:0%||0.00/103k[00:00<?,?B/s]



..parsed-literal::

special_tokens_map.json:0%||0.00/510[00:00<?,?B/s]



..parsed-literal::

tokenizer_config.json:0%||0.00/1.32k[00:00<?,?B/s]



..parsed-literal::

vocab.json:0%||0.00/2.78M[00:00<?,?B/s]



..parsed-literal::

tokenizer.json:0%||0.00/7.03M[00:00<?,?B/s]



..parsed-literal::

merges.txt:0%||0.00/1.67M[00:00<?,?B/s]



..parsed-literal::

model.safetensors:0%||0.00/2.10G[00:00<?,?B/s]


LoadPyTorchmodel
------------------

`backtotop⬆️<#table-of-contents>`__

ForcreatingPyTorchmodelweshoulduse``from_pretrained``methodof
``AutoModelForCausalLM``modelclass.Modelweightsarealready
downloadedfromHuggingFacehubusing``snapshot_download``functionon
previousstep.

..code::ipython3

importtransformers
fromtransformersimportAutoModelForCausalLM,AutoTokenizer
fromPILimportImage
importwarnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model=AutoModelForCausalLM.from_pretrained(model_local_dir,trust_remote_code=True)
tokenizer=AutoTokenizer.from_pretrained(model_local_dir,trust_remote_code=True)


..parsed-literal::

2024-07-1301:15:06.266352:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:15:06.301452:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:15:06.954075:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


RunPyTorchModelInference
---------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtorch
importrequests

prompt="Describethisimageindetail"

messages=[{"role":"user","content":f"<image>\n{prompt}"}]
text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

text_chunks=[tokenizer(chunk).input_idsforchunkintext.split("<image>")]
input_ids=torch.tensor(text_chunks[0]+[-200]+text_chunks[1],dtype=torch.long).unsqueeze(0)
url="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/8bf7d9f2-018a-4498-bec4-55f17c273ecc"
image=Image.open(requests.get(url,stream=True).raw)
image_tensor=model.process_images([image],model.config)
print(prompt)
image


..parsed-literal::

Describethisimageindetail




..image::nano-llava-multimodal-chatbot-with-output_files/nano-llava-multimodal-chatbot-with-output_7_1.png



..code::ipython3

fromtransformersimportTextStreamer

streamer=TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)

output_ids=model.generate(input_ids,images=image_tensor,max_new_tokens=128,use_cache=True,streamer=streamer)


..parsed-literal::

Thisimagecapturesadelightfulscenefeaturingawhite,fluffylambwithaplayfulexpression.Thelambispositionedtowardsthecenteroftheimage,itsbodyfillingmostoftheframefromlefttoright.Thelambhasacharminglyexpressiveface,withapairofblackeyesthatappeartobesquintingslightly.Ithasasmall,round,pinknose,anditsearsarealsopink,whichcontrastswiththerestofitswhitefur.Thelamb'slegsarewhite,andthelowerpartofitsbodyisfluffy,addingtoitsadorableappearance.
Thelamb'sfaceisquiteexpressive,withitseyeslookingdownand


ConvertandOptimizemodel
--------------------------

`backtotop⬆️<#table-of-contents>`__

Ourmodelconversionandoptimizationconsistoffollowingsteps:1.
ConvertmodeltoOpenVINOformatandsaveitondisk.2.Compressmodel
weightsusingNNCF

Let’sconsidereachstepmoredeeply.

ConvertmodeltoOpenVINOIRformat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ConvertmodeltoOpenVINOformatusingconversionhelperfunction
definedbellow.Wewilluse`OpenVINOModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
forconversionPyTorchmodeltoOpenVINOIntermediateRepresentation
format.``ov.convert_model``functionacceptsPyTorchmodelinstanceand
exampleinputfortracingandreturnsreadytouseOpenVINOModelobject
thatcanbecompiledondeviceusing``core.compile_model``orsavedon
diskfornextusagewithhelp``ov.save_model``function.Dependsfrom
generationstep,modelacceptsdifferentinputsandactivatesdifferent
partsofpipeline.Forpreservingthesamelevelofflexibility,wewill
splitmodelonparts:ImageEncoder,TextEmbeddings,LanguageModeland
converteachpartseparately.

CompressModelweightsto4and8bitsusingNNCF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

importgc
importwarnings
importtorch
importopenvinoasov
importnncf
fromtypingimportOptional,Tuple

warnings.filterwarnings("ignore")


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


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..code::ipython3

ifcompression_mode.value=="INT4":
ov_out_path=Path("ov_nanollava/INT4_compressed_weights")
llava_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT4_ASYM,group_size=128,ratio=0.8)
else:
ov_out_path=Path("ov_nanollava/INT8_compressed_weights")
llava_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT8)

image_encoder_wc_parameters=dict(mode=nncf.CompressWeightsMode.INT8)

ov_out_path.mkdir(exist_ok=True,parents=True)
model.config.save_pretrained(ov_out_path)
vision_tower=model.get_vision_tower()
ifnotvision_tower.is_loaded:
vision_tower.load_model()

image_encoder_path=ov_out_path/"image_encoder.xml"
token_embedding_model_path=ov_out_path/"token_embed.xml"
model_path=ov_out_path/"llava_with_past.xml"

model.eval()
model.config.use_cache=True
model.config.torchscript=True

ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ImageEncoderisrepresentedinnanoLLaVAbypretrainedSigLIPmodel.
Imageencoderisresponsibleforencodinginputimagesintoembedding
space.

..code::ipython3

ifnotimage_encoder_path.exists():
model.forward=model.encode_images
withtorch.no_grad():
ov_model=ov.convert_model(
model,
example_input=torch.zeros((1,3,384,384)),
input=[(-1,3,384,384)],
)
ifimage_encoder_wc_parametersisnotNone:
print("Applyingweightcompressiontoimageencoder")
ov_model=nncf.compress_weights(ov_model,**image_encoder_wc_parameters)
ov.save_model(ov_model,image_encoder_path)
cleanup_torchscript_cache()
delov_model
gc.collect()
print("ImageEncodermodelsuccessfullyconverted")


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

['images']
Applyingweightcompressiontoimageencoder
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(159/159)│100%(159/159)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

ImageEncodermodelsuccessfullyconverted


TextEmbeddings
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

InLLMs,inputembeddingisapartoflanguagemodel,butforLLaVAthe
firststephiddenstateproducedbythismodelpartshouldbeintegrated
withimageembeddingsintocommonembeddingspace.Forabilitytoreuse
thismodelpartandavoidintroductionofextrallmmodelinstance,we
willuseitseparately.

..code::ipython3

ifnottoken_embedding_model_path.exists():
withtorch.no_grad():
ov_model=ov.convert_model(model.get_model().embed_tokens,example_input=torch.ones((1,10),dtype=torch.long))
ov.save_model(ov_model,token_embedding_model_path)
cleanup_torchscript_cache()
delov_model
gc.collect()
print("TokenEmbeddingmodelsuccessfullyconverted")


..parsed-literal::

['input']
TokenEmbeddingmodelsuccessfullyconverted


LanguageModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LanguageModelisresponsibleforgenerationanswerinLLaVA.Thispart
isverysimilartostandardLLMfortextgeneration.Ourmodeluses
`Qwen/Qwen1.5-0.5B<https://huggingface.co/Qwen/Qwen1.5-0.5B>`__asbase
LLM.Tooptimizethegenerationprocessandusememorymoreefficiently,
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

..code::ipython3

ifnotmodel_path.exists():
model.forward=super(type(model),model).forward
example_input={"attention_mask":torch.ones([2,10],dtype=torch.int64),"position_ids":torch.tensor([[8,9],[8,9]],dtype=torch.int64)}

dynamic_shapes={
"input_embeds":{0:"batch_size",1:"seq_len"},
"attention_mask":{0:"batch_size",1:"prev_seq_len+seq_len"},
"position_ids":{0:"batch_size",1:"seq_len"},
}
input_embeds=torch.zeros((2,2,model.config.hidden_size))

input_names=["attention_mask","position_ids"]
output_names=["logits"]

past_key_values=[]
foriinrange(model.config.num_hidden_layers):
kv=[torch.randn([2,model.config.num_key_value_heads,8,model.config.hidden_size//model.config.num_attention_heads])for_inrange(2)]
past_key_values.append(kv)
input_names.extend([f"past_key_values.{i}.key",f"past_key_values.{i}.value"])
output_names.extend([f"present.{i}.key",f"present.{i}.value"])
dynamic_shapes[input_names[-2]]={0:"batch_size",2:"seq_len"}
dynamic_shapes[input_names[-1]]={0:"batch_size",2:"seq_len"}

example_input["past_key_values"]=past_key_values
example_input["inputs_embeds"]=input_embeds
input_names.append("inputs_embeds")
dynamic_shapes["inputs_embeds"]={0:"batch_size",1:"seq_len"}
ov_model=ov.convert_model(model,example_input=example_input)
ov_model=postprocess_converted_model(
ov_model,example_input=example_input.values(),input_names=input_names,output_names=output_names,dynamic_shapes=dynamic_shapes
)

ifllava_wc_parametersisnotNone:
print("ApplyingweightcompressiontosecondstageLLavamodel")
ov_model=nncf.compress_weights(ov_model,**llava_wc_parameters)
ov.save_model(ov_model,model_path)
cleanup_torchscript_cache()
delov_model
gc.collect()

print("LLaVAmodelsuccessfullyconverted")
delmodel
gc.collect();


..parsed-literal::

['attention_mask','position_ids','past_key_values','inputs_embeds']
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
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│47%(48/169)│20%(47/168)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│53%(121/169)│80%(121/168)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

LLaVAmodelsuccessfullyconverted


Preparemodelinferencepipeline
--------------------------------

`backtotop⬆️<#table-of-contents>`__

``OVLlavaQwen2ForCausalLM``classprovidesease-to-useinterfacefor
usingmodelingenerationscenario.Itisbasedon
``transformers.generation.GenerationMixin``thatgivesusopportunityto
reuseallreachcapabilitiesforgenerationimplementedinHuggingFace
Transformerslibrary.Moredetailsaboutthisinterfacecanbefoundin
`HuggingFace
documentation<https://huggingface.co/docs/transformers/main_classes/text_generation>`__.

..code::ipython3

fromtransformers.generationimportGenerationConfig,GenerationMixin
fromtransformers.modeling_outputsimportCausalLMOutputWithPast
fromtransformersimportAutoConfig
fromtransformers.image_processing_utilsimportBatchFeature,get_size_dict
fromtransformers.image_transformsimport(
convert_to_rgb,
normalize,
rescale,
resize,
to_channel_dimension_format,
)
fromtransformers.image_utilsimport(
ChannelDimension,
PILImageResampling,
to_numpy_array,
)
importnumpyasnp
importtorch
fromtypingimportDict
fromfunctoolsimportpartial,reduce

IGNORE_INDEX=-100
IMAGE_TOKEN_INDEX=-200


classImageProcessor:
def__init__(
self,
image_mean=(0.5,0.5,0.5),
image_std=(0.5,0.5,0.5),
size=(384,384),
crop_size:Dict[str,int]=None,
resample=PILImageResampling.BICUBIC,
rescale_factor=1/255,
data_format=ChannelDimension.FIRST,
):
crop_size=crop_sizeifcrop_sizeisnotNoneelse{"height":384,"width":384}
crop_size=get_size_dict(crop_size,default_to_square=True,param_name="crop_size")

self.image_mean=image_mean
self.image_std=image_std
self.size=size
self.resample=resample
self.rescale_factor=rescale_factor
self.data_format=data_format
self.crop_size=crop_size

defpreprocess(self,images,return_tensors):
ifisinstance(images,Image.Image):
images=[images]
else:
assertisinstance(images,list)

transforms=[
convert_to_rgb,
to_numpy_array,
partial(resize,size=self.size,resample=self.resample,data_format=self.data_format),
partial(rescale,scale=self.rescale_factor,data_format=self.data_format),
partial(normalize,mean=self.image_mean,std=self.image_std,data_format=self.data_format),
partial(to_channel_dimension_format,channel_dim=self.data_format,input_channel_dim=self.data_format),
]

images=reduce(lambdax,f:[*map(f,x)],transforms,images)
data={"pixel_values":images}

returnBatchFeature(data=data,tensor_type=return_tensors)


classOVLlavaQwen2ForCausalLM(GenerationMixin):
def__init__(self,core,model_dir,device):
self.image_encoder=core.compile_model(model_dir/"image_encoder.xml",device)
self.embed_tokens=core.compile_model(model_dir/"token_embed.xml",device)
self.model=core.read_model(model_dir/"llava_with_past.xml")
self.input_names={key.get_any_name():idxforidx,keyinenumerate(self.model.inputs)}
self.output_names={key.get_any_name():idxforidx,keyinenumerate(self.model.outputs)}
self.key_value_input_names=[keyforkeyinself.input_namesif"key_values"inkey]
self.key_value_output_names=[keyforkeyinself.output_namesif"present"inkey]
compiled_model=core.compile_model(self.model,device)
self.request=compiled_model.create_infer_request()
self.config=AutoConfig.from_pretrained(model_dir)
self.generation_config=GenerationConfig.from_model_config(self.config)
self.main_input_name="input_ids"
self.device=torch.device("cpu")
self.num_pkv=2
self.image_processor=ImageProcessor()
self._supports_cache_class=False

defcan_generate(self):
"""ReturnsTruetovalidatethecheckthatthemodelusing`GenerationMixin.generate()`canindeedgenerate."""
returnTrue

def__call__(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
position_ids:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->CausalLMOutputWithPast:
returnself.forward(input_ids,images,attention_mask,position_ids,past_key_values)

defforward(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
position_ids:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->CausalLMOutputWithPast:
"""Generalinferencemethod"""
inputs=self.prepare_inputs_for_multimodal(input_ids,position_ids,attention_mask,past_key_values,images)

#Runinference
self.request.start_async(inputs,share_inputs=True)
self.request.wait()

logits=torch.from_numpy(self.request.get_tensor("logits").data)

#Tupleoflengthequalto:numberoflayer*numberofpast_key_valueperdecoderlayer(2correspondstotheself-attentionlayer)
past_key_values=tuple(self.request.get_tensor(key).dataforkeyinself.key_value_output_names)
#Tupleoftupleoflength`n_layers`,witheachtupleoflengthequalto2(k/vofself-attention)

past_key_values=tuple(past_key_values[i:i+self.num_pkv]foriinrange(0,len(past_key_values),self.num_pkv))
returnCausalLMOutputWithPast(logits=logits,past_key_values=past_key_values)

defprepare_inputs_for_multimodal(self,input_ids,position_ids,attention_mask,past_key_values,images):
inputs={}
ifpast_key_valuesisNone:
past_key_values=self._dummy_past_key_values(input_ids.shape[0])
else:
past_key_values=tuple(past_key_valueforpkv_per_layerinpast_key_valuesforpast_key_valueinpkv_per_layer)
inputs.update(zip(self.key_value_input_names,past_key_values))

ifimagesisNoneorinput_ids.shape[1]==1:
target_shape=past_key_values[-1][-1].shape[-2]+1ifpast_key_valuesisnotNoneelseinput_ids.shape[1]
attention_mask=torch.cat(
(
attention_mask,
torch.ones((attention_mask.shape[0],target_shape-attention_mask.shape[1]),dtype=attention_mask.dtype,device=attention_mask.device),
),
dim=1,
)
position_ids=torch.sum(attention_mask,dim=1).unsqueeze(-1)-1
inputs_embeds=self.embed_tokens(input_ids)[0]
inputs["attention_mask"]=attention_mask.numpy()
inputs["position_ids"]=position_ids.numpy()
inputs["inputs_embeds"]=inputs_embeds

returninputs

iftype(images)islistorimages.ndim==5:
concat_images=torch.cat([imageforimageinimages],dim=0)
image_features=self.encode_images(concat_images)
split_sizes=[image.shape[0]forimageinimages]
image_features=torch.split(image_features,split_sizes,dim=0)
image_features=[x.flatten(0,1).to(self.device)forxinimage_features]
else:
image_features=self.encode_images(images).to(self.device)

#Let'sjustadddummytensorsiftheydonotexist,
#itisaheadachetodealwithNoneallthetime.
#Butitisnotideal,andifyouhaveabetteridea,
#pleaseopenanissue/submitaPR,thanks.
labels=None
_attention_mask=attention_mask
ifattention_maskisNone:
attention_mask=torch.ones_like(input_ids,dtype=torch.bool)
else:
attention_mask=attention_mask.bool()
ifposition_idsisNone:
position_ids=torch.arange(0,input_ids.shape[1],dtype=torch.long,device=input_ids.device)
iflabelsisNone:
labels=torch.full_like(input_ids,IGNORE_INDEX)

#removethepaddingusingattention_mask--TODO:doublecheck
input_ids=[cur_input_ids[cur_attention_mask]forcur_input_ids,cur_attention_maskinzip(input_ids,attention_mask)]
labels=[cur_labels[cur_attention_mask]forcur_labels,cur_attention_maskinzip(labels,attention_mask)]

new_input_embeds=[]
new_labels=[]
cur_image_idx=0
forbatch_idx,cur_input_idsinenumerate(input_ids):
num_images=(cur_input_ids==IMAGE_TOKEN_INDEX).sum()
ifnum_images==0:
cur_image_features=image_features[cur_image_idx]
cur_input_embeds_1=self.embed_tokens(cur_input_ids)
cur_input_embeds=torch.cat([cur_input_embeds_1,cur_image_features[0:0]],dim=0)
new_input_embeds.append(cur_input_embeds)
new_labels.append(labels[batch_idx])
cur_image_idx+=1
continue

image_token_indices=[-1]+torch.where(cur_input_ids==IMAGE_TOKEN_INDEX)[0].tolist()+[cur_input_ids.shape[0]]
cur_input_ids_noim=[]
cur_labels=labels[batch_idx]
cur_labels_noim=[]
foriinrange(len(image_token_indices)-1):
cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
split_sizes=[x.shape[0]forxincur_labels_noim]
cur_input_embeds=torch.from_numpy(self.embed_tokens(torch.cat(cur_input_ids_noim).unsqueeze(0))[0])[0]
cur_input_embeds_no_im=torch.split(cur_input_embeds,split_sizes,dim=0)
cur_new_input_embeds=[]
cur_new_labels=[]

foriinrange(num_images+1):
cur_new_input_embeds.append(cur_input_embeds_no_im[i])
cur_new_labels.append(cur_labels_noim[i])
ifi<num_images:
cur_image_features=image_features[cur_image_idx]
cur_image_idx+=1
cur_new_input_embeds.append(cur_image_features)
cur_new_labels.append(torch.full((cur_image_features.shape[0],),IGNORE_INDEX,device=cur_labels.device,dtype=cur_labels.dtype))

cur_new_input_embeds=torch.cat(cur_new_input_embeds)
cur_new_labels=torch.cat(cur_new_labels)

new_input_embeds.append(cur_new_input_embeds)
new_labels.append(cur_new_labels)

#Truncatesequencestomaxlengthasimageembeddingscanmakethesequencelonger
tokenizer_model_max_length=getattr(self.config,"tokenizer_model_max_length",None)
iftokenizer_model_max_lengthisnotNone:
new_input_embeds=[x[:tokenizer_model_max_length]forxinnew_input_embeds]
new_labels=[x[:tokenizer_model_max_length]forxinnew_labels]

#Combinethem
max_len=max(x.shape[0]forxinnew_input_embeds)
batch_size=len(new_input_embeds)

new_input_embeds_padded=[]
new_labels_padded=torch.full((batch_size,max_len),IGNORE_INDEX,dtype=new_labels[0].dtype,device=new_labels[0].device)
attention_mask=torch.zeros((batch_size,max_len),dtype=attention_mask.dtype,device=attention_mask.device)
position_ids=torch.zeros((batch_size,max_len),dtype=position_ids.dtype,device=position_ids.device)

fori,(cur_new_embed,cur_new_labels)inenumerate(zip(new_input_embeds,new_labels)):
cur_len=cur_new_embed.shape[0]
ifgetattr(self.config,"tokenizer_padding_side","right")=="left":
new_input_embeds_padded.append(
torch.cat(
(torch.zeros((max_len-cur_len,cur_new_embed.shape[1]),dtype=cur_new_embed.dtype,device=cur_new_embed.device),cur_new_embed),dim=0
)
)
ifcur_len>0:
new_labels_padded[i,-cur_len:]=cur_new_labels
attention_mask[i,-cur_len:]=True
position_ids[i,-cur_len:]=torch.arange(0,cur_len,dtype=position_ids.dtype,device=position_ids.device)
else:
new_input_embeds_padded.append(
torch.cat(
(cur_new_embed,torch.zeros((max_len-cur_len,cur_new_embed.shape[1]),dtype=cur_new_embed.dtype,device=cur_new_embed.device)),dim=0
)
)
ifcur_len>0:
new_labels_padded[i,:cur_len]=cur_new_labels
attention_mask[i,:cur_len]=True
position_ids[i,:cur_len]=torch.arange(0,cur_len,dtype=position_ids.dtype,device=position_ids.device)

new_input_embeds=torch.stack(new_input_embeds_padded,dim=0)
attention_mask=attention_mask.to(dtype=_attention_mask.dtype)
inputs["inputs_embeds"]=new_input_embeds.numpy()
inputs["attention_mask"]=attention_mask.numpy()
inputs["position_ids"]=position_ids.numpy()

returninputs

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
return{
"input_ids":input_ids,
"attention_mask":attention_mask,
"position_ids":kwargs.get("position_ids",None),
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

def_dummy_past_key_values(self,batch_size):
pkv=[]
forinput_nameinself.key_value_input_names:
input_t=self.model.input(input_name)
input_shape=self.model.input(input_name).get_partial_shape()
input_shape[0]=batch_size
input_shape[2]=0
pkv.append(ov.Tensor(input_t.get_element_type(),input_shape.get_shape()))

returnpkv

defencode_images(self,images):
returntorch.from_numpy(self.image_encoder(images)[0])

defexpand2square(self,pil_img,background_color):
width,height=pil_img.size
ifwidth==height:
returnpil_img
elifwidth>height:
result=Image.new(pil_img.mode,(width,width),background_color)
result.paste(pil_img,(0,(width-height)//2))
returnresult
else:
result=Image.new(pil_img.mode,(height,height),background_color)
result.paste(pil_img,((height-width)//2,0))
returnresult

defprocess_images(self,images,model_cfg):
image_aspect_ratio=getattr(model_cfg,"image_aspect_ratio",None)
new_images=[]
ifimage_aspect_ratio=="pad":
forimageinimages:
image=self.expand2square(image,tuple(int(x*255)forxinself.image_processor.image_mean))
image=self.image_processor.preprocess(image,return_tensors="pt")["pixel_values"][0]
new_images.append(image)
else:
returnself.image_processor(images,return_tensors="pt")["pixel_values"]
ifall(x.shape==new_images[0].shapeforxinnew_images):
new_images=torch.stack(new_images,dim=0)
returnnew_images

RunOpenVINOModelInference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Selectdevice
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

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

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

ov_model=OVLlavaQwen2ForCausalLM(core,ov_out_path,device.value)

..code::ipython3

streamer=TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)

output_ids=ov_model.generate(input_ids,images=image_tensor,max_new_tokens=128,use_cache=True,streamer=streamer)


..parsed-literal::

Theimagefeaturesawhite,fluffylambwithaplayfulexpression.Thelambispositionedinthecenteroftheimage,anditappearstobeinmotion,asifit'srunning.Thelamb'sfurisfluffyandwhite,andithasacute,adorableappearance.Thelamb'seyesarewideopen,andithasabig,blacknose.Thelamb'searsarealsovisible,andithasacute,adorableexpression.Thelamb'smouthisopen,anditseemstobesmiling.Thelamb'slegsarealsovisible,anditappearstobeinmotion,asifit'srunning.Thelamb


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
importtime
fromtransformersimportTextIteratorStreamer,StoppingCriteria
fromthreadingimportThread
importrequests

example_image_urls=[
(
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
"bee.jpg",
),
(
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
"baklava.png",
),
("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d","small.png"),
("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754","chart.png"),
]
forurl,file_nameinexample_image_urls:
ifnotPath(file_name).exists():
Image.open(requests.get(url,stream=True).raw).save(file_name)


classKeywordsStoppingCriteria(StoppingCriteria):
def__init__(self,keywords,tokenizer,input_ids):
self.keywords=keywords
self.keyword_ids=[]
self.max_keyword_len=0
forkeywordinkeywords:
cur_keyword_ids=tokenizer(keyword).input_ids
iflen(cur_keyword_ids)>1andcur_keyword_ids[0]==tokenizer.bos_token_id:
cur_keyword_ids=cur_keyword_ids[1:]
iflen(cur_keyword_ids)>self.max_keyword_len:
self.max_keyword_len=len(cur_keyword_ids)
self.keyword_ids.append(torch.tensor(cur_keyword_ids))
self.tokenizer=tokenizer
self.start_len=input_ids.shape[1]

defcall_for_batch(self,output_ids:torch.LongTensor,scores:torch.FloatTensor,**kwargs)->bool:
offset=min(output_ids.shape[1]-self.start_len,self.max_keyword_len)
self.keyword_ids=[keyword_id.to(output_ids.device)forkeyword_idinself.keyword_ids]
forkeyword_idinself.keyword_ids:
truncated_output_ids=output_ids[0,-keyword_id.shape[0]:]
iftorch.equal(truncated_output_ids,keyword_id):
returnTrue
outputs=self.tokenizer.batch_decode(output_ids[:,-offset:],skip_special_tokens=True)[0]
forkeywordinself.keywords:
ifkeywordinoutputs:
returnTrue
returnFalse

def__call__(self,output_ids:torch.LongTensor,scores:torch.FloatTensor,**kwargs)->bool:
outputs=[]
foriinrange(output_ids.shape[0]):
outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0),scores))
returnall(outputs)


defbot_streaming(message,history):
messages=[]
ifmessage["files"]:
image=message["files"][-1]["path"]ifisinstance(message["files"][-1],dict)elsemessage["files"][-1]
else:
for_,histinenumerate(history):
ifisinstance(hist[0],tuple):
image=hist[0][0]

iflen(history)>0andimageisnotNone:
messages.append({"role":"user","content":f"<image>\n{history[1][0]}"})
messages.append({"role":"assistant","content":history[1][1]})
forhuman,assistantinhistory[2:]:
ifassistantisNone:
continue
messages.append({"role":"user","content":human})
messages.append({"role":"assistant","content":assistant})
messages.append({"role":"user","content":message["text"]})
eliflen(history)>0andimageisNone:
forhuman,assistantinhistory:
ifassistantisNone:
continue
messages.append({"role":"user","content":human})
messages.append({"role":"assistant","content":assistant})
messages.append({"role":"user","content":message["text"]})
eliflen(history)==0andimageisnotNone:
messages.append({"role":"user","content":f"<image>\n{message['text']}"})
eliflen(history)==0andimageisNone:
messages.append({"role":"user","content":message["text"]})

print(messages)
image=Image.open(image).convert("RGB")
text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
text_chunks=[tokenizer(chunk).input_idsforchunkintext.split("<image>")]
input_ids=torch.tensor(text_chunks[0]+[-200]+text_chunks[1],dtype=torch.long).unsqueeze(0)
stop_str="<|im_end|>"
keywords=[stop_str]
stopping_criteria=KeywordsStoppingCriteria(keywords,tokenizer,input_ids)
streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)

image_tensor=ov_model.process_images([image],ov_model.config)
generation_kwargs=dict(
input_ids=input_ids,images=image_tensor,streamer=streamer,max_new_tokens=128,stopping_criteria=[stopping_criteria],temperature=0.01
)
thread=Thread(target=ov_model.generate,kwargs=generation_kwargs)
thread.start()

buffer=""
fornew_textinstreamer:
buffer+=new_text
generated_text_without_prompt=buffer[:]
time.sleep(0.04)
yieldgenerated_text_without_prompt


demo=gr.ChatInterface(
fn=bot_streaming,
title="🚀nanoLLaVA",
examples=[
{"text":"Whatisontheflower?","files":["./bee.jpg"]},
{"text":"Howtomakethispastry?","files":["./baklava.png"]},
{"text":"Whatisthetextsaying?","files":["./small.png"]},
{"text":"Whatdoesthechartdisplay?","files":["./chart.png"]},
],
description="Try[nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA)usingOpenVINOinthisdemo.Uploadanimageandstartchattingaboutit,orsimplytryoneoftheexamplesbelow.Ifyoudon'tuploadanimage,youwillreceiveanerror.",
stop_btn="StopGeneration",
multimodal=True,
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


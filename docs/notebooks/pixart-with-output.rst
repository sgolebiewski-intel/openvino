PixArt-α:FastTrainingofDiffusionTransformerforPhotorealisticText-to-ImageSynthesiswithOpenVINO
=========================================================================================================

`Thispaper<https://arxiv.org/abs/2310.00426>`__introduces
`PIXART-α<https://github.com/PixArt-alpha/PixArt-alpha>`__,a
Transformer-basedT2Idiffusionmodelwhoseimagegenerationqualityis
competitivewithstate-of-the-artimagegenerators,reaching
near-commercialapplicationstandards.Additionally,itsupports
high-resolutionimagesynthesisupto1024pxresolutionwithlow
trainingcost.Toachievethisgoal,threecoredesignsareproposed:1.
Trainingstrategydecomposition:Wedevisethreedistincttrainingsteps
thatseparatelyoptimizepixeldependency,text-imagealignment,and
imageaestheticquality;2.EfficientT2ITransformer:Weincorporate
cross-attentionmodulesintoDiffusionTransformer(DiT)toinjecttext
conditionsandstreamlinethecomputation-intensiveclass-condition
branch;3.High-informativedata:Weemphasizethesignificanceof
conceptdensityintext-imagepairsandleveragealargeVision-Language
modeltoauto-labeldensepseudo-captionstoassisttext-imagealignment
learning.

|image0|

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Loadtheoriginalmodel<#load-the-original-model>`__
-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__

-`Converttextencoder<#convert-text-encoder>`__
-`Converttransformer<#convert-transformer>`__
-`ConvertVAEdecoder<#convert-vae-decoder>`__

-`Compilingmodels<#compiling-models>`__
-`Buildingthepipeline<#building-the-pipeline>`__
-`Interactiveinference<#interactive-inference>`__

..|image0|image::https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/resolve/main/asset/images/teaser.png

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"diffusers>=0.14.0"sentencepiece"datasets>=2.14.6""transformers>=4.25.1""gradio>=4.19""torch>=2.1"Pillowopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall--pre-Uqopenvino--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-dev2024.2.0requiresopenvino==2024.2.0,butyouhaveopenvino2024.4.0.dev20240712whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Loadandruntheoriginalpipeline
----------------------------------

`backtotop⬆️<#table-of-contents>`__

Weuse
`PixArt-LCM-XL-2-1024-MS<https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS>`__
thatusesLCMs.`LCMs<https://arxiv.org/abs/2310.04378>`__isa
diffusiondistillationmethodwhichpredict``PF-ODE's``solution
directlyinlatentspace,achievingsuperfastinferencewithfewsteps.

..code::ipython3

importtorch
fromdiffusersimportPixArtAlphaPipeline


pipe=PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS",use_safetensors=True)

prompt="AsmallcactuswithahappyfaceintheSaharadesert."
generator=torch.Generator().manual_seed(42)

image=pipe(prompt,guidance_scale=0.0,num_inference_steps=4,generator=generator).images[0]


..parsed-literal::

2024-07-1301:36:32.634457:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:36:32.670663:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:36:33.345290:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Loadingpipelinecomponents...:0%||0/5[00:00<?,?it/s]


..parsed-literal::

Youareusingthedefaultlegacybehaviourofthe<class'transformers.models.t5.tokenization_t5.T5Tokenizer'>.Thisisexpected,andsimplymeansthatthe`legacy`(previous)behaviorwillbeusedsonothingchangesforyou.Ifyouwanttousethenewbehaviour,set`legacy=False`.Thisshouldonlybesetifyouunderstandwhatitmeans,andthoroughlyreadthereasonwhythiswasaddedasexplainedinhttps://github.com/huggingface/transformers/pull/24565



..parsed-literal::

Loadingcheckpointshards:0%||0/4[00:00<?,?it/s]


..parsed-literal::

SomeweightsofthemodelcheckpointwerenotusedwheninitializingPixArtTransformer2DModel:
['caption_projection.y_embedding']
TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

image




..image::pixart-with-output_files/pixart-with-output_5_0.png



ConvertthemodeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__

Let’sdefinetheconversionfunctionforPyTorchmodules.Weuse
``ov.convert_model``functiontoobtainOpenVINOIntermediate
Representationobjectand``ov.save_model``functiontosaveitasXML
file.

..code::ipython3

frompathlibimportPath

importnumpyasnp
importtorch

importopenvinoasov


defconvert(model:torch.nn.Module,xml_path:str,example_input):
xml_path=Path(xml_path)
ifnotxml_path.exists():
xml_path.parent.mkdir(parents=True,exist_ok=True)
model.eval()
withtorch.no_grad():
converted_model=ov.convert_model(model,example_input=example_input)
ov.save_model(converted_model,xml_path)

#cleanupmemory
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

PixArt-αconsistsofpuretransformerblocksforlatentdiffusion:It
candirectlygenerate1024pximagesfromtextpromptswithinasingle
samplingprocess.

|image0|.

Duringinferenceitusestextencoder``T5EncoderModel``,transformer
``Transformer2DModel``andVAEdecoder``AutoencoderKL``.Let’sconvert
themodelsfromthepipelineonebyone.

..|image0|image::https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/resolve/main/asset/images/model.png

..code::ipython3

MODEL_DIR=Path("model")

TEXT_ENCODER_PATH=MODEL_DIR/"text_encoder.xml"
TRANSFORMER_OV_PATH=MODEL_DIR/"transformer_ir.xml"
VAE_DECODER_PATH=MODEL_DIR/"vae_decoder.xml"

Converttextencoder
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

example_input={
"input_ids":torch.zeros(1,120,dtype=torch.int64),
"attention_mask":torch.zeros(1,120,dtype=torch.int64),
}

convert(pipe.text_encoder,TEXT_ENCODER_PATH,example_input)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(


..parsed-literal::

['input_ids','attention_mask']


Converttransformer
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classTransformerWrapper(torch.nn.Module):
def__init__(self,transformer):
super().__init__()
self.transformer=transformer

defforward(self,hidden_states=None,timestep=None,encoder_hidden_states=None,encoder_attention_mask=None,resolution=None,aspect_ratio=None):

returnself.transformer.forward(
hidden_states,
timestep=timestep,
encoder_hidden_states=encoder_hidden_states,
encoder_attention_mask=encoder_attention_mask,
added_cond_kwargs={"resolution":resolution,"aspect_ratio":aspect_ratio},
)


example_input={
"hidden_states":torch.rand([2,4,128,128],dtype=torch.float32),
"timestep":torch.tensor([999,999]),
"encoder_hidden_states":torch.rand([2,120,4096],dtype=torch.float32),
"encoder_attention_mask":torch.rand([2,120],dtype=torch.float32),
"resolution":torch.tensor([[1024.0,1024.0],[1024.0,1024.0]]),
"aspect_ratio":torch.tensor([[1.0],[1.0]]),
}


w_transformer=TransformerWrapper(pipe.transformer)
convert(w_transformer,TRANSFORMER_OV_PATH,example_input)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/embeddings.py:219:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifself.height!=heightorself.width!=width:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/attention_processor.py:682:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcurrent_length!=target_length:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/attention_processor.py:697:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.shape[0]<batch_size*head_size:


..parsed-literal::

['hidden_states','timestep','encoder_hidden_states','encoder_attention_mask','resolution','aspect_ratio']


ConvertVAEdecoder
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classVAEDecoderWrapper(torch.nn.Module):

def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latents):
returnself.vae.decode(latents,return_dict=False)


convert(VAEDecoderWrapper(pipe.vae),VAE_DECODER_PATH,(torch.zeros((1,4,128,128))))


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifhidden_states.shape[0]>=64:


..parsed-literal::

['latents']


Compilingmodels
----------------

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

..code::ipython3

importipywidgetsaswidgets

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



..code::ipython3

compiled_model=core.compile_model(TRANSFORMER_OV_PATH)
compiled_vae=core.compile_model(VAE_DECODER_PATH)
compiled_text_encoder=core.compile_model(TEXT_ENCODER_PATH)

Buildingthepipeline
---------------------

`backtotop⬆️<#table-of-contents>`__

Let’screatecallablewrapperclassesforcompiledmodelstoallow
interactionwithoriginalpipelines.Notethatallofwrapperclasses
return``torch.Tensor``\sinsteadof``np.array``\s.

..code::ipython3

fromcollectionsimportnamedtuple

EncoderOutput=namedtuple("EncoderOutput","last_hidden_state")


classTextEncoderWrapper(torch.nn.Module):
def__init__(self,text_encoder,dtype):
super().__init__()
self.text_encoder=text_encoder
self.dtype=dtype

defforward(self,input_ids=None,attention_mask=None):
inputs={
"input_ids":input_ids,
"attention_mask":attention_mask,
}
last_hidden_state=self.text_encoder(inputs)[0]
returnEncoderOutput(torch.from_numpy(last_hidden_state))

..code::ipython3

classTransformerWrapper(torch.nn.Module):
def__init__(self,transformer,config):
super().__init__()
self.transformer=transformer
self.config=config

defforward(
self,
hidden_states=None,
timestep=None,
encoder_hidden_states=None,
encoder_attention_mask=None,
resolution=None,
aspect_ratio=None,
added_cond_kwargs=None,
**kwargs
):
inputs={
"hidden_states":hidden_states,
"timestep":timestep,
"encoder_hidden_states":encoder_hidden_states,
"encoder_attention_mask":encoder_attention_mask,
}
resolution=added_cond_kwargs["resolution"]
aspect_ratio=added_cond_kwargs["aspect_ratio"]
ifresolutionisnotNone:
inputs["resolution"]=resolution
inputs["aspect_ratio"]=aspect_ratio
outputs=self.transformer(inputs)[0]

return[torch.from_numpy(outputs)]

..code::ipython3

classVAEWrapper(torch.nn.Module):
def__init__(self,vae,config):
super().__init__()
self.vae=vae
self.config=config

defdecode(self,latents=None,**kwargs):
inputs={
"latents":latents,
}

outs=self.vae(inputs)
outs=namedtuple("VAE","sample")(torch.from_numpy(outs[0]))

returnouts

Andinsertwrappersinstancesinthepipeline:

..code::ipython3

pipe.__dict__["_internal_dict"]["_execution_device"]=pipe._execution_device#thisistoavoidsomeproblemthatcanoccurinthepipeline

pipe.register_modules(
text_encoder=TextEncoderWrapper(compiled_text_encoder,pipe.text_encoder.dtype),
transformer=TransformerWrapper(compiled_model,pipe.transformer.config),
vae=VAEWrapper(compiled_vae,pipe.vae.config),
)

..code::ipython3

generator=torch.Generator().manual_seed(42)

image=pipe(prompt=prompt,guidance_scale=0.0,num_inference_steps=4,generator=generator).images[0]


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140:FutureWarning:Accessingconfigattribute`_execution_device`directlyvia'PixArtAlphaPipeline'objectattributeisdeprecated.Pleaseaccess'_execution_device'over'PixArtAlphaPipeline'sconfigobjectinstead,e.g.'scheduler.config._execution_device'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

image




..image::pixart-with-output_files/pixart-with-output_26_0.png



Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defgenerate(prompt,seed,negative_prompt,num_inference_steps):
generator=torch.Generator().manual_seed(seed)
image=pipe(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,generator=generator,guidance_scale=0.0).images[0]
returnimage


demo=gr.Interface(
generate,
[
gr.Textbox(label="Caption"),
gr.Slider(0,np.iinfo(np.int32).max,label="Seed"),
gr.Textbox(label="Negativeprompt"),
gr.Slider(2,20,step=1,label="Numberofinferencesteps",value=4),
],
"image",
examples=[
["AsmallcactuswithahappyfaceintheSaharadesert.",42],
["anastronautsittinginadiner,eatingfries,cinematic,analogfilm",42],
[
"Pirateshiptrappedinacosmicmaelstromnebula,renderedincosmicbeachwhirlpoolengine,volumetriclighting,spectacular,ambientlights,lightpollution,cinematicatmosphere,artnouveaustyle,illustrationartartworkbySenseiJaye,intricatedetail.",
0,
],
["professionalportraitphotoofananthropomorphiccatwearingfancygentlemanhatandjacketwalkinginautumnforest.",0],
],
allow_flagging="never",
)
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


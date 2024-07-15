ImagegenerationwithStableCascadeandOpenVINO
=================================================

`StableCascade<https://huggingface.co/stabilityai/stable-cascade>`__
isbuiltuponthe
`Würstchen<https://openreview.net/forum?id=gU58d5QeGv>`__architecture
anditsmaindifferencetoothermodelslikeStableDiffusionisthatit
isworkingatamuchsmallerlatentspace.Whyisthisimportant?The
smallerthelatentspace,thefasteryoucanruninferenceandthe
cheaperthetrainingbecomes.Howsmallisthelatentspace?Stable
Diffusionusesacompressionfactorof8,resultingina1024x1024image
beingencodedto128x128.StableCascadeachievesacompressionfactor
of42,meaningthatitispossibletoencodea1024x1024imageto24x24,
whilemaintainingcrispreconstructions.Thetext-conditionalmodelis
thentrainedinthehighlycompressedlatentspace.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Loadtheoriginalmodel<#load-the-original-model>`__

-`Infertheoriginalmodel<#infer-the-original-model>`__

-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__

-`Priorpipeline<#prior-pipeline>`__
-`Decoderpipeline<#decoder-pipeline>`__

-`Selectinferencedevice<#select-inference-device>`__
-`Buildingthepipeline<#building-the-pipeline>`__
-`Inference<#inference>`__
-`Interactiveinference<#interactive-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"diffusers>=0.27.0"acceleratedatasetsgradiotransformers"nncf>=2.10.0""openvino>=2024.1.0""torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Loadandruntheoriginalpipeline
----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtorch
fromdiffusersimportStableCascadeDecoderPipeline,StableCascadePriorPipeline

prompt="animageofashibainu,donningaspacesuitandhelmet"
negative_prompt=""

prior=StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",torch_dtype=torch.float32)
decoder=StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",torch_dtype=torch.float32)


..parsed-literal::

2024-07-1303:34:38.692876:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1303:34:38.727848:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1303:34:39.399629:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:20:FutureWarning:`VQEncoderOutput`isdeprecatedandwillberemovedinversion0.31.Importing`VQEncoderOutput`from`diffusers.models.vq_model`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.autoencoders.vq_modelimportVQEncoderOutput`,instead.
deprecate("VQEncoderOutput","0.31",deprecation_message)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:25:FutureWarning:`VQModel`isdeprecatedandwillberemovedinversion0.31.Importing`VQModel`from`diffusers.models.vq_model`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.autoencoders.vq_modelimportVQModel`,instead.
deprecate("VQModel","0.31",deprecation_message)



..parsed-literal::

Loadingpipelinecomponents...:0%||0/6[00:00<?,?it/s]


..parsed-literal::

TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.



..parsed-literal::

Loadingpipelinecomponents...:0%||0/5[00:00<?,?it/s]


Toreducememoryusage,weskiptheoriginalinference.Ifyouwantrun
it,turnit.

..code::ipython3

importipywidgetsaswidgets


run_original_inference=widgets.Checkbox(
value=False,
description="Runoriginalinference",
disabled=False,
)

run_original_inference




..parsed-literal::

Checkbox(value=False,description='Runoriginalinference')



..code::ipython3

ifrun_original_inference.value:
prior.to(torch.device("cpu"))
prior_output=prior(
prompt=prompt,
height=1024,
width=1024,
negative_prompt=negative_prompt,
guidance_scale=4.0,
num_images_per_prompt=1,
num_inference_steps=20,
)

decoder_output=decoder(
image_embeddings=prior_output.image_embeddings,
prompt=prompt,
negative_prompt=negative_prompt,
guidance_scale=0.0,
output_type="pil",
num_inference_steps=10,
).images[0]
display(decoder_output)

ConvertthemodeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__

StableCascadehas2components:-Priorstage``prior``:create
low-dimensionallatentspacerepresentationoftheimageusing
text-conditionalLDM-Decoderstage``decoder``:usingrepresentation
fromPriorStage,producealatentimageinlatentspaceofhigher
dimensionalityusingLDMandusingVQGAN-decoder,decodethelatent
imagetoyieldafull-resolutionoutputimage.

Let’sdefinetheconversionfunctionforPyTorchmodules.Weuse
``ov.convert_model``functiontoobtainOpenVINOIntermediate
Representationobjectand``ov.save_model``functiontosaveitasXML
file.Weuse``nncf.compress_weights``to`compressmodel
weights<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html#compress-model-weights>`__
to8-bittoreducemodelsize.

..code::ipython3

importgc
frompathlibimportPath

importopenvinoasov
importnncf


MODELS_DIR=Path("models")


defconvert(model:torch.nn.Module,xml_path:str,example_input,input_shape=None):
xml_path=Path(xml_path)
ifnotxml_path.exists():
model.eval()
xml_path.parent.mkdir(parents=True,exist_ok=True)
withtorch.no_grad():
ifnotinput_shape:
converted_model=ov.convert_model(model,example_input=example_input)
else:
converted_model=ov.convert_model(model,example_input=example_input,input=input_shape)
converted_model=nncf.compress_weights(converted_model)
ov.save_model(converted_model,xml_path)
delconverted_model

#cleanupmemory
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

gc.collect()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Priorpipeline
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thispipelineconsistsoftextencoderandpriordiffusionmodel.From
here,wealwaysusefixedshapesinconversionbyusingan
``input_shape``parametertogeneratealessmemory-demandingmodel.

..code::ipython3

PRIOR_TEXT_ENCODER_OV_PATH=MODELS_DIR/"prior_text_encoder_model.xml"

prior.text_encoder.config.output_hidden_states=True


classTextEncoderWrapper(torch.nn.Module):
def__init__(self,text_encoder):
super().__init__()
self.text_encoder=text_encoder

defforward(self,input_ids,attention_mask):
outputs=self.text_encoder(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True)
returnoutputs["text_embeds"],outputs["last_hidden_state"],outputs["hidden_states"]


convert(
TextEncoderWrapper(prior.text_encoder),
PRIOR_TEXT_ENCODER_OV_PATH,
example_input={
"input_ids":torch.zeros(1,77,dtype=torch.int32),
"attention_mask":torch.zeros(1,77),
},
input_shape={"input_ids":((1,77),),"attention_mask":((1,77),)},
)
delprior.text_encoder
gc.collect();


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifinput_shape[-1]>1orself.sliding_windowisnotNone:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifpast_key_values_length>0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:279:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz*self.num_heads,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:287:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_attention_mask.size()!=(bsz,1,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:296:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.size()!=(bsz,1,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:319:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):


..parsed-literal::

['input_ids','attention_mask']
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(194/194)│100%(194/194)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

PRIOR_PRIOR_MODEL_OV_PATH=MODELS_DIR/"prior_prior_model.xml"

convert(
prior.prior,
PRIOR_PRIOR_MODEL_OV_PATH,
example_input={
"sample":torch.zeros(2,16,24,24),
"timestep_ratio":torch.ones(2),
"clip_text_pooled":torch.zeros(2,1,1280),
"clip_text":torch.zeros(2,77,1280),
"clip_img":torch.zeros(2,1,768),
},
input_shape=[((-1,16,24,24),),((-1),),((-1,1,1280),),((-1,77,1280),),(-1,1,768)],
)
delprior.prior
gc.collect();


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_stable_cascade.py:550:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifskipisnotNoneand(x.size(-1)!=skip.size(-1)orx.size(-2)!=skip.size(-2)):


..parsed-literal::

['sample','timestep_ratio','clip_text_pooled','clip_text','clip_img']
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(711/711)│100%(711/711)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



Decoderpipeline
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Decoderpipelineconsistsof3parts:decoder,textencoderandVQGAN.

..code::ipython3

DECODER_TEXT_ENCODER_MODEL_OV_PATH=MODELS_DIR/"decoder_text_encoder_model.xml"

convert(
TextEncoderWrapper(decoder.text_encoder),
DECODER_TEXT_ENCODER_MODEL_OV_PATH,
example_input={
"input_ids":torch.zeros(1,77,dtype=torch.int32),
"attention_mask":torch.zeros(1,77),
},
input_shape={"input_ids":((1,77),),"attention_mask":((1,77),)},
)

deldecoder.text_encoder
gc.collect();


..parsed-literal::

['input_ids','attention_mask']
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(194/194)│100%(194/194)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

DECODER_DECODER_MODEL_OV_PATH=MODELS_DIR/"decoder_decoder_model.xml"

convert(
decoder.decoder,
DECODER_DECODER_MODEL_OV_PATH,
example_input={
"sample":torch.zeros(1,4,256,256),
"timestep_ratio":torch.ones(1),
"clip_text_pooled":torch.zeros(1,1,1280),
"effnet":torch.zeros(1,16,24,24),
},
input_shape=[((-1,4,256,256),),((-1),),((-1,1,1280),),((-1,16,24,24),)],
)
deldecoder.decoder
gc.collect();


..parsed-literal::

['sample','timestep_ratio','clip_text_pooled','effnet']
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(855/855)│100%(855/855)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

VQGAN_PATH=MODELS_DIR/"vqgan_model.xml"


classVqganDecoderWrapper(torch.nn.Module):
def__init__(self,vqgan):
super().__init__()
self.vqgan=vqgan

defforward(self,h):
returnself.vqgan.decode(h)


convert(
VqganDecoderWrapper(decoder.vqgan),
VQGAN_PATH,
example_input=torch.zeros(1,4,256,256),
input_shape=(1,4,256,256),
)
deldecoder.vqgan
gc.collect();


..parsed-literal::

['h']
INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(42/42)│100%(42/42)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

..code::ipython3

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



Buildingthepipeline
---------------------

`backtotop⬆️<#table-of-contents>`__

Let’screatecallablewrapperclassesforcompiledmodelstoallow
interactionwithoriginalpipelines.Notethatallofwrapperclasses
return``torch.Tensor``\sinsteadof``np.array``\s.

..code::ipython3

fromcollectionsimportnamedtuple


BaseModelOutputWithPooling=namedtuple("BaseModelOutputWithPooling",["text_embeds","last_hidden_state","hidden_states"])


classTextEncoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,text_encoder_path,device):
self.text_encoder=core.compile_model(text_encoder_path,device.value)

def__call__(self,input_ids,attention_mask,output_hidden_states=True):
output=self.text_encoder({"input_ids":input_ids,"attention_mask":attention_mask})
text_embeds=output[0]
last_hidden_state=output[1]
hidden_states=list(output.values())[1:]
returnBaseModelOutputWithPooling(torch.from_numpy(text_embeds),torch.from_numpy(last_hidden_state),[torch.from_numpy(hs)forhsinhidden_states])

..code::ipython3

classPriorPriorWrapper:
def__init__(self,prior_path,device):
self.prior=core.compile_model(prior_path,device.value)
self.config=namedtuple("PriorWrapperConfig",["clip_image_in_channels","in_channels"])(768,16)#accessedintheoriginalworkflow
self.parameters=lambda:(torch.zeros(i,dtype=torch.float32)foriinrange(1))#accessedintheoriginalworkflow

def__call__(self,sample,timestep_ratio,clip_text_pooled,clip_text=None,clip_img=None,**kwargs):
inputs={
"sample":sample,
"timestep_ratio":timestep_ratio,
"clip_text_pooled":clip_text_pooled,
"clip_text":clip_text,
"clip_img":clip_img,
}
output=self.prior(inputs)
return[torch.from_numpy(output[0])]

..code::ipython3

classDecoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,decoder_path,device):
self.decoder=core.compile_model(decoder_path,device.value)

def__call__(self,sample,timestep_ratio,clip_text_pooled,effnet,**kwargs):
inputs={"sample":sample,"timestep_ratio":timestep_ratio,"clip_text_pooled":clip_text_pooled,"effnet":effnet}
output=self.decoder(inputs)
return[torch.from_numpy(output[0])]

..code::ipython3

VqganOutput=namedtuple("VqganOutput","sample")


classVqganWrapper:
config=namedtuple("VqganWrapperConfig","scale_factor")(0.3764)#accessedintheoriginalworkflow

def__init__(self,vqgan_path,device):
self.vqgan=core.compile_model(vqgan_path,device.value)

defdecode(self,h):
output=self.vqgan(h)[0]
output=torch.tensor(output)
returnVqganOutput(output)

Andinsertwrappersinstancesinthepipeline:

..code::ipython3

prior.text_encoder=TextEncoderWrapper(PRIOR_TEXT_ENCODER_OV_PATH,device)
prior.prior=PriorPriorWrapper(PRIOR_PRIOR_MODEL_OV_PATH,device)
decoder.decoder=DecoderWrapper(DECODER_DECODER_MODEL_OV_PATH,device)
decoder.text_encoder=TextEncoderWrapper(DECODER_TEXT_ENCODER_MODEL_OV_PATH,device)
decoder.vqgan=VqganWrapper(VQGAN_PATH,device)

Inference
---------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

prior_output=prior(
prompt=prompt,
height=1024,
width=1024,
negative_prompt=negative_prompt,
guidance_scale=4.0,
num_images_per_prompt=1,
num_inference_steps=20,
)

decoder_output=decoder(
image_embeddings=prior_output.image_embeddings,
prompt=prompt,
negative_prompt=negative_prompt,
guidance_scale=0.0,
output_type="pil",
num_inference_steps=10,
).images[0]
display(decoder_output)



..parsed-literal::

0%||0/20[00:00<?,?it/s]



..parsed-literal::

0%||0/10[00:00<?,?it/s]



..image::stable-cascade-image-generation-with-output_files/stable-cascade-image-generation-with-output_29_2.png


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defgenerate(prompt,negative_prompt,prior_guidance_scale,decoder_guidance_scale,seed):
generator=torch.Generator().manual_seed(seed)
prior_output=prior(
prompt=prompt,
height=1024,
width=1024,
negative_prompt=negative_prompt,
guidance_scale=prior_guidance_scale,
num_images_per_prompt=1,
num_inference_steps=20,
generator=generator,
)

decoder_output=decoder(
image_embeddings=prior_output.image_embeddings,
prompt=prompt,
negative_prompt=negative_prompt,
guidance_scale=decoder_guidance_scale,
output_type="pil",
num_inference_steps=10,
generator=generator,
).images[0]

returndecoder_output

..code::ipython3

importgradioasgr
importnumpyasnp


demo=gr.Interface(
generate,
[
gr.Textbox(label="Prompt"),
gr.Textbox(label="Negativeprompt"),
gr.Slider(
0,
20,
step=1,
label="Priorguidancescale",
info="Higherguidancescaleencouragestogenerateimagesthatareclosely"
"linkedtothetext`prompt`,usuallyattheexpenseoflowerimagequality.Appliestothepriorpipeline",
),
gr.Slider(
0,
20,
step=1,
label="Decoderguidancescale",
info="Higherguidancescaleencouragestogenerateimagesthatareclosely"
"linkedtothetext`prompt`,usuallyattheexpenseoflowerimagequality.Appliestothedecoderpipeline",
),
gr.Slider(0,np.iinfo(np.int32).max,label="Seed",step=1),
],
"image",
examples=[["Animageofashibainu,donningaspacesuitandhelmet","",4,0,0],["Anarmchairintheshapeofanavocado","",4,0,0]],
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


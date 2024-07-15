LightweightimagegenerationwithaMUSEdandOpenVINO
=====================================================

`Amused<https://huggingface.co/docs/diffusers/api/pipelines/amused>`__
isalightweighttexttoimagemodelbasedoffofthe
`muse<https://arxiv.org/pdf/2301.00704.pdf>`__architecture.Amusedis
particularlyusefulinapplicationsthatrequirealightweightandfast
modelsuchasgeneratingmanyimagesquicklyatonce.

AmusedisaVQVAEtokenbasedtransformerthatcangenerateanimagein
fewerforwardpassesthanmanydiffusionmodels.Incontrastwithmuse,
itusesthesmallertextencoderCLIP-L/14insteadoft5-xxl.Duetoits
smallparametercountandfewforwardpassgenerationprocess,amused
cangeneratemanyimagesquickly.Thisbenefitisseenparticularlyat
largerbatchsizes.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Loadandruntheoriginal
pipeline<#load-and-run-the-original-pipeline>`__
-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__

-`ConverttheTextEncoder<#convert-the-text-encoder>`__
-`ConverttheU-ViTtransformer<#convert-the-u-vit-transformer>`__
-`ConvertVQ-GANdecoder
(VQVAE)<#convert-vq-gan-decoder-vqvae>`__

-`Compilingmodelsandprepare
pipeline<#compiling-models-and-prepare-pipeline>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`Runmodelquantization<#run-model-quantization>`__
-`ComputeInceptionScoresandinference
time<#compute-inception-scores-and-inference-time>`__

-`Interactiveinference<#interactive-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-qtransformers"diffusers>=0.25.0""openvino>=2023.2.0""accelerate>=0.20.3""gradio>=4.19""torch>=2.1""pillow""torchmetrics""torch-fidelity"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"nncf>=2.9.0"datasets


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Loadandruntheoriginalpipeline
----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtorch
fromdiffusersimportAmusedPipeline


pipe=AmusedPipeline.from_pretrained(
"amused/amused-256",
)

prompt="kindsmilingghost"
image=pipe(prompt,generator=torch.Generator("cpu").manual_seed(8)).images[0]
image.save("text2image_256.png")



..parsed-literal::

Loadingpipelinecomponents...:0%||0/5[00:00<?,?it/s]



..parsed-literal::

0%||0/12[00:00<?,?it/s]


..code::ipython3

image




..image::amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_6_0.png



ConvertthemodeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__

aMUSEdconsistsofthreeseparatelytrainedcomponents:apre-trained
CLIP-L/14textencoder,aVQ-GAN,andaU-ViT.

..figure::https://cdn-uploads.huggingface.co/production/uploads/5dfcb1aada6d0311fd3d5448/97ca2Vqm7jBfCAzq20TtF.png
:alt:image_png

image_png

Duringinference,theU-ViTisconditionedonthetextencoder’shidden
statesanditerativelypredictsvaluesforallmaskedtokens.Thecosine
maskingscheduledeterminesapercentageofthemostconfidenttoken
predictionstobefixedaftereveryiteration.After12iterations,all
tokenshavebeenpredictedandaredecodedbytheVQ-GANintoimage
pixels.

Definepathsforconvertedmodels:

..code::ipython3

frompathlibimportPath


TRANSFORMER_OV_PATH=Path("models/transformer_ir.xml")
TEXT_ENCODER_OV_PATH=Path("models/text_encoder_ir.xml")
VQVAE_OV_PATH=Path("models/vqvae_ir.xml")

DefinetheconversionfunctionforPyTorchmodules.Weuse
``ov.convert_model``functiontoobtainOpenVINOIntermediate
Representationobjectand``ov.save_model``functiontosaveitasXML
file.

..code::ipython3

importtorch

importopenvinoasov


defconvert(model:torch.nn.Module,xml_path:str,example_input):
xml_path=Path(xml_path)
ifnotxml_path.exists():
xml_path.parent.mkdir(parents=True,exist_ok=True)
withtorch.no_grad():
converted_model=ov.convert_model(model,example_input=example_input)
ov.save_model(converted_model,xml_path,compress_to_fp16=False)

#cleanupmemory
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

ConverttheTextEncoder
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classTextEncoderWrapper(torch.nn.Module):
def__init__(self,text_encoder):
super().__init__()
self.text_encoder=text_encoder

defforward(self,input_ids=None,return_dict=None,output_hidden_states=None):
outputs=self.text_encoder(
input_ids=input_ids,
return_dict=return_dict,
output_hidden_states=output_hidden_states,
)

returnoutputs.text_embeds,outputs.last_hidden_state,outputs.hidden_states


input_ids=pipe.tokenizer(
prompt,
return_tensors="pt",
padding="max_length",
truncation=True,
max_length=pipe.tokenizer.model_max_length,
)

input_example={
"input_ids":input_ids.input_ids,
"return_dict":torch.tensor(True),
"output_hidden_states":torch.tensor(True),
}

convert(TextEncoderWrapper(pipe.text_encoder),TEXT_ENCODER_OV_PATH,input_example)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifinput_shape[-1]>1orself.sliding_windowisnotNone:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifpast_key_values_length>0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:621:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
encoder_states=()ifoutput_hidden_stateselseNone
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:626:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifoutput_hidden_states:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:275:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz*self.num_heads,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:283:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_attention_mask.size()!=(bsz,1,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:315:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:649:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifoutput_hidden_states:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:652:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnotreturn_dict:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:744:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnotreturn_dict:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:1231:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnotreturn_dict:


ConverttheU-ViTtransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classTransformerWrapper(torch.nn.Module):
def__init__(self,transformer):
super().__init__()
self.transformer=transformer

defforward(
self,
latents=None,
micro_conds=None,
pooled_text_emb=None,
encoder_hidden_states=None,
):
returnself.transformer(
latents,
micro_conds=micro_conds,
pooled_text_emb=pooled_text_emb,
encoder_hidden_states=encoder_hidden_states,
)


shape=(1,16,16)
latents=torch.full(shape,pipe.scheduler.config.mask_token_id,dtype=torch.long)
latents=torch.cat([latents]*2)


example_input={
"latents":latents,
"micro_conds":torch.rand([2,5],dtype=torch.float32),
"pooled_text_emb":torch.rand([2,768],dtype=torch.float32),
"encoder_hidden_states":torch.rand([2,77,768],dtype=torch.float32),
}


pipe.transformer.eval()
w_transformer=TransformerWrapper(pipe.transformer)
convert(w_transformer,TRANSFORMER_OV_PATH,example_input)

ConvertVQ-GANdecoder(VQVAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Function``get_latents``is
neededtoreturnreallatentsfortheconversion.DuetotheVQVAE
implementationautogeneratedtensoroftherequiredshapeisnot
suitable.Thisfunctionrepeatspartof``AmusedPipeline``.

..code::ipython3

defget_latents():
shape=(1,16,16)
latents=torch.full(shape,pipe.scheduler.config.mask_token_id,dtype=torch.long)
model_input=torch.cat([latents]*2)

model_output=pipe.transformer(
model_input,
micro_conds=torch.rand([2,5],dtype=torch.float32),
pooled_text_emb=torch.rand([2,768],dtype=torch.float32),
encoder_hidden_states=torch.rand([2,77,768],dtype=torch.float32),
)
guidance_scale=10.0
uncond_logits,cond_logits=model_output.chunk(2)
model_output=uncond_logits+guidance_scale*(cond_logits-uncond_logits)

latents=pipe.scheduler.step(
model_output=model_output,
timestep=torch.tensor(0),
sample=latents,
).prev_sample

returnlatents


classVQVAEWrapper(torch.nn.Module):
def__init__(self,vqvae):
super().__init__()
self.vqvae=vqvae

defforward(self,latents=None,force_not_quantize=True,shape=None):
outputs=self.vqvae.decode(
latents,
force_not_quantize=force_not_quantize,
shape=shape.tolist(),
)

returnoutputs


latents=get_latents()
example_vqvae_input={
"latents":latents,
"force_not_quantize":torch.tensor(True),
"shape":torch.tensor((1,16,16,64)),
}

convert(VQVAEWrapper(pipe.vqvae),VQVAE_OV_PATH,example_vqvae_input)


..parsed-literal::

/tmp/ipykernel_114139/3779428577.py:34:TracerWarning:ConvertingatensortoaPythonlistmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
shape=shape.tolist(),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/autoencoders/vq_model.py:144:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnotforce_not_quantize:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifhidden_states.shape[0]>=64:


Compilingmodelsandpreparepipeline
-------------------------------------

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

ov_text_encoder=core.compile_model(TEXT_ENCODER_OV_PATH,device.value)
ov_transformer=core.compile_model(TRANSFORMER_OV_PATH,device.value)
ov_vqvae=core.compile_model(VQVAE_OV_PATH,device.value)

Let’screatecallablewrapperclassesforcompiledmodelstoallow
interactionwithoriginal``AmusedPipeline``class.Notethatallof
wrapperclassesreturn``torch.Tensor``\sinsteadof``np.array``\s.

..code::ipython3

fromcollectionsimportnamedtuple


classConvTextEncoderWrapper(torch.nn.Module):
def__init__(self,text_encoder,config):
super().__init__()
self.config=config
self.text_encoder=text_encoder

defforward(self,input_ids=None,return_dict=None,output_hidden_states=None):
inputs={
"input_ids":input_ids,
"return_dict":return_dict,
"output_hidden_states":output_hidden_states,
}

outs=self.text_encoder(inputs)

outputs=namedtuple("CLIPTextModelOutput",("text_embeds","last_hidden_state","hidden_states"))

text_embeds=torch.from_numpy(outs[0])
last_hidden_state=torch.from_numpy(outs[1])
hidden_states=list(torch.from_numpy(out)foroutinouts.values())[2:]

returnoutputs(text_embeds,last_hidden_state,hidden_states)

..code::ipython3

classConvTransformerWrapper(torch.nn.Module):
def__init__(self,transformer,config):
super().__init__()
self.config=config
self.transformer=transformer

defforward(self,latents=None,micro_conds=None,pooled_text_emb=None,encoder_hidden_states=None,**kwargs):
outputs=self.transformer(
{
"latents":latents,
"micro_conds":micro_conds,
"pooled_text_emb":pooled_text_emb,
"encoder_hidden_states":encoder_hidden_states,
},
share_inputs=False,
)

returntorch.from_numpy(outputs[0])

..code::ipython3

classConvVQVAEWrapper(torch.nn.Module):
def__init__(self,vqvae,dtype,config):
super().__init__()
self.vqvae=vqvae
self.dtype=dtype
self.config=config

defdecode(self,latents=None,force_not_quantize=True,shape=None):
inputs={
"latents":latents,
"force_not_quantize":force_not_quantize,
"shape":torch.tensor(shape),
}

outs=self.vqvae(inputs)
outs=namedtuple("VQVAE","sample")(torch.from_numpy(outs[0]))

returnouts

Andinsertwrappersinstancesinthepipeline:

..code::ipython3

prompt="kindsmilingghost"

transformer=pipe.transformer
vqvae=pipe.vqvae
text_encoder=pipe.text_encoder

pipe.__dict__["_internal_dict"]["_execution_device"]=pipe._execution_device#thisistoavoidsomeproblemthatcanoccurinthepipeline
pipe.register_modules(
text_encoder=ConvTextEncoderWrapper(ov_text_encoder,text_encoder.config),
transformer=ConvTransformerWrapper(ov_transformer,transformer.config),
vqvae=ConvVQVAEWrapper(ov_vqvae,vqvae.dtype,vqvae.config),
)

image=pipe(prompt,generator=torch.Generator("cpu").manual_seed(8)).images[0]
image.save("text2image_256.png")


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140:FutureWarning:Accessingconfigattribute`_execution_device`directlyvia'AmusedPipeline'objectattributeisdeprecated.Pleaseaccess'_execution_device'over'AmusedPipeline'sconfigobjectinstead,e.g.'scheduler.config._execution_device'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)



..parsed-literal::

0%||0/12[00:00<?,?it/s]


..code::ipython3

image




..image::amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_28_0.png



Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``Amused``pipelinestructure,thevisiontransformermodel
takesupsignificantportionoftheoverallpipelineexecutiontime.Now
wewillshowyouhowtooptimizetheUNetpartusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__toreduce
computationcostandspeedupthepipeline.Quantizingtherestofthe
pipelinedoesnotsignificantlyimproveinferenceperformancebutcan
leadtoasubstantialdegradationofgenerationsquality.

Wealsoestimatethequalityofgenerationsproducedbyoptimized
pipelinewith`Inception
Score<https://en.wikipedia.org/wiki/Inception_score>`__whichisoften
usedtomeasurequalityoftext-to-imagegenerationsystems.

Thestepsarethefollowing:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize()``onthemodel.
3.Savethequantizedmodelusing``openvino.save_model()``function.
4.CompareinferencetimeandInceptionscorefororiginalandquantized
pipelines.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

QUANTIZED_TRANSFORMER_OV_PATH=Path(str(TRANSFORMER_OV_PATH).replace(".xml","_quantized.xml"))

skip_for_device="GPU"indevice.value
to_quantize=widgets.Checkbox(value=notskip_for_device,description="Quantization",disabled=skip_for_device)
to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



..code::ipython3

importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparecalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`conceptual_captions<https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforcalibrationwecustomize``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.autoimporttqdm
fromtypingimportAny,Dict,List
importpickle
importnumpyasnp


defdisable_progress_bar(pipeline,disable=True):
ifnothasattr(pipeline,"_progress_bar_config"):
pipeline._progress_bar_config={'disable':disable}
else:
pipeline._progress_bar_config['disable']=disable


classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,data_cache:List[Any]=None,keep_prob:float=0.5):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheisnotNoneelse[]
self.keep_prob=keep_prob

def__call__(self,*args,**kwargs):
ifnp.random.rand()<=self.keep_prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)


defcollect_calibration_data(ov_transformer_model,calibration_dataset_size:int)->List[Dict]:
calibration_dataset_filepath=Path(f"calibration_data/{calibration_dataset_size}.pkl")
ifnotcalibration_dataset_filepath.exists():
calibration_data=[]
pipe.transformer.transformer=CompiledModelDecorator(ov_transformer_model,calibration_data,keep_prob=1.0)
disable_progress_bar(pipe)

dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True).shuffle(seed=42)

#Runinferencefordatacollection
pbar=tqdm(total=calibration_dataset_size)
forbatchindataset:
prompt=batch["caption"]
iflen(prompt)>pipe.tokenizer.model_max_length:
continue
pipe(prompt,generator=torch.Generator('cpu').manual_seed(0))
pbar.update(len(calibration_data)-pbar.n)
ifpbar.n>=calibration_dataset_size:
break

pipe.transformer.transformer=ov_transformer_model
disable_progress_bar(pipe,disable=False)

calibration_dataset_filepath.parent.mkdir(exist_ok=True,parents=True)
withopen(calibration_dataset_filepath,'wb')asf:
pickle.dump(calibration_data,f)

withopen(calibration_dataset_filepath,'rb')asf:
calibration_data=pickle.load(f)
returncalibration_data

Runmodelquantization
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Runcalibrationdatacollectionandquantizethevisiontransformer
model.

..code::ipython3

%%skipnot$to_quantize.value

fromnncf.quantization.advanced_parametersimportAdvancedSmoothQuantParameters
fromnncf.quantization.range_estimatorimportRangeEstimatorParameters,StatisticsCollectorParameters,StatisticsType,\
AggregatorType
importnncf

CALIBRATION_DATASET_SIZE=12*25

ifnotQUANTIZED_TRANSFORMER_OV_PATH.exists():
calibration_data=collect_calibration_data(ov_transformer,CALIBRATION_DATASET_SIZE)
quantized_model=nncf.quantize(
core.read_model(TRANSFORMER_OV_PATH),
nncf.Dataset(calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
subset_size=len(calibration_data),
#Weignoreconvolutionstoimprovequalityofgenerationswithoutsignificantdropininferencespeed
ignored_scope=nncf.IgnoredScope(types=["Convolution"]),
#Valueof0.85wasobtainedusinggridsearchbasedonInceptionScorecomputedbelow
advanced_parameters=nncf.AdvancedQuantizationParameters(
smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=0.85),
#Duringactivationstatisticscollectionweignore1%ofoutlierswhichimprovesquantizationquality
activations_range_estimator_params=RangeEstimatorParameters(
min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN,
aggregator_type=AggregatorType.MEAN_NO_OUTLIERS,
quantile_outlier_prob=0.01),
max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX,
aggregator_type=AggregatorType.MEAN_NO_OUTLIERS,
quantile_outlier_prob=0.01)
)
)
)
ov.save_model(quantized_model,QUANTIZED_TRANSFORMER_OV_PATH)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino



..parsed-literal::

0%||0/300[00:00<?,?it/s]


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140:FutureWarning:Accessingconfigattribute`_execution_device`directlyvia'AmusedPipeline'objectattributeisdeprecated.Pleaseaccess'_execution_device'over'AmusedPipeline'sconfigobjectinstead,e.g.'scheduler.config._execution_device'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)



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

INFO:nncf:3ignorednodeswerefoundbytypesintheNNCFGraph
INFO:nncf:182ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:Notaddingactivationinputquantizerforoperation:120__module.transformer.embed.conv/aten::_convolution/Convolution
INFO:nncf:Notaddingactivationinputquantizerforoperation:2154__module.transformer.mlm_layer.conv1/aten::_convolution/Convolution
INFO:nncf:Notaddingactivationinputquantizerforoperation:2993__module.transformer.mlm_layer.conv2/aten::_convolution/Convolution



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:92:RuntimeWarning:invalidvalueencounteredinmultiply
returnTensor(self.data*unwrap_tensor_data(other))



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



Demogenerationwithquantizedpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..code::ipython3

%%skipnot$to_quantize.value

original_ov_transformer_model=pipe.transformer.transformer
pipe.transformer.transformer=core.compile_model(QUANTIZED_TRANSFORMER_OV_PATH,device.value)

image=pipe(prompt,generator=torch.Generator('cpu').manual_seed(8)).images[0]
image.save('text2image_256_quantized.png')

pipe.transformer.transformer=original_ov_transformer_model

display(image)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140:FutureWarning:Accessingconfigattribute`_execution_device`directlyvia'AmusedPipeline'objectattributeisdeprecated.Pleaseaccess'_execution_device'over'AmusedPipeline'sconfigobjectinstead,e.g.'scheduler.config._execution_device'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)



..parsed-literal::

0%||0/12[00:00<?,?it/s]



..image::amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_37_2.png


ComputeInceptionScoresandinferencetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Belowwecompute`Inception
Score<https://en.wikipedia.org/wiki/Inception_score>`__oforiginaland
quantizedpipelinesonasmallsubsetofimages.Imagesaregenerated
frompromptsof``conceptual_captions``validationset.Wealsomeasure
thetimeittooktogeneratetheimagesforcomparisonreasons.

Pleasenotethatthevalidationdatasetsizeissmallandservesonlyas
aroughestimateofgenerationquality.

..code::ipython3

%%skipnot$to_quantize.value

fromtorchmetrics.image.inceptionimportInceptionScore
fromtorchvisionimporttransformsastransforms
fromitertoolsimportislice
importtime

VALIDATION_DATASET_SIZE=100

defcompute_inception_score(ov_transformer_model_path,validation_set_size,batch_size=100):
original_ov_transformer_model=pipe.transformer.transformer
pipe.transformer.transformer=core.compile_model(ov_transformer_model_path,device.value)

disable_progress_bar(pipe)
dataset=datasets.load_dataset("google-research-datasets/conceptual_captions","unlabeled",split="validation",trust_remote_code=True).shuffle(seed=42)
dataset=islice(dataset,validation_set_size)

inception_score=InceptionScore(normalize=True,splits=1)

images=[]
infer_times=[]
forbatchintqdm(dataset,total=validation_set_size,desc="ComputingInceptionScore"):
prompt=batch["caption"]
iflen(prompt)>pipe.tokenizer.model_max_length:
continue
start_time=time.perf_counter()
image=pipe(prompt,generator=torch.Generator('cpu').manual_seed(0)).images[0]
infer_times.append(time.perf_counter()-start_time)
image=transforms.ToTensor()(image)
images.append(image)

mean_perf_time=sum(infer_times)/len(infer_times)

whilelen(images)>0:
images_batch=torch.stack(images[-batch_size:])
images=images[:-batch_size]
inception_score.update(images_batch)
kl_mean,kl_std=inception_score.compute()

pipe.transformer.transformer=original_ov_transformer_model
disable_progress_bar(pipe,disable=False)

returnkl_mean,mean_perf_time


original_inception_score,original_time=compute_inception_score(TRANSFORMER_OV_PATH,VALIDATION_DATASET_SIZE)
print(f"OriginalpipelineInceptionScore:{original_inception_score}")
quantized_inception_score,quantized_time=compute_inception_score(QUANTIZED_TRANSFORMER_OV_PATH,VALIDATION_DATASET_SIZE)
print(f"QuantizedpipelineInceptionScore:{quantized_inception_score}")
print(f"Quantizationspeed-up:{original_time/quantized_time:.2f}x")


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchmetrics/utilities/prints.py:43:UserWarning:Metric`InceptionScore`willsaveallextractedfeaturesinbuffer.Forlargedatasetsthismayleadtolargememoryfootprint.
warnings.warn(*args,**kwargs)#noqa:B028



..parsed-literal::

ComputingInceptionScore:0%||0/100[00:00<?,?it/s]


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140:FutureWarning:Accessingconfigattribute`_execution_device`directlyvia'AmusedPipeline'objectattributeisdeprecated.Pleaseaccess'_execution_device'over'AmusedPipeline'sconfigobjectinstead,e.g.'scheduler.config._execution_device'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchmetrics/image/inception.py:176:UserWarning:std():degreesoffreedomis<=0.Correctionshouldbestrictlylessthanthereductionfactor(inputnumeldividedbyoutputnumel).(Triggeredinternallyat../aten/src/ATen/native/ReduceOps.cpp:1807.)
returnkl.mean(),kl.std()


..parsed-literal::

OriginalpipelineInceptionScore:11.146076202392578



..parsed-literal::

ComputingInceptionScore:0%||0/100[00:00<?,?it/s]


..parsed-literal::

QuantizedpipelineInceptionScore:9.630992889404297
Quantizationspeed-up:2.09x


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

Belowyoucanselectwhichpipelinetorun:originalorquantized.

..code::ipython3

quantized_model_present=QUANTIZED_TRANSFORMER_OV_PATH.exists()

use_quantized_model=widgets.Checkbox(
value=Trueifquantized_model_presentelseFalse,
description="Usequantizedpipeline",
disabled=notquantized_model_present,
)

use_quantized_model




..parsed-literal::

Checkbox(value=True,description='Usequantizedpipeline')



..code::ipython3

importgradioasgr
importnumpyasnp

pipe.transformer.transformer=core.compile_model(
QUANTIZED_TRANSFORMER_OV_PATHifuse_quantized_model.valueelseTRANSFORMER_OV_PATH,
device.value,
)


defgenerate(prompt,seed,_=gr.Progress(track_tqdm=True)):
image=pipe(prompt,generator=torch.Generator("cpu").manual_seed(seed)).images[0]
returnimage


demo=gr.Interface(
generate,
[
gr.Textbox(label="Prompt"),
gr.Slider(0,np.iinfo(np.int32).max,label="Seed",step=1),
],
"image",
examples=[
["happysnowman",88],
["greenghostrider",0],
["kindsmilingghost",8],
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


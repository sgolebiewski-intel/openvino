OneStepSketchtoImagetranslationwithpix2pix-turboandOpenVINO
====================================================================

Diffusionmodelsachieveremarkableresultsinimagegeneration.They
areablesynthesizehigh-qualityimagesguidedbyuserinstructions.In
thesametime,majorityofdiffusion-basedimagegenerationapproaches
aretime-consumingduetotheiterativedenoisingprocess.Pix2Pix-turbo
modelwasproposedin`One-StepImageTranslationwithText-to-Image
Modelspaper<https://arxiv.org/abs/2403.12036>`__foraddressing
slownessofdiffusionprocessinimage-to-imagetranslationtask.Itis
basedon`SD-Turbo<https://huggingface.co/stabilityai/sd-turbo>`__,a
fastgenerativetext-to-imagemodelthatcansynthesizephotorealistic
imagesfromatextpromptinasinglenetworkevaluation.Usingonly
singleinference,pix2pix-turboachievescomparablebyqualityresults
withrecentworkssuchasControlNetforSketch2PhotoandEdge2Imagefor
50steps.

|image0|

Inthistutorialyouwilllearnhowtoturnsketchestoimagesusing
`Pix2Pix-Turbo<https://github.com/GaParmar/img2img-turbo>`__and
OpenVINO.####Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchmodel<#load-pytorch-model>`__
-`ConvertPyTorchmodeltoOpenvinoIntermediateRepresentation
format<#convert-pytorch-model-to-openvino-intermediate-representation-format>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Compilemodel<#compile-model>`__
-`Runmodelinference<#run-model-inference>`__
-`Interactivedemo<#interactive-demo>`__

..|image0|image::https://github.com/GaParmar/img2img-turbo/raw/main/assets/gen_variations.jpg

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Clone`modelrepository<https://github.com/GaParmar/img2img-turbo>`__
andinstallrequiredpackages.

..code::ipython3

%pipinstall-q"openvino>=2024.1.0""torch>=2.1"torchvision"diffusers==0.25.1""peft==0.6.2"transformerstqdmpillowopencv-python"gradio==3.43.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

repo_dir=Path("img2img-turbo")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/GaParmar/img2img-turbo.git

pix2pix_turbo_py_path=repo_dir/"src/pix2pix_turbo.py"
model_py_path=repo_dir/"src/model.py"
orig_pix2pix_turbo_path=pix2pix_turbo_py_path.parent/("orig_"+pix2pix_turbo_py_path.name)
orig_model_py_path=model_py_path.parent/("orig_"+model_py_path.name)

ifnotorig_pix2pix_turbo_path.exists():
pix2pix_turbo_py_path.rename(orig_pix2pix_turbo_path)

withorig_pix2pix_turbo_path.open("r")asf:
data=f.read()
data=data.replace("cuda","cpu")
withpix2pix_turbo_py_path.open("w")asout_f:
out_f.write(data)

ifnotorig_model_py_path.exists():
model_py_path.rename(orig_model_py_path)

withorig_model_py_path.open("r")asf:
data=f.read()
data=data.replace("cuda","cpu")
withmodel_py_path.open("w")asout_f:
out_f.write(data)
%cd$repo_dir


..parsed-literal::

Cloninginto'img2img-turbo'...
remote:Enumeratingobjects:205,done.[K
remote:Countingobjects:100%(70/70),done.[K
remote:Compressingobjects:100%(26/26),done.[K
remote:Total205(delta53),reused46(delta44),pack-reused135[K
Receivingobjects:100%(205/205),31.89MiB|19.13MiB/s,done.
Resolvingdeltas:100%(96/96),done.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/sketch-to-image-pix2pix-turbo/img2img-turbo


LoadPyTorchmodel
------------------

`backtotop⬆️<#table-of-contents>`__

Pix2Pix-turboarchitectureillustratedonthediagrambelow.Model
combinesthreeseparatemodulesintheoriginallatentdiffusionmodels
intoasingleend-to-endnetworkwithsmalltrainableweights.This
architectureallowstranslationtheinputimagextotheoutputy,while
retainingtheinputscenestructure.AuthorsuseLoRAadaptersineach
module,introduceskipconnectionsandZero-Convolutionsbetweeninput
andoutput,andretrainthefirstlayeroftheU-Net.Blueboxeson
diagramindicatetrainablelayers.Semi-transparentlayersarefrozen.
|model_diagram|

..|model_diagram|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/18f1a442-8547-4edd-85b0-d8bd1a99bdf1

..code::ipython3

importrequests
importcopy
fromtqdmimporttqdm
importtorch
fromtransformersimportAutoTokenizer,CLIPTextModel
fromdiffusersimportAutoencoderKL,UNet2DConditionModel
fromdiffusers.models.autoencoders.vaeimportDiagonalGaussianDistribution
fromdiffusers.utils.peft_utilsimportset_weights_and_activate_adapters
frompeftimportLoraConfig
importtypes

fromsrc.modelimportmake_1step_sched
fromsrc.pix2pix_turboimportTwinConv

tokenizer=AutoTokenizer.from_pretrained("stabilityai/sd-turbo",subfolder="tokenizer")


deftokenize_prompt(prompt):
caption_tokens=tokenizer(prompt,max_length=tokenizer.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids
returncaption_tokens


def_vae_encoder_fwd(self,sample):
sample=self.conv_in(sample)
l_blocks=[]
#down
fordown_blockinself.down_blocks:
l_blocks.append(sample)
sample=down_block(sample)
#middle
sample=self.mid_block(sample)
sample=self.conv_norm_out(sample)
sample=self.conv_act(sample)
sample=self.conv_out(sample)
current_down_blocks=l_blocks
returnsample,current_down_blocks


def_vae_decoder_fwd(self,sample,incoming_skip_acts,latent_embeds=None):
sample=self.conv_in(sample)
upscale_dtype=next(iter(self.up_blocks.parameters())).dtype
#middle
sample=self.mid_block(sample,latent_embeds)
sample=sample.to(upscale_dtype)
ifnotself.ignore_skip:
skip_convs=[self.skip_conv_1,self.skip_conv_2,self.skip_conv_3,self.skip_conv_4]
#up
foridx,up_blockinenumerate(self.up_blocks):
skip_in=skip_convs[idx](incoming_skip_acts[::-1][idx]*self.gamma)
#addskip
sample=sample+skip_in
sample=up_block(sample,latent_embeds)
else:
foridx,up_blockinenumerate(self.up_blocks):
sample=up_block(sample,latent_embeds)
#post-process
iflatent_embedsisNone:
sample=self.conv_norm_out(sample)
else:
sample=self.conv_norm_out(sample,latent_embeds)
sample=self.conv_act(sample)
sample=self.conv_out(sample)
returnsample


defvae_encode(self,x:torch.FloatTensor):
"""
Encodeabatchofimagesintolatents.

Args:
x(`torch.FloatTensor`):Inputbatchofimages.

Returns:
Thelatentrepresentationsoftheencodedimages.If`return_dict`isTrue,a
[`~models.autoencoder_kl.AutoencoderKLOutput`]isreturned,otherwiseaplain`tuple`isreturned.
"""
h,down_blocks=self.encoder(x)

moments=self.quant_conv(h)
posterior=DiagonalGaussianDistribution(moments)

return(posterior,down_blocks)


defvae_decode(self,z:torch.FloatTensor,skip_acts):
decoded=self._decode(z,skip_acts)[0]
return(decoded,)


defvae__decode(self,z:torch.FloatTensor,skip_acts):
z=self.post_quant_conv(z)
dec=self.decoder(z,skip_acts)

return(dec,)


classPix2PixTurbo(torch.nn.Module):
def__init__(self,pretrained_name=None,pretrained_path=None,ckpt_folder="checkpoints",lora_rank_unet=8,lora_rank_vae=4):
super().__init__()
self.text_encoder=CLIPTextModel.from_pretrained("stabilityai/sd-turbo",subfolder="text_encoder").cpu()
self.sched=make_1step_sched()

vae=AutoencoderKL.from_pretrained("stabilityai/sd-turbo",subfolder="vae")
vae.encoder.forward=types.MethodType(_vae_encoder_fwd,vae.encoder)
vae.decoder.forward=types.MethodType(_vae_decoder_fwd,vae.decoder)
vae.encode=types.MethodType(vae_encode,vae)
vae.decode=types.MethodType(vae_decode,vae)
vae._decode=types.MethodType(vae__decode,vae)
#addtheskipconnectionconvs
vae.decoder.skip_conv_1=torch.nn.Conv2d(512,512,kernel_size=(1,1),stride=(1,1),bias=False).cpu()
vae.decoder.skip_conv_2=torch.nn.Conv2d(256,512,kernel_size=(1,1),stride=(1,1),bias=False).cpu()
vae.decoder.skip_conv_3=torch.nn.Conv2d(128,512,kernel_size=(1,1),stride=(1,1),bias=False).cpu()
vae.decoder.skip_conv_4=torch.nn.Conv2d(128,256,kernel_size=(1,1),stride=(1,1),bias=False).cpu()
vae.decoder.ignore_skip=False
unet=UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo",subfolder="unet")
ckpt_folder=Path(ckpt_folder)

ifpretrained_name=="edge_to_image":
url="https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
ckpt_folder.mkdir(exist_ok=True)
outf=ckpt_folder/"edge_to_image_loras.pkl"
ifnotoutf:
print(f"Downloadingcheckpointto{outf}")
response=requests.get(url,stream=True)
total_size_in_bytes=int(response.headers.get("content-length",0))
block_size=1024#1Kibibyte
progress_bar=tqdm(total=total_size_in_bytes,unit="iB",unit_scale=True)
withopen(outf,"wb")asfile:
fordatainresponse.iter_content(block_size):
progress_bar.update(len(data))
file.write(data)
progress_bar.close()
iftotal_size_in_bytes!=0andprogress_bar.n!=total_size_in_bytes:
print("ERROR,somethingwentwrong")
print(f"Downloadedsuccessfullyto{outf}")
p_ckpt=outf
sd=torch.load(p_ckpt,map_location="cpu")
unet_lora_config=LoraConfig(r=sd["rank_unet"],init_lora_weights="gaussian",target_modules=sd["unet_lora_target_modules"])
vae_lora_config=LoraConfig(r=sd["rank_vae"],init_lora_weights="gaussian",target_modules=sd["vae_lora_target_modules"])
vae.add_adapter(vae_lora_config,adapter_name="vae_skip")
_sd_vae=vae.state_dict()
forkinsd["state_dict_vae"]:
_sd_vae[k]=sd["state_dict_vae"][k]
vae.load_state_dict(_sd_vae)
unet.add_adapter(unet_lora_config)
_sd_unet=unet.state_dict()
forkinsd["state_dict_unet"]:
_sd_unet[k]=sd["state_dict_unet"][k]
unet.load_state_dict(_sd_unet)

elifpretrained_name=="sketch_to_image_stochastic":
#downloadfromurl
url="https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
ckpt_folder.mkdir(exist_ok=True)
outf=ckpt_folder/"sketch_to_image_stochastic_lora.pkl"
ifnotoutf.exists():
print(f"Downloadingcheckpointto{outf}")
response=requests.get(url,stream=True)
total_size_in_bytes=int(response.headers.get("content-length",0))
block_size=1024#1Kibibyte
progress_bar=tqdm(total=total_size_in_bytes,unit="iB",unit_scale=True)
withopen(outf,"wb")asfile:
fordatainresponse.iter_content(block_size):
progress_bar.update(len(data))
file.write(data)
progress_bar.close()
iftotal_size_in_bytes!=0andprogress_bar.n!=total_size_in_bytes:
print("ERROR,somethingwentwrong")
print(f"Downloadedsuccessfullyto{outf}")
p_ckpt=outf
convin_pretrained=copy.deepcopy(unet.conv_in)
unet.conv_in=TwinConv(convin_pretrained,unet.conv_in)
sd=torch.load(p_ckpt,map_location="cpu")
unet_lora_config=LoraConfig(r=sd["rank_unet"],init_lora_weights="gaussian",target_modules=sd["unet_lora_target_modules"])
vae_lora_config=LoraConfig(r=sd["rank_vae"],init_lora_weights="gaussian",target_modules=sd["vae_lora_target_modules"])
vae.add_adapter(vae_lora_config,adapter_name="vae_skip")
_sd_vae=vae.state_dict()
forkinsd["state_dict_vae"]:
ifknotin_sd_vae:
continue
_sd_vae[k]=sd["state_dict_vae"][k]

vae.load_state_dict(_sd_vae)
unet.add_adapter(unet_lora_config)
_sd_unet=unet.state_dict()
forkinsd["state_dict_unet"]:
_sd_unet[k]=sd["state_dict_unet"][k]
unet.load_state_dict(_sd_unet)

elifpretrained_pathisnotNone:
sd=torch.load(pretrained_path,map_location="cpu")
unet_lora_config=LoraConfig(r=sd["rank_unet"],init_lora_weights="gaussian",target_modules=sd["unet_lora_target_modules"])
vae_lora_config=LoraConfig(r=sd["rank_vae"],init_lora_weights="gaussian",target_modules=sd["vae_lora_target_modules"])
vae.add_adapter(vae_lora_config,adapter_name="vae_skip")
_sd_vae=vae.state_dict()
forkinsd["state_dict_vae"]:
_sd_vae[k]=sd["state_dict_vae"][k]
vae.load_state_dict(_sd_vae)
unet.add_adapter(unet_lora_config)
_sd_unet=unet.state_dict()
forkinsd["state_dict_unet"]:
_sd_unet[k]=sd["state_dict_unet"][k]
unet.load_state_dict(_sd_unet)

#unet.enable_xformers_memory_efficient_attention()
unet.to("cpu")
vae.to("cpu")
self.unet,self.vae=unet,vae
self.vae.decoder.gamma=1
self.timesteps=torch.tensor([999],device="cpu").long()
self.text_encoder.requires_grad_(False)

defset_r(self,r):
self.unet.set_adapters(["default"],weights=[r])
set_weights_and_activate_adapters(self.vae,["vae_skip"],[r])
self.r=r
self.unet.conv_in.r=r
self.vae.decoder.gamma=r

defforward(self,c_t,prompt_tokens,noise_map):
caption_enc=self.text_encoder(prompt_tokens)[0]
#scaletheloraweightsbasedonthervalue
sample,current_down_blocks=self.vae.encode(c_t)
encoded_control=sample.sample()*self.vae.config.scaling_factor
#combinetheinputandnoise
unet_input=encoded_control*self.r+noise_map*(1-self.r)

unet_output=self.unet(
unet_input,
self.timesteps,
encoder_hidden_states=caption_enc,
).sample
x_denoised=self.sched.step(unet_output,self.timesteps,unet_input,return_dict=True).prev_sample
output_image=(self.vae.decode(x_denoised/self.vae.config.scaling_factor,current_down_blocks)[0]).clamp(-1,1)
returnoutput_image


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(


..code::ipython3

ov_model_path=Path("model/pix2pix-turbo.xml")

pt_model=None

ifnotov_model_path.exists():
pt_model=Pix2PixTurbo("sketch_to_image_stochastic")
pt_model.set_r(0.4)
pt_model.eval()


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(


..parsed-literal::

Downloadingcheckpointtocheckpoints/sketch_to_image_stochastic_lora.pkl


..parsed-literal::

100%|██████████|525M/525M[33:51<00:00,258kiB/s]


..parsed-literal::

Downloadedsuccessfullytocheckpoints/sketch_to_image_stochastic_lora.pkl


ConvertPyTorchmodeltoOpenvinoIntermediateRepresentationformat
--------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

StartingfromOpenVINO2023.0release,OpenVINOsupportsdirectPyTorch
modelsconversionto`OpenVINOIntermediateRepresentation(IR)
format<https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
totaketheadvantageofadvancedOpenVINOoptimizationtoolsand
features.Youneedtoprovideamodelobject,inputdataformodel
tracingto`OpenVINOModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html>`__.
``ov.convert_model``functionconvertPyTorchmodelinstanceto
``ov.Model``objectthatcanbeusedforcompilationondeviceorsaved
ondiskusing``ov.save_model``incompressedtoFP16format.

..code::ipython3

importgc
importopenvinoasov

ifnotov_model_path.exists():
example_input=[torch.ones((1,3,512,512)),torch.ones([1,77],dtype=torch.int64),torch.ones([1,4,64,64])]
withtorch.no_grad():
ov_model=ov.convert_model(pt_model,example_input=example_input,input=[[1,3,512,512],[1,77],[1,4,64,64]])
ov.save_model(ov_model,ov_model_path)
delov_model
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()
delpt_model
gc.collect();


..parsed-literal::

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
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:319:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:135:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:144:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unet_2d_condition.py:915:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifdim%default_overall_up_factor!=0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:149:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:165:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifhidden_states.shape[0]>=64:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/schedulers/scheduling_ddpm.py:433:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifmodel_output.shape[1]==sample.shape[1]*2andself.variance_typein["learned","learned_range"]:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/schedulers/scheduling_ddpm.py:440:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
alpha_prod_t_prev=self.alphas_cumprod[prev_t]ifprev_t>=0elseself.one
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/schedulers/scheduling_ddpm.py:479:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ift>0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/schedulers/scheduling_ddpm.py:330:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
alpha_prod_t_prev=self.alphas_cumprod[prev_t]ifprev_t>=0elseself.one
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Tracehadnondeterministicnodes.Didyouforgetcall.eval()onyourmodel?Nodes:
	%20785:Float(1,4,64,64,strides=[16384,4096,64,1],requires_grad=0,device=cpu)=aten::randn(%20779,%20780,%20781,%20782,%20783,%20784)#/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/torch_utils.py:80:0
	%35917:Float(1,4,64,64,strides=[16384,4096,64,1],requires_grad=0,device=cpu)=aten::randn(%35911,%35912,%35913,%35914,%35915,%35916)#/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/torch_utils.py:80:0
Thismaycauseerrorsintracechecking.Todisabletracechecking,passcheck_trace=Falsetotorch.jit.trace()
_check_trace(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Outputnr1.ofthetracedfunctiondoesnotmatchthecorrespondingoutputofthePythonfunction.Detailederror:
Tensor-likesarenotclose!

Mismatchedelements:35/786432(0.0%)
Greatestabsolutedifference:1.6555190086364746e-05atindex(0,2,421,41)(upto1e-05allowed)
Greatestrelativedifference:7.15815554884626e-05atindex(0,2,421,41)(upto1e-05allowed)
_check_trace(


..parsed-literal::

['c_t','prompt_tokens','noise_map']


Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

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



Compilemodel
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

compiled_model=core.compile_model(ov_model_path,device.value)

Runmodelinference
-------------------

`backtotop⬆️<#table-of-contents>`__

Now,let’strymodelinactionandturnsimplecatsketchinto
professionalartwork.

..code::ipython3

fromdiffusers.utilsimportload_image

sketch_image=load_image("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/f964a51d-34e8-411a-98f4-5f97a28f56b0")

sketch_image




..image::sketch-to-image-pix2pix-turbo-with-output_files/sketch-to-image-pix2pix-turbo-with-output_14_0.png



..code::ipython3

importtorchvision.transforms.functionalasF

torch.manual_seed(145)
c_t=torch.unsqueeze(F.to_tensor(sketch_image)>0.5,0)
noise=torch.randn((1,4,512//8,512//8))

..code::ipython3

prompt_template="animeartwork{prompt}.animestyle,keyvisual,vibrant,studioanime,highlydetailed"
prompt=prompt_template.replace("{prompt}","fluffymagiccat")

prompt_tokens=tokenize_prompt(prompt)

..code::ipython3

result=compiled_model([1-c_t.to(torch.float32),prompt_tokens,noise])[0]

..code::ipython3

fromPILimportImage
importnumpyasnp

image_tensor=(result[0]*0.5+0.5)*255
image=np.transpose(image_tensor,(1,2,0)).astype(np.uint8)
Image.fromarray(image)




..image::sketch-to-image-pix2pix-turbo-with-output_files/sketch-to-image-pix2pix-turbo-with-output_18_0.png



Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Inthissection,youcantrymodelonownpaintings.

**Instructions:**\*Enteratextprompt(e.g. cat)\*Startsketching,
usingpencilanderaserbuttons\*Changetheimagestyleusingastyle
template\*Trydifferentseedstogeneratedifferentresults\*
Downloadresultsusingdownloadbutton

..code::ipython3

importrandom
importbase64
fromioimportBytesIO
importgradioasgr

style_list=[
{
"name":"Cinematic",
"prompt":"cinematicstill{prompt}.emotional,harmonious,vignette,highlydetailed,highbudget,bokeh,cinemascope,moody,epic,gorgeous,filmgrain,grainy",
},
{
"name":"3DModel",
"prompt":"professional3dmodel{prompt}.octanerender,highlydetailed,volumetric,dramaticlighting",
},
{
"name":"Anime",
"prompt":"animeartwork{prompt}.animestyle,keyvisual,vibrant,studioanime,highlydetailed",
},
{
"name":"DigitalArt",
"prompt":"conceptart{prompt}.digitalartwork,illustrative,painterly,mattepainting,highlydetailed",
},
{
"name":"Photographic",
"prompt":"cinematicphoto{prompt}.35mmphotograph,film,bokeh,professional,4k,highlydetailed",
},
{
"name":"Pixelart",
"prompt":"pixel-art{prompt}.low-res,blocky,pixelartstyle,8-bitgraphics",
},
{
"name":"Fantasyart",
"prompt":"etherealfantasyconceptartof{prompt}.magnificent,celestial,ethereal,painterly,epic,majestic,magical,fantasyart,coverart,dreamy",
},
{
"name":"Neonpunk",
"prompt":"neonpunkstyle{prompt}.cyberpunk,vaporwave,neon,vibes,vibrant,stunninglybeautiful,crisp,detailed,sleek,ultramodern,magentahighlights,darkpurpleshadows,highcontrast,cinematic,ultradetailed,intricate,professional",
},
{
"name":"Manga",
"prompt":"mangastyle{prompt}.vibrant,high-energy,detailed,iconic,Japanesecomicstyle",
},
]

styles={k["name"]:k["prompt"]forkinstyle_list}
STYLE_NAMES=list(styles.keys())
DEFAULT_STYLE_NAME="Fantasyart"
MAX_SEED=np.iinfo(np.int32).max


defpil_image_to_data_uri(img,format="PNG"):
buffered=BytesIO()
img.save(buffered,format=format)
img_str=base64.b64encode(buffered.getvalue()).decode()
returnf"data:image/{format.lower()};base64,{img_str}"


defrun(image,prompt,prompt_template,style_name,seed):
print(f"prompt:{prompt}")
print("sketchupdated")
ifimageisNone:
ones=Image.new("L",(512,512),255)
temp_uri=pil_image_to_data_uri(ones)
returnones,gr.update(link=temp_uri),gr.update(link=temp_uri)
prompt=prompt_template.replace("{prompt}",prompt)
image=image.convert("RGB")
image_t=F.to_tensor(image)>0.5
print(f"seed={seed}")
caption_tokens=tokenizer(prompt,max_length=tokenizer.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids.cpu()
withtorch.no_grad():
c_t=image_t.unsqueeze(0)
torch.manual_seed(seed)
B,C,H,W=c_t.shape
noise=torch.randn((1,4,H//8,W//8))
output_image=torch.from_numpy(compiled_model([c_t.to(torch.float32),caption_tokens,noise])[0])
output_pil=F.to_pil_image(output_image[0].cpu()*0.5+0.5)
input_sketch_uri=pil_image_to_data_uri(Image.fromarray(255-np.array(image)))
output_image_uri=pil_image_to_data_uri(output_pil)
return(
output_pil,
gr.update(link=input_sketch_uri),
gr.update(link=output_image_uri),
)


defupdate_canvas(use_line,use_eraser):
ifuse_eraser:
_color="#ffffff"
brush_size=20
ifuse_line:
_color="#000000"
brush_size=4
returngr.update(brush_radius=brush_size,brush_color=_color,interactive=True)


defupload_sketch(file):
_img=Image.open(file.name)
_img=_img.convert("L")
returngr.update(value=_img,source="upload",interactive=True)


scripts="""
async()=>{
globalThis.theSketchDownloadFunction=()=>{
console.log("test")
varlink=document.createElement("a");
dataUri=document.getElementById('download_sketch').href
link.setAttribute("href",dataUri)
link.setAttribute("download","sketch.png")
document.body.appendChild(link);//RequiredforFirefox
link.click();
document.body.removeChild(link);//Cleanup

//alsocalltheoutputdownloadfunction
theOutputDownloadFunction();
returnfalse
}

globalThis.theOutputDownloadFunction=()=>{
console.log("testoutputdownloadfunction")
varlink=document.createElement("a");
dataUri=document.getElementById('download_output').href
link.setAttribute("href",dataUri);
link.setAttribute("download","output.png");
document.body.appendChild(link);//RequiredforFirefox
link.click();
document.body.removeChild(link);//Cleanup
returnfalse
}

globalThis.UNDO_SKETCH_FUNCTION=()=>{
console.log("undosketchfunction")
varbutton_undo=document.querySelector('#input_image>div.image-container.svelte-p3y7hu>div.svelte-s6ybro>button:nth-child(1)');
//Createanew'click'event
varevent=newMouseEvent('click',{
'view':window,
'bubbles':true,
'cancelable':true
});
button_undo.dispatchEvent(event);
}

globalThis.DELETE_SKETCH_FUNCTION=()=>{
console.log("deletesketchfunction")
varbutton_del=document.querySelector('#input_image>div.image-container.svelte-p3y7hu>div.svelte-s6ybro>button:nth-child(2)');
//Createanew'click'event
varevent=newMouseEvent('click',{
'view':window,
'bubbles':true,
'cancelable':true
});
button_del.dispatchEvent(event);
}

globalThis.togglePencil=()=>{
el_pencil=document.getElementById('my-toggle-pencil');
el_pencil.classList.toggle('clicked');
//simulateaclickonthegradiobutton
btn_gradio=document.querySelector("#cb-line>label>input");
varevent=newMouseEvent('click',{
'view':window,
'bubbles':true,
'cancelable':true
});
btn_gradio.dispatchEvent(event);
if(el_pencil.classList.contains('clicked')){
document.getElementById('my-toggle-eraser').classList.remove('clicked');
document.getElementById('my-div-pencil').style.backgroundColor="gray";
document.getElementById('my-div-eraser').style.backgroundColor="white";
}
else{
document.getElementById('my-toggle-eraser').classList.add('clicked');
document.getElementById('my-div-pencil').style.backgroundColor="white";
document.getElementById('my-div-eraser').style.backgroundColor="gray";
}
}

globalThis.toggleEraser=()=>{
element=document.getElementById('my-toggle-eraser');
element.classList.toggle('clicked');
//simulateaclickonthegradiobutton
btn_gradio=document.querySelector("#cb-eraser>label>input");
varevent=newMouseEvent('click',{
'view':window,
'bubbles':true,
'cancelable':true
});
btn_gradio.dispatchEvent(event);
if(element.classList.contains('clicked')){
document.getElementById('my-toggle-pencil').classList.remove('clicked');
document.getElementById('my-div-pencil').style.backgroundColor="white";
document.getElementById('my-div-eraser').style.backgroundColor="gray";
}
else{
document.getElementById('my-toggle-pencil').classList.add('clicked');
document.getElementById('my-div-pencil').style.backgroundColor="gray";
document.getElementById('my-div-eraser').style.backgroundColor="white";
}
}
}
"""

withgr.Blocks(css="style.css")asdemo:
#thesearehiddenbuttonsthatareusedtotriggerthecanvaschanges
line=gr.Checkbox(label="line",value=False,elem_id="cb-line")
eraser=gr.Checkbox(label="eraser",value=False,elem_id="cb-eraser")
withgr.Row(elem_id="main_row"):
withgr.Column(elem_id="column_input"):
gr.Markdown("##INPUT",elem_id="input_header")
image=gr.Image(
source="canvas",
tool="color-sketch",
type="pil",
image_mode="L",
invert_colors=True,
shape=(512,512),
brush_radius=4,
height=440,
width=440,
brush_color="#000000",
interactive=True,
show_download_button=True,
elem_id="input_image",
show_label=False,
)
download_sketch=gr.Button("Downloadsketch",scale=1,elem_id="download_sketch")

gr.HTML(
"""
<divclass="button-row">
<divid="my-div-pencil"class="pad2"><buttonid="my-toggle-pencil"onclick="returntogglePencil(this)"></button></div>
<divid="my-div-eraser"class="pad2"><buttonid="my-toggle-eraser"onclick="returntoggleEraser(this)"></button></div>
<divclass="pad2"><buttonid="my-button-undo"onclick="returnUNDO_SKETCH_FUNCTION(this)"></button></div>
<divclass="pad2"><buttonid="my-button-clear"onclick="returnDELETE_SKETCH_FUNCTION(this)"></button></div>
<divclass="pad2"><buttonhref="TODO"download="image"id="my-button-down"onclick='returntheSketchDownloadFunction()'></button></div>
</div>
"""
)
#gr.Markdown("##Prompt",elem_id="tools_header")
prompt=gr.Textbox(label="Prompt",value="",show_label=True)
withgr.Row():
style=gr.Dropdown(
label="Style",
choices=STYLE_NAMES,
value=DEFAULT_STYLE_NAME,
scale=1,
)
prompt_temp=gr.Textbox(
label="PromptStyleTemplate",
value=styles[DEFAULT_STYLE_NAME],
scale=2,
max_lines=1,
)

withgr.Row():
seed=gr.Textbox(label="Seed",value=42,scale=1,min_width=50)
randomize_seed=gr.Button("Random",scale=1,min_width=50)

withgr.Column(elem_id="column_process",min_width=50,scale=0.4):
gr.Markdown("##pix2pix-turbo",elem_id="description")
run_button=gr.Button("Run",min_width=50)

withgr.Column(elem_id="column_output"):
gr.Markdown("##OUTPUT",elem_id="output_header")
result=gr.Image(
label="Result",
height=440,
width=440,
elem_id="output_image",
show_label=False,
show_download_button=True,
)
download_output=gr.Button("Downloadoutput",elem_id="download_output")
gr.Markdown("###Instructions")
gr.Markdown("**1**.Enteratextprompt(e.g.cat)")
gr.Markdown("**2**.Startsketching")
gr.Markdown("**3**.Changetheimagestyleusingastyletemplate")
gr.Markdown("**4**.Trydifferentseedstogeneratedifferentresults")

eraser.change(
fn=lambdax:gr.update(value=notx),
inputs=[eraser],
outputs=[line],
queue=False,
api_name=False,
).then(update_canvas,[line,eraser],[image])
line.change(
fn=lambdax:gr.update(value=notx),
inputs=[line],
outputs=[eraser],
queue=False,
api_name=False,
).then(update_canvas,[line,eraser],[image])

demo.load(None,None,None,_js=scripts)
randomize_seed.click(
lambdax:random.randint(0,MAX_SEED),
inputs=[],
outputs=seed,
queue=False,
api_name=False,
)
inputs=[image,prompt,prompt_temp,style,seed]
outputs=[result,download_sketch,download_output]
prompt.submit(fn=run,inputs=inputs,outputs=outputs,api_name=False)
style.change(
lambdax:styles[x],
inputs=[style],
outputs=[prompt_temp],
queue=False,
api_name=False,
).then(
fn=run,
inputs=inputs,
outputs=outputs,
api_name=False,
)
run_button.click(fn=run,inputs=inputs,outputs=outputs,api_name=False)
image.change(run,inputs=inputs,outputs=outputs,queue=False,api_name=False)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

/tmp/ipykernel_173952/1555011934.py:259:GradioDeprecationWarning:'scale'valueshouldbeaninteger.Using0.4willcauseissues.
withgr.Column(elem_id="column_process",min_width=50,scale=0.4):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/gradio/utils.py:776:UserWarning:Expected1argumentsforfunction<function<lambda>at0x7fda5d68fca0>,received0.
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/gradio/utils.py:780:UserWarning:Expectedatleast1argumentsforfunction<function<lambda>at0x7fda5d68fca0>,received0.
warnings.warn(


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


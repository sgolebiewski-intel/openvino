TripoSRfeedforward3DreconstructionfromasingleimageandOpenVINO
======================================================================

`TripoSR<https://huggingface.co/spaces/stabilityai/TripoSR>`__isa
state-of-the-artopen-sourcemodelforfastfeedforward3D
reconstructionfromasingleimage,developedincollaborationbetween
`TripoAI<https://www.tripo3d.ai/>`__and`Stability
AI<https://stability.ai/news/triposr-3d-generation>`__.

Youcanfind`thesourcecodeon
GitHub<https://github.com/VAST-AI-Research/TripoSR>`__and`demoon
HuggingFace<https://huggingface.co/spaces/stabilityai/TripoSR>`__.
Also,youcanreadthepaper`TripoSR:Fast3DObjectReconstruction
fromaSingleImage<https://arxiv.org/abs/2403.02151>`__.

..figure::https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/figures/teaser800.gif
:alt:TeaserVideo

TeaserVideo

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Gettheoriginalmodel<#get-the-original-model>`__
-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__
-`Compilingmodelsandprepare
pipeline<#compiling-models-and-prepare-pipeline>`__
-`Interactiveinference<#interactive-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"gradio>=4.19""torch==2.2.2"rembgtrimesheinops"omegaconf>=2.3.0""transformers>=4.35.0""openvino>=2024.0.0"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/tatsy/torchmcubes.git"


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
descript-audiotools0.7.2requiresprotobuf<3.20,>=3.9.2,butyouhaveprotobuf3.20.3whichisincompatible.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.2.2+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
torchaudio2.3.1+cpurequirestorch==2.3.1,butyouhavetorch2.2.2+cpuwhichisincompatible.
torchvision0.18.1+cpurequirestorch==2.3.1,butyouhavetorch2.2.2+cpuwhichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importsys
frompathlibimportPath

ifnotPath("TripoSR").exists():
!gitclonehttps://huggingface.co/spaces/stabilityai/TripoSR

sys.path.append("TripoSR")


..parsed-literal::

Cloninginto'TripoSR'...
remote:Enumeratingobjects:117,done.[K
remote:Countingobjects:100%(113/113),done.[K
remote:Compressingobjects:100%(111/111),done.[K
remote:Total117(delta36),reused0(delta0),pack-reused4(from1)[K
Receivingobjects:100%(117/117),569.16KiB|2.61MiB/s,done.
Resolvingdeltas:100%(36/36),done.


Gettheoriginalmodel
----------------------

..code::ipython3

importos

fromtsr.systemimportTSR


model=TSR.from_pretrained(
"stabilityai/TripoSR",
config_name="config.yaml",
weight_name="model.ckpt",
)
model.renderer.set_chunk_size(131072)
model.to("cpu")




..parsed-literal::

TSR(
(image_tokenizer):DINOSingleImageTokenizer(
(model):ViTModel(
(embeddings):ViTEmbeddings(
(patch_embeddings):ViTPatchEmbeddings(
(projection):Conv2d(3,768,kernel_size=(16,16),stride=(16,16))
)
(dropout):Dropout(p=0.0,inplace=False)
)
(encoder):ViTEncoder(
(layer):ModuleList(
(0-11):12xViTLayer(
(attention):ViTAttention(
(attention):ViTSelfAttention(
(query):Linear(in_features=768,out_features=768,bias=True)
(key):Linear(in_features=768,out_features=768,bias=True)
(value):Linear(in_features=768,out_features=768,bias=True)
(dropout):Dropout(p=0.0,inplace=False)
)
(output):ViTSelfOutput(
(dense):Linear(in_features=768,out_features=768,bias=True)
(dropout):Dropout(p=0.0,inplace=False)
)
)
(intermediate):ViTIntermediate(
(dense):Linear(in_features=768,out_features=3072,bias=True)
(intermediate_act_fn):GELUActivation()
)
(output):ViTOutput(
(dense):Linear(in_features=3072,out_features=768,bias=True)
(dropout):Dropout(p=0.0,inplace=False)
)
(layernorm_before):LayerNorm((768,),eps=1e-12,elementwise_affine=True)
(layernorm_after):LayerNorm((768,),eps=1e-12,elementwise_affine=True)
)
)
)
(layernorm):LayerNorm((768,),eps=1e-12,elementwise_affine=True)
(pooler):ViTPooler(
(dense):Linear(in_features=768,out_features=768,bias=True)
(activation):Tanh()
)
)
)
(tokenizer):Triplane1DTokenizer()
(backbone):Transformer1D(
(norm):GroupNorm(32,1024,eps=1e-06,affine=True)
(proj_in):Linear(in_features=1024,out_features=1024,bias=True)
(transformer_blocks):ModuleList(
(0-15):16xBasicTransformerBlock(
(norm1):LayerNorm((1024,),eps=1e-05,elementwise_affine=True)
(attn1):Attention(
(to_q):Linear(in_features=1024,out_features=1024,bias=False)
(to_k):Linear(in_features=1024,out_features=1024,bias=False)
(to_v):Linear(in_features=1024,out_features=1024,bias=False)
(to_out):ModuleList(
(0):Linear(in_features=1024,out_features=1024,bias=True)
(1):Dropout(p=0.0,inplace=False)
)
)
(norm2):LayerNorm((1024,),eps=1e-05,elementwise_affine=True)
(attn2):Attention(
(to_q):Linear(in_features=1024,out_features=1024,bias=False)
(to_k):Linear(in_features=768,out_features=1024,bias=False)
(to_v):Linear(in_features=768,out_features=1024,bias=False)
(to_out):ModuleList(
(0):Linear(in_features=1024,out_features=1024,bias=True)
(1):Dropout(p=0.0,inplace=False)
)
)
(norm3):LayerNorm((1024,),eps=1e-05,elementwise_affine=True)
(ff):FeedForward(
(net):ModuleList(
(0):GEGLU(
(proj):Linear(in_features=1024,out_features=8192,bias=True)
)
(1):Dropout(p=0.0,inplace=False)
(2):Linear(in_features=4096,out_features=1024,bias=True)
)
)
)
)
(proj_out):Linear(in_features=1024,out_features=1024,bias=True)
)
(post_processor):TriplaneUpsampleNetwork(
(upsample):ConvTranspose2d(1024,40,kernel_size=(2,2),stride=(2,2))
)
(decoder):NeRFMLP(
(layers):Sequential(
(0):Linear(in_features=120,out_features=64,bias=True)
(1):SiLU(inplace=True)
(2):Linear(in_features=64,out_features=64,bias=True)
(3):SiLU(inplace=True)
(4):Linear(in_features=64,out_features=64,bias=True)
(5):SiLU(inplace=True)
(6):Linear(in_features=64,out_features=64,bias=True)
(7):SiLU(inplace=True)
(8):Linear(in_features=64,out_features=64,bias=True)
(9):SiLU(inplace=True)
(10):Linear(in_features=64,out_features=64,bias=True)
(11):SiLU(inplace=True)
(12):Linear(in_features=64,out_features=64,bias=True)
(13):SiLU(inplace=True)
(14):Linear(in_features=64,out_features=64,bias=True)
(15):SiLU(inplace=True)
(16):Linear(in_features=64,out_features=64,bias=True)
(17):SiLU(inplace=True)
(18):Linear(in_features=64,out_features=4,bias=True)
)
)
(renderer):TriplaneNeRFRenderer()
)



ConvertthemodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

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

Theoriginalmodelisapipelineofseveralmodels.Thereare
``image_tokenizer``,``tokenizer``,``backbone``and``post_processor``.
``image_tokenizer``contains``ViTModel``thatconsistsof
``ViTPatchEmbeddings``,``ViTEncoder``and``ViTPooler``.``tokenizer``
is``Triplane1DTokenizer``,``backbone``is``Transformer1D``,
``post_processor``is``TriplaneUpsampleNetwork``.Convertallinternal
modelsonebyone.

..code::ipython3

VIT_PATCH_EMBEDDINGS_OV_PATH=Path("models/vit_patch_embeddings_ir.xml")


classPatchEmbedingWrapper(torch.nn.Module):
def__init__(self,patch_embeddings):
super().__init__()
self.patch_embeddings=patch_embeddings

defforward(self,pixel_values,interpolate_pos_encoding=True):
outputs=self.patch_embeddings(pixel_values=pixel_values,interpolate_pos_encoding=True)
returnoutputs


example_input={
"pixel_values":torch.rand([1,3,512,512],dtype=torch.float32),
}

convert(
PatchEmbedingWrapper(model.image_tokenizer.model.embeddings.patch_embeddings),
VIT_PATCH_EMBEDDINGS_OV_PATH,
example_input,
)


..parsed-literal::

['pixel_values']


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py:167:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnum_channels!=self.num_channels:


..code::ipython3

VIT_ENCODER_OV_PATH=Path("models/vit_encoder_ir.xml")


classEncoderWrapper(torch.nn.Module):
def__init__(self,encoder):
super().__init__()
self.encoder=encoder

defforward(
self,
hidden_states=None,
head_mask=None,
output_attentions=False,
output_hidden_states=False,
return_dict=False,
):
outputs=self.encoder(
hidden_states=hidden_states,
)

returnoutputs.last_hidden_state


example_input={
"hidden_states":torch.rand([1,1025,768],dtype=torch.float32),
}

convert(
EncoderWrapper(model.image_tokenizer.model.encoder),
VIT_ENCODER_OV_PATH,
example_input,
)


..parsed-literal::

['hidden_states']


..code::ipython3

VIT_POOLER_OV_PATH=Path("models/vit_pooler_ir.xml")
convert(
model.image_tokenizer.model.pooler,
VIT_POOLER_OV_PATH,
torch.rand([1,1025,768],dtype=torch.float32),
)


..parsed-literal::

['hidden_states']


..code::ipython3

TOKENIZER_OV_PATH=Path("models/tokenizer_ir.xml")
convert(model.tokenizer,TOKENIZER_OV_PATH,torch.tensor(1))


..parsed-literal::

['batch_size']


..code::ipython3

example_input={
"hidden_states":torch.rand([1,1024,3072],dtype=torch.float32),
"encoder_hidden_states":torch.rand([1,1025,768],dtype=torch.float32),
}

BACKBONE_OV_PATH=Path("models/backbone_ir.xml")
convert(model.backbone,BACKBONE_OV_PATH,example_input)


..parsed-literal::

['hidden_states','encoder_hidden_states']


..code::ipython3

POST_PROCESSOR_OV_PATH=Path("models/post_processor_ir.xml")
convert(
model.post_processor,
POST_PROCESSOR_OV_PATH,
torch.rand([1,3,1024,32,32],dtype=torch.float32),
)


..parsed-literal::

['triplanes']


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

compiled_vit_patch_embeddings=core.compile_model(VIT_PATCH_EMBEDDINGS_OV_PATH,device.value)
compiled_vit_model_encoder=core.compile_model(VIT_ENCODER_OV_PATH,device.value)
compiled_vit_model_pooler=core.compile_model(VIT_POOLER_OV_PATH,device.value)

compiled_tokenizer=core.compile_model(TOKENIZER_OV_PATH,device.value)
compiled_backbone=core.compile_model(BACKBONE_OV_PATH,device.value)
compiled_post_processor=core.compile_model(POST_PROCESSOR_OV_PATH,device.value)

Let’screatecallablewrapperclassesforcompiledmodelstoallow
interactionwithoriginal``TSR``class.Notethatallofwrapper
classesreturn``torch.Tensor``\sinsteadof``np.array``\s.

..code::ipython3

fromcollectionsimportnamedtuple


classVitPatchEmdeddingsWrapper(torch.nn.Module):
def__init__(self,vit_patch_embeddings,model):
super().__init__()
self.vit_patch_embeddings=vit_patch_embeddings
self.projection=model.projection

defforward(self,pixel_values,interpolate_pos_encoding=False):
inputs={
"pixel_values":pixel_values,
}
outs=self.vit_patch_embeddings(inputs)[0]

returntorch.from_numpy(outs)


classVitModelEncoderWrapper(torch.nn.Module):
def__init__(self,vit_model_encoder):
super().__init__()
self.vit_model_encoder=vit_model_encoder

defforward(
self,
hidden_states,
head_mask,
output_attentions=False,
output_hidden_states=False,
return_dict=False,
):
inputs={
"hidden_states":hidden_states.detach().numpy(),
}

outs=self.vit_model_encoder(inputs)
outputs=namedtuple("BaseModelOutput",("last_hidden_state","hidden_states","attentions"))

returnoutputs(torch.from_numpy(outs[0]),None,None)


classVitModelPoolerWrapper(torch.nn.Module):
def__init__(self,vit_model_pooler):
super().__init__()
self.vit_model_pooler=vit_model_pooler

defforward(self,hidden_states):
outs=self.vit_model_pooler(hidden_states.detach().numpy())[0]

returntorch.from_numpy(outs)


classTokenizerWrapper(torch.nn.Module):
def__init__(self,tokenizer,model):
super().__init__()
self.tokenizer=tokenizer
self.detokenize=model.detokenize

defforward(self,batch_size):
outs=self.tokenizer(batch_size)[0]

returntorch.from_numpy(outs)


classBackboneWrapper(torch.nn.Module):
def__init__(self,backbone):
super().__init__()
self.backbone=backbone

defforward(self,hidden_states,encoder_hidden_states):
inputs={
"hidden_states":hidden_states,
"encoder_hidden_states":encoder_hidden_states.detach().numpy(),
}

outs=self.backbone(inputs)[0]

returntorch.from_numpy(outs)


classPostProcessorWrapper(torch.nn.Module):
def__init__(self,post_processor):
super().__init__()
self.post_processor=post_processor

defforward(self,triplanes):
outs=self.post_processor(triplanes)[0]

returntorch.from_numpy(outs)

Replaceallmodelsintheoriginalmodelbywrappersinstances:

..code::ipython3

model.image_tokenizer.model.embeddings.patch_embeddings=VitPatchEmdeddingsWrapper(
compiled_vit_patch_embeddings,
model.image_tokenizer.model.embeddings.patch_embeddings,
)
model.image_tokenizer.model.encoder=VitModelEncoderWrapper(compiled_vit_model_encoder)
model.image_tokenizer.model.pooler=VitModelPoolerWrapper(compiled_vit_model_pooler)

model.tokenizer=TokenizerWrapper(compiled_tokenizer,model.tokenizer)
model.backbone=BackboneWrapper(compiled_backbone)
model.post_processor=PostProcessorWrapper(compiled_post_processor)

Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtempfile

importgradioasgr
importnumpyasnp
importrembg
fromPILimportImage

fromtsr.utilsimportremove_background,resize_foreground,to_gradio_3d_orientation


rembg_session=rembg.new_session()


defcheck_input_image(input_image):
ifinput_imageisNone:
raisegr.Error("Noimageuploaded!")


defpreprocess(input_image,do_remove_background,foreground_ratio):
deffill_background(image):
image=np.array(image).astype(np.float32)/255.0
image=image[:,:,:3]*image[:,:,3:4]+(1-image[:,:,3:4])*0.5
image=Image.fromarray((image*255.0).astype(np.uint8))
returnimage

ifdo_remove_background:
image=input_image.convert("RGB")
image=remove_background(image,rembg_session)
image=resize_foreground(image,foreground_ratio)
image=fill_background(image)
else:
image=input_image
ifimage.mode=="RGBA":
image=fill_background(image)
returnimage


defgenerate(image):
scene_codes=model(image,"cpu")#thedeviceisprovidedfortheimageprocessor
mesh=model.extract_mesh(scene_codes)[0]
mesh=to_gradio_3d_orientation(mesh)
mesh_path=tempfile.NamedTemporaryFile(suffix=".obj",delete=False)
mesh.export(mesh_path.name)
returnmesh_path.name


withgr.Blocks()asdemo:
withgr.Row(variant="panel"):
withgr.Column():
withgr.Row():
input_image=gr.Image(
label="InputImage",
image_mode="RGBA",
sources="upload",
type="pil",
elem_id="content_image",
)
processed_image=gr.Image(label="ProcessedImage",interactive=False)
withgr.Row():
withgr.Group():
do_remove_background=gr.Checkbox(label="RemoveBackground",value=True)
foreground_ratio=gr.Slider(
label="ForegroundRatio",
minimum=0.5,
maximum=1.0,
value=0.85,
step=0.05,
)
withgr.Row():
submit=gr.Button("Generate",elem_id="generate",variant="primary")
withgr.Column():
withgr.Tab("Model"):
output_model=gr.Model3D(
label="OutputModel",
interactive=False,
)
withgr.Row(variant="panel"):
gr.Examples(
examples=[os.path.join("TripoSR/examples",img_name)forimg_nameinsorted(os.listdir("TripoSR/examples"))],
inputs=[input_image],
outputs=[processed_image,output_model],
label="Examples",
examples_per_page=20,
)
submit.click(fn=check_input_image,inputs=[input_image]).success(
fn=preprocess,
inputs=[input_image,do_remove_background,foreground_ratio],
outputs=[processed_image],
).success(
fn=generate,
inputs=[processed_image],
outputs=[output_model],
)

try:
demo.launch(debug=False,height=680)
exceptException:
demo.queue().launch(share=True,debug=False,height=680)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="680"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


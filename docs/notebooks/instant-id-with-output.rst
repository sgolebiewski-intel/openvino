InstantID:Zero-shotIdentity-PreservingGenerationusingOpenVINO
==================================================================

Nowadayshasbeensignificantprogressinpersonalizedimagesynthesis
withmethodssuchasTextualInversion,DreamBooth,andLoRA.However,
theirreal-worldapplicabilityishinderedbyhighstoragedemands,
lengthyfine-tuningprocesses,andtheneedformultiplereference
images.Conversely,existingIDembedding-basedmethods,whilerequiring
onlyasingleforwardinference,facechallenges:theyeither
necessitateextensivefine-tuningacrossnumerousmodelparameters,lack
compatibilitywithcommunitypre-trainedmodels,orfailtomaintain
highfacefidelity.

`InstantID<https://instantid.github.io/>`__isatuning-freemethodto
achieveID-Preservinggenerationwithonlysingleimage,supporting
variousdownstreamtasks.|applications.png|

GivenonlyonereferenceIDimage,InstantIDaimstogeneratecustomized
imageswithvariousposesorstylesfromasinglereferenceIDimage
whileensuringhighfidelity.Followingfigureprovidesanoverviewof
themethod.Itincorporatesthreecrucialcomponents:

1.AnIDembeddingthatcapturesrobustsemanticfaceinformation;
2.Alightweightadaptedmodulewithdecoupledcross-attention,
facilitatingtheuseofanimageasavisualprompt;
3.AnIdentityNetthatencodesthedetailedfeaturesfromthereference
facialimagewithadditionalspatialcontrol.

..figure::https://instantid.github.io/static/documents/pipeline.png
:alt:instantid-components.png

instantid-components.png

ThedifferenceInstantIDfrompreviousworksinthefollowingaspects:
1.donotinvolveUNettraining,soitcanpreservethegeneration
abilityoftheoriginaltext-to-imagemodelandbecompatiblewith
existingpre-trainedmodelsandControlNetsinthecommunity;2.doesn’t
requiretest-timetuning,soforaspecificcharacter,thereisnoneed
tocollectmultipleimagesforfine-tuning,onlyasingleimageneedsto
beinferredonce;3.achievebetterfacefidelity,andretainthe
editabilityoftext.

Youcanfindmoredetailsabouttheapproachwith`projectweb
page<https://instantid.github.io/>`__,
`paper<https://arxiv.org/abs/2401.07519>`__and`original
repository<https://github.com/InstantID/InstantID>`__

Inthistutorial,weconsiderhowtouseInstantIDwithOpenVINO.An
additionalpartdemonstrateshowtorunoptimizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__tospeedup
pipeline.####Tableofcontents:-`Prerequisites<#prerequisites>`__-
`ConvertandprepareFace
IdentityNet<#convert-and-prepare-face-identitynet>`__-`Select
InferenceDeviceforFace
Recognition<#select-inference-device-for-face-recognition>`__-
`PerformFaceIdentityextraction<#perform-face-identity-extraction>`__
-`PrepareInstantIDpipeline<#prepare-instantid-pipeline>`__-
`ConvertInstantIDpipelinecomponentstoOpenVINOIntermediate
Representation
format<#convert-instantid-pipeline-components-to-openvino-intermediate-representation-format>`__
-`ControlNet<#controlnet>`__-`Unet<#unet>`__-`VAE
Decoder<#vae-decoder>`__-`TextEncoders<#text-encoders>`__-`Image
ProjectionModel<#image-projection-model>`__-`PrepareOpenVINO
InstantIDPipeline<#prepare-openvino-instantid-pipeline>`__-`Run
OpenVINOpipelineinference<#run-openvino-pipeline-inference>`__-
`Selectinferencedevicefor
InstantID<#select-inference-device-for-instantid>`__-`Create
pipeline<#create-pipeline>`__-`Runinference<#run-inference>`__-
`Quantization<#quantization>`__-`Preparecalibration
datasets<#prepare-calibration-datasets>`__-`Run
quantization<#run-quantization>`__-`RunControlNet
Quantization<#run-controlnet-quantization>`__-`RunUNetHybrid
Quantization<#run-unet-hybrid-quantization>`__-`RunWeights
Compression<#run-weights-compression>`__-`Comparemodelfile
sizes<#compare-model-file-sizes>`__-`Compareinferencetimeofthe
FP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__-
`Interactivedemo<#interactive-demo>`__

..|applications.png|image::https://github.com/InstantID/InstantID/blob/main/assets/applications.png?raw=true

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath
importsys

repo_dir=Path("InstantID")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/InstantID/InstantID.git

sys.path.append(str(repo_dir))

..code::ipython3

%pipinstall-q"openvino>=2023.3.0"opencv-pythontransformersdiffusersaccelerategdown"scikit-image>=0.19.2""gradio>=4.19""nncf>=2.9.0""datasets>=2.14.6""peft==0.6.2"

ConvertandprepareFaceIdentityNet
------------------------------------

`backtotop⬆️<#table-of-contents>`__

Forgettingfaceembeddingsandposekeypoints,InstantIDuses
`InsightFace<https://github.com/deepinsight/insightface>`__face
analyticlibrary.ItsmodelsaredistributedinONNXformatandcanbe
runwithOpenVINO.Forpreparingthefaceimage,weneedtodetectthe
boundingboxesandkeypointsforthefaceusingtheRetinaFacemodel,
cropthedetectedface,alignthefacelocationusinglandmarks,and
provideeachfaceintotheArcfacefaceembeddingmodelforgettingthe
person’sidentityembeddings.

ThecodebelowdownloadstheInsightFaceAntelopev2modelkitand
providesasimpleinterfacecompatiblewithInsightFaceforgettingface
recognitionresults.

..code::ipython3

MODELS_DIR=Path("models")
face_detector_path=MODELS_DIR/"antelopev2"/"scrfd_10g_bnkps.onnx"
face_embeddings_path=MODELS_DIR/"antelopev2"/"glintr100.onnx"

..code::ipython3

fromzipfileimportZipFile
importgdown

archive_file=Path("antelopev2.zip")

ifnotface_detector_path.exists()orface_embeddings_path.exists():
ifnotarchive_file.exists():
gdown.download(
"https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8",
str(archive_file),
)
withZipFile(archive_file,"r")aszip_face_models:
zip_face_models.extractall(MODELS_DIR)

..code::ipython3

importcv2
importnumpyasnp
fromskimageimporttransformastrans


defsoftmax(z):
assertlen(z.shape)==2
s=np.max(z,axis=1)
s=s[:,np.newaxis]#necessarysteptodobroadcasting
e_x=np.exp(z-s)
div=np.sum(e_x,axis=1)
div=div[:,np.newaxis]#dito
returne_x/div


defdistance2bbox(points,distance,max_shape=None):
"""Decodedistancepredictiontoboundingbox.

Args:
points(Tensor):Shape(n,2),[x,y].
distance(Tensor):Distancefromthegivenpointto4
boundaries(left,top,right,bottom).
max_shape(tuple):Shapeoftheimage.

Returns:
Tensor:Decodedbboxes.
"""
x1=points[:,0]-distance[:,0]
y1=points[:,1]-distance[:,1]
x2=points[:,0]+distance[:,2]
y2=points[:,1]+distance[:,3]
ifmax_shapeisnotNone:
x1=x1.clamp(min=0,max=max_shape[1])
y1=y1.clamp(min=0,max=max_shape[0])
x2=x2.clamp(min=0,max=max_shape[1])
y2=y2.clamp(min=0,max=max_shape[0])
returnnp.stack([x1,y1,x2,y2],axis=-1)


defdistance2kps(points,distance,max_shape=None):
"""Decodedistancepredictiontoboundingbox.

Args:
points(Tensor):Shape(n,2),[x,y].
distance(Tensor):Distancefromthegivenpointto4
boundaries(left,top,right,bottom).
max_shape(tuple):Shapeoftheimage.

Returns:
Tensor:Decodedbboxes.
"""
preds=[]
foriinrange(0,distance.shape[1],2):
px=points[:,i%2]+distance[:,i]
py=points[:,i%2+1]+distance[:,i+1]
ifmax_shapeisnotNone:
px=px.clamp(min=0,max=max_shape[1])
py=py.clamp(min=0,max=max_shape[0])
preds.append(px)
preds.append(py)
returnnp.stack(preds,axis=-1)


defprepare_input(image,std,mean,reverse_channels=True):
normalized_image=(image.astype(np.float32)-mean)/std
ifreverse_channels:
normalized_image=normalized_image[:,:,::-1]
input_tensor=np.expand_dims(np.transpose(normalized_image,(2,0,1)),0)
returninput_tensor


classRetinaFace:
def__init__(self,ov_model):
self.taskname="detection"
self.ov_model=ov_model
self.center_cache={}
self.nms_thresh=0.4
self.det_thresh=0.5
self._init_vars()

def_init_vars(self):
self.input_size=(640,640)
outputs=self.ov_model.outputs
self.input_mean=127.5
self.input_std=128.0
#print(self.output_names)
#assertlen(outputs)==10orlen(outputs)==15
self.use_kps=False
self._anchor_ratio=1.0
self._num_anchors=1
iflen(outputs)==6:
self.fmc=3
self._feat_stride_fpn=[8,16,32]
self._num_anchors=2
eliflen(outputs)==9:
self.fmc=3
self._feat_stride_fpn=[8,16,32]
self._num_anchors=2
self.use_kps=True
eliflen(outputs)==10:
self.fmc=5
self._feat_stride_fpn=[8,16,32,64,128]
self._num_anchors=1
eliflen(outputs)==15:
self.fmc=5
self._feat_stride_fpn=[8,16,32,64,128]
self._num_anchors=1
self.use_kps=True

defprepare(self,**kwargs):
nms_thresh=kwargs.get("nms_thresh",None)
ifnms_threshisnotNone:
self.nms_thresh=nms_thresh
det_thresh=kwargs.get("det_thresh",None)
ifdet_threshisnotNone:
self.det_thresh=det_thresh
input_size=kwargs.get("input_size",None)
ifinput_sizeisnotNone:
ifself.input_sizeisnotNone:
print("warning:det_sizeisalreadysetindetectionmodel,ignore")
else:
self.input_size=input_size

defforward(self,img,threshold):
scores_list=[]
bboxes_list=[]
kpss_list=[]
blob=prepare_input(img,self.input_mean,self.input_std,True)
net_outs=self.ov_model(blob)

input_height=blob.shape[2]
input_width=blob.shape[3]
fmc=self.fmc
foridx,strideinenumerate(self._feat_stride_fpn):
scores=net_outs[idx]
bbox_preds=net_outs[idx+fmc]
bbox_preds=bbox_preds*stride
ifself.use_kps:
kps_preds=net_outs[idx+fmc*2]*stride
height=input_height//stride
width=input_width//stride
key=(height,width,stride)
ifkeyinself.center_cache:
anchor_centers=self.center_cache[key]
else:
anchor_centers=np.stack(np.mgrid[:height,:width][::-1],axis=-1).astype(np.float32)
anchor_centers=(anchor_centers*stride).reshape((-1,2))
ifself._num_anchors>1:
anchor_centers=np.stack([anchor_centers]*self._num_anchors,axis=1).reshape((-1,2))
iflen(self.center_cache)<100:
self.center_cache[key]=anchor_centers

pos_inds=np.where(scores>=threshold)[0]
bboxes=distance2bbox(anchor_centers,bbox_preds)
pos_scores=scores[pos_inds]
pos_bboxes=bboxes[pos_inds]
scores_list.append(pos_scores)
bboxes_list.append(pos_bboxes)
ifself.use_kps:
kpss=distance2kps(anchor_centers,kps_preds)
#kpss=kps_preds
kpss=kpss.reshape((kpss.shape[0],-1,2))
pos_kpss=kpss[pos_inds]
kpss_list.append(pos_kpss)
returnscores_list,bboxes_list,kpss_list

defdetect(self,img,input_size=None,max_num=0,metric="default"):
assertinput_sizeisnotNoneorself.input_sizeisnotNone
input_size=self.input_sizeifinput_sizeisNoneelseinput_size

im_ratio=float(img.shape[0])/img.shape[1]
model_ratio=float(input_size[1])/input_size[0]
ifim_ratio>model_ratio:
new_height=input_size[1]
new_width=int(new_height/im_ratio)
else:
new_width=input_size[0]
new_height=int(new_width*im_ratio)
det_scale=float(new_height)/img.shape[0]
resized_img=cv2.resize(img,(new_width,new_height))
det_img=np.zeros((input_size[1],input_size[0],3),dtype=np.uint8)
det_img[:new_height,:new_width,:]=resized_img

scores_list,bboxes_list,kpss_list=self.forward(det_img,self.det_thresh)

scores=np.vstack(scores_list)
scores_ravel=scores.ravel()
order=scores_ravel.argsort()[::-1]
bboxes=np.vstack(bboxes_list)/det_scale
ifself.use_kps:
kpss=np.vstack(kpss_list)/det_scale
pre_det=np.hstack((bboxes,scores)).astype(np.float32,copy=False)
pre_det=pre_det[order,:]
keep=self.nms(pre_det)
det=pre_det[keep,:]
ifself.use_kps:
kpss=kpss[order,:,:]
kpss=kpss[keep,:,:]
else:
kpss=None
ifmax_num>0anddet.shape[0]>max_num:
area=(det[:,2]-det[:,0])*(det[:,3]-det[:,1])
img_center=img.shape[0]//2,img.shape[1]//2
offsets=np.vstack(
[
(det[:,0]+det[:,2])/2-img_center[1],
(det[:,1]+det[:,3])/2-img_center[0],
]
)
offset_dist_squared=np.sum(np.power(offsets,2.0),0)
ifmetric=="max":
values=area
else:
values=area-offset_dist_squared*2.0#someextraweightonthecentering
bindex=np.argsort(values)[::-1]#someextraweightonthecentering
bindex=bindex[0:max_num]
det=det[bindex,:]
ifkpssisnotNone:
kpss=kpss[bindex,:]
returndet,kpss

defnms(self,dets):
thresh=self.nms_thresh
x1=dets[:,0]
y1=dets[:,1]
x2=dets[:,2]
y2=dets[:,3]
scores=dets[:,4]

areas=(x2-x1+1)*(y2-y1+1)
order=scores.argsort()[::-1]

keep=[]
whileorder.size>0:
i=order[0]
keep.append(i)
xx1=np.maximum(x1[i],x1[order[1:]])
yy1=np.maximum(y1[i],y1[order[1:]])
xx2=np.minimum(x2[i],x2[order[1:]])
yy2=np.minimum(y2[i],y2[order[1:]])

w=np.maximum(0.0,xx2-xx1+1)
h=np.maximum(0.0,yy2-yy1+1)
inter=w*h
ovr=inter/(areas[i]+areas[order[1:]]-inter)

inds=np.where(ovr<=thresh)[0]
order=order[inds+1]

returnkeep


arcface_dst=np.array(
[
[38.2946,51.6963],
[73.5318,51.5014],
[56.0252,71.7366],
[41.5493,92.3655],
[70.7299,92.2041],
],
dtype=np.float32,
)


defestimate_norm(lmk,image_size=112,mode="arcface"):
assertlmk.shape==(5,2)
assertimage_size%112==0orimage_size%128==0
ifimage_size%112==0:
ratio=float(image_size)/112.0
diff_x=0
else:
ratio=float(image_size)/128.0
diff_x=8.0*ratio
dst=arcface_dst*ratio
dst[:,0]+=diff_x
tform=trans.SimilarityTransform()
tform.estimate(lmk,dst)
M=tform.params[0:2,:]
returnM


defnorm_crop(img,landmark,image_size=112,mode="arcface"):
M=estimate_norm(landmark,image_size,mode)
warped=cv2.warpAffine(img,M,(image_size,image_size),borderValue=0.0)
returnwarped


classFaceEmbeddings:
def__init__(self,ov_model):
self.ov_model=ov_model
self.taskname="recognition"
input_mean=127.5
input_std=127.5
self.input_mean=input_mean
self.input_std=input_std
input_shape=self.ov_model.inputs[0].partial_shape
self.input_size=(input_shape[3].get_length(),input_shape[2].get_length())
self.input_shape=input_shape

defget(self,img,kps):
aimg=norm_crop(img,landmark=kps,image_size=self.input_size[0])
embedding=self.get_feat(aimg).flatten()
returnembedding

defget_feat(self,imgs):
ifnotisinstance(imgs,list):
imgs=[imgs]
input_size=self.input_size
blob=np.concatenate([prepare_input(cv2.resize(img,input_size),self.input_mean,self.input_std,True)forimginimgs])

net_out=self.ov_model(blob)[0]
returnnet_out

defforward(self,batch_data):
blob=(batch_data-self.input_mean)/self.input_std
net_out=self.ov_model(blob)[0]
returnnet_out


classOVFaceAnalysis:
def__init__(self,detect_model,embedding_model):
self.det_model=RetinaFace(detect_model)
self.embed_model=FaceEmbeddings(embedding_model)

defget(self,img,max_num=0):
bboxes,kpss=self.det_model.detect(img,max_num=max_num,metric="default")
ifbboxes.shape[0]==0:
return[]
ret=[]
foriinrange(bboxes.shape[0]):
bbox=bboxes[i,0:4]
det_score=bboxes[i,4]
kps=None
ifkpssisnotNone:
kps=kpss[i]
embedding=self.embed_model.get(img,kps)
ret.append({"bbox":bbox,"score":det_score,"kps":kps,"embedding":embedding})
returnret

Now,let’sseemodelsinferenceresult

SelectInferenceDeviceforFaceRecognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov
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

core=ov.Core()
face_detector=core.compile_model(face_detector_path,device.value)
face_embedding=core.compile_model(face_embeddings_path,device.value)

..code::ipython3

app=OVFaceAnalysis(face_detector,face_embedding)

PerformFaceIdentityextraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,wecanapplyour``OVFaceAnalysis``pipelineonanimagefor
collectionfaceembeddingsandkeypointsforreflectiononthe
generatedimage

..code::ipython3

importPIL.Image
frompipeline_stable_diffusion_xl_instantidimportdraw_kps


defget_face_info(face_image:PIL.Image.Image):
r"""
Retrievefaceinformationfromtheinputfaceimage.

Args:
face_image(PIL.Image.Image):
Animagecontainingaface.

Returns:
face_emb(numpy.ndarray):
Facialembeddingextractedfromthefaceimage.
face_kps(PIL.Image.Image):
Facialkeypointsdrawnonthefaceimage.
"""
face_image=face_image.resize((832,800))
#preparefaceemb
face_info=app.get(cv2.cvtColor(np.array(face_image),cv2.COLOR_RGB2BGR))
iflen(face_info)==0:
raiseRuntimeError("Couldn'tfindthefaceontheimage")
face_info=sorted(
face_info,
key=lambdax:(x["bbox"][2]-x["bbox"][0])*x["bbox"][3]-x["bbox"][1],
)[
-1
]#onlyusethemaximumface
face_emb=face_info["embedding"]
face_kps=draw_kps(face_image,face_info["kps"])
returnface_emb,face_kps

..code::ipython3

fromdiffusers.utilsimportload_image

face_image=load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg")

face_emb,face_kps=get_face_info(face_image)

..code::ipython3

face_image




..image::instant-id-with-output_files/instant-id-with-output_15_0.png



..code::ipython3

face_kps




..image::instant-id-with-output_files/instant-id-with-output_16_0.png



PrepareInstantIDpipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

ThecodebelowdownloadsInstantIDpipelineparts-ControlNetforface
poseandIP-Adapterforaddingfaceembeddingstoprompt

..code::ipython3

fromhuggingface_hubimporthf_hub_download

hf_hub_download(
repo_id="InstantX/InstantID",
filename="ControlNetModel/config.json",
local_dir="./checkpoints",
)
hf_hub_download(
repo_id="InstantX/InstantID",
filename="ControlNetModel/diffusion_pytorch_model.safetensors",
local_dir="./checkpoints",
)
hf_hub_download(repo_id="InstantX/InstantID",filename="ip-adapter.bin",local_dir="./checkpoints");

Asitwasdiscussedinmodeldescription,InstantIDdoesnotrequired
diffusionmodelfine-tuningandcanbeappliedonexistingStable
Diffusionpipeline.Wewilluse
`stable-diffusion-xl-bas-1-0<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__
asbasictext-to-imagediffusionpipeline.Wealsoapply`LCM
LoRA<https://huggingface.co/latent-consistency/lcm-lora-sdxl>`__to
speedupthegenerationprocess.Previously,wealreadyconsideredhowto
convertandrunSDXLmodelforText-to-ImageandImage-to-Image
generationusingOptimum-Intellibrary(pleasecheckoutthisnotebook
for`details<stable-diffusion-xl-with-output.html>`__),now
wewilluseitincombinationwithControlNetandconvertitusing
OpenVINOModelConversionAPI.

..code::ipython3

fromdiffusers.modelsimportControlNetModel
fromdiffusersimportLCMScheduler
frompipeline_stable_diffusion_xl_instantidimportStableDiffusionXLInstantIDPipeline

importtorch
fromPILimportImage
importgc


ov_controlnet_path=MODELS_DIR/"controlnet.xml"
ov_unet_path=MODELS_DIR/"unet.xml"
ov_vae_decoder_path=MODELS_DIR/"vae_decoder.xml"
ov_text_encoder_path=MODELS_DIR/"text_encoder.xml"
ov_text_encoder_2_path=MODELS_DIR/"text_encoder_2.xml"
ov_image_proj_encoder_path=MODELS_DIR/"image_proj_model.xml"

required_pipeline_parts=[
ov_controlnet_path,
ov_unet_path,
ov_vae_decoder_path,
ov_text_encoder_path,
ov_text_encoder_2_path,
ov_image_proj_encoder_path,
]


defload_pytorch_pipeline(sdxl_id="stabilityai/stable-diffusion-xl-base-1.0"):
#preparemodelsunder./checkpoints
face_adapter=Path("checkpoints/ip-adapter.bin")
controlnet_path=Path("checkpoints/ControlNetModel")

#loadIdentityNet
controlnet=ControlNetModel.from_pretrained(controlnet_path)

pipe=StableDiffusionXLInstantIDPipeline.from_pretrained(sdxl_id,controlnet=controlnet)

#loadadapter
pipe.load_ip_adapter_instantid(face_adapter)
#loadlcmlora
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.fuse_lora()
scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe.set_ip_adapter_scale(0.8)

controlnet,unet,vae=pipe.controlnet,pipe.unet,pipe.vae
text_encoder,text_encoder_2,tokenizer,tokenizer_2=(
pipe.text_encoder,
pipe.text_encoder_2,
pipe.tokenizer,
pipe.tokenizer_2,
)
image_proj_model=pipe.image_proj_model
return(
controlnet,
unet,
vae,
text_encoder,
text_encoder_2,
tokenizer,
tokenizer_2,
image_proj_model,
scheduler,
)


load_torch_models=any([notpath.exists()forpathinrequired_pipeline_parts])

ifload_torch_models:
(
controlnet,
unet,
vae,
text_encoder,
text_encoder_2,
tokenizer,
tokenizer_2,
image_proj_model,
scheduler,
)=load_pytorch_pipeline()
tokenizer.save_pretrained(MODELS_DIR/"tokenizer")
tokenizer_2.save_pretrained(MODELS_DIR/"tokenizer_2")
scheduler.save_pretrained(MODELS_DIR/"scheduler")
else:
(
controlnet,
unet,
vae,
text_encoder,
text_encoder_2,
tokenizer,
tokenizer_2,
image_proj_model,
scheduler,
)=(None,None,None,None,None,None,None,None,None)

gc.collect();

ConvertInstantIDpipelinecomponentstoOpenVINOIntermediateRepresentationformat
------------------------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodels
conversiondirectly.Weneedtoprovideamodelobject,inputdatafor
modeltracingto``ov.convert_model``functiontoobtainOpenVINO
``ov.Model``objectinstance.Modelcanbesavedondiskfornext
deploymentusing``ov.save_model``function.

Thepipelineconsistsofthefollowinglistofimportantparts:

-ImageProjectionmodelforgettingimagepromptembeddings.Itis
similarwithIP-Adapterapproachdescribedin`this
tutorial<stable-diffusion-ip-adapter-with-output.html>`__,
butinsteadofimage,itusesfaceembeddingsasinputforimage
promptencoding.
-TextEncodersforcreatingtextembeddingstogenerateanimagefrom
atextprompt.
-ControlNetforconditioningbyfacekeypointsimagefortranslation
faceposeongeneratedimage.
-Unetforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

ControlNet
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ControlNetwasintroducedin`AddingConditionalControlto
Text-to-ImageDiffusionModels<https://arxiv.org/abs/2302.05543>`__
paper.Itprovidesaframeworkthatenablessupportforvariousspatial
contextssuchasadepthmap,asegmentationmap,ascribble,andkey
pointsthatcanserveasadditionalconditioningstoDiffusionmodels
suchasStableDiffusion.Inthis
`tutorial<controlnet-stable-diffusion-with-output.html>`__
wealreadyconsideredhowtoconvertanduseControlNetwithStable
Diffusionpipeline.TheprocessofusageControlNetforStableDiffusion
XLremainswithoutchanges.

..code::ipython3

importopenvinoasov
fromfunctoolsimportpartial


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


controlnet_example_input={
"sample":torch.ones((2,4,100,100)),
"timestep":torch.tensor(1,dtype=torch.float32),
"encoder_hidden_states":torch.randn((2,77,2048)),
"controlnet_cond":torch.randn((2,3,800,800)),
"conditioning_scale":torch.tensor(0.8,dtype=torch.float32),
"added_cond_kwargs":{
"text_embeds":torch.zeros((2,1280)),
"time_ids":torch.ones((2,6),dtype=torch.int32),
},
}


ifnotov_controlnet_path.exists():
controlnet.forward=partial(controlnet.forward,return_dict=False)
withtorch.no_grad():
ov_controlnet=ov.convert_model(controlnet,example_input=controlnet_example_input)
ov_controlnet.inputs[-1].get_node().set_element_type(ov.Type.f32)
ov_controlnet.inputs[-1].get_node().set_partial_shape(ov.PartialShape([-1,6]))
ov_controlnet.validate_nodes_and_infer_types()
ov.save_model(ov_controlnet,ov_controlnet_path)
cleanup_torchscript_cache()
delov_controlnet
gc.collect()

ifnotov_unet_path.exists():
down_block_res_samples,mid_block_res_sample=controlnet(**controlnet_example_input)
else:
down_block_res_samples,mid_block_res_sample=None,None

delcontrolnet
gc.collect();

Unet
~~~~

`backtotop⬆️<#table-of-contents>`__

ComparedwithStableDiffusion,StableDiffusionXLUnethasan
additionalinputforthe``time_ids``condition.AsweuseControlNet
andImageProjectionModel,thesemodels’outputsalsocontributeto
preparingmodelinputforUnet.

..code::ipython3

fromtypingimportTuple


classUnetWrapper(torch.nn.Module):
def__init__(
self,
unet,
sample_dtype=torch.float32,
timestep_dtype=torch.int64,
encoder_hidden_states_dtype=torch.float32,
down_block_additional_residuals_dtype=torch.float32,
mid_block_additional_residual_dtype=torch.float32,
text_embeds_dtype=torch.float32,
time_ids_dtype=torch.int32,
):
super().__init__()
self.unet=unet
self.sample_dtype=sample_dtype
self.timestep_dtype=timestep_dtype
self.encoder_hidden_states_dtype=encoder_hidden_states_dtype
self.down_block_additional_residuals_dtype=down_block_additional_residuals_dtype
self.mid_block_additional_residual_dtype=mid_block_additional_residual_dtype
self.text_embeds_dtype=text_embeds_dtype
self.time_ids_dtype=time_ids_dtype

defforward(
self,
sample:torch.Tensor,
timestep:torch.Tensor,
encoder_hidden_states:torch.Tensor,
down_block_additional_residuals:Tuple[torch.Tensor],
mid_block_additional_residual:torch.Tensor,
text_embeds:torch.Tensor,
time_ids:torch.Tensor,
):
sample.to(self.sample_dtype)
timestep.to(self.timestep_dtype)
encoder_hidden_states.to(self.encoder_hidden_states_dtype)
down_block_additional_residuals=[res.to(self.down_block_additional_residuals_dtype)forresindown_block_additional_residuals]
mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
added_cond_kwargs={
"text_embeds":text_embeds.to(self.text_embeds_dtype),
"time_ids":time_ids.to(self.time_ids_dtype),
}

returnself.unet(
sample,
timestep,
encoder_hidden_states,
down_block_additional_residuals=down_block_additional_residuals,
mid_block_additional_residual=mid_block_additional_residual,
added_cond_kwargs=added_cond_kwargs,
)


ifnotov_unet_path.exists():
unet_example_input={
"sample":torch.ones((2,4,100,100)),
"timestep":torch.tensor(1,dtype=torch.float32),
"encoder_hidden_states":torch.randn((2,77,2048)),
"down_block_additional_residuals":down_block_res_samples,
"mid_block_additional_residual":mid_block_res_sample,
"text_embeds":torch.zeros((2,1280)),
"time_ids":torch.ones((2,6),dtype=torch.int32),
}
unet=UnetWrapper(unet)
withtorch.no_grad():
ov_unet=ov.convert_model(unet,example_input=unet_example_input)
foriinrange(3,len(ov_unet.inputs)-2):
ov_unet.inputs[i].get_node().set_element_type(ov.Type.f32)

ov_unet.validate_nodes_and_infer_types()
ov.save_model(ov_unet,ov_unet_path)
delov_unet
cleanup_torchscript_cache()
gc.collect()

delunet
gc.collect();

VAEDecoder
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheVAEmodelhastwoparts,anencoderandadecoder.Theencoderis
usedtoconverttheimageintoalowdimensionallatentrepresentation,
whichwillserveastheinputtotheU-Netmodel.Thedecoder,
conversely,transformsthelatentrepresentationbackintoanimage.For
InstantIDpipelinewewilluseVAEonlyfordecodingUnetgenerated
image,itmeansthatwecanskipVAEencoderpartconversion.

..code::ipython3

classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae_decoder):
super().__init__()
self.vae=vae_decoder

defforward(self,latents):
returnself.vae.decode(latents)


ifnotov_vae_decoder_path.exists():
vae_decoder=VAEDecoderWrapper(vae)

withtorch.no_grad():
ov_vae_decoder=ov.convert_model(vae_decoder,example_input=torch.zeros((1,4,64,64)))
ov.save_model(ov_vae_decoder,ov_vae_decoder_path)
delov_vae_decoder
cleanup_torchscript_cache()
delvae_decoder
gc.collect()

delvae
gc.collect();

TextEncoders
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetext-encoderisresponsiblefortransformingtheinputprompt,for
example,“aphotoofanastronautridingahorse”intoanembedding
spacethatcanbeunderstoodbytheU-Net.Itisusuallyasimple
transformer-basedencoderthatmapsasequenceofinputtokenstoa
sequenceoflatenttextembeddings.

..code::ipython3

inputs={"input_ids":torch.ones((1,77),dtype=torch.long)}

ifnotov_text_encoder_path.exists():
text_encoder.eval()
text_encoder.config.output_hidden_states=True
text_encoder.config.return_dict=False
withtorch.no_grad():
ov_text_encoder=ov.convert_model(text_encoder,example_input=inputs)
ov.save_model(ov_text_encoder,ov_text_encoder_path)
delov_text_encoder
cleanup_torchscript_cache()
gc.collect()

deltext_encoder
gc.collect()

ifnotov_text_encoder_2_path.exists():
text_encoder_2.eval()
text_encoder_2.config.output_hidden_states=True
text_encoder_2.config.return_dict=False
withtorch.no_grad():
ov_text_encoder=ov.convert_model(text_encoder_2,example_input=inputs)
ov.save_model(ov_text_encoder,ov_text_encoder_2_path)
delov_text_encoder
cleanup_torchscript_cache()
deltext_encoder_2
gc.collect();

ImageProjectionModel
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Imageprojectionmodelisresponsibletotransformingfaceembeddingsto
imagepromptembeddings

..code::ipython3

ifnotov_image_proj_encoder_path.exists():
withtorch.no_grad():
ov_image_encoder=ov.convert_model(image_proj_model,example_input=torch.zeros((2,1,512)))
ov.save_model(ov_image_encoder,ov_image_proj_encoder_path)
delov_image_encoder
cleanup_torchscript_cache()
delimage_proj_model
gc.collect();

PrepareOpenVINOInstantIDPipeline
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importnumpyasnp
fromdiffusersimportStableDiffusionXLControlNetPipeline
fromdiffusers.pipelines.stable_diffusion_xlimportStableDiffusionXLPipelineOutput
fromtypingimportAny,Callable,Dict,List,Optional,Tuple,Union

importnumpyasnp
importtorch

fromdiffusers.image_processorimportPipelineImageInput,VaeImageProcessor


classOVStableDiffusionXLInstantIDPipeline(StableDiffusionXLControlNetPipeline):
def__init__(
self,
text_encoder,
text_encoder_2,
image_proj_model,
controlnet,
unet,
vae_decoder,
tokenizer,
tokenizer_2,
scheduler,
):
self.text_encoder=text_encoder
self.text_encoder_2=text_encoder_2
self.tokenizer=tokenizer
self.tokenizer_2=tokenizer_2
self.image_proj_model=image_proj_model
self.controlnet=controlnet
self.unet=unet
self.vae_decoder=vae_decoder
self.scheduler=scheduler
self.image_proj_model_in_features=512
self.vae_scale_factor=8
self.vae_scaling_factor=0.13025
self.image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,do_convert_rgb=True)
self.control_image_processor=VaeImageProcessor(
vae_scale_factor=self.vae_scale_factor,
do_convert_rgb=True,
do_normalize=False,
)
self._internal_dict={}
self._progress_bar_config={}

def_encode_prompt_image_emb(self,prompt_image_emb,num_images_per_prompt,do_classifier_free_guidance):
ifisinstance(prompt_image_emb,torch.Tensor):
prompt_image_emb=prompt_image_emb.clone().detach()
else:
prompt_image_emb=torch.tensor(prompt_image_emb)

prompt_image_emb=prompt_image_emb.reshape([1,-1,self.image_proj_model_in_features])

ifdo_classifier_free_guidance:
prompt_image_emb=torch.cat([torch.zeros_like(prompt_image_emb),prompt_image_emb],dim=0)
else:
prompt_image_emb=torch.cat([prompt_image_emb],dim=0)
prompt_image_emb=self.image_proj_model(prompt_image_emb)[0]

bs_embed,seq_len,_=prompt_image_emb.shape
prompt_image_emb=np.tile(prompt_image_emb,(1,num_images_per_prompt,1))
prompt_image_emb=prompt_image_emb.reshape(bs_embed*num_images_per_prompt,seq_len,-1)

returnprompt_image_emb

def__call__(
self,
prompt:Union[str,List[str]]=None,
prompt_2:Optional[Union[str,List[str]]]=None,
image:PipelineImageInput=None,
height:Optional[int]=None,
width:Optional[int]=None,
num_inference_steps:int=50,
guidance_scale:float=5.0,
negative_prompt:Optional[Union[str,List[str]]]=None,
negative_prompt_2:Optional[Union[str,List[str]]]=None,
num_images_per_prompt:Optional[int]=1,
eta:float=0.0,
generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
latents:Optional[torch.FloatTensor]=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
negative_pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
image_embeds:Optional[torch.FloatTensor]=None,
output_type:Optional[str]="pil",
return_dict:bool=True,
cross_attention_kwargs:Optional[Dict[str,Any]]=None,
controlnet_conditioning_scale:Union[float,List[float]]=1.0,
guess_mode:bool=False,
control_guidance_start:Union[float,List[float]]=0.0,
control_guidance_end:Union[float,List[float]]=1.0,
original_size:Tuple[int,int]=None,
crops_coords_top_left:Tuple[int,int]=(0,0),
target_size:Tuple[int,int]=None,
negative_original_size:Optional[Tuple[int,int]]=None,
negative_crops_coords_top_left:Tuple[int,int]=(0,0),
negative_target_size:Optional[Tuple[int,int]]=None,
clip_skip:Optional[int]=None,
callback_on_step_end:Optional[Callable[[int,int,Dict],None]]=None,
callback_on_step_end_tensor_inputs:List[str]=["latents"],
#IPadapter
ip_adapter_scale=None,
**kwargs,
):
r"""
Thecallfunctiontothepipelineforgeneration.

Args:
prompt(`str`or`List[str]`,*optional*):
Thepromptorpromptstoguideimagegeneration.Ifnotdefined,youneedtopass`prompt_embeds`.
prompt_2(`str`or`List[str]`,*optional*):
Thepromptorpromptstobesentto`tokenizer_2`and`text_encoder_2`.Ifnotdefined,`prompt`is
usedinbothtext-encoders.
image(`torch.FloatTensor`,`PIL.Image.Image`,`np.ndarray`,`List[torch.FloatTensor]`,`List[PIL.Image.Image]`,`List[np.ndarray]`,:
`List[List[torch.FloatTensor]]`,`List[List[np.ndarray]]`or`List[List[PIL.Image.Image]]`):
TheControlNetinputconditiontoprovideguidancetothe`unet`forgeneration.Ifthetypeis
specifiedas`torch.FloatTensor`,itispassedtoControlNetasis.`PIL.Image.Image`canalsobe
acceptedasanimage.Thedimensionsoftheoutputimagedefaultsto`image`'sdimensions.Ifheight__module.unet.up_blocks.0.upsamplers.0.conv.base_layer/aten::_convolu
and/orwidtharepassed,`image`isresizedaccordingly.IfmultipleControlNetsarespecifiedin
`init`,imagesmustbepassedasalistsuchthateachelementofthelistcanbecorrectlybatchedfor
inputtoasingleControlNet.
height(`int`,*optional*,defaultsto`self.unet.config.sample_size*self.vae_scale_factor`):
Theheightinpixelsofthegeneratedimage.Anythingbelow512pixelswon'tworkwellfor
[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
andcheckpointsthatarenotspecificallyfine-tunedonlowresolutions.
width(`int`,*optional*,defaultsto`self.unet.config.sample_size*self.vae_scale_factor`):
Thewidthinpixelsofthegeneratedimage.Anythingbelow512pixelswon'tworkwellfor
[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
andcheckpointsthatarenotspecificallyfine-tunedonlowresolutions.
num_inference_steps(`int`,*optional*,defaultsto50):
Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.
guidance_scale(`float`,*optional*,defaultsto5.0):
Ahigherguidancescalevalueencouragesthemodeltogenerateimagescloselylinkedtothetext
`prompt`attheexpenseoflowerimagequality.Guidancescaleisenabledwhen`guidance_scale>1`.
negative_prompt(`str`or`List[str]`,*optional*):
Thepromptorpromptstoguidewhattonotincludeinimagegeneration.Ifnotdefined,youneedto
pass`negative_prompt_embeds`instead.Ignoredwhennotusingguidance(`guidance_scale<1`).
negative_prompt_2(`str`or`List[str]`,*optional*):
Thepromptorpromptstoguidewhattonotincludeinimagegeneration.Thisissentto`tokenizer_2`
and`text_encoder_2`.Ifnotdefined,`negative_prompt`isusedinbothtext-encoders.
num_images_per_prompt(`int`,*optional*,defaultsto1):
Thenumberofimagestogenerateperprompt.
eta(`float`,*optional*,defaultsto0.0):
Correspondstoparametereta(η)fromthe[DDIM](https://arxiv.org/abs/2010.02502)paper.Onlyapplies
tothe[`~schedulers.DDIMScheduler`],andisignoredinotherschedulers.
generator(`torch.Generator`or`List[torch.Generator]`,*optional*):
A[`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)tomake
generationdeterministic.
latents(`torch.FloatTensor`,*optional*):
Pre-generatednoisylatentssampledfromaGaussiandistribution,tobeusedasinputsforimage
generation.Canbeusedtotweakthesamegenerationwithdifferentprompts.Ifnotprovided,alatents
tensorisgeneratedbysamplingusingthesuppliedrandom`generator`.
prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedtextembeddings.Canbeusedtoeasilytweaktextinputs(promptweighting).Ifnot
provided,textembeddingsaregeneratedfromthe`prompt`inputargument.
negative_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatednegativetextembeddings.Canbeusedtoeasilytweaktextinputs(promptweighting).If
notprovided,`negative_prompt_embeds`aregeneratedfromthe`negative_prompt`inputargument.
pooled_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedpooledtextembeddings.Canbeusedtoeasilytweaktextinputs(promptweighting).If
notprovided,pooledtextembeddingsaregeneratedfrom`prompt`inputargument.
negative_pooled_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatednegativepooledtextembeddings.Canbeusedtoeasilytweaktextinputs(prompt
weighting).Ifnotprovided,pooled`negative_prompt_embeds`aregeneratedfrom`negative_prompt`input
argument.
image_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedimageembeddings.
output_type(`str`,*optional*,defaultsto`"pil"`):
Theoutputformatofthegeneratedimage.Choosebetween`PIL.Image`or`np.array`.
return_dict(`bool`,*optional*,defaultsto`True`):
Whetherornottoreturna[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]insteadofa
plaintuple.
controlnet_conditioning_scale(`float`or`List[float]`,*optional*,defaultsto1.0):
TheoutputsoftheControlNetaremultipliedby`controlnet_conditioning_scale`beforetheyareadded
totheresidualintheoriginal`unet`.IfmultipleControlNetsarespecifiedin`init`,youcanset
thecorrespondingscaleasalist.
control_guidance_start(`float`or`List[float]`,*optional*,defaultsto0.0):
ThepercentageoftotalstepsatwhichtheControlNetstartsapplying.
control_guidance_end(`float`or`List[float]`,*optional*,defaultsto1.0):
ThepercentageoftotalstepsatwhichtheControlNetstopsapplying.
original_size(`Tuple[int]`,*optional*,defaultsto(1024,1024)):
If`original_size`isnotthesameas`target_size`theimagewillappeartobedown-orupsampled.
`original_size`defaultsto`(height,width)`ifnotspecified.PartofSDXL'smicro-conditioningas
explainedinsection2.2of
[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
crops_coords_top_left(`Tuple[int]`,*optional*,defaultsto(0,0)):
`crops_coords_top_left`canbeusedtogenerateanimagethatappearstobe"cropped"fromtheposition
`crops_coords_top_left`downwards.Favorable,well-centeredimagesareusuallyachievedbysetting
`crops_coords_top_left`to(0,0).PartofSDXL'smicro-conditioningasexplainedinsection2.2of
[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
target_size(`Tuple[int]`,*optional*,defaultsto(1024,1024)):
Formostcases,`target_size`shouldbesettothedesiredheightandwidthofthegeneratedimage.If
notspecifieditwilldefaultto`(height,width)`.PartofSDXL'smicro-conditioningasexplainedin
section2.2of[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
negative_original_size(`Tuple[int]`,*optional*,defaultsto(1024,1024)):
Tonegativelyconditionthegenerationprocessbasedonaspecificimageresolution.PartofSDXL's
micro-conditioningasexplainedinsection2.2of
[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).Formore
information,refertoencode_prothisissuethread:https://github.com/huggingface/diffusers/issues/4208.
negative_crops_coords_top_left(`Tuple[int]`,*optional*,defaultsto(0,0)):
Tonegativelyconditionthegenerationprocessbasedonaspecificcropcoordinates.PartofSDXL's
micro-conditioningasexplainedinsection2.2of
[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).Formore
information,refertothisissuethread:https://github.com/huggingface/diffusers/issues/4208.
negative_target_size(`Tuple[int]`,*optional*,defaultsto(1024,1024)):
Tonegativelyconditionthegenerationprocessbasedonatargetimageresolution.Itshouldbeassame
asthe`target_size`formostcases.PartofSDXL'smicro-conditioningasexplainedinsection2.2of
[https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).Formore
information,refertothisissuethread:https://github.com/huggingface/diffusers/issues/4208.
clip_skip(`int`,*optional*):
NumberoflayerstobeskippedfromCLIPwhilecomputingthepromptembeddings.Avalueof1meansthat
theoutputofthepre-finallayerwillbeusedforcomputingthepromptembeddings.

Examples:

Returns:
[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]or`tuple`:
If`return_dict`is`True`,[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]isreturned,
otherwisea`tuple`isreturnedcontainingtheoutputimages.
"""

do_classifier_free_guidance=guidance_scale>=1.0
#alignformatforcontrolguidance
ifnotisinstance(control_guidance_start,list)andisinstance(control_guidance_end,list):
control_guidance_start=len(control_guidance_end)*[control_guidance_start]
elifnotisinstance(control_guidance_end,list)andisinstance(control_guidance_start,list):
control_guidance_end=len(control_guidance_start)*[control_guidance_end]
elifnotisinstance(control_guidance_start,list)andnotisinstance(control_guidance_end,list):
control_guidance_start,control_guidance_end=(
[control_guidance_start],
[control_guidance_end],
)

#2.Definecallparameters
ifpromptisnotNoneandisinstance(prompt,str):
batch_size=1
elifpromptisnotNoneandisinstance(prompt,list):
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

(
prompt_embeds,
negative_prompt_embeds,
pooled_prompt_embeds,
negative_pooled_prompt_embeds,
)=self.encode_prompt(
prompt,
prompt_2,
num_images_per_prompt,
do_classifier_free_guidance,
negative_prompt,
negative_prompt_2,
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
pooled_prompt_embeds=pooled_prompt_embeds,
negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
lora_scale=None,
clip_skip=clip_skip,
)

#3.2Encodeimageprompt
prompt_image_emb=self._encode_prompt_image_emb(image_embeds,num_images_per_prompt,do_classifier_free_guidance)

#4.Prepareimage
image=self.prepare_image(
image=image,
width=width,
height=height,
batch_size=batch_size*num_images_per_prompt,
num_images_per_prompt=num_images_per_prompt,
do_classifier_free_guidance=do_classifier_free_guidance,
guess_mode=guess_mode,
)
height,width=image.shape[-2:]

#5.Preparetimesteps
self.scheduler.set_timesteps(num_inference_steps)
timesteps=self.scheduler.timesteps

#6.Preparelatentvariables
num_channels_latents=4
latents=self.prepare_latents(
int(batch_size)*int(num_images_per_prompt),
int(num_channels_latents),
int(height),
int(width),
dtype=torch.float32,
device=torch.device("cpu"),
generator=generator,
latents=latents,
)

#7.Prepareextrastepkwargs.
extra_step_kwargs=self.prepare_extra_step_kwargs(generator,eta)
#7.1Createtensorstatingwhichcontrolnetstokeep
controlnet_keep=[]
foriinrange(len(timesteps)):
keeps=[1.0-float(i/len(timesteps)<sor(i+1)/len(timesteps)>e)fors,einzip(control_guidance_start,control_guidance_end)]
controlnet_keep.append(keeps)

#7.2Prepareaddedtimeids&embeddings
ifisinstance(image,list):
original_size=original_sizeorimage[0].shape[-2:]
else:
original_size=original_sizeorimage.shape[-2:]
target_size=target_sizeor(height,width)

add_text_embeds=pooled_prompt_embeds
ifself.text_encoder_2isNone:
text_encoder_projection_dim=pooled_prompt_embeds.shape[-1]
else:
text_encoder_projection_dim=1280

add_time_ids=self._get_add_time_ids(
original_size,
crops_coords_top_left,
target_size,
text_encoder_projection_dim=text_encoder_projection_dim,
)

ifnegative_original_sizeisnotNoneandnegative_target_sizeisnotNone:
negative_add_time_ids=self._get_add_time_ids(
negative_original_size,
negative_crops_coords_top_left,
negative_target_size,
text_encoder_projection_dim=text_encoder_projection_dim,
)
else:
negative_add_time_ids=add_time_ids

ifdo_classifier_free_guidance:
prompt_embeds=np.concatenate([negative_prompt_embeds,prompt_embeds],axis=0)
add_text_embeds=np.concatenate([negative_pooled_prompt_embeds,add_text_embeds],axis=0)
add_time_ids=np.concatenate([negative_add_time_ids,add_time_ids],axis=0)

add_time_ids=np.tile(add_time_ids,(batch_size*num_images_per_prompt,1))
encoder_hidden_states=np.concatenate([prompt_embeds,prompt_image_emb],axis=1)

#8.Denoisingloop
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=torch.cat([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

#controlnet(s)inference
control_model_input=latent_model_input

cond_scale=controlnet_conditioning_scale

controlnet_outputs=self.controlnet(
[
control_model_input,
t,
prompt_image_emb,
image,
cond_scale,
add_text_embeds,
add_time_ids,
]
)

controlnet_additional_blocks=list(controlnet_outputs.values())

#predictthenoiseresidual
noise_pred=self.unet(
[
latent_model_input,
t,
encoder_hidden_states,
*controlnet_additional_blocks,
add_text_embeds,
add_time_ids,
]
)[0]

#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred[0],noise_pred[1]
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(
torch.from_numpy(noise_pred),
t,
latents,
**extra_step_kwargs,
return_dict=False,
)[0]
progress_bar.update()

ifnotoutput_type=="latent":
image=self.vae_decoder(latents/self.vae_scaling_factor)[0]
else:
image=latents

ifnotoutput_type=="latent":
image=self.image_processor.postprocess(torch.from_numpy(image),output_type=output_type)

ifnotreturn_dict:
return(image,)

returnStableDiffusionXLPipelineOutput(images=image)

defencode_prompt(
self,
prompt:str,
prompt_2:Optional[str]=None,
num_images_per_prompt:int=1,
do_classifier_free_guidance:bool=True,
negative_prompt:Optional[str]=None,
negative_prompt_2:Optional[str]=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
negative_pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
lora_scale:Optional[float]=None,
clip_skip:Optional[int]=None,
):
r"""
Encodesthepromptintotextencoderhiddenstates.

Args:
prompt(`str`or`List[str]`,*optional*):
prompttobeencoded
prompt_2(`str`or`List[str]`,*optional*):
Thepromptorpromptstobesenttothe`tokenizer_2`and`text_encoder_2`.Ifnotdefined,`prompt`is
usedinbothtext-encoders
num_images_per_prompt(`int`):
numberofimagesthatshouldbegeneratedperprompt
do_classifier_free_guidance(`bool`):
whethertouseclassifierfreeguidanceornot
negative_prompt(`str`or`List[str]`,*optional*):
Thepromptorpromptsnottoguidetheimagegeneration.Ifnotdefined,onehastopass
`negative_prompt_embeds`instead.Ignoredwhennotusingguidance(i.e.,ignoredif`guidance_scale`is
lessthan`1`).
negative_prompt_2(`str`or`List[str]`,*optional*):
Thepromptorpromptsnottoguidetheimagegenerationtobesentto`tokenizer_2`and
`text_encoder_2`.Ifnotdefined,`negative_prompt`isusedinbothtext-encoders
prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedtextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*promptweighting.Ifnot
provided,textembeddingswillbegeneratedfrom`prompt`inputargument.
negative_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatednegativetextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*prompt
weighting.Ifnotprovided,negative_prompt_embedswillbegeneratedfrom`negative_prompt`input
argument.
pooled_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedpooledtextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*promptweighting.
Ifnotprovided,pooledtextembeddingswillbegeneratedfrom`prompt`inputargument.
negative_pooled_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatednegativepooledtextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*prompt
weighting.Ifnotprovided,poolednegative_prompt_embedswillbegeneratedfrom`negative_prompt`
inputargument.
lora_scale(`float`,*optional*):
AlorascalethatwillbeappliedtoallLoRAlayersofthetextencoderifLoRAlayersareloaded.
clip_skip(`int`,*optional*):
NumberoflayerstobeskippedfromCLIPwhilecomputingthepromptembeddings.Avalueof1meansthat
theoutputofthepre-finallayerwillbeusedforcomputingthepromptembeddings.
"""
prompt=[prompt]ifisinstance(prompt,str)elseprompt

ifpromptisnotNone:
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

#Definetokenizersandtextencoders
tokenizers=[self.tokenizer,self.tokenizer_2]ifself.tokenizerisnotNoneelse[self.tokenizer_2]
text_encoders=[self.text_encoder,self.text_encoder_2]ifself.text_encoderisnotNoneelse[self.text_encoder_2]

ifprompt_embedsisNone:
prompt_2=prompt_2orprompt
prompt_2=[prompt_2]ifisinstance(prompt_2,str)elseprompt_2

#textualinversion:procecssmulti-vectortokensifnecessary
prompt_embeds_list=[]
prompts=[prompt,prompt_2]
forprompt,tokenizer,text_encoderinzip(prompts,tokenizers,text_encoders):
text_inputs=tokenizer(
prompt,
padding="max_length",
max_length=tokenizer.model_max_length,
truncation=True,
return_tensors="pt",
)

text_input_ids=text_inputs.input_ids

prompt_embeds=text_encoder(text_input_ids)

#WeareonlyALWAYSinterestedinthepooledoutputofthefinaltextencoder
pooled_prompt_embeds=prompt_embeds[0]
hidden_states=list(prompt_embeds.values())[1:]
ifclip_skipisNone:
prompt_embeds=hidden_states[-2]
else:
#"2"becauseSDXLalwaysindexesfromthepenultimatelayer.
prompt_embeds=hidden_states[-(clip_skip+2)]

prompt_embeds_list.append(prompt_embeds)

prompt_embeds=np.concatenate(prompt_embeds_list,axis=-1)

#getunconditionalembeddingsforclassifierfreeguidance
zero_out_negative_prompt=negative_promptisNone
ifdo_classifier_free_guidanceandnegative_prompt_embedsisNoneandzero_out_negative_prompt:
negative_prompt_embeds=np.zeros_like(prompt_embeds)
negative_pooled_prompt_embeds=np.zeros_like(pooled_prompt_embeds)
elifdo_classifier_free_guidanceandnegative_prompt_embedsisNone:
negative_prompt=negative_promptor""
negative_prompt_2=negative_prompt_2ornegative_prompt

#normalizestrtolist
negative_prompt=batch_size*[negative_prompt]ifisinstance(negative_prompt,str)elsenegative_prompt
negative_prompt_2=batch_size*[negative_prompt_2]ifisinstance(negative_prompt_2,str)elsenegative_prompt_2

uncond_tokens:List[str]
ifpromptisnotNoneandtype(prompt)isnottype(negative_prompt):
raiseTypeError(f"`negative_prompt`shouldbethesametypeto`prompt`,butgot{type(negative_prompt)}!="f"{type(prompt)}.")
elifbatch_size!=len(negative_prompt):
raiseValueError(
f"`negative_prompt`:{negative_prompt}hasbatchsize{len(negative_prompt)},but`prompt`:"
f"{prompt}hasbatchsize{batch_size}.Pleasemakesurethatpassed`negative_prompt`matches"
"thebatchsizeof`prompt`."
)
else:
uncond_tokens=[negative_prompt,negative_prompt_2]

negative_prompt_embeds_list=[]
fornegative_prompt,tokenizer,text_encoderinzip(uncond_tokens,tokenizers,text_encoders):
max_length=prompt_embeds.shape[1]
uncond_input=tokenizer(
negative_prompt,
padding="max_length",
max_length=max_length,
truncation=True,
return_tensors="pt",
)

negative_prompt_embeds=text_encoder(uncond_input.input_ids)
#WeareonlyALWAYSinterestedinthepooledoutputofthefinaltextencoder
negative_pooled_prompt_embeds=negative_prompt_embeds[0]
hidden_states=list(negative_prompt_embeds.values())[1:]
negative_prompt_embeds=hidden_states[-2]

negative_prompt_embeds_list.append(negative_prompt_embeds)

negative_prompt_embeds=np.concatenate(negative_prompt_embeds_list,axis=-1)

bs_embed,seq_len,_=prompt_embeds.shape
#duplicatetextembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
prompt_embeds=np.tile(prompt_embeds,(1,num_images_per_prompt,1))
prompt_embeds=prompt_embeds.reshape(bs_embed*num_images_per_prompt,seq_len,-1)

ifdo_classifier_free_guidance:
#duplicateunconditionalembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
seq_len=negative_prompt_embeds.shape[1]
negative_prompt_embeds=np.tile(negative_prompt_embeds,(1,num_images_per_prompt,1))
negative_prompt_embeds=negative_prompt_embeds.reshape(batch_size*num_images_per_prompt,seq_len,-1)

pooled_prompt_embeds=np.tile(pooled_prompt_embeds,(1,num_images_per_prompt)).reshape(bs_embed*num_images_per_prompt,-1)
ifdo_classifier_free_guidance:
negative_pooled_prompt_embeds=np.tile(negative_pooled_prompt_embeds,(1,num_images_per_prompt)).reshape(bs_embed*num_images_per_prompt,-1)

return(
prompt_embeds,
negative_prompt_embeds,
pooled_prompt_embeds,
negative_pooled_prompt_embeds,
)

defprepare_image(
self,
image,
width,
height,
batch_size,
num_images_per_prompt,
do_classifier_free_guidance=False,
guess_mode=False,
):
image=self.control_image_processor.preprocess(image,height=height,width=width).to(dtype=torch.float32)
image_batch_size=image.shape[0]

ifimage_batch_size==1:
repeat_by=batch_size
else:
#imagebatchsizeisthesameaspromptbatchsize
repeat_by=num_images_per_prompt

image=image.repeat_interleave(repeat_by,dim=0)

ifdo_classifier_free_guidanceandnotguess_mode:
image=torch.cat([image]*2)

returnimage

def_get_add_time_ids(
self,
original_size,
crops_coords_top_left,
target_size,
text_encoder_projection_dim,
):
add_time_ids=list(original_size+crops_coords_top_left+target_size)
add_time_ids=torch.tensor([add_time_ids])
returnadd_time_ids

RunOpenVINOpipelineinference
-------------------------------

`backtotop⬆️<#table-of-contents>`__

SelectinferencedeviceforInstantID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

text_encoder=core.compile_model(ov_text_encoder_path,device.value)
text_encoder_2=core.compile_model(ov_text_encoder_2_path,device.value)
vae_decoder=core.compile_model(ov_vae_decoder_path,device.value)
unet=core.compile_model(ov_unet_path,device.value)
controlnet=core.compile_model(ov_controlnet_path,device.value)
image_proj_model=core.compile_model(ov_image_proj_encoder_path,device.value)

..code::ipython3

fromtransformersimportAutoTokenizer

tokenizer=AutoTokenizer.from_pretrained(MODELS_DIR/"tokenizer")
tokenizer_2=AutoTokenizer.from_pretrained(MODELS_DIR/"tokenizer_2")
scheduler=LCMScheduler.from_pretrained(MODELS_DIR/"scheduler")


..parsed-literal::

Theconfigattributes{'interpolation_type':'linear','skip_prk_steps':True,'use_karras_sigmas':False}werepassedtoLCMScheduler,butarenotexpectedandwillbeignored.Pleaseverifyyourscheduler_config.jsonconfigurationfile.


Createpipeline
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_pipe=OVStableDiffusionXLInstantIDPipeline(
text_encoder,
text_encoder_2,
image_proj_model,
controlnet,
unet,
vae_decoder,
tokenizer,
tokenizer_2,
scheduler,
)

Runinference
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

prompt="Animegirl"
negative_prompt=""

image=ov_pipe(
prompt,
image_embeds=face_emb,
image=face_kps,
num_inference_steps=4,
negative_prompt=negative_prompt,
guidance_scale=0.5,
generator=torch.Generator(device="cpu").manual_seed(1749781188),
).images[0]



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

image




..image::instant-id-with-output_files/instant-id-with-output_41_0.png



Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``OVStableDiffusionXLInstantIDPipeline``structure,
ControlNetandUNetmodelsareusedinthecyclerepeatinginferenceon
eachdiffusionstep,whileotherpartsofpipelinetakepartonlyonce.
Nowwewillshowyouhowtooptimizepipelineusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__toreducememoryand
computationcost.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

skip_for_device="GPU"indevice.value
to_quantize=widgets.Checkbox(value=notskip_for_device,description="Quantization",disabled=skip_for_device)
to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

int8_pipe=None

%load_extskip_kernel_extension

Preparecalibrationdatasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`wider_face<https://huggingface.co/datasets/wider_face>`__dataset
fromHuggingFaceascalibrationdata.Weusepromptsbelowtoguide
imagegenerationandtodeterminewhatnottoincludeintheresulting
image.

..code::ipython3

%%skipnot$to_quantize.value

negative_prompts=[
"blurryunrealoccluded",
"lowcontrastdisfigureduncenteredmangled",
"amateuroutofframelowqualitynsfw",
"uglyunderexposedjpegartifacts",
"lowsaturationdisturbingcontent",
"overexposedseveredistortion",
"amateurNSFW",
"uglymutilatedoutofframedisfigured",
]
prompts=[
"aNaruto-styleimageofayoungboy,incorporatingdynamicactionlines,intenseenergyeffects,andasenseofmovementandpower",
"ananime-stylegirl,withvibrant,otherworldlycolors,fantasticalelements,andasenseofawe",
"analogfilmphotoofaman.fadedfilm,desaturated,35mmphoto,grainy,vignette,vintage,Kodachrome,Lomography,stained,highlydetailed,foundfootage,masterpiece,bestquality",
"Applyastainingfiltertogivetheimpressionofaged,worn-outfilmwhilemaintainingsharpdetailonaportraitofawoman",
"amodernpictureofaboyanantiquefeelthroughselectivedesaturation,grainaddition,andawarmtone,mimickingthestyleofoldphotographs",
"adreamy,etherealportraitofayounggirl,featuringsoft,pastelcolors,ablurredbackground,andatouchofbokeh",
"adynamic,action-packedimageofaboyinmotion,usingmotionblur,panning,andothertechniquestoconveyasenseofspeedandenergy",
"adramatic,cinematicimageofaboy,usingcolorgrading,contrastadjustments,andawidescreenaspectratio,tocreateasenseofepicscaleandgrandeur",
"aportraitofawomaninthestyleofPicasso'scubism,featuringfragmentedshapes,boldlines,andavibrantcolorpalette",
"anartworkinthestyleofPicasso'sBluePeriod,featuringasomber,melancholicportraitofaperson,withmutedcolors,elongatedforms,andasenseofintrospectionandcontemplation",
]

..code::ipython3

%%skipnot$to_quantize.value

importdatasets

num_inference_steps=4
subset_size=200

ov_int8_unet_path=MODELS_DIR/'unet_optimized.xml'
ov_int8_controlnet_path=MODELS_DIR/'controlnet_optimized.xml'

num_samples=int(np.ceil(subset_size/num_inference_steps))
dataset=datasets.load_dataset("wider_face",split="train",streaming=True,trust_remote_code=True).shuffle(seed=42)
face_info=[]
forbatchindataset:
try:
face_info.append(get_face_info(batch["image"]))
exceptRuntimeError:
continue
iflen(face_info)>num_samples:
break

Tocollectintermediatemodelinputsforcalibrationweshouldcustomize
``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

fromtqdm.notebookimporttqdm
fromtransformersimportset_seed

set_seed(42)

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,keep_prob:float=1.0):
super().__init__(compiled_model)
self.data_cache=[]
self.keep_prob=np.clip(keep_prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()<=self.keep_prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)


defcollect_calibration_data(pipeline,face_info,subset_size):
original_unet=pipeline.unet
pipeline.unet=CompiledModelDecorator(original_unet)
pipeline.set_progress_bar_config(disable=True)

pbar=tqdm(total=subset_size)
forface_emb,face_kpsinface_info:
negative_prompt=np.random.choice(negative_prompts)
prompt=np.random.choice(prompts)
_=pipeline(
prompt,
image_embeds=face_emb,
image=face_kps,
num_inference_steps=num_inference_steps,
negative_prompt=negative_prompt,
guidance_scale=0.5,
generator=torch.Generator(device="cpu").manual_seed(1749781188)
)
collected_subset_size=len(pipeline.unet.data_cache)
pbar.update(collected_subset_size-pbar.n)

calibration_dataset=pipeline.unet.data_cache[:subset_size]
pipeline.set_progress_bar_config(disable=False)
pipeline.unet=original_unet
returncalibration_dataset


..code::ipython3

%%skipnot$to_quantize.value

ifnot(ov_int8_unet_path.exists()andov_int8_controlnet_path.exists()):
unet_calibration_data=collect_calibration_data(ov_pipe,face_info,subset_size=subset_size)

..code::ipython3

%%skipnot$to_quantize.value

defprepare_controlnet_dataset(pipeline,face_info,unet_calibration_data):
controlnet_calibration_data=[]
i=0
forface_emb,face_kpsinface_info:
prompt_image_emb=pipeline._encode_prompt_image_emb(
face_emb,num_images_per_prompt=1,do_classifier_free_guidance=False
)
image=pipeline.prepare_image(
image=face_kps,
width=None,
height=None,
batch_size=1,
num_images_per_prompt=1,
do_classifier_free_guidance=False,
guess_mode=False,
)
fordatainunet_calibration_data[i:i+num_inference_steps]:
controlnet_inputs=[data[0],data[1],prompt_image_emb,image,1.0,data[-2],data[-1]]
controlnet_calibration_data.append(controlnet_inputs)
i+=num_inference_steps
returncontrolnet_calibration_data


..code::ipython3

%%skipnot$to_quantize.value

ifnotov_int8_controlnet_path.exists():
controlnet_calibration_data=prepare_controlnet_dataset(ov_pipe,face_info,unet_calibration_data)

RunQuantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

RunControlNetQuantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Quantizationofthefirst``Convolution``layerimpactsthegeneration
results.Werecommendusing``IgnoredScope``tokeepaccuracysensitive
layersinFP16precision.

..code::ipython3

%%skipnot$to_quantize.value

importnncf

ifnotov_int8_controlnet_path.exists():
controlnet=core.read_model(ov_controlnet_path)
quantized_controlnet=nncf.quantize(
model=controlnet,
calibration_dataset=nncf.Dataset(controlnet_calibration_data),
subset_size=subset_size,
ignored_scope=nncf.IgnoredScope(names=["__module.model.conv_in/aten::_convolution/Convolution"]),
model_type=nncf.ModelType.TRANSFORMER,
)
ov.save_model(quantized_controlnet,ov_int8_controlnet_path)

RunUNetHybridQuantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Ontheonehand,post-trainingquantizationoftheUNetmodelrequires
morethan~100Gbandleadstoaccuracydrop.Ontheotherhand,the
weightcompressiondoesn’timproveperformancewhenapplyingtoStable
Diffusionmodels,becausethesizeofactivationsiscomparableto
weights.Thatiswhytheproposalistoapplyquantizationinhybrid
modewhichmeansthatwequantize:(1)weightsofMatMulandEmbedding
layersand(2)activationsofotherlayers.Thestepsarethefollowing:

1.Createacalibrationdatasetforquantization.
2.Collectoperationswithweights.
3.Run``nncf.compress_model()``tocompressonlythemodelweights.
4.Run``nncf.quantize()``onthecompressedmodelwithweighted
operationsignoredbyproviding``ignored_scope``parameter.
5.Savethe``INT8``modelusing``openvino.save_model()``function.

..code::ipython3

%%skipnot$to_quantize.value

fromcollectionsimportdeque

defget_operation_const_op(operation,const_port_id:int):
node=operation.input_value(const_port_id).get_node()
queue=deque([node])
constant_node=None
allowed_propagation_types_list=["Convert","FakeQuantize","Reshape"]

whilelen(queue)!=0:
curr_node=queue.popleft()
ifcurr_node.get_type_name()=="Constant":
constant_node=curr_node
break
iflen(curr_node.inputs())==0:
break
ifcurr_node.get_type_name()inallowed_propagation_types_list:
queue.append(curr_node.input_value(0).get_node())

returnconstant_node


defis_embedding(node)->bool:
allowed_types_list=["f16","f32","f64"]
const_port_id=0
input_tensor=node.input_value(const_port_id)
ifinput_tensor.get_element_type().get_type_name()inallowed_types_list:
const_node=get_operation_const_op(node,const_port_id)
ifconst_nodeisnotNone:
returnTrue

returnFalse


defcollect_ops_with_weights(model):
ops_with_weights=[]
foropinmodel.get_ops():
ifop.get_type_name()=="MatMul":
constant_node_0=get_operation_const_op(op,const_port_id=0)
constant_node_1=get_operation_const_op(op,const_port_id=1)
ifconstant_node_0orconstant_node_1:
ops_with_weights.append(op.get_friendly_name())
ifop.get_type_name()=="Gather"andis_embedding(op):
ops_with_weights.append(op.get_friendly_name())

returnops_with_weights

..code::ipython3

%%skipnot$to_quantize.value

ifnotov_int8_unet_path.exists():
unet=core.read_model(ov_unet_path)
unet_ignored_scope=collect_ops_with_weights(unet)
compressed_unet=nncf.compress_weights(unet,ignored_scope=nncf.IgnoredScope(types=['Convolution']))
quantized_unet=nncf.quantize(
model=compressed_unet,
calibration_dataset=nncf.Dataset(unet_calibration_data),
subset_size=subset_size,
model_type=nncf.ModelType.TRANSFORMER,
ignored_scope=nncf.IgnoredScope(names=unet_ignored_scope),
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=-1)
)
ov.save_model(quantized_unet,ov_int8_unet_path)

RunWeightsCompression
^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Quantizingofthe``TextEncoders``and``VAEDecoder``doesnot
significantlyimproveinferenceperformancebutcanleadtoa
substantialdegradationofaccuracy.Theweightcompressionwillbe
appliedtofootprintreduction.

..code::ipython3

%%skipnot$to_quantize.value

ov_int8_text_encoder_path=MODELS_DIR/'text_encoder_optimized.xml'
ov_int8_text_encoder_2_path=MODELS_DIR/'text_encoder_2_optimized.xml'
ov_int8_vae_decoder_path=MODELS_DIR/'vae_decoder_optimized.xml'

ifnotov_int8_text_encoder_path.exists():
text_encoder=core.read_model(ov_text_encoder_path)
compressed_text_encoder=nncf.compress_weights(text_encoder)
ov.save_model(compressed_text_encoder,ov_int8_text_encoder_path)

ifnotov_int8_text_encoder_2_path.exists():
text_encoder_2=core.read_model(ov_text_encoder_2_path)
compressed_text_encoder_2=nncf.compress_weights(text_encoder_2)
ov.save_model(compressed_text_encoder_2,ov_int8_text_encoder_2_path)

ifnotov_int8_vae_decoder_path.exists():
vae_decoder=core.read_model(ov_vae_decoder_path)
compressed_vae_decoder=nncf.compress_weights(vae_decoder)
ov.save_model(compressed_vae_decoder,ov_int8_vae_decoder_path)

Let’scomparetheimagesgeneratedbytheoriginalandoptimized
pipelines.

..code::ipython3

%%skipnot$to_quantize.value

optimized_controlnet=core.compile_model(ov_int8_controlnet_path,device.value)
optimized_unet=core.compile_model(ov_int8_unet_path,device.value)
optimized_text_encoder=core.compile_model(ov_int8_text_encoder_path,device.value)
optimized_text_encoder_2=core.compile_model(ov_int8_text_encoder_2_path,device.value)
optimized_vae_decoder=core.compile_model(ov_int8_vae_decoder_path,device.value)

int8_pipe=OVStableDiffusionXLInstantIDPipeline(
optimized_text_encoder,
optimized_text_encoder_2,
image_proj_model,
optimized_controlnet,
optimized_unet,
optimized_vae_decoder,
tokenizer,
tokenizer_2,
scheduler,
)

..code::ipython3

%%skipnot$to_quantize.value

int8_image=int8_pipe(
prompt,
image_embeds=face_emb,
image=face_kps,
num_inference_steps=4,
negative_prompt=negative_prompt,
guidance_scale=0.5,
generator=torch.Generator(device="cpu").manual_seed(1749781188)
).images[0]



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

#%%skipnot$to_quantize.value

importmatplotlib.pyplotasplt


defvisualize_results(orig_img:Image,optimized_img:Image):
"""
Helperfunctionforresultsvisualization

Parameters:
orig_img(Image.Image):generatedimageusingFP16models
optimized_img(Image.Image):generatedimageusingquantizedmodels
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
orig_title="FP16pipeline"
control_title="INT8pipeline"
figsize=(20,20)
fig,axs=plt.subplots(1,2,figsize=figsize,sharex="all",sharey="all")
list_axes=list(axs.flat)
forainlist_axes:
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
list_axes[0].imshow(np.array(orig_img))
list_axes[1].imshow(np.array(optimized_img))
list_axes[0].set_title(orig_title,fontsize=15)
list_axes[1].set_title(control_title,fontsize=15)

fig.subplots_adjust(wspace=0.01,hspace=0.01)
fig.tight_layout()
returnfig

..code::ipython3

%%skipnot$to_quantize.value

visualize_results(image,int8_image)



..image::instant-id-with-output_files/instant-id-with-output_66_0.png


Comparemodelfilesizes
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_model_paths=[ov_unet_path,ov_controlnet_path,ov_text_encoder_path,ov_text_encoder_2_path,ov_vae_decoder_path]
int8_model_paths=[ov_int8_unet_path,ov_int8_controlnet_path,ov_int8_text_encoder_path,ov_int8_text_encoder_2_path,ov_int8_vae_decoder_path]

forfp16_path,int8_pathinzip(fp16_model_paths,int8_model_paths):
fp16_ir_model_size=fp16_path.with_suffix(".bin").stat().st_size
int8_model_size=int8_path.with_suffix(".bin").stat().st_size
print(f"{fp16_path.stem}compressionrate:{fp16_ir_model_size/int8_model_size:.3f}")


..parsed-literal::

unetcompressionrate:1.996
controlnetcompressionrate:1.995
text_encodercompressionrate:1.992
text_encoder_2compressionrate:1.995
vae_decodercompressionrate:1.997


CompareinferencetimeoftheFP16andINT8pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
pipelines,weusemeaninferencetimeon5samples.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

%%skipnot$to_quantize.value

importtime

defcalculate_inference_time(pipeline,face_info):
inference_time=[]
pipeline.set_progress_bar_config(disable=True)
foriinrange(5):
face_emb,face_kps=face_info[i]
prompt=np.random.choice(prompts)
negative_prompt=np.random.choice(negative_prompts)
start=time.perf_counter()
_=pipeline(
prompt,
image_embeds=face_emb,
image=face_kps,
num_inference_steps=4,
negative_prompt=negative_prompt,
guidance_scale=0.5,
generator=torch.Generator(device="cpu").manual_seed(1749781188)
)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
pipeline.set_progress_bar_config(disable=False)
returnnp.mean(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_latency=calculate_inference_time(ov_pipe,face_info)
print(f"FP16pipeline:{fp_latency:.3f}seconds")
int8_latency=calculate_inference_time(int8_pipe,face_info)
print(f"INT8pipeline:{int8_latency:.3f}seconds")
print(f"Performancespeed-up:{fp_latency/int8_latency:.3f}")


..parsed-literal::

FP16pipeline:17.595seconds
INT8pipeline:15.258seconds
Performancespeed-up:1.153


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodels
tolaunchtheinteractivedemo.

..code::ipython3

quantized_models_present=int8_pipeisnotNone

use_quantized_models=widgets.Checkbox(
value=quantized_models_present,
description="Usequantizedmodels",
disabled=notquantized_models_present,
)

use_quantized_models

..code::ipython3

importgradioasgr
fromtypingimportTuple
importrandom
importPIL
importsys

sys.path.append("./InstantID/gradio_demo")

fromstyle_templateimportstyles

#globalvariable
MAX_SEED=np.iinfo(np.int32).max
STYLE_NAMES=list(styles.keys())
DEFAULT_STYLE_NAME="Watercolor"


example_image_urls=[
"https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00002-3104853212.png",
"https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/images%207/00171-2728008415.png",
"https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00003-3962843561.png",
"https://huggingface.co/datasets/EnD-Diffusers/AI_Faces/resolve/main/00005-3104853215.png",
"https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png",
]

examples_dir=Path("examples")

examples=[
[examples_dir/"face_0.png","Awomaninreddress","FilmNoir",""],
[examples_dir/"face_1.png","photoofabusinesslady","VibrantColor",""],
[examples_dir/"face_2.png","famousrockstarposter","(Nostyle)",""],
[examples_dir/"face_3.png","aperson","Neon",""],
[examples_dir/"face_4.png","agirl","Snow",""],
]

pipeline=int8_pipeifuse_quantized_models.valueelseov_pipe


ifnotexamples_dir.exists():
examples_dir.mkdir()
forimg_id,img_urlinenumerate(example_image_urls):
load_image(img_url).save(examples_dir/f"face_{img_id}.png")


defrandomize_seed_fn(seed:int,randomize_seed:bool)->int:
ifrandomize_seed:
seed=random.randint(0,MAX_SEED)
returnseed


defconvert_from_cv2_to_image(img:np.ndarray)->PIL.Image:
returnImage.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


defconvert_from_image_to_cv2(img:PIL.Image)->np.ndarray:
returncv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)


defresize_img(
input_image,
max_side=1024,
min_side=800,
size=None,
pad_to_max_side=False,
mode=PIL.Image.BILINEAR,
base_pixel_number=64,
):
w,h=input_image.size
ifsizeisnotNone:
w_resize_new,h_resize_new=size
else:
ratio=min_side/min(h,w)
w,h=round(ratio*w),round(ratio*h)
ratio=max_side/max(h,w)
input_image=input_image.resize([round(ratio*w),round(ratio*h)],mode)
w_resize_new=(round(ratio*w)//base_pixel_number)*base_pixel_number
h_resize_new=(round(ratio*h)//base_pixel_number)*base_pixel_number
input_image=input_image.resize([w_resize_new,h_resize_new],mode)

ifpad_to_max_side:
res=np.ones([max_side,max_side,3],dtype=np.uint8)*255
offset_x=(max_side-w_resize_new)//2
offset_y=(max_side-h_resize_new)//2
res[offset_y:offset_y+h_resize_new,offset_x:offset_x+w_resize_new]=np.array(input_image)
input_image=Image.fromarray(res)
returninput_image


defapply_style(style_name:str,positive:str,negative:str="")->Tuple[str,str]:
p,n=styles.get(style_name,styles[DEFAULT_STYLE_NAME])
returnp.replace("{prompt}",positive),n+""+negative


defgenerate_image(
face_image,
pose_image,
prompt,
negative_prompt,
style_name,
num_steps,
identitynet_strength_ratio,
guidance_scale,
seed,
progress=gr.Progress(track_tqdm=True),
):
ifpromptisNone:
prompt="aperson"

#applythestyletemplate
prompt,negative_prompt=apply_style(style_name,prompt,negative_prompt)

#face_image=load_image(face_image_path)
face_image=resize_img(face_image)
face_image_cv2=convert_from_image_to_cv2(face_image)
height,width,_=face_image_cv2.shape

#Extractfacefeatures
face_info=app.get(face_image_cv2)

iflen(face_info)==0:
raisegr.Error("Cannotfindanyfaceintheimage!Pleaseuploadanotherpersonimage")

face_info=sorted(
face_info,
key=lambdax:(x["bbox"][2]-x["bbox"][0])*x["bbox"][3]-x["bbox"][1],
)[
-1
]#onlyusethemaximumface
face_emb=face_info["embedding"]
face_kps=draw_kps(convert_from_cv2_to_image(face_image_cv2),face_info["kps"])

ifpose_imageisnotNone:
#pose_image=load_image(pose_image_path)
pose_image=resize_img(pose_image)
pose_image_cv2=convert_from_image_to_cv2(pose_image)

face_info=app.get(pose_image_cv2)

iflen(face_info)==0:
raisegr.Error("Cannotfindanyfaceinthereferenceimage!Pleaseuploadanotherpersonimage")

face_info=face_info[-1]
face_kps=draw_kps(pose_image,face_info["kps"])

width,height=face_kps.size

generator=torch.Generator(device="cpu").manual_seed(seed)

print("Startinference...")
print(f"[Debug]Prompt:{prompt},\n[Debug]NegPrompt:{negative_prompt}")
images=pipeline(
prompt=prompt,
negative_prompt=negative_prompt,
image_embeds=face_emb,
image=face_kps,
controlnet_conditioning_scale=float(identitynet_strength_ratio),
num_inference_steps=num_steps,
guidance_scale=guidance_scale,
height=height,
width=width,
generator=generator,
).images

returnimages[0]


###Description
title=r"""
<h1align="center">InstantID:Zero-shotIdentity-PreservingGeneration</h1>
"""

description=r"""

Howtouse:<br>
1.Uploadanimagewithaface.Forimageswithmultiplefaces,wewillonlydetectthelargestface.Ensurethefaceisnottoosmallandisclearlyvisiblewithoutsignificantobstructionsorblurring.
2.(Optional)Youcanuploadanotherimageasareferenceforthefacepose.Ifyoudon't,wewillusethefirstdetectedfaceimagetoextractfaciallandmarks.Ifyouuseacroppedfaceatstep1,itisrecommendedtouploadittodefineanewfacepose.
3.Enteratextprompt,asdoneinnormaltext-to-imagemodels.
4.Clickthe<b>Submit</b>buttontobegincustomization.
5.Shareyourcustomizedphotowithyourfriendsandenjoy!😊
"""


css="""
.gradio-container{width:85%!important}
"""
withgr.Blocks(css=css)asdemo:
#description
gr.Markdown(title)
gr.Markdown(description)

withgr.Row():
withgr.Column():
#uploadfaceimage
face_file=gr.Image(label="Uploadaphotoofyourface",type="pil")

#optional:uploadareferenceposeimage
pose_file=gr.Image(label="Uploadareferenceposeimage(optional)",type="pil")

#prompt
prompt=gr.Textbox(
label="Prompt",
info="Givesimplepromptisenoughtoachievegoodfacefidelity",
placeholder="Aphotoofaperson",
value="",
)

submit=gr.Button("Submit",variant="primary")
style=gr.Dropdown(label="Styletemplate",choices=STYLE_NAMES,value=DEFAULT_STYLE_NAME)

#strength
identitynet_strength_ratio=gr.Slider(
label="IdentityNetstrength(forfidelity)",
minimum=0,
maximum=1.5,
step=0.05,
value=0.80,
)

withgr.Accordion(open=False,label="AdvancedOptions"):
negative_prompt=gr.Textbox(
label="NegativePrompt",
placeholder="lowquality",
value="(lowres,lowquality,worstquality:1.2),(text:1.2),watermark,(frame:1.2),deformed,ugly,deformedeyes,blur,outoffocus,blurry,deformedcat,deformed,photo,anthropomorphiccat,monochrome,petcollar,gun,weapon,blue,3d,drones,drone,buildingsinbackground,green",
)
num_steps=gr.Slider(
label="Numberofsamplesteps",
minimum=1,
maximum=10,
step=1,
value=4,
)
guidance_scale=gr.Slider(label="Guidancescale",minimum=0.1,maximum=10.0,step=0.1,value=0)
seed=gr.Slider(
label="Seed",
minimum=0,
maximum=MAX_SEED,
step=1,
value=42,
)
randomize_seed=gr.Checkbox(label="Randomizeseed",value=True)
gr.Examples(
examples=examples,
inputs=[face_file,prompt,style,negative_prompt],
)

withgr.Column():
gallery=gr.Image(label="GeneratedImage")

submit.click(
fn=randomize_seed_fn,
inputs=[seed,randomize_seed],
outputs=seed,
api_name=False,
).then(
fn=generate_image,
inputs=[
face_file,
pose_file,
prompt,
negative_prompt,
style,
num_steps,
identitynet_strength_ratio,
guidance_scale,
seed,
],
outputs=[gallery],
)

..code::ipython3

if__name__=="__main__":
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

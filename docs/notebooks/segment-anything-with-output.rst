ObjectmasksfrompromptswithSAMandOpenVINO
===============================================

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Background<#background>`__
-`Prerequisites<#prerequisites>`__
-`ConvertmodeltoOpenVINOIntermediate
Representation<#convert-model-to-openvino-intermediate-representation>`__

-`DownloadmodelcheckpointandcreatePyTorch
model<#download-model-checkpoint-and-create-pytorch-model>`__
-`ImageEncoder<#image-encoder>`__
-`Maskpredictor<#mask-predictor>`__

-`RunOpenVINOmodelininteractivesegmentation
mode<#run-openvino-model-in-interactive-segmentation-mode>`__

-`ExampleImage<#example-image>`__
-`Preprocessingandvisualization
utilities<#preprocessing-and-visualization-utilities>`__
-`Imageencoding<#image-encoding>`__
-`Examplepointinput<#example-point-input>`__
-`Examplewithmultiplepoints<#example-with-multiple-points>`__
-`Exampleboxandpointinputwithnegative
label<#example-box-and-point-input-with-negative-label>`__

-`Interactivesegmentation<#interactive-segmentation>`__
-`RunOpenVINOmodelinautomaticmaskgeneration
mode<#run-openvino-model-in-automatic-mask-generation-mode>`__
-`OptimizeencoderusingNNCFPost-trainingQuantization
API<#optimize-encoder-using-nncf-post-training-quantization-api>`__

-`Prepareacalibrationdataset<#prepare-a-calibration-dataset>`__
-`RunquantizationandserializeOpenVINOIR
model<#run-quantization-and-serialize-openvino-ir-model>`__
-`ValidateQuantizedModel
Inference<#validate-quantized-model-inference>`__
-`ComparePerformanceoftheOriginalandQuantized
Models<#compare-performance-of-the-original-and-quantized-models>`__

Segmentation-identifyingwhichimagepixelsbelongtoanobject-isa
coretaskincomputervisionandisusedinabroadarrayof
applications,fromanalyzingscientificimagerytoeditingphotos.But
creatinganaccuratesegmentationmodelforspecifictaskstypically
requireshighlyspecializedworkbytechnicalexpertswithaccesstoAI
traininginfrastructureandlargevolumesofcarefullyannotated
in-domaindata.Reducingtheneedfortask-specificmodelingexpertise,
trainingcompute,andcustomdataannotationforimagesegmentationis
themaingoalofthe`Segment
Anything<https://arxiv.org/abs/2304.02643>`__project.

The`SegmentAnythingModel
(SAM)<https://github.com/facebookresearch/segment-anything>`__predicts
objectmasksgivenpromptsthatindicatethedesiredobject.SAMhas
learnedageneralnotionofwhatobjectsare,anditcangeneratemasks
foranyobjectinanyimageoranyvideo,evenincludingobjectsand
imagetypesthatithadnotencounteredduringtraining.SAMisgeneral
enoughtocoverabroadsetofusecasesandcanbeusedoutofthebox
onnewimage“domains”(e.g. underwaterphotos,MRIorcellmicroscopy)
withoutrequiringadditionaltraining(acapabilityoftenreferredtoas
zero-shottransfer).Thisnotebookshowsanexampleofhowtoconvert
anduseSegmentAnythingModelinOpenVINOformat,allowingittorunon
avarietyofplatformsthatsupportanOpenVINO.

Background
----------

`backtotop⬆️<#table-of-contents>`__

Previously,tosolveanykindofsegmentationproblem,thereweretwo
classesofapproaches.Thefirst,interactivesegmentation,allowedfor
segmentinganyclassofobjectbutrequiredapersontoguidethemethod
byiterativerefiningamask.Thesecond,automaticsegmentation,
allowedforsegmentationofspecificobjectcategoriesdefinedaheadof
time(e.g.,catsorchairs)butrequiredsubstantialamountsofmanually
annotatedobjectstotrain(e.g.,thousandsoreventensofthousandsof
examplesofsegmentedcats),alongwiththecomputeresourcesand
technicalexpertisetotrainthesegmentationmodel.Neitherapproach
providedageneral,fullyautomaticapproachtosegmentation.

SegmentAnythingModelisageneralizationofthesetwoclassesof
approaches.Itisasinglemodelthatcaneasilyperformboth
interactivesegmentationandautomaticsegmentation.TheSegment
AnythingModel(SAM)produceshighqualityobjectmasksfrominput
promptssuchaspointsorboxes,anditcanbeusedtogeneratemasks
forallobjectsinanimage.Ithasbeentrainedona
`dataset<https://segment-anything.com/dataset/index.html>`__of11
millionimagesand1.1billionmasks,andhasstrongzero-shot
performanceonavarietyofsegmentationtasks.Themodelconsistsof3
parts:

-**ImageEncoder**-VisionTransformermodel(VIT)pretrainedusing
MaskedAutoEncodersapproach(MAE)forencodingimagetoembedding
space.Theimageencoderrunsonceperimageandcanbeappliedprior
topromptingthemodel.
-**PromptEncoder**-Encoderforsegmentationcondition.Asa
conditioncanbeused:

-points-setofpointsrelatedtoobjectwhichshouldbe
segmented.Promptencoderconvertspointstoembeddingusing
positionalencoding.
-boxes-boundingboxwhereobjectforsegmentationislocated.
Similartopoints,coordinatesofboundingboxencodedvia
positionalencoding.
-segmentationmask-providedbyusersegmentationmaskisembedded
usingconvolutionsandsummedelement-wisewiththeimage
embedding.
-text-encodedbyCLIPmodeltextrepresentation

-**MaskDecoder**-Themaskdecoderefficientlymapstheimage
embedding,promptembeddings,andanoutputtokentoamask.

ThediagrambelowdemonstratestheprocessofmaskgenerationusingSAM:
|model_diagram|

Themodelfirstconvertstheimageintoanimageembeddingthatallows
highqualitymaskstobeefficientlyproducedfromaprompt.Themodel
returnsmultiplemaskswhichfittotheprovidedpromptanditsscore.
Theprovidedmaskscanbeoverlappedareasasitshownondiagram,itis
usefulforcomplicatedcaseswhenpromptcanbeinterpretedindifferent
manner,e.g. segmentwholeobjectoronlyitsspecificpartorwhen
providedpointattheintersectionofmultipleobjects.Themodel’s
promptableinterfaceallowsittobeusedinflexiblewaysthatmakea
widerangeofsegmentationtaskspossiblesimplybyengineeringthe
rightpromptforthemodel(clicks,boxes,text,andsoon).

Moredetailsaboutapproachcanbefoundinthe
`paper<https://arxiv.org/abs/2304.02643>`__,original
`repo<https://github.com/facebookresearch/segment-anything>`__and
`MetaAIblog
post<https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/>`__

..|model_diagram|image::https://raw.githubusercontent.com/facebookresearch/segment-anything/main/assets/model_diagram.png

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"segment_anything""gradio>=4.13""openvino>=2023.1.0""nncf>=2.7.0""torch>=2.1""torchvision>=0.16"Pillowopencv-pythontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

ConvertmodeltoOpenVINOIntermediateRepresentation
-----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

DownloadmodelcheckpointandcreatePyTorchmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThereareseveralSegmentAnythingModel
`checkpoints<https://github.com/facebookresearch/segment-anything#model-checkpoints>`__
availablefordownloadingInthistutorialwewillusemodelbasedon
``vit_b``,butthedemonstratedapproachisverygeneralandapplicable
tootherSAMmodels.SetthemodelURL,pathforsavingcheckpointand
modeltypebelowtoaSAMmodelcheckpoint,thenloadthemodelusing
``sam_model_registry``.

..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

checkpoint="sam_vit_b_01ec64.pth"
model_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
model_type="vit_b"

download_file(model_url)


..parsed-literal::

'sam_vit_b_01ec64.pth'alreadyexists.




..parsed-literal::

PosixPath('/home/ea/work/openvino_notebooks/notebooks/segment-anything/sam_vit_b_01ec64.pth')



..code::ipython3

fromsegment_anythingimportsam_model_registry

sam=sam_model_registry[model_type](checkpoint=checkpoint)

Aswealreadydiscussed,ImageEncoderpartcanbeusedonceperimage,
thenchangingprompt,promptencoderandmaskdecodercanberun
multipletimestoretrievedifferentobjectsfromthesameimage.Taking
intoaccountthisfact,wesplitmodelon2independentparts:
image_encoderandmask_predictor(combinationofPromptEncoderandMask
Decoder).

ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ImageEncoderinputistensorwithshape``1x3x1024x1024``in``NCHW``
format,containsimageforsegmentation.ImageEncoderoutputisimage
embeddings,tensorwithshape``1x256x64x64``

..code::ipython3

importwarnings
frompathlibimportPath
importtorch
importopenvinoasov

core=ov.Core()

ov_encoder_path=Path("sam_image_encoder.xml")
ifnotov_encoder_path.exists():
withwarnings.catch_warnings():
warnings.filterwarnings("ignore",category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore",category=UserWarning)

ov_encoder_model=ov.convert_model(
sam.image_encoder,
example_input=torch.zeros(1,3,1024,1024),
input=([1,3,1024,1024],),
)
ov.save_model(ov_encoder_model,ov_encoder_path)
else:
ov_encoder_model=core.read_model(ov_encoder_path)

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=2,options=('CPU','GPU','AUTO'),value='AUTO')



..code::ipython3

ov_encoder=core.compile_model(ov_encoder_model,device.value)

Maskpredictor
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thisnotebookexpectsthemodelwasexportedwiththeparameter
``return_single_mask=True``.Itmeansthatmodelwillonlyreturnthe
bestmask,insteadofreturningmultiplemasks.Forhighresolution
imagesthiscanimproveruntimewhenupscalingmasksisexpensive.

Combinedpromptencoderandmaskdecodermodelhasfollowinglistof
inputs:

-``image_embeddings``:Theimageembeddingfrom``image_encoder``.Has
abatchindexoflength1.
-``point_coords``:Coordinatesofsparseinputprompts,corresponding
tobothpointinputsandboxinputs.Boxesareencodedusingtwo
points,oneforthetop-leftcornerandoneforthebottom-right
corner.*Coordinatesmustalreadybetransformedtolong-side1024.*
Hasabatchindexoflength1.
-``point_labels``:Labelsforthesparseinputprompts.0isa
negativeinputpoint,1isapositiveinputpoint,2isatop-left
boxcorner,3isabottom-rightboxcorner,and-1isapadding
point.\*Ifthereisnoboxinput,asinglepaddingpointwithlabel
-1andcoordinates(0.0,0.0)shouldbeconcatenated.

Modeloutputs:

-``masks``-predictedmasksresizedtooriginalimagesize,toobtain
abinarymask,shouldbecomparedwith``threshold``(usuallyequal
0.0).
-``iou_predictions``-intersectionoverunionpredictions
-``low_res_masks``-predictedmasksbeforepostprocessing,canbe
usedasmaskinputformodel.

..code::ipython3

fromtypingimportTuple


classSamExportableModel(torch.nn.Module):
def__init__(
self,
model,
return_single_mask:bool,
use_stability_score:bool=False,
return_extra_metrics:bool=False,
)->None:
super().__init__()
self.mask_decoder=model.mask_decoder
self.model=model
self.img_size=model.image_encoder.img_size
self.return_single_mask=return_single_mask
self.use_stability_score=use_stability_score
self.stability_score_offset=1.0
self.return_extra_metrics=return_extra_metrics

def_embed_points(self,point_coords:torch.Tensor,point_labels:torch.Tensor)->torch.Tensor:
point_coords=point_coords+0.5
point_coords=point_coords/self.img_size
point_embedding=self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
point_labels=point_labels.unsqueeze(-1).expand_as(point_embedding)

point_embedding=point_embedding*(point_labels!=-1).to(torch.float32)
point_embedding=point_embedding+self.model.prompt_encoder.not_a_point_embed.weight*(point_labels==-1).to(torch.float32)

foriinrange(self.model.prompt_encoder.num_point_embeddings):
point_embedding=point_embedding+self.model.prompt_encoder.point_embeddings[i].weight*(point_labels==i).to(torch.float32)

returnpoint_embedding

deft_embed_masks(self,input_mask:torch.Tensor)->torch.Tensor:
mask_embedding=self.model.prompt_encoder.mask_downscaling(input_mask)
returnmask_embedding

defmask_postprocessing(self,masks:torch.Tensor)->torch.Tensor:
masks=torch.nn.functional.interpolate(
masks,
size=(self.img_size,self.img_size),
mode="bilinear",
align_corners=False,
)
returnmasks

defselect_masks(self,masks:torch.Tensor,iou_preds:torch.Tensor,num_points:int)->Tuple[torch.Tensor,torch.Tensor]:
#Determineifweshouldreturnthemulticlickmaskornotfromthenumberofpoints.
#Thereweightingisusedtoavoidcontrolflow.
score_reweight=torch.tensor([[1000]+[0]*(self.model.mask_decoder.num_mask_tokens-1)]).to(iou_preds.device)
score=iou_preds+(num_points-2.5)*score_reweight
best_idx=torch.argmax(score,dim=1)
masks=masks[torch.arange(masks.shape[0]),best_idx,:,:].unsqueeze(1)
iou_preds=iou_preds[torch.arange(masks.shape[0]),best_idx].unsqueeze(1)

returnmasks,iou_preds

@torch.no_grad()
defforward(
self,
image_embeddings:torch.Tensor,
point_coords:torch.Tensor,
point_labels:torch.Tensor,
mask_input:torch.Tensor=None,
):
sparse_embedding=self._embed_points(point_coords,point_labels)
ifmask_inputisNone:
dense_embedding=self.model.prompt_encoder.no_mask_embed.weight.reshape(1,-1,1,1).expand(
point_coords.shape[0],-1,image_embeddings.shape[0],64
)
else:
dense_embedding=self._embed_masks(mask_input)

masks,scores=self.model.mask_decoder.predict_masks(
image_embeddings=image_embeddings,
image_pe=self.model.prompt_encoder.get_dense_pe(),
sparse_prompt_embeddings=sparse_embedding,
dense_prompt_embeddings=dense_embedding,
)

ifself.use_stability_score:
scores=calculate_stability_score(masks,self.model.mask_threshold,self.stability_score_offset)

ifself.return_single_mask:
masks,scores=self.select_masks(masks,scores,point_coords.shape[1])

upscaled_masks=self.mask_postprocessing(masks)

ifself.return_extra_metrics:
stability_scores=calculate_stability_score(upscaled_masks,self.model.mask_threshold,self.stability_score_offset)
areas=(upscaled_masks>self.model.mask_threshold).sum(-1).sum(-1)
returnupscaled_masks,scores,stability_scores,areas,masks

returnupscaled_masks,scores


ov_model_path=Path("sam_mask_predictor.xml")
ifnotov_model_path.exists():
exportable_model=SamExportableModel(sam,return_single_mask=True)
embed_dim=sam.prompt_encoder.embed_dim
embed_size=sam.prompt_encoder.image_embedding_size
dummy_inputs={
"image_embeddings":torch.randn(1,embed_dim,*embed_size,dtype=torch.float),
"point_coords":torch.randint(low=0,high=1024,size=(1,5,2),dtype=torch.float),
"point_labels":torch.randint(low=0,high=4,size=(1,5),dtype=torch.float),
}
withwarnings.catch_warnings():
warnings.filterwarnings("ignore",category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore",category=UserWarning)
ov_model=ov.convert_model(exportable_model,example_input=dummy_inputs)
ov.save_model(ov_model,ov_model_path)
else:
ov_model=core.read_model(ov_model_path)

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=2,options=('CPU','GPU','AUTO'),value='AUTO')



..code::ipython3

ov_predictor=core.compile_model(ov_model,device.value)

RunOpenVINOmodelininteractivesegmentationmode
---------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ExampleImage
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importnumpyasnp
importcv2
importmatplotlib.pyplotasplt

download_file("https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg")
image=cv2.imread("truck.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


..parsed-literal::

'truck.jpg'alreadyexists.


..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_21_0.png


Preprocessingandvisualizationutilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ToprepareinputforImageEncoderweshould:

1.ConvertBGRimagetoRGB
2.ResizeimagesavingaspectratiowherelongestsizeequaltoImage
Encoderinputsize-1024.
3.Normalizeimagesubtractmeanvalues(123.675,116.28,103.53)and
dividebystd(58.395,57.12,57.375)
4.TransposeHWCdatalayouttoCHWandaddbatchdimension.
5.Addzeropaddingtoinputtensorbyheightorwidth(dependson
aspectratio)accordingImageEncoderexpectedinputshape.

Thesestepsareapplicabletoallavailablemodels

..code::ipython3

fromcopyimportdeepcopy
fromtypingimportTuple
fromtorchvision.transforms.functionalimportresize,to_pil_image


classResizeLongestSide:
"""
Resizesimagestolongestside'target_length',aswellasprovides
methodsforresizingcoordinatesandboxes.Providesmethodsfor
transformingnumpyarrays.
"""

def__init__(self,target_length:int)->None:
self.target_length=target_length

defapply_image(self,image:np.ndarray)->np.ndarray:
"""
ExpectsanumpyarraywithshapeHxWxCinuint8format.
"""
target_size=self.get_preprocess_shape(image.shape[0],image.shape[1],self.target_length)
returnnp.array(resize(to_pil_image(image),target_size))

defapply_coords(self,coords:np.ndarray,original_size:Tuple[int,...])->np.ndarray:
"""
Expectsanumpyarrayoflength2inthefinaldimension.Requiresthe
originalimagesizein(H,W)format.
"""
old_h,old_w=original_size
new_h,new_w=self.get_preprocess_shape(original_size[0],original_size[1],self.target_length)
coords=deepcopy(coords).astype(float)
coords[...,0]=coords[...,0]*(new_w/old_w)
coords[...,1]=coords[...,1]*(new_h/old_h)
returncoords

defapply_boxes(self,boxes:np.ndarray,original_size:Tuple[int,...])->np.ndarray:
"""
ExpectsanumpyarrayshapeBx4.Requirestheoriginalimagesize
in(H,W)format.
"""
boxes=self.apply_coords(boxes.reshape(-1,2,2),original_size)
returnboxes.reshape(-1,4)

@staticmethod
defget_preprocess_shape(oldh:int,oldw:int,long_side_length:int)->Tuple[int,int]:
"""
Computetheoutputsizegiveninputsizeandtargetlongsidelength.
"""
scale=long_side_length*1.0/max(oldh,oldw)
newh,neww=oldh*scale,oldw*scale
neww=int(neww+0.5)
newh=int(newh+0.5)
return(newh,neww)


resizer=ResizeLongestSide(1024)


defpreprocess_image(image:np.ndarray):
resized_image=resizer.apply_image(image)
resized_image=(resized_image.astype(np.float32)-[123.675,116.28,103.53])/[
58.395,
57.12,
57.375,
]
resized_image=np.expand_dims(np.transpose(resized_image,(2,0,1)).astype(np.float32),0)

#Pad
h,w=resized_image.shape[-2:]
padh=1024-h
padw=1024-w
x=np.pad(resized_image,((0,0),(0,0),(0,padh),(0,padw)))
returnx


defpostprocess_masks(masks:np.ndarray,orig_size):
size_before_pad=resizer.get_preprocess_shape(orig_size[0],orig_size[1],masks.shape[-1])
masks=masks[...,:int(size_before_pad[0]),:int(size_before_pad[1])]
masks=torch.nn.functional.interpolate(torch.from_numpy(masks),size=orig_size,mode="bilinear",align_corners=False).numpy()
returnmasks

..code::ipython3

defshow_mask(mask,ax):
color=np.array([30/255,144/255,255/255,0.6])
h,w=mask.shape[-2:]
mask_image=mask.reshape(h,w,1)*color.reshape(1,1,-1)
ax.imshow(mask_image)


defshow_points(coords,labels,ax,marker_size=375):
pos_points=coords[labels==1]
neg_points=coords[labels==0]
ax.scatter(
pos_points[:,0],
pos_points[:,1],
color="green",
marker="*",
s=marker_size,
edgecolor="white",
linewidth=1.25,
)
ax.scatter(
neg_points[:,0],
neg_points[:,1],
color="red",
marker="*",
s=marker_size,
edgecolor="white",
linewidth=1.25,
)


defshow_box(box,ax):
x0,y0=box[0],box[1]
w,h=box[2]-box[0],box[3]-box[1]
ax.add_patch(plt.Rectangle((x0,y0),w,h,edgecolor="green",facecolor=(0,0,0,0),lw=2))

Imageencoding
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tostartworkwithimage,weshouldpreprocessitandobtainimage
embeddingsusing``ov_encoder``.Wewillusethesameimageforall
experiments,soitispossibletogenerateimageembeddingonceandthen
reusethem.

..code::ipython3

preprocessed_image=preprocess_image(image)
encoding_results=ov_encoder(preprocessed_image)

image_embeddings=encoding_results[ov_encoder.output(0)]

Now,wecantrytoprovidedifferentpromptsformaskgeneration

Examplepointinput
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthisexampleweselectonepoint.Thegreenstarsymbolshowits
locationontheimagebelow.

..code::ipython3

input_point=np.array([[500,375]])
input_label=np.array([1])

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_28_0.png


Addabatchindex,concatenateapaddingpoint,andtransformitto
inputtensorcoordinatesystem.

..code::ipython3

coord=np.concatenate([input_point,np.array([[0.0,0.0]])],axis=0)[None,:,:]
label=np.concatenate([input_label,np.array([-1])],axis=0)[None,:].astype(np.float32)
coord=resizer.apply_coords(coord,image.shape[:2]).astype(np.float32)

Packagetheinputstoruninthemaskpredictor.

..code::ipython3

inputs={
"image_embeddings":image_embeddings,
"point_coords":coord,
"point_labels":label,
}

Predictamaskandthresholdittogetbinarymask(0-noobject,1-
object).

..code::ipython3

results=ov_predictor(inputs)

masks=results[ov_predictor.output(0)]
masks=postprocess_masks(masks,image.shape[:-1])
masks=masks>0.0

..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks,plt.gca())
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_35_0.png


Examplewithmultiplepoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

inthisexample,weprovideadditionalpointforcoverlargerobject
area.

..code::ipython3

input_point=np.array([[500,375],[1125,625],[575,750],[1405,575]])
input_label=np.array([1,1,1,1])

Now,promptformodellookslikerepresentedonthisimage:

..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_39_0.png


Transformthepointsasinthepreviousexample.

..code::ipython3

coord=np.concatenate([input_point,np.array([[0.0,0.0]])],axis=0)[None,:,:]
label=np.concatenate([input_label,np.array([-1])],axis=0)[None,:].astype(np.float32)

coord=resizer.apply_coords(coord,image.shape[:2]).astype(np.float32)

Packageinputs,thenpredictandthresholdthemask.

..code::ipython3

inputs={
"image_embeddings":image_embeddings,
"point_coords":coord,
"point_labels":label,
}

results=ov_predictor(inputs)

masks=results[ov_predictor.output(0)]
masks=postprocess_masks(masks,image.shape[:-1])
masks=masks>0.0

..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks,plt.gca())
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_44_0.png


Great!Lookslikenow,predictedmaskcoverwholetruck.

Exampleboxandpointinputwithnegativelabel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthisexamplewedefineinputpromptusingboundingboxandpoint
insideit.Theboundingboxrepresentedassetofpointsofitsleft
uppercornerandrightlowercorner.Label0forpointspeakthatthis
pointshouldbeexcludedfrommask.

..code::ipython3

input_box=np.array([425,600,700,875])
input_point=np.array([[575,750]])
input_label=np.array([0])

..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
show_box(input_box,plt.gca())
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_48_0.png


Addabatchindex,concatenateaboxandpointinputs,addthe
appropriatelabelsfortheboxcorners,andtransform.Thereisno
paddingpointsincetheinputincludesaboxinput.

..code::ipython3

box_coords=input_box.reshape(2,2)
box_labels=np.array([2,3])

coord=np.concatenate([input_point,box_coords],axis=0)[None,:,:]
label=np.concatenate([input_label,box_labels],axis=0)[None,:].astype(np.float32)

coord=resizer.apply_coords(coord,image.shape[:2]).astype(np.float32)

Packageinputs,thenpredictandthresholdthemask.

..code::ipython3

inputs={
"image_embeddings":image_embeddings,
"point_coords":coord,
"point_labels":label,
}

results=ov_predictor(inputs)

masks=results[ov_predictor.output(0)]
masks=postprocess_masks(masks,image.shape[:-1])
masks=masks>0.0

..code::ipython3

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks[0],plt.gca())
show_box(input_box,plt.gca())
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_53_0.png


Interactivesegmentation
------------------------

`backtotop⬆️<#table-of-contents>`__

Now,youcantrySAMonownimage.Uploadimagetoinputwindowand
clickondesiredpoint,modelpredictsegmentbasedonyourimageand
point.

..code::ipython3

importgradioasgr


classSegmenter:
def__init__(self,ov_encoder,ov_predictor):
self.encoder=ov_encoder
self.predictor=ov_predictor
self._img_embeddings=None

defset_image(self,img:np.ndarray):
ifself._img_embeddingsisnotNone:
delself._img_embeddings
preprocessed_image=preprocess_image(img)
encoding_results=self.encoder(preprocessed_image)
image_embeddings=encoding_results[ov_encoder.output(0)]
self._img_embeddings=image_embeddings
returnimg

defget_mask(self,points,img):
coord=np.array(points)
coord=np.concatenate([coord,np.array([[0,0]])],axis=0)
coord=coord[None,:,:]
label=np.concatenate([np.ones(len(points)),np.array([-1])],axis=0)[None,:].astype(np.float32)
coord=resizer.apply_coords(coord,img.shape[:2]).astype(np.float32)
ifself._img_embeddingsisNone:
self.set_image(img)
inputs={
"image_embeddings":self._img_embeddings,
"point_coords":coord,
"point_labels":label,
}

results=self.predictor(inputs)
masks=results[ov_predictor.output(0)]
masks=postprocess_masks(masks,img.shape[:-1])

masks=masks>0.0
mask=masks[0]
mask=np.transpose(mask,(1,2,0))
returnmask


segmenter=Segmenter(ov_encoder,ov_predictor)


withgr.Blocks()asdemo:
withgr.Row():
input_img=gr.Image(label="Input",type="numpy",height=480,width=480)
output_img=gr.Image(label="SelectedSegment",type="numpy",height=480,width=480)

defon_image_change(img):
segmenter.set_image(img)
returnimg

defget_select_coords(img,evt:gr.SelectData):
pixels_in_queue=set()
h,w=img.shape[:2]
pixels_in_queue.add((evt.index[0],evt.index[1]))
out=img.copy()
whilelen(pixels_in_queue)>0:
pixels=list(pixels_in_queue)
pixels_in_queue=set()
color=np.random.randint(0,255,size=(1,1,3))
mask=segmenter.get_mask(pixels,img)
mask_image=out.copy()
mask_image[mask.squeeze(-1)]=color
out=cv2.addWeighted(out.astype(np.float32),0.7,mask_image.astype(np.float32),0.3,0.0)
out=out.astype(np.uint8)
returnout

input_img.select(get_select_coords,[input_img],output_img)
input_img.upload(on_image_change,[input_img],[input_img])

if__name__=="__main__":
try:
demo.launch()
exceptException:
demo.launch(share=True)


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


RunOpenVINOmodelinautomaticmaskgenerationmode
----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

SinceSAMcanefficientlyprocessprompts,masksfortheentireimage
canbegeneratedbysamplingalargenumberofpromptsoveranimage.
``automatic_mask_generation``functionimplementsthiscapability.It
worksbysamplingsingle-pointinputpromptsinagridovertheimage,
fromeachofwhichSAMcanpredictmultiplemasks.Then,masksare
filteredforqualityanddeduplicatedusingnon-maximalsuppression.
Additionaloptionsallowforfurtherimprovementofmaskqualityand
quantity,suchasrunningpredictiononmultiplecropsoftheimageor
postprocessingmaskstoremovesmalldisconnectedregionsandholes.

..code::ipython3

fromsegment_anything.utils.amgimport(
MaskData,
generate_crop_boxes,
uncrop_boxes_xyxy,
uncrop_masks,
uncrop_points,
calculate_stability_score,
rle_to_mask,
batched_mask_to_box,
mask_to_rle_pytorch,
is_box_near_crop_edge,
batch_iterator,
remove_small_regions,
build_all_layer_point_grids,
box_xyxy_to_xywh,
area_from_rle,
)
fromtorchvision.ops.boxesimportbatched_nms,box_area
fromtypingimportTuple,List,Dict,Any

..code::ipython3

defprocess_batch(
image_embedding:np.ndarray,
points:np.ndarray,
im_size:Tuple[int,...],
crop_box:List[int],
orig_size:Tuple[int,...],
iou_thresh,
mask_threshold,
stability_score_offset,
stability_score_thresh,
)->MaskData:
orig_h,orig_w=orig_size

#Runmodelonthisbatch
transformed_points=resizer.apply_coords(points,im_size)
in_points=transformed_points
in_labels=np.ones(in_points.shape[0],dtype=int)

inputs={
"image_embeddings":image_embedding,
"point_coords":in_points[:,None,:],
"point_labels":in_labels[:,None],
}
res=ov_predictor(inputs)
masks=postprocess_masks(res[ov_predictor.output(0)],orig_size)
masks=torch.from_numpy(masks)
iou_preds=torch.from_numpy(res[ov_predictor.output(1)])

#SerializepredictionsandstoreinMaskData
data=MaskData(
masks=masks.flatten(0,1),
iou_preds=iou_preds.flatten(0,1),
points=torch.as_tensor(points.repeat(masks.shape[1],axis=0)),
)
delmasks

#FilterbypredictedIoU
ifiou_thresh>0.0:
keep_mask=data["iou_preds"]>iou_thresh
data.filter(keep_mask)

#Calculatestabilityscore
data["stability_score"]=calculate_stability_score(data["masks"],mask_threshold,stability_score_offset)
ifstability_score_thresh>0.0:
keep_mask=data["stability_score"]>=stability_score_thresh
data.filter(keep_mask)

#Thresholdmasksandcalculateboxes
data["masks"]=data["masks"]>mask_threshold
data["boxes"]=batched_mask_to_box(data["masks"])

#Filterboxesthattouchcropboundaries
keep_mask=~is_box_near_crop_edge(data["boxes"],crop_box,[0,0,orig_w,orig_h])
ifnottorch.all(keep_mask):
data.filter(keep_mask)

#CompresstoRLE
data["masks"]=uncrop_masks(data["masks"],crop_box,orig_h,orig_w)
data["rles"]=mask_to_rle_pytorch(data["masks"])
deldata["masks"]

returndata

..code::ipython3

defprocess_crop(
image:np.ndarray,
point_grids,
crop_box:List[int],
crop_layer_idx:int,
orig_size:Tuple[int,...],
box_nms_thresh:float=0.7,
mask_threshold:float=0.0,
points_per_batch:int=64,
pred_iou_thresh:float=0.88,
stability_score_thresh:float=0.95,
stability_score_offset:float=1.0,
)->MaskData:
#Croptheimageandcalculateembeddings
x0,y0,x1,y1=crop_box
cropped_im=image[y0:y1,x0:x1,:]
cropped_im_size=cropped_im.shape[:2]
preprocessed_cropped_im=preprocess_image(cropped_im)
crop_embeddings=ov_encoder(preprocessed_cropped_im)[ov_encoder.output(0)]

#Getpointsforthiscrop
points_scale=np.array(cropped_im_size)[None,::-1]
points_for_image=point_grids[crop_layer_idx]*points_scale

#Generatemasksforthiscropinbatches
data=MaskData()
for(points,)inbatch_iterator(points_per_batch,points_for_image):
batch_data=process_batch(
crop_embeddings,
points,
cropped_im_size,
crop_box,
orig_size,
pred_iou_thresh,
mask_threshold,
stability_score_offset,
stability_score_thresh,
)
data.cat(batch_data)
delbatch_data

#Removeduplicateswithinthiscrop.
keep_by_nms=batched_nms(
data["boxes"].float(),
data["iou_preds"],
torch.zeros(len(data["boxes"])),#categories
iou_threshold=box_nms_thresh,
)
data.filter(keep_by_nms)

#Returntotheoriginalimageframe
data["boxes"]=uncrop_boxes_xyxy(data["boxes"],crop_box)
data["points"]=uncrop_points(data["points"],crop_box)
data["crop_boxes"]=torch.tensor([crop_boxfor_inrange(len(data["rles"]))])

returndata

..code::ipython3

defgenerate_masks(image:np.ndarray,point_grids,crop_n_layers,crop_overlap_ratio,crop_nms_thresh)->MaskData:
orig_size=image.shape[:2]
crop_boxes,layer_idxs=generate_crop_boxes(orig_size,crop_n_layers,crop_overlap_ratio)

#Iterateoverimagecrops
data=MaskData()
forcrop_box,layer_idxinzip(crop_boxes,layer_idxs):
crop_data=process_crop(image,point_grids,crop_box,layer_idx,orig_size)
data.cat(crop_data)

#Removeduplicatemasksbetweencrops
iflen(crop_boxes)>1:
#Prefermasksfromsmallercrops
scores=1/box_area(data["crop_boxes"])
scores=scores.to(data["boxes"].device)
keep_by_nms=batched_nms(
data["boxes"].float(),
scores,
torch.zeros(len(data["boxes"])),#categories
iou_threshold=crop_nms_thresh,
)
data.filter(keep_by_nms)

data.to_numpy()
returndata

..code::ipython3

defpostprocess_small_regions(mask_data:MaskData,min_area:int,nms_thresh:float)->MaskData:
"""
Removessmalldisconnectedregionsandholesinmasks,thenreruns
boxNMStoremoveanynewduplicates.

Editsmask_datainplace.

Requiresopen-cvasadependency.
"""
iflen(mask_data["rles"])==0:
returnmask_data

#Filtersmalldisconnectedregionsandholes
new_masks=[]
scores=[]
forrleinmask_data["rles"]:
mask=rle_to_mask(rle)

mask,changed=remove_small_regions(mask,min_area,mode="holes")
unchanged=notchanged
mask,changed=remove_small_regions(mask,min_area,mode="islands")
unchanged=unchangedandnotchanged

new_masks.append(torch.as_tensor(mask).unsqueeze(0))
#Givescore=0tochangedmasksandscore=1tounchangedmasks
#soNMSwillpreferonesthatdidn'tneedpostprocessing
scores.append(float(unchanged))

#Recalculateboxesandremoveanynewduplicates
masks=torch.cat(new_masks,dim=0)
boxes=batched_mask_to_box(masks)
keep_by_nms=batched_nms(
boxes.float(),
torch.as_tensor(scores),
torch.zeros(len(boxes)),#categories
iou_threshold=nms_thresh,
)

#OnlyrecalculateRLEsformasksthathavechanged
fori_maskinkeep_by_nms:
ifscores[i_mask]==0.0:
mask_torch=masks[i_mask].unsqueeze(0)
mask_data["rles"][i_mask]=mask_to_rle_pytorch(mask_torch)[0]
#updateresdirectly
mask_data["boxes"][i_mask]=boxes[i_mask]
mask_data.filter(keep_by_nms)

returnmask_data

Thereareseveraltunableparametersinautomaticmaskgenerationthat
controlhowdenselypointsaresampledandwhatthethresholdsarefor
removinglowqualityorduplicatemasks.Additionally,generationcanbe
automaticallyrunoncropsoftheimagetogetimprovedperformanceon
smallerobjects,andpost-processingcanremovestraypixelsandholes

..code::ipython3

defautomatic_mask_generation(
image:np.ndarray,
min_mask_region_area:int=0,
points_per_side:int=32,
crop_n_layers:int=0,
crop_n_points_downscale_factor:int=1,
crop_overlap_ratio:float=512/1500,
box_nms_thresh:float=0.7,
crop_nms_thresh:float=0.7,
)->List[Dict[str,Any]]:
"""
Generatesmasksforthegivenimage.

Arguments:
image(np.ndarray):Theimagetogeneratemasksfor,inHWCuint8format.

Returns:
list(dict(str,any)):Alistoverrecordsformasks.Eachrecordis
adictcontainingthefollowingkeys:
segmentation(dict(str,any)ornp.ndarray):Themask.If
output_mode='binary_mask',isanarrayofshapeHW.Otherwise,
isadictionarycontainingtheRLE.
bbox(list(float)):Theboxaroundthemask,inXYWHformat.
area(int):Theareainpixelsofthemask.
predicted_iou(float):Themodel'sownpredictionofthemask's
quality.Thisisfilteredbythepred_iou_threshparameter.
point_coords(list(list(float))):Thepointcoordinatesinput
tothemodeltogeneratethismask.
stability_score(float):Ameasureofthemask'squality.This
isfilteredonusingthestability_score_threshparameter.
crop_box(list(float)):Thecropoftheimageusedtogenerate
themask,giveninXYWHformat.
"""
point_grids=build_all_layer_point_grids(
points_per_side,
crop_n_layers,
crop_n_points_downscale_factor,
)
mask_data=generate_masks(image,point_grids,crop_n_layers,crop_overlap_ratio,crop_nms_thresh)

#Filtersmalldisconnectedregionsandholesinmasks
ifmin_mask_region_area>0:
mask_data=postprocess_small_regions(
mask_data,
min_mask_region_area,
max(box_nms_thresh,crop_nms_thresh),
)

mask_data["segmentations"]=[rle_to_mask(rle)forrleinmask_data["rles"]]

#Writemaskrecords
curr_anns=[]
foridxinrange(len(mask_data["segmentations"])):
ann={
"segmentation":mask_data["segmentations"][idx],
"area":area_from_rle(mask_data["rles"][idx]),
"bbox":box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
"predicted_iou":mask_data["iou_preds"][idx].item(),
"point_coords":[mask_data["points"][idx].tolist()],
"stability_score":mask_data["stability_score"][idx].item(),
"crop_box":box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
}
curr_anns.append(ann)

returncurr_anns

..code::ipython3

prediction=automatic_mask_generation(image)

``automatic_mask_generation``returnsalistovermasks,whereeachmask
isadictionarycontainingvariousdataaboutthemask.Thesekeysare:

-``segmentation``:themask
-``area``:theareaofthemaskinpixels
-``bbox``:theboundaryboxofthemaskinXYWHformat
-``predicted_iou``:themodel’sownpredictionforthequalityofthe
mask
-``point_coords``:thesampledinputpointthatgeneratedthismask
-``stability_score``:anadditionalmeasureofmaskquality
-``crop_box``:thecropoftheimageusedtogeneratethismaskin
XYWHformat

..code::ipython3

print(f"Numberofdetectedmasks:{len(prediction)}")
print(f"Annotationkeys:{prediction[0].keys()}")


..parsed-literal::

Numberofdetectedmasks:48
Annotationkeys:dict_keys(['segmentation','area','bbox','predicted_iou','point_coords','stability_score','crop_box'])


..code::ipython3

fromtqdm.notebookimporttqdm


defdraw_anns(image,anns):
iflen(anns)==0:
return
segments_image=image.copy()
sorted_anns=sorted(anns,key=(lambdax:x["area"]),reverse=True)
forannintqdm(sorted_anns):
mask=ann["segmentation"]
mask_color=np.random.randint(0,255,size=(1,1,3)).astype(np.uint8)
segments_image[mask]=mask_color
returncv2.addWeighted(image.astype(np.float32),0.7,segments_image.astype(np.float32),0.3,0.0)

..code::ipython3

importPIL

out=draw_anns(image,prediction)
cv2.imwrite("result.png",out[:,:,::-1])

PIL.Image.open("result.png")



..parsed-literal::

0%||0/48[00:00<?,?it/s]




..image::segment-anything-with-output_files/segment-anything-with-output_68_1.png



OptimizeencoderusingNNCFPost-trainingQuantizationAPI
----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.

SinceencodercostingmuchmoretimethanotherpartsinSAMinference
pipeline,wewilluse8-bitquantizationinpost-trainingmode(without
thefine-tuningpipeline)tooptimizeencoderofSAM.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeOpenVINOIRmodel,usingthe``openvino.save_model``
function.

Prepareacalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

DownloadCOCOdataset.Sincethedatasetisusedtocalibratethe
model’sparameterinsteadoffine-tuningit,wedon’tneedtodownload
thelabelfiles.

..code::ipython3

fromzipfileimportZipFile

DATA_URL="https://ultralytics.com/assets/coco128.zip"
OUT_DIR=Path(".")

download_file(DATA_URL,directory=OUT_DIR,show_progress=True)

ifnot(OUT_DIR/"coco128/images/train2017").exists():
withZipFile("coco128.zip","r")aszip_ref:
zip_ref.extractall(OUT_DIR)


..parsed-literal::

'coco128.zip'alreadyexists.


Createaninstanceofthe``nncf.Dataset``classthatrepresentsthe
calibrationdataset.ForPyTorch,wecanpassaninstanceofthe
``torch.utils.data.DataLoader``object.

..code::ipython3

importtorch.utils.dataasdata


classCOCOLoader(data.Dataset):
def__init__(self,images_path):
self.images=list(Path(images_path).iterdir())

def__getitem__(self,index):
image_path=self.images[index]
image=cv2.imread(str(image_path))
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
returnimage

def__len__(self):
returnlen(self.images)


coco_dataset=COCOLoader(OUT_DIR/"coco128/images/train2017")
calibration_loader=torch.utils.data.DataLoader(coco_dataset)

Thetransformationfunctionisafunctionthattakesasamplefromthe
datasetandreturnsdatathatcanbepassedtothemodelforinference.

..code::ipython3

importnncf


deftransform_fn(image_data):
"""
Quantizationtransformfunction.Extractsandpreprocessinputdatafromdataloaderitemforquantization.
Parameters:
image_data:imagedataproducedbyDataLoaderduringiteration
Returns:
input_tensor:inputdatainDictformatformodelquantization
"""
image=image_data.numpy()
processed_image=preprocess_image(np.squeeze(image))
returnprocessed_image


calibration_dataset=nncf.Dataset(calibration_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


RunquantizationandserializeOpenVINOIRmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Itisavailableformodelsinthefollowing
frameworks:``PyTorch``,``TensorFlow2.x``,``ONNX``,and
``OpenVINOIR``.

Optionally,someadditionalparametersfortheconfiguration
quantizationprocess(numberofsamplesforquantization,preset,model
type,etc.)canbeprovided.``model_type``canbeusedtospecify
quantizationschemerequiredforspecifictypeofthemodel.For
example,TransformermodelssuchasSAMrequireaspecialquantization
schemetopreserveaccuracyafterquantization.Toachieveabetter
result,wewillusea``mixed``quantizationpreset.Itprovides
symmetricquantizationofweightsandasymmetricquantizationof
activations.

**Note**:Modelpost-trainingquantizationistime-consumingprocess.
Bepatient,itcantakeseveralminutesdependingonyourhardware.

..code::ipython3

model=core.read_model(ov_encoder_path)
quantized_model=nncf.quantize(
model,
calibration_dataset,
model_type=nncf.parameters.ModelType.TRANSFORMER,
subset_size=128,
)
print("modelquantizationfinished")


..parsed-literal::

2023-09-1120:39:36.145499:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-09-1120:39:36.181406:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-09-1120:39:36.769588:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
Statisticscollection:100%|██████████████████|128/128[02:12<00:00,1.03s/it]
ApplyingSmoothQuant:100%|████████████████████|48/48[00:01<00:00,32.29it/s]


..parsed-literal::

INFO:nncf:36ignorednodeswasfoundbynameintheNNCFGraph


..parsed-literal::

Statisticscollection:100%|██████████████████|128/128[04:36<00:00,2.16s/it]
ApplyingFastBiascorrection:100%|████████████|49/49[00:28<00:00,1.72it/s]

..parsed-literal::

modelquantizationfinished


..parsed-literal::




..code::ipython3

ov_encoder_path_int8="sam_image_encoder_int8.xml"
ov.save_model(quantized_model,ov_encoder_path_int8)

ValidateQuantizedModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Wecanreusethepreviouscodetovalidatetheoutputof``INT8``model.

..code::ipython3

#LoadINT8modelandrunpipelineagain
ov_encoder_model_int8=core.read_model(ov_encoder_path_int8)
ov_encoder_int8=core.compile_model(ov_encoder_model_int8,device.value)
encoding_results=ov_encoder_int8(preprocessed_image)
image_embeddings=encoding_results[ov_encoder_int8.output(0)]

input_point=np.array([[500,375]])
input_label=np.array([1])
coord=np.concatenate([input_point,np.array([[0.0,0.0]])],axis=0)[None,:,:]
label=np.concatenate([input_label,np.array([-1])],axis=0)[None,:].astype(np.float32)

coord=resizer.apply_coords(coord,image.shape[:2]).astype(np.float32)
inputs={
"image_embeddings":image_embeddings,
"point_coords":coord,
"point_labels":label,
}
results=ov_predictor(inputs)

masks=results[ov_predictor.output(0)]
masks=postprocess_masks(masks,image.shape[:-1])
masks=masks>0.0
plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks,plt.gca())
show_points(input_point,input_label,plt.gca())
plt.axis("off")
plt.show()



..image::segment-anything-with-output_files/segment-anything-with-output_80_0.png


Run``INT8``modelinautomaticmaskgenerationmode

..code::ipython3

ov_encoder=ov_encoder_int8
prediction=automatic_mask_generation(image)
out=draw_anns(image,prediction)
cv2.imwrite("result_int8.png",out[:,:,::-1])
PIL.Image.open("result_int8.png")



..parsed-literal::

0%||0/47[00:00<?,?it/s]




..image::segment-anything-with-output_files/segment-anything-with-output_82_1.png



ComparePerformanceoftheOriginalandQuantizedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Finally,usetheOpenVINO
`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
tomeasuretheinferenceperformanceofthe``FP32``and``INT8``
models.

..code::ipython3

#InferenceFP32model(OpenVINOIR)
!benchmark_app-m$ov_encoder_path-d$device.value


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
[INFO]OpenVINO:
[INFO]Build.................................2023.1.0-12050-e33de350633
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2023.1.0-12050-e33de350633
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook31.21ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,1024,1024]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.neck.3/aten::add/Add_2933):f32/[...]/[1,256,64,64]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,1024,1024]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.neck.3/aten::add/Add_2933):f32/[...]/[1,256,64,64]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook956.62ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model474
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:36
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]NETWORK_NAME:Model474
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:False
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook3347.39ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:132iterations
[INFO]Duration:135907.17ms
[INFO]Latency:
[INFO]Median:12159.63ms
[INFO]Average:12098.43ms
[INFO]Min:7652.77ms
[INFO]Max:13027.98ms
[INFO]Throughput:0.97FPS


..code::ipython3

#InferenceINT8model(OpenVINOIR)
!benchmark_app-m$ov_encoder_path_int8-d$device.value


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[WARNING]Defaultduration120secondsisusedforunknowndeviceAUTO
[INFO]OpenVINO:
[INFO]Build.................................2023.1.0-12050-e33de350633
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2023.1.0-12050-e33de350633
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook40.67ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,1024,1024]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.neck.3/aten::add/Add_2933):f32/[...]/[1,256,64,64]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,1024,1024]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.neck.3/aten::add/Add_2933):f32/[...]/[1,256,64,64]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1151.47ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model474
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:36
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]NETWORK_NAME:Model474
[INFO]NUM_STREAMS:12
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:False
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:120000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook1951.78ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:216iterations
[INFO]Duration:130123.96ms
[INFO]Latency:
[INFO]Median:7192.03ms
[INFO]Average:7197.18ms
[INFO]Min:6134.35ms
[INFO]Max:7888.28ms
[INFO]Throughput:1.66FPS


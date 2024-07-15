VisualContentSearchusingMobileCLIPandOpenVINO
===================================================

Semanticvisualcontentsearchisamachinelearningtaskthatuses
eitheratextqueryoraninputimagetosearchadatabaseofimages
(photogallery,video)tofindimagesthataresemanticallysimilarto
thesearchquery.Historically,buildingarobustsearchenginefor
imageswasdifficult.Onecouldsearchbyfeaturessuchasfilenameand
imagemetadata,anduseanycontextaroundanimage(i.e. alttextor
surroundingtextifanimageappearsinapassageoftext)toprovide
therichersearchingfeature.Thiswasbeforetheadventofneural
networksthatcanidentifysemanticallyrelatedimagestoagivenuser
query.

`ContrastiveLanguage-ImagePre-Training
(CLIP)<https://arxiv.org/abs/2103.00020>`__modelsprovidethemeans
throughwhichyoucanimplementasemanticsearchenginewithafew
dozenlinesofcode.TheCLIPmodelhasbeentrainedonmillionsof
pairsoftextandimages,encodingsemanticsfromimagesandtext
combined.UsingCLIP,youcanprovideatextqueryandCLIPwillreturn
theimagesmostrelatedtothequery.

Inthistutorial,weconsiderhowtouseMobileCLIPtoimplementa
visualcontentsearchengineforfindingrelevantframesinvideo.####
Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`Selectmodel<#select-model>`__
-`Runmodelinference<#run-model-inference>`__

-`Prepareimagegallery<#prepare-image-gallery>`__
-`Preparemodel<#prepare-model>`__
-`Performsearch<#perform-search>`__

-`ConvertModeltoOpenVINOIntermediateRepresentation
format<#convert-model-to-openvino-intermediate-representation-format>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectdeviceforimage
encoder<#select-device-for-image-encoder>`__
-`Selectdevicefortext
encoder<#select-device-for-text-encoder>`__
-`Performsearch<#perform-search>`__

-`InteractiveDemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__##Prerequisites

..code::ipython3

frompathlibimportPath

repo_dir=Path("./ml-mobileclip")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/apple/ml-mobileclip.git


..parsed-literal::

Cloninginto'ml-mobileclip'...
remote:Enumeratingobjects:68,done.[K
remote:Countingobjects:100%(68/68),done.[K
remote:Compressingobjects:100%(51/51),done.[K
remote:Total68(delta19),reused65(delta16),pack-reused0[K
Unpackingobjects:100%(68/68),447.59KiB|4.03MiB/s,done.


..code::ipython3

%pipinstall-q"./ml-mobileclip"--no-deps

%pipinstall-q"clip-benchmark>=1.4.0""datasets>=2.8.0""open-clip-torch>=2.20.0""timm>=0.9.5""torch>=1.13.1""torchvision>=0.14.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu

%pipinstall-q"openvino>=2024.0.0""gradio>=4.19""matplotlib""Pillow""altair""pandas""opencv-python""tqdm"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.3.1+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Selectmodel
------------

`backtotop⬆️<#table-of-contents>`__

Forstartingwork,weshouldselectmodelthatwillbeusedinour
demonstration.Bydefault,wewillusetheMobileCLIPmodel,butfor
comparisonpurposes,youcanselectdifferentmodelsamong:

-**CLIP**-CLIP(ContrastiveLanguage-ImagePre-Training)isaneural
networktrainedonvarious(image,text)pairs.Itcanbeinstructed
innaturallanguagetopredictthemostrelevanttextsnippet,given
animage,withoutdirectlyoptimizingforthetask.CLIPusesa
`ViT<https://arxiv.org/abs/2010.11929>`__liketransformertoget
visualfeaturesandacausallanguagemodeltogetthetextfeatures.
Thetextandvisualfeaturesarethenprojectedintoalatentspace
withidenticaldimensions.Thedotproductbetweentheprojected
imageandtextfeaturesisthenusedasasimilarityscore.Youcan
findmoreinformationaboutthismodelinthe`research
paper<https://arxiv.org/abs/2103.00020>`__,`OpenAI
blog<https://openai.com/blog/clip/>`__,`model
card<https://github.com/openai/CLIP/blob/main/model-card.md>`__and
GitHub`repository<https://github.com/openai/CLIP>`__.
-**SigLIP**-TheSigLIPmodelwasproposedin`SigmoidLossfor
LanguageImagePre-Training<https://arxiv.org/abs/2303.15343>`__.
SigLIPproposestoreplacethelossfunctionusedin
`CLIP<https://github.com/openai/CLIP>`__(ContrastiveLanguage–Image
Pre-training)byasimplepairwisesigmoidloss.Thisresultsin
betterperformanceintermsofzero-shotclassificationaccuracyon
ImageNet.Youcanfindmoreinformationaboutthismodelinthe
`researchpaper<https://arxiv.org/abs/2303.15343>`__and`GitHub
repository<https://github.com/google-research/big_vision>`__,
-**MobileCLIP**-MobileCLIP–anewfamilyofefficientimage-text
modelsoptimizedforruntimeperformancealongwithanoveland
efficienttrainingapproach,namelymulti-modalreinforcedtraining.
ThesmallestvariantMobileCLIP-S0obtainssimilarzero-shot
performanceasOpenAI’sCLIPViT-b16modelwhilebeingseveraltimes
fasterand2.8xsmaller.Moredetailsaboutmodelcanbefoundin
`researchpaper<https://arxiv.org/pdf/2311.17049.pdf>`__and`GitHub
repository<https://github.com/apple/ml-mobileclip>`__.

..code::ipython3

importipywidgetsaswidgets

model_dir=Path("checkpoints")

supported_models={
"MobileCLIP":{
"mobileclip_s0":{
"model_name":"mobileclip_s0",
"pretrained":model_dir/"mobileclip_s0.pt",
"url":"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
"image_size":256,
},
"mobileclip_s1":{
"model_name":"mobileclip_s1",
"pretrained":model_dir/"mobileclip_s1.pt",
"url":"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt",
"image_size":256,
},
"mobileclip_s2":{
"model_name":"mobileclip_s0",
"pretrained":model_dir/"mobileclip_s2.pt",
"url":"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt",
"image_size":256,
},
"mobileclip_b":{
"model_name":"mobileclip_b",
"pretrained":model_dir/"mobileclip_b.pt",
"url":"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt",
"image_size":224,
},
"mobileclip_blt":{
"model_name":"mobileclip_b",
"pretrained":model_dir/"mobileclip_blt.pt",
"url":"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt",
"image_size":224,
},
},
"CLIP":{
"clip-vit-b-32":{
"model_name":"ViT-B-32",
"pretrained":"laion2b_s34b_b79k",
"image_size":224,
},
"clip-vit-b-16":{
"image_name":"ViT-B-16",
"pretrained":"openai",
"image_size":224,
},
"clip-vit-l-14":{
"image_name":"ViT-L-14",
"pretrained":"datacomp_xl_s13b_b90k",
"image_size":224,
},
"clip-vit-h-14":{
"image_name":"ViT-H-14",
"pretrained":"laion2b_s32b_b79k",
"image_size":224,
},
},
"SigLIP":{
"siglip-vit-b-16":{
"model_name":"ViT-B-16-SigLIP",
"pretrained":"webli",
"image_size":224,
},
"siglip-vit-l-16":{
"model_name":"ViT-L-16-SigLIP-256",
"pretrained":"webli",
"image_size":256,
},
},
}


model_type=widgets.Dropdown(options=supported_models.keys(),default="MobileCLIP",description="Modeltype:")
model_type




..parsed-literal::

Dropdown(description='Modeltype:',options=('MobileCLIP','CLIP','SigLIP'),value='MobileCLIP')



..code::ipython3

available_models=supported_models[model_type.value]

model_checkpoint=widgets.Dropdown(
options=available_models.keys(),
default=list(available_models),
description="Model:",
)

model_checkpoint




..parsed-literal::

Dropdown(description='Model:',options=('mobileclip_s0','mobileclip_s1','mobileclip_s2','mobileclip_b','mo…



..code::ipython3

importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

model_config=available_models[model_checkpoint.value]

Runmodelinference
-------------------

`backtotop⬆️<#table-of-contents>`__

Now,let’sseemodelinaction.Wewilltrytofindimage,wheresome
specificobjectisrepresentedusingembeddings.Embeddingsarea
numericrepresentationofdatasuchastextandimages.Themodel
learnedtoencodesemanticsaboutthecontentsofimagesinembedding
format.Thisabilityturnsthemodelintoapowerfulforsolvingvarious
tasksincludingimage-textretrieval.Toreachourgoalweshould:

1.Calculateembeddingsforalloftheimagesinourdataset;
2.Calculateatextembeddingforauserquery(i.e. “blackdog”or
“car”);
3.Comparethetextembeddingtotheimageembeddingstofindrelated
embeddings.

Theclosertwoembeddingsare,themoresimilarthecontentsthey
representare.

Prepareimagegallery
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtypingimportList
importmatplotlib.pyplotasplt
importnumpyasnp
fromPILimportImage


defvisualize_result(images:List,query:str="",selected:List[int]=None):
"""
Utilityfunctionforvisualizationclassificationresults
params:
images(List[Image])-listofimagesforvisualization
query(str)-titleforvisualization
selected(List[int])-listofselectedimageindicesfromimages
returns:
matplotlib.Figure
"""
figsize=(20,5)
fig,axs=plt.subplots(1,4,figsize=figsize,sharex="all",sharey="all")
fig.patch.set_facecolor("white")
list_axes=list(axs.flat)
ifquery:
fig.suptitle(query,fontsize=20)
foridx,ainenumerate(list_axes):
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
a.imshow(images[idx])
ifselectedisnotNoneandidxnotinselected:
mask=np.ones_like(np.array(images[idx]))
a.imshow(mask,"jet",interpolation="none",alpha=0.75)
returnfig


images_urls=[
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/282ce53e-912d-41aa-ab48-2a001c022d74",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/9bb40168-82b5-4b11-ada6-d8df104c736c",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/0747b6db-12c3-4252-9a6a-057dcf8f3d4e",
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png",
]
image_names=["red_panda.png","cat.png","raccoon.png","dog.png"]
sample_path=Path("data")
sample_path.mkdir(parents=True,exist_ok=True)

images=[]
forimage_name,image_urlinzip(image_names,images_urls):
image_path=sample_path/image_name
ifnotimage_path.exists():
download_file(image_url,filename=image_name,directory=sample_path)
images.append(Image.open(image_path).convert("RGB").resize((640,420)))

input_labels=["cat"]
text_descriptions=[f"Thisisaphotoofa{label}"forlabelininput_labels]

visualize_result(images,"imagegallery");



..parsed-literal::

data/red_panda.png:0%||0.00/50.6k[00:00<?,?B/s]



..parsed-literal::

data/cat.png:0%||0.00/54.5k[00:00<?,?B/s]



..parsed-literal::

data/raccoon.png:0%||0.00/106k[00:00<?,?B/s]



..parsed-literal::

data/dog.png:0%||0.00/716k[00:00<?,?B/s]



..image::mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_10_4.png


Preparemodel
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thecodebellowdownloadmodelweights,createmodelclassinstanceand
preprocessingutilities

..code::ipython3

importtorch
importtime
fromPILimportImage
importmobileclip
importopen_clip

#instantiatemodel
model_name=model_config["model_name"]
pretrained=model_config["pretrained"]
ifmodel_type.value=="MobileCLIP":
model_dir.mkdir(exist_ok=True)
model_url=model_config["url"]
download_file(model_url,directory=model_dir)
model,_,preprocess=mobileclip.create_model_and_transforms(model_name,pretrained=pretrained)
tokenizer=mobileclip.get_tokenizer(model_name)
else:
model,_,preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained)
tokenizer=open_clip.get_tokenizer(model_name)



..parsed-literal::

checkpoints/mobileclip_s0.pt:0%||0.00/206M[00:00<?,?B/s]


Performsearch
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

image_tensor=torch.stack([preprocess(image)forimageinimages])
text=tokenizer(text_descriptions)


withtorch.no_grad():
#calculateimageembeddings
image_encoding_start=time.perf_counter()
image_features=model.encode_image(image_tensor)
image_encoding_end=time.perf_counter()
print(f"Imageencodingtook{image_encoding_end-image_encoding_start:.3}ms")
#calculatetextembeddings
text_encoding_start=time.perf_counter()
text_features=model.encode_text(text)
text_encoding_end=time.perf_counter()
print(f"Textencodingtook{text_encoding_end-text_encoding_start:.3}ms")

#normalizeembeddings
image_features/=image_features.norm(dim=-1,keepdim=True)
text_features/=text_features.norm(dim=-1,keepdim=True)

#calcualtesimilarityscore
image_probs=(100.0*text_features@image_features.T).softmax(dim=-1)
selected_image=[torch.argmax(image_probs).item()]

visualize_result(images,input_labels[0],selected_image);


..parsed-literal::

Imageencodingtook0.123ms
Textencodingtook0.0159ms



..image::mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_14_1.png


ConvertModeltoOpenVINOIntermediateRepresentationformat
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ForbestresultswithOpenVINO,itisrecommendedtoconvertthemodel
toOpenVINOIRformat.OpenVINOsupportsPyTorchviaModelconversion
API.ToconvertthePyTorchmodeltoOpenVINOIRformatwewilluse
``ov.convert_model``of`modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
The``ov.convert_model``PythonfunctionreturnsanOpenVINOModel
objectreadytoloadonthedeviceandstartmakingpredictions.

Ourmodelconsistfrom2parts-imageencoderandtextencoderthatcan
beusedseparately.Let’sconverteachparttoOpenVINO.

..code::ipython3

importtypes
importtorch.nn.functionalasF


defse_block_forward(self,inputs):
"""Applyforwardpass."""
b,c,h,w=inputs.size()
x=F.avg_pool2d(inputs,kernel_size=[8,8])
x=self.reduce(x)
x=F.relu(x)
x=self.expand(x)
x=torch.sigmoid(x)
x=x.view(-1,c,1,1)
returninputs*x

..code::ipython3

importopenvinoasov
importgc

ov_models_dir=Path("ov_models")
ov_models_dir.mkdir(exist_ok=True)

image_encoder_path=ov_models_dir/f"{model_checkpoint.value}_im_encoder.xml"

ifnotimage_encoder_path.exists():
if"mobileclip_s"inmodel_name:
model.image_encoder.model.conv_exp.se.forward=types.MethodType(se_block_forward,model.image_encoder.model.conv_exp.se)
model.forward=model.encode_image
ov_image_encoder=ov.convert_model(
model,
example_input=image_tensor,
input=[-1,3,image_tensor.shape[2],image_tensor.shape[3]],
)
ov.save_model(ov_image_encoder,image_encoder_path)
delov_image_encoder
gc.collect()

text_encoder_path=ov_models_dir/f"{model_checkpoint.value}_text_encoder.xml"

ifnottext_encoder_path.exists():
model.forward=model.encode_text
ov_text_encoder=ov.convert_model(model,example_input=text,input=[-1,text.shape[1]])
ov.save_model(ov_text_encoder,text_encoder_path)
delov_text_encoder
gc.collect()

delmodel
gc.collect();


..parsed-literal::

['image']


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/mobileclip/modules/common/transformer.py:125:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifseq_len!=self.num_embeddings:


..parsed-literal::

['text']


RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Selectdeviceforimageencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()

importipywidgetsaswidgets

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

ov_compiled_image_encoder=core.compile_model(image_encoder_path,device.value)
ov_compiled_image_encoder(image_tensor);

Selectdevicefortextencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

ov_compiled_text_encoder=core.compile_model(text_encoder_path,device.value)
ov_compiled_text_encoder(text);

Performsearch
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

image_encoding_start=time.perf_counter()
image_features=torch.from_numpy(ov_compiled_image_encoder(image_tensor)[0])
image_encoding_end=time.perf_counter()
print(f"Imageencodingtook{image_encoding_end-image_encoding_start:.3}ms")
text_encoding_start=time.perf_counter()
text_features=torch.from_numpy(ov_compiled_text_encoder(text)[0])
text_encoding_end=time.perf_counter()
print(f"Textencodingtook{text_encoding_end-text_encoding_start:.3}ms")
image_features/=image_features.norm(dim=-1,keepdim=True)
text_features/=text_features.norm(dim=-1,keepdim=True)

image_probs=(100.0*text_features@image_features.T).softmax(dim=-1)
selected_image=[torch.argmax(image_probs).item()]

visualize_result(images,input_labels[0],selected_image);


..parsed-literal::

Imageencodingtook0.0321ms
Textencodingtook0.00763ms



..image::mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_25_1.png


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Inthispart,youcantrydifferentsupportedbytutorialmodelsin
searchingframesinthevideobytextqueryorimage.Uploadvideoand
providetextqueryorreferenceimageforsearchandmodelwillfindthe
mostrelevantframesaccordingtoprovidedquery.Pleasenote,different
modelscanrequiredifferentoptimalthresholdforsearch.

..code::ipython3

importaltairasalt
importcv2
importgradioasgr
importpandasaspd
importtorch
fromPILimportImage
fromtorch.utils.dataimportDataLoader,Dataset
fromtorchvision.transforms.functionalimportto_pil_image,to_tensor
fromtorchvision.transformsimport(
CenterCrop,
Compose,
InterpolationMode,
Resize,
ToTensor,
)
fromopen_clip.transformimportimage_transform


current_device=device.value
current_model=image_encoder_path.name.split("_im_encoder")[0]

available_converted_models=[model_file.name.split("_im_encoder")[0]formodel_fileinov_models_dir.glob("*_im_encoder.xml")]
available_devices=list(core.available_devices)+["AUTO"]

download_file(
"https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
directory=sample_path,
)
download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4",
directory=sample_path,
filename="coco.mp4",
)


defget_preprocess_and_tokenizer(model_name):
if"mobileclip"inmodel_name:
resolution=supported_models["MobileCLIP"][model_name]["image_size"]
resize_size=resolution
centercrop_size=resolution
aug_list=[
Resize(
resize_size,
interpolation=InterpolationMode.BILINEAR,
),
CenterCrop(centercrop_size),
ToTensor(),
]
preprocess=Compose(aug_list)
tokenizer=mobileclip.get_tokenizer(supported_models["MobileCLIP"][model_name]["model_name"])
else:
model_configs=supported_models["SigLIP"]if"siglip"inmodel_nameelsesupported_models["CLIP"]
resize_size=model_configs[model_name]["image_size"]
preprocess=image_transform((resize_size,resize_size),is_train=False,resize_mode="longest")
tokenizer=open_clip.get_tokenizer(model_configs[model_name]["model_name"])

returnpreprocess,tokenizer


defrun(
path:str,
text_search:str,
image_search:Image.Image,
model_name:str,
device:str,
thresh:float,
stride:int,
batch_size:int,
):
assertpath,"Aninputvideoshouldbeprovided"
asserttext_searchisnotNoneorimage_searchisnotNone,"Atextorimagequeryshouldbeprovided"
globalcurrent_model
globalcurrent_device
globalpreprocess
globaltokenizer
globalov_compiled_image_encoder
globalov_compiled_text_encoder

ifcurrent_model!=model_nameordevice!=current_device:
ov_compiled_image_encoder=core.compile_model(ov_models_dir/f"{model_name}_im_encoder.xml",device)
ov_compiled_text_encoder=core.compile_model(ov_models_dir/f"{model_name}_text_encoder.xml",device)
preprocess,tokenizer=get_preprocess_and_tokenizer(model_name)
current_model=model_name
current_device=device
#Loadvideo
dataset=LoadVideo(path,transforms=preprocess,vid_stride=stride)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)

#Getimagequeryfeatures
ifimage_search:
image=preprocess(image_search).unsqueeze(0)
query_features=torch.from_numpy(ov_compiled_image_encoder(image)[0])
query_features/=query_features.norm(dim=-1,keepdim=True)
#Gettextqueryfeatures
else:
#Tokenizesearchphrase
text=tokenizer([text_search])
#Encodetextquery
query_features=torch.from_numpy(ov_compiled_text_encoder(text)[0])
query_features/=query_features.norm(dim=-1,keepdim=True)
#Encodeeachframeandcomparewithqueryfeatures
matches=[]
matches_probs=[]
res=pd.DataFrame(columns=["Frame","Timestamp","Similarity"])
forimage,orig,frame,timestampindataloader:
withtorch.no_grad():
image_features=torch.from_numpy(ov_compiled_image_encoder(image)[0])

image_features/=image_features.norm(dim=-1,keepdim=True)
probs=query_features.cpu().numpy()@image_features.cpu().numpy().T
probs=probs[0]

#Saveframesimilarityvalues
df=pd.DataFrame(
{
"Frame":frame.tolist(),
"Timestamp":torch.round(timestamp/1000,decimals=2).tolist(),
"Similarity":probs.tolist(),
}
)
res=pd.concat([res,df])

#Checkifframeisoverthreshold
fori,pinenumerate(probs):
ifp>thresh:
matches.append(to_pil_image(orig[i]))
matches_probs.append(p)

print(f"Frames:{frame.tolist()}-Probs:{probs}")

#Createplotofsimilarityvalues
lines=(
alt.Chart(res)
.mark_line(color="firebrick")
.encode(
alt.X("Timestamp",title="Timestamp(seconds)"),
alt.Y("Similarity",scale=alt.Scale(zero=False)),
)
).properties(width=600)
rule=alt.Chart().mark_rule(strokeDash=[6,3],size=2).encode(y=alt.datum(thresh))

selected_frames=np.argsort(-1*np.array(matches_probs))[:20]
matched_sorted_frames=[matches[idx]foridxinselected_frames]

return(
lines+rule,
matched_sorted_frames,
)#Onlyreturnupto20imagestonotcrashtheUI


classLoadVideo(Dataset):
def__init__(self,path,transforms,vid_stride=1):
self.transforms=transforms
self.vid_stride=vid_stride
self.cur_frame=0
self.cap=cv2.VideoCapture(path)
self.total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)/self.vid_stride)

def__getitem__(self,_):
#Readvideo
#Skipoverframes
for_inrange(self.vid_stride):
self.cap.grab()
self.cur_frame+=1

#Readframe
_,img=self.cap.retrieve()
timestamp=self.cap.get(cv2.CAP_PROP_POS_MSEC)

#ConverttoPIL
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=Image.fromarray(np.uint8(img))

#Applytransforms
img_t=self.transforms(img)

returnimg_t,to_tensor(img),self.cur_frame,timestamp

def__len__(self):
returnself.total_frames


desc_text="""
Searchthecontent'sofavideowithatextdescription.
__Note__:Longvideos(overafewminutes)maycauseUIperformanceissues.
"""
text_app=gr.Interface(
description=desc_text,
fn=run,
inputs=[
gr.Video(label="Video"),
gr.Textbox(label="TextSearchQuery"),
gr.Image(label="ImageSearchQuery",visible=False),
gr.Dropdown(
label="Model",
choices=available_converted_models,
value=model_checkpoint.value,
),
gr.Dropdown(label="Device",choices=available_devices,value=device.value),
gr.Slider(label="Threshold",maximum=1.0,value=0.2),
gr.Slider(label="Frame-rateStride",value=4,step=1),
gr.Slider(label="BatchSize",value=4,step=1),
],
outputs=[
gr.Plot(label="SimilarityPlot"),
gr.Gallery(label="MatchedFrames",columns=2,object_fit="contain",height="auto"),
],
examples=[[sample_path/"car-detection.mp4","whitecar"]],
allow_flagging="never",
)

desc_image="""
Searchthecontent'sofavideowithanimagequery.
__Note__:Longvideos(overafewminutes)maycauseUIperformanceissues.
"""
image_app=gr.Interface(
description=desc_image,
fn=run,
inputs=[
gr.Video(label="Video"),
gr.Textbox(label="TextSearchQuery",visible=False),
gr.Image(label="ImageSearchQuery",type="pil"),
gr.Dropdown(
label="Model",
choices=available_converted_models,
value=model_checkpoint.value,
),
gr.Dropdown(label="Device",choices=available_devices,value=device.value),
gr.Slider(label="Threshold",maximum=1.0,value=0.2),
gr.Slider(label="Frame-rateStride",value=4,step=1),
gr.Slider(label="BatchSize",value=4,step=1),
],
outputs=[
gr.Plot(label="SimilarityPlot"),
gr.Gallery(label="MatchedFrames",columns=2,object_fit="contain",height="auto"),
],
allow_flagging="never",
examples=[[sample_path/"coco.mp4",None,sample_path/"dog.png"]],
)
demo=gr.TabbedInterface(
interface_list=[text_app,image_app],
tab_names=["TextQuerySearch","ImageQuerySearch"],
title="CLIPVideoContentSearch",
)


try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/



..parsed-literal::

data/car-detection.mp4:0%||0.00/2.68M[00:00<?,?B/s]



..parsed-literal::

data/coco.mp4:0%||0.00/877k[00:00<?,?B/s]


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


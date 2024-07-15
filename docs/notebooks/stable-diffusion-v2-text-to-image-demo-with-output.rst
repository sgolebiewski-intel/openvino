StableDiffusionText-to-ImageDemo
===================================

StableDiffusionisaninnovativegenerativeAItechniquethatallowsus
togenerateandmanipulateimagesininterestingways,including
generatingimagefromtextandrestoringmissingpartsofpictures
(inpainting)!

StableDiffusionv2providesgreatfunctionalityoverpreviousversions,
includingbeingabletousemoredata,employmoretraining,andhas
lessrestrictivefilteringofthedataset.Allofthesefeaturesgiveus
promisingresultsforselectingawiderangeofinputtextprompts!

**Note:**Thisisashorterversionofthe
`stable-diffusion-v2-text-to-image<stable-diffusion-v2-with-output.html>`__
notebookfordemopurposesandtogetstartedquickly.Thisversiondoes
nothavethefullimplementationofthehelperutilitiesneededto
convertthemodelsfromPyTorchtoONNXtoOpenVINO,andtheOpenVINO
``OVStableDiffusionPipeline``withinthenotebookdirectly.Ifyouwould
liketoseethefullimplementationofstablediffusionfortextto
image,pleasevisit
`stable-diffusion-v2-text-to-image<stable-diffusion-v2-with-output.html>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Step0:Installandimport
prerequisites<#step-0-install-and-import-prerequisites>`__
-`Step1:StableDiffusionv2Fundamental
components<#step-1-stable-diffusion-v2-fundamental-components>`__

-`Step1.1:Retrievecomponentsfrom
HuggingFace<#step-1-1-retrieve-components-from-huggingface>`__

-`Step2:Convertthemodelsto
OpenVINO<#step-2-convert-the-models-to-openvino>`__
-`Step3:Text-to-ImageGenerationInference
Pipeline<#step-3-text-to-image-generation-inference-pipeline>`__

-`Step3.1:LoadandUnderstandTexttoImageOpenVINO
models<#step-3-1-load-and-understand-text-to-image-openvino-models>`__
-`Step3.2:Selectinference
device<#step-3-2-select-inference-device>`__
-`Step3.3:RunText-to-Image
generation<#step-3-3-run-text-to-image-generation>`__

Step0:Installandimportprerequisites
----------------------------------------

`backtotop⬆️<#table-of-contents>`__

ToworkwithStableDiffusionv2,wewilluseHuggingFace’s
`Diffusers<https://github.com/huggingface/diffusers>`__library.

ToexperimentwithStableDiffusionmodels,Diffusersexposesthe
`StableDiffusionPipeline<https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
and``StableDiffusionInpaintPipeline``,similartothe`otherDiffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.

..code::ipython3

%pipinstall-q"diffusers>=0.14.0""openvino>=2023.1.0""transformers>=4.31"accelerate"torch>=2.1"Pillowopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)
WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)
WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)
WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)
WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)
WARNING:Ignoringinvaliddistribution-orch(/home/ea/work/ov_venv/lib/python3.8/site-packages)

[notice]Anewreleaseofpipavailable:22.3->23.2.1
[notice]Toupdate,run:pipinstall--upgradepip
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Step1:StableDiffusionv2Fundamentalcomponents
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

StableDiffusionpipelinesforbothTexttoImageandInpaintingconsist
ofthreeimportantparts:

1.ATextEncodertocreateconditions:forexample,generatinganimage
fromatextpromptorperforminginpaintingtocreateaninfinite
zoomeffect.
2.AU-Netforstep-by-stepdenoisingoflatentimagerepresentation.
3.AnAutoencoder(VAE)fordecodingthelatentspacetoanimage.

Dependingonthepipeline,theparametersforthesepartscandiffer,
whichwe’llexploreinthisdemo!

Step1.1:RetrievecomponentsfromHuggingFace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Let’sstartbyretrievingthesecomponentsfromHuggingFace!

Thecodebelowdemonstrateshowtocreate``StableDiffusionPipeline``
using``stable-diffusion-2-1``.

..code::ipython3

#RetrievetheTexttoImageStableDiffusionpipelinecomponents
fromdiffusersimportStableDiffusionPipeline

pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")

#forreducingmemoryconsumptiongetallcomponentsfrompipelineindependently
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()

conf=pipe.scheduler.config

delpipe


..parsed-literal::

2023-09-1211:59:21.971103:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-09-1211:59:22.005818:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-09-1211:59:22.607625:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Loadingpipelinecomponents...:0%||0/6[00:00<?,?it/s]


Step2:ConvertthemodelstoOpenVINO
--------------------------------------

`backtotop⬆️<#table-of-contents>`__

Nowthatwe’veretrievedthethreepartsforbothofthesepipelines,we
nowneedto:

1.ConverttheoriginalPyTorchmodelstoOpenVINOformatusingModel
ConversionAPI

::

ov_model_part=ov.convert_model(model_part,example_input=input_data)

2.SaveOpenVINOmodelsondisk:

::

ov.save_model(ov_model_part,xml_file_path)

WecanthenrunourStableDiffusionv2texttoimageandinpainting
pipelinesinOpenVINOonourowndata!

..code::ipython3

frompathlibimportPath

#Defineadirtosavetext-to-imagemodels
txt2img_model_dir=Path("sd2.1")
txt2img_model_dir.mkdir(exist_ok=True)

..code::ipython3

fromimplementation.conversion_helper_utilsimport(
convert_encoder,
convert_unet,
convert_vae_decoder,
convert_vae_encoder,
)

#ConverttheText-to-ImagemodelsfromPyTorch->Onnx->OpenVINO
#1.ConverttheTextEncoder
txt_encoder_ov_path=txt2img_model_dir/"text_encoder.xml"
convert_encoder(text_encoder,txt_encoder_ov_path)
#2.ConverttheU-NET
unet_ov_path=txt2img_model_dir/"unet.xml"
convert_unet(unet,unet_ov_path,num_channels=4,width=96,height=96)
#3.ConverttheVAEencoder
vae_encoder_ov_path=txt2img_model_dir/"vae_encoder.xml"
convert_vae_encoder(vae,vae_encoder_ov_path,width=768,height=768)
#4.ConverttheVAEdecoder
vae_decoder_ov_path=txt2img_model_dir/"vae_decoder.xml"
convert_vae_decoder(vae,vae_decoder_ov_path,width=96,height=96)

Step3:Text-to-ImageGenerationInferencePipeline
---------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Step3.1:LoadandUnderstandTexttoImageOpenVINOmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Step3.2:Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=2,options=('CPU','GPU','AUTO'),value='AUTO')



Let’screateinstancesofourOpenVINOModelforTexttoImage.

..code::ipython3

text_enc=core.compile_model(txt_encoder_ov_path,device.value)

..code::ipython3

unet_model=core.compile_model(unet_ov_path,device.value)

..code::ipython3

vae_encoder=core.compile_model(vae_encoder_ov_path,device.value)
vae_decoder=core.compile_model(vae_decoder_ov_path,device.value)

Next,wewilldefineafewkeyelementstocreatetheinference
pipeline,asdepictedinthediagrambelow:

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/ec454103-0d28-48e3-a18e-b55da3fab381
:alt:text2img-stable-diffusion

text2img-stable-diffusion

Aspartofthe``OVStableDiffusionPipeline()``class:

1.Thestablediffusionpipelinetakesbothalatentseedandatext
promptasinput.Thelatentseedisusedtogeneraterandomlatent
imagerepresentations,andthetextpromptisprovidedtoOpenAI’s
CLIPtotransformthesetotextembeddings.

2.Next,theU-Netmodeliterativelydenoisestherandomlatentimage
representationswhilebeingconditionedonthetextembeddings.The
outputoftheU-Net,beingthenoiseresidual,isusedtocomputea
denoisedlatentimagerepresentationviaascheduleralgorithm.In
thiscaseweusethe``LMSDiscreteScheduler``.

..code::ipython3

fromdiffusers.schedulersimportLMSDiscreteScheduler
fromtransformersimportCLIPTokenizer
fromimplementation.ov_stable_diffusion_pipelineimportOVStableDiffusionPipeline

scheduler=LMSDiscreteScheduler.from_config(conf)
tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

ov_pipe=OVStableDiffusionPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=unet_model,
vae_encoder=vae_encoder,
vae_decoder=vae_decoder,
scheduler=scheduler,
)


..parsed-literal::

/home/ea/work/openvino_notebooks/notebooks/stable-diffusion-v2/implementation/ov_stable_diffusion_pipeline.py:10:FutureWarning:Importing`DiffusionPipeline`or`ImagePipelineOutput`fromdiffusers.pipeline_utilsisdeprecated.Pleaseimportfromdiffusers.pipelines.pipeline_utilsinstead.
fromdiffusers.pipeline_utilsimportDiffusionPipeline


Step3.3:RunText-to-Imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,let’sdefinesometextpromptsforimagegenerationandrunour
inferencepipeline.

Wecanalsochangeourrandomgeneratorseedforlatentstate
initializationandnumberofsteps(highersteps=moreprecise
results).

Exampleprompts:

-“valleyintheAlpsatsunset,epicvista,beautifullandscape,4k,
8k”
-"cityfilledwithcyborgs,modern,industrial,4k,8k

Toimproveimagegenerationquality,wecanusenegativeprompting.
Whilepositivepromptssteerdiffusiontowardtheimagesassociatedwith
it,negativepromptsdeclaresundesiredconceptsforthegeneration
image,e.g. ifwewanttohavecolorfulandbrightimages,agrayscale
imagewillberesultwhichwewanttoavoid.Inthiscase,agrayscale
canbetreatedasnegativeprompt.Thepositiveandnegativepromptare
inequalfooting.Youcanalwaysuseonewithorwithouttheother.More
explanationofhowitworkscanbefoundinthis
`article<https://stable-diffusion-art.com/how-negative-prompt-work/>`__.

..code::ipython3

importipywidgetsaswidgets

text_prompt=widgets.Textarea(
value="valleyintheAlpsatsunset,epicvista,beautifullandscape,4k,8k",
description="positiveprompt",
layout=widgets.Layout(width="auto"),
)
negative_prompt=widgets.Textarea(
value="frames,borderline,text,charachter,duplicate,error,outofframe,watermark,lowquality,ugly,deformed,blur",
description="negativeprompt",
layout=widgets.Layout(width="auto"),
)
num_steps=widgets.IntSlider(min=1,max=50,value=25,description="steps:")
seed=widgets.IntSlider(min=0,max=10000000,description="seed:",value=42)
widgets.VBox([text_prompt,negative_prompt,seed,num_steps])




..parsed-literal::

VBox(children=(Textarea(value='valleyintheAlpsatsunset,epicvista,beautifullandscape,4k,8k',descrip…



..code::ipython3

#Runinferencepipeline
result=ov_pipe(
text_prompt.value,
negative_prompt=negative_prompt.value,
num_inference_steps=num_steps.value,
seed=seed.value,
)



..parsed-literal::

0%||0/25[00:00<?,?it/s]


..code::ipython3

final_image=result["sample"][0]
final_image.save("result.png")
final_image




..image::stable-diffusion-v2-text-to-image-demo-with-output_files/stable-diffusion-v2-text-to-image-demo-with-output_24_0.png



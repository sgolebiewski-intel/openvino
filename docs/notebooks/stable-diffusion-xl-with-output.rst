ImagegenerationwithStableDiffusionXLandOpenVINO
======================================================

StableDiffusionXLorSDXListhelatestimagegenerationmodelthatis
tailoredtowardsmorephotorealisticoutputswithmoredetailedimagery
andcompositioncomparedtopreviousStableDiffusionmodels,including
StableDiffusion2.1.

WithStableDiffusionXLyoucannowmakemorerealisticimageswith
improvedfacegeneration,producelegibletextwithinimages,andcreate
moreaestheticallypleasingartusingshorterprompts.

..figure::https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/pipeline.png
:alt:pipeline

pipeline

`SDXL<https://arxiv.org/abs/2307.01952>`__consistsofan`ensembleof
experts<https://arxiv.org/abs/2211.01324>`__pipelineforlatent
diffusion:Inthefirststep,thebasemodelisusedtogenerate(noisy)
latents,whicharethenfurtherprocessedwitharefinementmodel
specializedforthefinaldenoisingsteps.Notethatthebasemodelcan
beusedasastandalonemoduleorinatwo-stagepipelineasfollows:
First,thebasemodelisusedtogeneratelatentsofthedesiredoutput
size.Inthesecondstep,weuseaspecializedhigh-resolutionmodeland
applyatechniquecalled
`SDEdit<https://arxiv.org/abs/2108.01073>`__\(alsoknownas“imageto
image”)tothelatentsgeneratedinthefirststep,usingthesame
prompt.

ComparedtopreviousversionsofStableDiffusion,SDXLleveragesa
threetimeslargerUNetbackbone:Theincreaseofmodelparametersis
mainlyduetomoreattentionblocksandalargercross-attentioncontext
asSDXLusesasecondtextencoder.Theauthorsdesignmultiplenovel
conditioningschemesandtrainSDXLonmultipleaspectratiosandalso
introducearefinementmodelthatisusedtoimprovethevisualfidelity
ofsamplesgeneratedbySDXLusingapost-hocimage-to-imagetechnique.
ThetestingofSDXLshowsdrasticallyimprovedperformancecomparedto
thepreviousversionsofStableDiffusionandachievesresults
competitivewiththoseofblack-boxstate-of-the-artimagegenerators.

Inthistutorial,weconsiderhowtoruntheSDXLmodelusingOpenVINO.

Wewilluseapre-trainedmodelfromthe`HuggingFace
Diffusers<https://huggingface.co/docs/diffusers/index>`__library.To
simplifytheuserexperience,the`HuggingFaceOptimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__libraryis
usedtoconvertthemodelstoOpenVINO™IRformat.

Thetutorialconsistsofthefollowingsteps:

-Installprerequisites
-DownloadtheStableDiffusionXLBasemodelfromapublicsource
usingthe`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-RunText2ImagegenerationpipelineusingStableDiffusionXLbase
-RunImage2ImagegenerationpipelineusingStableDiffusionXLbase
-DownloadandconverttheStableDiffusionXLRefinermodelfroma
publicsourceusingthe`OpenVINOintegrationwithHuggingFace
Optimum<https://huggingface.co/blog/openvino>`__.
-Run2-stagesStableDiffusionXLpipeline

..

**Note**:Somedemonstratedmodelscanrequireatleast64GBRAMfor
conversionandrunning.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Installprerequisites<#install-prerequisites>`__
-`SDXLBasemodel<#sdxl-base-model>`__

-`SelectinferencedeviceSDXLBase
model<#select-inference-device-sdxl-base-model>`__
-`RunText2Imagegeneration
pipeline<#run-text2image-generation-pipeline>`__
-`Text2imageGenerationInteractive
Demo<#text2image-generation-interactive-demo>`__
-`RunImage2Imagegeneration
pipeline<#run-image2image-generation-pipeline>`__

-`SelectinferencedeviceSDXLRefiner
model<#select-inference-device-sdxl-refiner-model>`__

-`Image2ImageGenerationInteractive
Demo<#image2image-generation-interactive-demo>`__

-`SDXLRefinermodel<#sdxl-refiner-model>`__

-`Selectinferencedevice<#select-inference-device>`__
-`RunText2Imagegenerationwith
Refinement<#run-text2image-generation-with-refinement>`__

Installprerequisites
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"torch>=2.1""diffusers>=0.18.0""invisible-watermark>=0.2.0""transformers>=4.33.0""accelerate""onnx""peft==0.6.2"
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git"
%pipinstall-q"openvino>=2023.1.0""gradio>=4.19""nncf>=2.9.0"

SDXLBasemodel
---------------

`backtotop⬆️<#table-of-contents>`__

Wewillstartwiththebasemodelpart,whichisresponsibleforthe
generationofimagesofthedesiredoutputsize.
`stable-diffusion-xl-base-1.0<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__
isavailablefordownloadingviathe`HuggingFace
hub<https://huggingface.co/models>`__.Italreadyprovidesa
ready-to-usemodelinOpenVINOformatcompatiblewith`Optimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__.

ToloadanOpenVINOmodelandrunaninferencewithOpenVINORuntime,
youneedtoreplacediffusers``StableDiffusionXLPipeline``withOptimum
``OVStableDiffusionXLPipeline``.IncaseyouwanttoloadaPyTorch
modelandconvertittotheOpenVINOformatonthefly,youcanset
``export=True``.

Youcansavethemodelondiskusingthe``save_pretrained``method.

..code::ipython3

frompathlibimportPath
fromoptimum.intel.openvinoimportOVStableDiffusionXLPipeline
importgc

model_id="stabilityai/stable-diffusion-xl-base-1.0"
model_dir=Path("openvino-sd-xl-base-1.0")

SelectinferencedeviceSDXLBasemodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



Pleaseselectbelowwhetheryouwouldliketouseweightcompressionto
reducememoryfootprint.`Optimum
Intel<https://huggingface.co/docs/optimum/en/intel/optimization_ov#weight-only-quantization>`__
supportsweightcompressionviaNNCFoutofthebox.For8-bit
compressionweprovide
``quantization_config=OVWeightQuantizationConfig(bits=8,...)``argument
to``from_pretrained()``methodcontainingnumberofbitsandother
compressionparameters.

..code::ipython3

compress_weights=widgets.Checkbox(
description="Applyweightcompression",
value=True,
)

compress_weights




..parsed-literal::

Checkbox(value=True,description='Applyweightcompression')



..code::ipython3

defget_quantization_config(compress_weights):
quantization_config=None
ifcompress_weights.value:
fromoptimum.intelimportOVWeightQuantizationConfig

quantization_config=OVWeightQuantizationConfig(bits=8)
returnquantization_config


quantization_config=get_quantization_config(compress_weights)

..code::ipython3

ifnotmodel_dir.exists():
text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_id,compile=False,device=device.value,quantization_config=quantization_config)
text2image_pipe.half()
text2image_pipe.save_pretrained(model_dir)
text2image_pipe.compile()
else:
text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(794/794)|100%(794/794)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(32/32)|100%(32/32)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(40/40)|100%(40/40)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(74/74)|100%(74/74)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|100%(195/195)|100%(195/195)|
+--------------+---------------------------+-----------------------------------+



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthevae_encodertoAUTO...
Compilingthetext_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...


RunText2Imagegenerationpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,wecanrunthemodelforthegenerationofimagesusingtext
prompts.Tospeedupevaluationandreducetherequiredmemorywe
decrease``num_inference_steps``andimagesize(using``height``and
``width``).Youcanmodifythemtosuityourneedsanddependonthe
targethardware.Wealsospecifieda``generator``parameterbasedona
numpyrandomstatewithaspecificseedforresultsreproducibility.

..code::ipython3

importnumpyasnp

prompt="cutecat4k,high-res,masterpiece,bestquality,softlighting,dynamicangle"
image=text2image_pipe(
prompt,
num_inference_steps=15,
height=512,
width=512,
generator=np.random.RandomState(314),
).images[0]
image.save("cat.png")
image



..parsed-literal::

0%||0/15[00:00<?,?it/s]




..image::stable-diffusion-xl-with-output_files/stable-diffusion-xl-with-output_13_1.png



Text2imageGenerationInteractiveDemo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr

iftext2image_pipeisNone:
text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)

prompt="cutecat4k,high-res,masterpiece,bestquality,softlighting,dynamicangle"


defgenerate_from_text(text,seed,num_steps):
result=text2image_pipe(
text,
num_inference_steps=num_steps,
generator=np.random.RandomState(seed),
height=512,
width=512,
).images[0]
returnresult


withgr.Blocks()asdemo:
withgr.Column():
positive_input=gr.Textbox(label="Textprompt")
withgr.Row():
seed_input=gr.Number(precision=0,label="Seed",value=42,minimum=0)
steps_input=gr.Slider(label="Steps",value=10)
btn=gr.Button()
out=gr.Image(label="Result",type="pil",width=512)
btn.click(generate_from_text,[positive_input,seed_input,steps_input],out)
gr.Examples(
[
[prompt,999,20],
[
"underwaterworldcoralreef,colorfuljellyfish,35mm,cinematiclighting,shallowdepthoffield,ultraquality,masterpiece,realistic",
89,
20,
],
[
"aphotorealistichappywhitepoodledog​​playinginthegrass,extremelydetailed,highres,8k,masterpiece,dynamicangle",
1569,
15,
],
[
"AstronautonMarswatchingsunset,bestquality,cinematiceffects,",
65245,
12,
],
[
"BlackandwhitestreetphotographyofarainynightinNewYork,reflectionsonwetpavement",
48199,
10,
],
],
[positive_input,seed_input,steps_input],
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
#ifyouwantcreatepubliclinkforsharingdemo,pleaseaddshare=True
demo.launch()

..code::ipython3

demo.close()
text2image_pipe=None
gc.collect();

RunImage2Imagegenerationpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WecanreusethealreadyconvertedmodelforrunningtheImage2Image
generationpipeline.Forthat,weshouldreplace
``OVStableDiffusionXLPipeline``with
``OVStableDiffusionXLImage2ImagePipeline``.

SelectinferencedeviceSDXLRefinermodel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



..code::ipython3

fromoptimum.intelimportOVStableDiffusionXLImg2ImgPipeline

image2image_pipe=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir,device=device.value)


..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthevae_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthetext_encodertoAUTO...


..code::ipython3

photo_prompt="professionalphotoofacat,extremelydetailed,hyperrealistic,bestquality,fullhd"
photo_image=image2image_pipe(
photo_prompt,
image=image,
num_inference_steps=25,
generator=np.random.RandomState(356),
).images[0]
photo_image.save("photo_cat.png")
photo_image



..parsed-literal::

0%||0/7[00:00<?,?it/s]




..image::stable-diffusion-xl-with-output_files/stable-diffusion-xl-with-output_21_1.png



Image2ImageGenerationInteractiveDemo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
fromdiffusers.utilsimportload_image
importnumpyasnp


load_image("https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/sd_xl/castle_friedrich.png").resize((512,512)).save(
"castle_friedrich.png"
)


ifimage2image_pipeisNone:
image2image_pipe=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir)


defgenerate_from_image(text,image,seed,num_steps):
result=image2image_pipe(
text,
image=image,
num_inference_steps=num_steps,
generator=np.random.RandomState(seed),
).images[0]
returnresult


withgr.Blocks()asdemo:
withgr.Column():
positive_input=gr.Textbox(label="Textprompt")
withgr.Row():
seed_input=gr.Number(precision=0,label="Seed",value=42,minimum=0)
steps_input=gr.Slider(label="Steps",value=10)
btn=gr.Button()
withgr.Row():
i2i_input=gr.Image(label="Inputimage",type="pil")
out=gr.Image(label="Result",type="pil",width=512)
btn.click(
generate_from_image,
[positive_input,i2i_input,seed_input,steps_input],
out,
)
gr.Examples(
[
["amazinglandscapefromlegends","castle_friedrich.png",971,60],
[
"MasterpieceofwatercolorpaintinginVanGoghstyle",
"cat.png",
37890,
40,
],
],
[positive_input,i2i_input,seed_input,steps_input],
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
#ifyouwantcreatepubliclinkforsharingdemo,pleaseaddshare=True
demo.launch()

..code::ipython3

demo.close()
delimage2image_pipe
gc.collect()

SDXLRefinermodel
------------------

`backtotop⬆️<#table-of-contents>`__

Aswediscussedabove,StableDiffusionXLcanbeusedina2-stages
approach:first,thebasemodelisusedtogeneratelatentsofthe
desiredoutputsize.Inthesecondstep,weuseaspecialized
high-resolutionmodelfortherefinementoflatentsgeneratedinthe
firststep,usingthesameprompt.TheStableDiffusionXLRefinermodel
isdesignedtotransformregularimagesintostunningmasterpieceswith
thehelpofuser-specifiedprompttext.Itcanbeusedtoimprovethe
qualityofimagegenerationaftertheStableDiffusionXLBase.The
refinermodelacceptslatentsproducedbytheSDXLbasemodelandtext
promptforimprovinggeneratedimage.

selectwhetheryouwouldliketouseweightcompressiontoreducememory
footprint

..code::ipython3

compress_weights

..code::ipython3

quantization_config=get_quantization_config(compress_weights)

..code::ipython3

fromoptimum.intelimport(
OVStableDiffusionXLImg2ImgPipeline,
OVStableDiffusionXLPipeline,
)
frompathlibimportPath

refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0"
refiner_model_dir=Path("openvino-sd-xl-refiner-1.0")


ifnotrefiner_model_dir.exists():
refiner=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_id,export=True,compile=False,quantization_config=quantization_config)
refiner.half()
refiner.save_pretrained(refiner_model_dir)
delrefiner
gc.collect()

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



RunText2ImagegenerationwithRefinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importnumpyasnp
importgc

model_dir=Path("openvino-sd-xl-base-1.0")
base=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)
prompt="cutecat4k,high-res,masterpiece,bestquality,softlighting,dynamicangle"
latents=base(
prompt,
num_inference_steps=15,
height=512,
width=512,
generator=np.random.RandomState(314),
output_type="latent",
).images[0]

delbase
gc.collect()


..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthetext_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthevae_encodertoAUTO...



..parsed-literal::

0%||0/15[00:00<?,?it/s]




..parsed-literal::

294



..code::ipython3

refiner=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_dir,device=device.value)


..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthevae_encodertoAUTO...


..code::ipython3

image=refiner(
prompt=prompt,
image=np.transpose(latents[None,:],(0,2,3,1)),
num_inference_steps=15,
generator=np.random.RandomState(314),
).images[0]
image.save("cat_refined.png")

image



..parsed-literal::

0%||0/4[00:00<?,?it/s]




..image::stable-diffusion-xl-with-output_files/stable-diffusion-xl-with-output_35_1.png



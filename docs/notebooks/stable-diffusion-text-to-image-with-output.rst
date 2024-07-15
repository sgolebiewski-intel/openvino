Text-to-ImageGenerationwithStableDiffusionandOpenVINO™
============================================================

StableDiffusionisatext-to-imagelatentdiffusionmodelcreatedby
theresearchersandengineersfrom
`CompVis<https://github.com/CompVis>`__,`Stability
AI<https://stability.ai/>`__and`LAION<https://laion.ai/>`__.Itis
trainedon512x512imagesfromasubsetofthe
`LAION-5B<https://laion.ai/blog/laion-5b/>`__database.Thismodeluses
afrozenCLIPViT-L/14textencodertoconditionthemodelontext
prompts.Withits860MUNetand123Mtextencoder.Seethe`model
card<https://huggingface.co/CompVis/stable-diffusion>`__formore
information.

Generaldiffusionmodelsaremachinelearningsystemsthataretrained
todenoiserandomgaussiannoisestepbystep,togettoasampleof
interest,suchasanimage.Diffusionmodelshaveshowntoachieve
state-of-the-artresultsforgeneratingimagedata.Butonedownsideof
diffusionmodelsisthatthereversedenoisingprocessisslow.In
addition,thesemodelsconsumealotofmemorybecausetheyoperatein
pixelspace,whichbecomesunreasonablyexpensivewhengenerating
high-resolutionimages.Therefore,itischallengingtotrainthese
modelsandalsousethemforinference.OpenVINObringscapabilitiesto
runmodelinferenceonIntelhardwareandopensthedoortothe
fantasticworldofdiffusionmodelsforeveryone!

Modelcapabilitiesarenotlimitedtext-to-imageonly,italsoisable
solveadditionaltasks,forexampletext-guidedimage-to-image
generationandinpainting.Thistutorialalsoconsidershowtorun
text-guidedimage-to-imagegenerationusingStableDiffusion.

Thisnotebookdemonstrateshowtoconvertandrunstablediffusionmodel
usingOpenVINO.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`PrepareInferencePipelines<#prepare-inference-pipelines>`__
-`Text-to-imagepipeline<#text-to-image-pipeline>`__

-`LoadStableDiffusionmodelandcreatetext-to-image
pipeline<#load-stable-diffusion-model-and-create-text-to-image-pipeline>`__
-`Text-to-Imagegeneration<#text-to-image-generation>`__
-`Interactivetext-to-image
demo<#interactive-text-to-image-demo>`__

-`Image-to-Imagepipeline<#image-to-image-pipeline>`__

-`Createimage-to-Image
pipeline<#create-image-to-image-pipeline>`__
-`Image-to-Imagegeneration<#image-to-image-generation>`__
-`Interactiveimage-to-image
demo<#interactive-image-to-image-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0""git+https://github.com/huggingface/optimum-intel.git"
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"diffusers>=0.9.0""torch>=2.1"
%pipinstall-q"huggingface-hub>=0.9.1"
%pipinstall-q"gradio>=4.19"
%pipinstall-qtransformersPillowopencv-pythontqdm


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


PrepareInferencePipelines
---------------------------

`backtotop⬆️<#table-of-contents>`__

Letusnowtakeacloserlookathowthemodelworksininferenceby
illustratingthelogicalflow.

..figure::https://user-images.githubusercontent.com/29454499/260981188-c112dd0a-5752-4515-adca-8b09bea5d14a.png
:alt:sd-pipeline

sd-pipeline

Asyoucanseefromthediagram,theonlydifferencebetween
Text-to-Imageandtext-guidedImage-to-Imagegenerationinapproachis
howinitiallatentstateisgenerated.IncaseofImage-to-Image
generation,youadditionallyhaveanimageencodedbyVAEencodermixed
withthenoiseproducedbyusinglatentseed,whileinText-to-Imageyou
useonlynoiseasinitiallatentstate.Thestablediffusionmodeltakes
bothalatentimagerepresentationofsize:math:`64\times64`anda
textpromptistransformedtotextembeddingsofsize
:math:`77\times768`viaCLIP’stextencoderasaninput.

Next,theU-Netiteratively*denoises*therandomlatentimage
representationswhilebeingconditionedonthetextembeddings.The
outputoftheU-Net,beingthenoiseresidual,isusedtocomputea
denoisedlatentimagerepresentationviaascheduleralgorithm.Many
differentscheduleralgorithmscanbeusedforthiscomputation,each
havingitsprosandcons.ForStableDiffusion,itisrecommendedtouse
oneof:

-`PNDM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py>`__
-`DDIM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py>`__
-`K-LMS
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py>`__\(you
willuseitinyourpipeline)

Theoryonhowthescheduleralgorithmfunctionworksisoutofscopefor
thisnotebook.Nonetheless,inshort,youshouldrememberthatyou
computethepredicteddenoisedimagerepresentationfromtheprevious
noiserepresentationandthepredictednoiseresidual.Formore
information,refertotherecommended`ElucidatingtheDesignSpaceof
Diffusion-BasedGenerativeModels<https://arxiv.org/abs/2206.00364>`__

The*denoising*processisrepeatedgivennumberoftimes(bydefault
50)tostep-by-stepretrievebetterlatentimagerepresentations.When
complete,thelatentimagerepresentationisdecodedbythedecoderpart
ofthevariationalautoencoder.

Text-to-imagepipeline
----------------------

`backtotop⬆️<#table-of-contents>`__

LoadStableDiffusionmodelandcreatetext-to-imagepipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WewillloadoptimizedStableDiffusionmodelfromtheHuggingFaceHub
andcreatepipelinetorunaninferencewithOpenVINORuntimeby
`Optimum
Intel<https://huggingface.co/docs/optimum/intel/inference#stable-diffusion>`__.

ForrunningtheStableDiffusionmodelwithOptimumIntel,wewilluse
the``optimum.intel.OVStableDiffusionPipeline``class,whichrepresents
theinferencepipeline.``OVStableDiffusionPipeline``initializedbythe
``from_pretrained``method.Itsupportson-the-flyconversionmodels
fromPyTorchusingthe``export=True``parameter.Aconvertedmodelcan
besavedondiskusingthe``save_pretrained``methodforthenext
running.

WhenStableDiffusionmodelsareexportedtotheOpenVINOformat,they
aredecomposedintothreecomponentsthatconsistoffourmodels
combinedduringinferenceintothepipeline:

-Thetextencoder

-Thetext-encoderisresponsiblefortransformingtheinput
prompt(forexample“aphotoofanastronautridingahorse”)into
anembeddingspacethatcanbeunderstoodbytheU-Net.Itis
usuallyasimpletransformer-basedencoderthatmapsasequenceof
inputtokenstoasequenceoflatenttextembeddings.

-TheU-NET

-Modelpredictsthe``sample``stateforthenextstep.

-TheVAEencoder

-Theencoderisusedtoconverttheimageintoalowdimensional
latentrepresentation,whichwillserveastheinputtotheU-Net
model.

-TheVAEdecoder

-Thedecodertransformsthelatentrepresentationbackintoan
image.

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

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

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

fromoptimum.intel.openvinoimportOVStableDiffusionPipeline
frompathlibimportPath

DEVICE=device.value

MODEL_ID="prompthero/openjourney"
MODEL_DIR=Path("diffusion_pipeline")

ifnotMODEL_DIR.exists():
ov_pipe=OVStableDiffusionPipeline.from_pretrained(MODEL_ID,export=True,device=DEVICE,compile=False)
ov_pipe.save_pretrained(MODEL_DIR)
else:
ov_pipe=OVStableDiffusionPipeline.from_pretrained(MODEL_DIR,device=DEVICE,compile=False)

ov_pipe.compile()


..parsed-literal::

Compilingthevae_decodertoCPU...
CompilingtheunettoCPU...
Compilingthetext_encodertoCPU...
Compilingthevae_encodertoCPU...


Text-to-Imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,youcandefineatextpromptforimagegenerationandruninference
pipeline.

**Note**:Considerincreasing``steps``togetmorepreciseresults.
Asuggestedvalueis``50``,butitwilltakelongertimetoprocess.

..code::ipython3

sample_text=(
"cyberpunkcityscapelikeTokyoNewYorkwithtallbuildingsatduskgoldenhourcinematiclighting,epiccomposition."
"Agoldendaylight,hyper-realisticenvironment."
"Hyperandintricatedetail,photo-realistic."
"Cinematicandvolumetriclight."
"Epicconceptart."
"OctanerenderandUnrealEngine,trendingonartstation"
)
text_prompt=widgets.Text(value=sample_text,description="yourtext")
num_steps=widgets.IntSlider(min=1,max=50,value=20,description="steps:")
seed=widgets.IntSlider(min=0,max=10000000,description="seed:",value=42)
widgets.VBox([text_prompt,num_steps,seed])




..parsed-literal::

VBox(children=(Text(value='cyberpunkcityscapelikeTokyoNewYorkwithtallbuildingsatduskgoldenhourci…



..code::ipython3

print("Pipelinesettings")
print(f"Inputtext:{text_prompt.value}")
print(f"Seed:{seed.value}")
print(f"Numberofsteps:{num_steps.value}")


..parsed-literal::

Pipelinesettings
Inputtext:cyberpunkcityscapelikeTokyoNewYorkwithtallbuildingsatduskgoldenhourcinematiclighting,epiccomposition.Agoldendaylight,hyper-realisticenvironment.Hyperandintricatedetail,photo-realistic.Cinematicandvolumetriclight.Epicconceptart.OctanerenderandUnrealEngine,trendingonartstation
Seed:42
Numberofsteps:20


Let’sgenerateanimageandsavethegenerationresults.Thepipeline
returnsoneorseveralresults:``images``containsfinalgenerated
image.Togetmorethanoneresult,youcansetthe
``num_images_per_prompt``parameter.

..code::ipython3

importnumpyasnp

np.random.seed(seed.value)

result=ov_pipe(text_prompt.value,num_inference_steps=num_steps.value)

final_image=result["images"][0]
final_image.save("result.png")



..parsed-literal::

0%||0/21[00:00<?,?it/s]


Nowisshowtime!

..code::ipython3

text="\n\t".join(text_prompt.value.split("."))
print("Inputtext:")
print("\t"+text)
display(final_image)


..parsed-literal::

Inputtext:
	cyberpunkcityscapelikeTokyoNewYorkwithtallbuildingsatduskgoldenhourcinematiclighting,epiccomposition
	Agoldendaylight,hyper-realisticenvironment
	Hyperandintricatedetail,photo-realistic
	Cinematicandvolumetriclight
	Epicconceptart
	OctanerenderandUnrealEngine,trendingonartstation



..image::stable-diffusion-text-to-image-with-output_files/stable-diffusion-text-to-image-with-output_16_1.png


Nice.Asyoucansee,thepicturehasquiteahighdefinition🔥.

Interactivetext-to-imagedemo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defgenerate_from_text(text,seed,num_steps,_=gr.Progress(track_tqdm=True)):
np.random.seed(seed)
result=ov_pipe(text,num_inference_steps=num_steps)
returnresult["images"][0]


withgr.Blocks()asdemo:
withgr.Tab("Text-to-Imagegeneration"):
withgr.Row():
withgr.Column():
text_input=gr.Textbox(lines=3,label="Text")
seed_input=gr.Slider(0,10000000,value=42,step=1,label="Seed")
steps_input=gr.Slider(1,50,value=20,step=1,label="Steps")
out=gr.Image(label="Result",type="pil")
btn=gr.Button()
btn.click(generate_from_text,[text_input,seed_input,steps_input],out)
gr.Examples([[sample_text,42,20]],[text_input,seed_input,steps_input])
try:
demo.queue().launch()
exceptException:
demo.queue().launch(share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

..code::ipython3

demo.close()
delov_pipe
np.random.seed(None)

Image-to-Imagepipeline
-----------------------

`backtotop⬆️<#table-of-contents>`__

Createimage-to-Imagepipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ForrunningtheStableDiffusionmodelwithOptimumIntel,wewilluse
the``optimum.intel.OVStableDiffusionImg2ImgPipeline``class,which
representstheinferencepipeline.Wewillusethesamemodelasfor
text-to-imagepipeline.Themodelhasalreadybeendownloadedfromthe
HuggingFaceHubandconvertedtoOpenVINOIRformatonprevioussteps,
sowecanjustloadit.

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



..code::ipython3

fromoptimum.intel.openvinoimportOVStableDiffusionImg2ImgPipeline
frompathlibimportPath

DEVICE=device.value

ov_pipe_i2i=OVStableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DIR,device=DEVICE,compile=False)
ov_pipe_i2i.compile()


..parsed-literal::

Compilingthevae_decodertoCPU...
CompilingtheunettoCPU...
Compilingthetext_encodertoCPU...
Compilingthevae_encodertoCPU...


Image-to-Imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Image-to-Imagegeneration,additionallytotextprompt,requires
providinginitialimage.Optionally,youcanalsochange``strength``
parameter,whichisavaluebetween0.0and1.0,thatcontrolsthe
amountofnoisethatisaddedtotheinputimage.Valuesthatapproach
1.0enablelotsofvariationsbutwillalsoproduceimagesthatarenot
semanticallyconsistentwiththeinput.

..code::ipython3

text_prompt_i2i=widgets.Text(value="amazingwatercolorpainting",description="yourtext")
num_steps_i2i=widgets.IntSlider(min=1,max=50,value=10,description="steps:")
seed_i2i=widgets.IntSlider(min=0,max=1024,description="seed:",value=42)
image_widget=widgets.FileUpload(
accept="",
multiple=False,
description="Uploadimage",
)
strength=widgets.FloatSlider(min=0,max=1,description="strength:",value=0.5)
widgets.VBox([text_prompt_i2i,seed_i2i,num_steps_i2i,image_widget,strength])




..parsed-literal::

VBox(children=(Text(value='amazingwatercolorpainting',description='yourtext'),IntSlider(value=42,descrip…



..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

..code::ipython3

importio
importPIL

default_image_path=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
filename="coco.jpg",
)

#readuploadedimage
image=PIL.Image.open(io.BytesIO(image_widget.value[-1]["content"])ifimage_widget.valueelsestr(default_image_path))
print("Pipelinesettings")
print(f"Inputtext:{text_prompt_i2i.value}")
print(f"Seed:{seed_i2i.value}")
print(f"Numberofsteps:{num_steps_i2i.value}")
print(f"Strength:{strength.value}")
print("Inputimage:")
display(image)


..parsed-literal::

'coco.jpg'alreadyexists.
Pipelinesettings
Inputtext:amazingwatercolorpainting
Seed:42
Numberofsteps:20
Strength:0.4
Inputimage:



..image::stable-diffusion-text-to-image-with-output_files/stable-diffusion-text-to-image-with-output_27_1.png


..code::ipython3

importPIL
importnumpyasnp


defscale_fit_to_window(dst_width:int,dst_height:int,image_width:int,image_height:int):
"""
Preprocessinghelperfunctionforcalculatingimagesizeforresizewithpeservingoriginalaspectratio
andfittingimagetospecificwindowsize

Parameters:
dst_width(int):destinationwindowwidth
dst_height(int):destinationwindowheight
image_width(int):sourceimagewidth
image_height(int):sourceimageheight
Returns:
result_width(int):calculatedwidthforresize
result_height(int):calculatedheightforresize
"""
im_scale=min(dst_height/image_height,dst_width/image_width)
returnint(im_scale*image_width),int(im_scale*image_height)


defpreprocess(image:PIL.Image.Image):
"""
Imagepreprocessingfunction.TakesimageinPIL.Imageformat,resizesittokeepaspectrationandfitstomodelinputwindow512x512,
thenconvertsittonp.ndarrayandaddspaddingwithzerosonrightorbottomsideofimage(dependsfromaspectratio),afterthat
convertsdatatofloat32datatypeandchangerangeofvaluesfrom[0,255]to[-1,1].
Thefunctionreturnspreprocessedinputtensorandpaddingsize,whichcanbeusedinpostprocessing.

Parameters:
image(PIL.Image.Image):inputimage
Returns:
image(np.ndarray):preprocessedimagetensor
meta(Dict):dictionarywithpreprocessingmetadatainfo
"""
src_width,src_height=image.size
dst_width,dst_height=scale_fit_to_window(512,512,src_width,src_height)
image=np.array(image.resize((dst_width,dst_height),resample=PIL.Image.Resampling.LANCZOS))[None,:]
pad_width=512-dst_width
pad_height=512-dst_height
pad=((0,0),(0,pad_height),(0,pad_width),(0,0))
image=np.pad(image,pad,mode="constant")
image=image.astype(np.float32)/255.0
image=2.0*image-1.0
returnimage,{"padding":pad,"src_width":src_width,"src_height":src_height}


defpostprocess(image:PIL.Image.Image,orig_width:int,orig_height:int):
"""
Imagepostprocessingfunction.TakesimageinPIL.Imageformatandmetricsoforiginalimage.Imageiscroppedandresizedtorestoreinitialsize.

Parameters:
image(PIL.Image.Image):inputimage
orig_width(int):originalimagewidth
orig_height(int):originalimageheight
Returns:
image(PIL.Image.Image):postprocessimage
"""
src_width,src_height=image.size
dst_width,dst_height=scale_fit_to_window(src_width,src_height,orig_width,orig_height)
image=image.crop((0,0,dst_width,dst_height))
image=image.resize((orig_width,orig_height))
returnimage

..code::ipython3

preprocessed_image,meta_data=preprocess(image)

np.random.seed(seed_i2i.value)

processed_image=ov_pipe_i2i(text_prompt_i2i.value,preprocessed_image,num_inference_steps=num_steps_i2i.value,strength=strength.value)



..parsed-literal::

0%||0/9[00:00<?,?steps/s]


..code::ipython3

final_image_i2i=postprocess(processed_image["images"][0],meta_data["src_width"],meta_data["src_height"])
final_image_i2i.save("result_i2i.png")

..code::ipython3

text_i2i="\n\t".join(text_prompt_i2i.value.split("."))
print("Inputtext:")
print("\t"+text_i2i)
display(final_image_i2i)


..parsed-literal::

Inputtext:
	amazingwatercolorpainting



..image::stable-diffusion-text-to-image-with-output_files/stable-diffusion-text-to-image-with-output_31_1.png


Interactiveimage-to-imagedemo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defgenerate_from_image(img,text,seed,num_steps,strength,_=gr.Progress(track_tqdm=True)):
preprocessed_img,meta_data=preprocess(img)
np.random.seed(seed)
result=ov_pipe_i2i(text,preprocessed_img,num_inference_steps=num_steps,strength=strength)
result_img=postprocess(result["images"][0],meta_data["src_width"],meta_data["src_height"])
returnresult_img


withgr.Blocks()asdemo:
withgr.Tab("Image-to-Imagegeneration"):
withgr.Row():
withgr.Column():
i2i_input=gr.Image(label="Image",type="pil")
i2i_text_input=gr.Textbox(lines=3,label="Text")
i2i_seed_input=gr.Slider(0,1024,value=42,step=1,label="Seed")
i2i_steps_input=gr.Slider(1,50,value=10,step=1,label="Steps")
strength_input=gr.Slider(0,1,value=0.5,label="Strength")
i2i_out=gr.Image(label="Result")
i2i_btn=gr.Button()
sample_i2i_text="amazingwatercolorpainting"
i2i_btn.click(
generate_from_image,
[
i2i_input,
i2i_text_input,
i2i_seed_input,
i2i_steps_input,
strength_input,
],
i2i_out,
)
gr.Examples(
[[str(default_image_path),sample_i2i_text,42,10,0.5]],
[
i2i_input,
i2i_text_input,
i2i_seed_input,
i2i_steps_input,
strength_input,
],
)

try:
demo.queue().launch()
exceptException:
demo.queue().launch(share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

..code::ipython3

demo.close()

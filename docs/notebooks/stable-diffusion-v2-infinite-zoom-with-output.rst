InfiniteZoomStableDiffusionv2andOpenVINO™
===============================================

StableDiffusionv2isthenextgenerationofStableDiffusionmodela
Text-to-Imagelatentdiffusionmodelcreatedbytheresearchersand
engineersfrom`StabilityAI<https://stability.ai/>`__and
`LAION<https://laion.ai/>`__.

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

Inpreviousnotebooks,wealreadydiscussedhowtorun`Text-to-Image
generationandImage-to-ImagegenerationusingStableDiffusion
v1<stable-diffusion-text-to-image-with-output.html>`__
and`controllingitsgenerationprocessusing
ControlNet<./controlnet-stable-diffusion/controlnet-stable-diffusion.ipynb>`__.
NowisturnofStableDiffusionv2.

StableDiffusionv2:What’snew?
--------------------------------

Thenewstablediffusionmodeloffersabunchofnewfeaturesinspired
bytheothermodelsthathaveemergedsincetheintroductionofthe
firstiteration.Someofthefeaturesthatcanbefoundinthenewmodel
are:

-Themodelcomeswithanewrobustencoder,OpenCLIP,createdbyLAION
andaidedbyStabilityAI;thisversionv2significantlyenhancesthe
producedphotosovertheV1versions.
-Themodelcannowgenerateimagesina768x768resolution,offering
moreinformationtobeshowninthegeneratedimages.
-Themodelfinetunedwith
`v-objective<https://arxiv.org/abs/2202.00512>`__.The
v-parameterizationisparticularlyusefulfornumericalstability
throughoutthediffusionprocesstoenableprogressivedistillation
formodels.Formodelsthatoperateathigherresolution,itisalso
discoveredthatthev-parameterizationavoidscolorshifting
artifactsthatareknowntoaffecthighresolutiondiffusionmodels,
andinthevideosettingitavoidstemporalcolorshiftingthat
sometimesappearswithepsilon-predictionusedinStableDiffusion
v1.
-Themodelalsocomeswithanewdiffusionmodelcapableofrunning
upscalingontheimagesgenerated.Upscaledimagescanbeadjustedup
to4timestheoriginalimage.Providedasseparatedmodel,formore
detailspleasecheck
`stable-diffusion-x4-upscaler<https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler>`__
-Themodelcomeswithanewrefineddeptharchitecturecapableof
preservingcontextfrompriorgenerationlayersinanimage-to-image
setting.Thisstructurepreservationhelpsgenerateimagesthat
preservingformsandshadowofobjects,butwithdifferentcontent.
-Themodelcomeswithanupdatedinpaintingmodulebuiltuponthe
previousmodel.Thistext-guidedinpaintingmakesswitchingoutparts
intheimageeasierthanbefore.

ThisnotebookdemonstrateshowtodownloadthemodelfromtheHugging
FaceHubandconvertedtoOpenVINOIRformatwith`Optimum
Intel<https://huggingface.co/docs/optimum/intel/inference#stable-diffusion>`__.
Andhowtousethemodeltogeneratesequenceofimagesforinfinite
zoomvideoeffect.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`StableDiffusionv2InfiniteZoom
Showcase<#stable-diffusion-v2-infinite-zoom-showcase>`__

-`StableDiffusionTextguided
Inpainting<#stable-diffusion-text-guided-inpainting>`__

-`Prerequisites<#prerequisites>`__
-`LoadStableDiffusionInpaintpipelineusingOptimum
Intel<#load-stable-diffusion-inpaint-pipeline-using-optimum-intel>`__
-`ZoomVideoGeneration<#zoom-video-generation>`__
-`RunInfiniteZoomvideo
generation<#run-infinite-zoom-video-generation>`__

StableDiffusionv2InfiniteZoomShowcase
------------------------------------------

`backtotop⬆️<#table-of-contents>`__

InthistutorialweconsiderhowtouseStableDiffusionv2modelfor
generationsequenceofimagesforinfinitezoomvideoeffect.Todo
this,wewillneed
`stabilityai/stable-diffusion-2-inpainting<https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__
model.

StableDiffusionTextguidedInpainting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inimageediting,inpaintingisaprocessofrestoringmissingpartsof
pictures.Mostcommonlyappliedtoreconstructingolddeteriorated
images,removingcracks,scratches,dustspots,orred-eyesfrom
photographs.

ButwiththepowerofAIandtheStableDiffusionmodel,inpaintingcan
beusedtoachievemorethanthat.Forexample,insteadofjust
restoringmissingpartsofanimage,itcanbeusedtorendersomething
entirelynewinanypartofanexistingpicture.Onlyyourimagination
limitsit.

TheworkflowdiagramexplainshowStableDiffusioninpaintingpipeline
forinpaintingworks:

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/9ac6de45-186f-4a3c-aa20-825825a337eb
:alt:sd2-inpainting

sd2-inpainting

ThepipelinehasalotofcommonwithText-to-Imagegenerationpipeline
discussedinprevioussection.Additionallytotextprompt,pipeline
acceptsinputsourceimageandmaskwhichprovidesanareaofimage
whichshouldbemodified.MaskedimageencodedbyVAEencoderinto
latentdiffusionspaceandconcatenatedwithrandomlygenerated(on
initialsteponly)orproducedbyU-Netlatentgeneratedimage
representationandusedasinputfornextstepdenoising.

Usingthisinpaintingfeature,decreasingimagebycertainmarginand
maskingthisborderforeverynewframewecancreateinterestingZoom
Outvideobasedonourprompt.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

installrequiredpackages

..code::ipython3

%pipinstall-q"diffusers>=0.14.0""transformers>=4.25.1""gradio>=4.19""openvino>=2024.2.0""torch>=2.1"Pillowopencv-python"git+https://github.com/huggingface/optimum-intel.git"--extra-index-urlhttps://download.pytorch.org/whl/cpu

LoadStableDiffusionInpaintpipelineusingOptimumIntel
----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

WewillloadoptimizedStableDiffusionmodelfromtheHuggingFaceHub
andcreatepipelinetorunaninferencewithOpenVINORuntimeby
`Optimum
Intel<https://huggingface.co/docs/optimum/intel/inference#stable-diffusion>`__.

ForrunningtheStableDiffusionmodelwithOptimumIntel,wewilluse
theoptimum.intel.OVStableDiffusionInpaintPipelineclass,which
representstheinferencepipeline.OVStableDiffusionInpaintPipeline
initializedbythefrom_pretrainedmethod.Itsupportson-the-fly
conversionmodelsfromPyTorchusingtheexport=Trueparameter.A
convertedmodelcanbesavedondiskusingthesave_pretrainedmethod
forthenextrunning.

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

..code::ipython3

fromoptimum.intel.openvinoimportOVStableDiffusionInpaintPipeline
frompathlibimportPath

DEVICE=device.value

MODEL_ID="stabilityai/stable-diffusion-2-inpainting"
MODEL_DIR=Path("sd2_inpainting")

ifnotMODEL_DIR.exists():
ov_pipe=OVStableDiffusionInpaintPipeline.from_pretrained(MODEL_ID,export=True,device=DEVICE,compile=False)
ov_pipe.save_pretrained(MODEL_DIR)
else:
ov_pipe=OVStableDiffusionInpaintPipeline.from_pretrained(MODEL_DIR,device=DEVICE,compile=False)

ov_pipe.compile()

ZoomVideoGeneration
---------------------

`backtotop⬆️<#table-of-contents>`__

Forachievingzoomeffect,wewilluseinpaintingtoexpandimages
beyondtheiroriginalborders.Werunour
``OVStableDiffusionInpaintPipeline``intheloop,whereeachnextframe
willaddedgestoprevious.Theframegenerationprocessillustratedon
diagrambelow:

..figure::https://user-images.githubusercontent.com/29454499/228739686-436f2759-4c79-42a2-a70f-959fb226834c.png
:alt:framegeneration)

framegeneration)

Afterprocessingcurrentframe,wedecreasesizeofcurrentimageby
masksizepixelsfromeachsideanduseitasinputfornextstep.
Changingsizeofmaskwecaninfluencethesizeofpaintingareaand
imagescaling.

Thereare2zoomingdirections:

-ZoomOut-moveawayfromobject
-ZoomIn-moveclosertoobject

ZoomInwillbeprocessedinthesamewayasZoomOut,butafter
generationisfinished,werecordframesinreversedorder.

..code::ipython3

fromtypingimportList,Union

importPIL
importcv2
fromtqdmimporttrange
importnumpyasnp


defgenerate_video(
pipe,
prompt:Union[str,List[str]],
negative_prompt:Union[str,List[str]],
guidance_scale:float=7.5,
num_inference_steps:int=20,
num_frames:int=20,
mask_width:int=128,
seed:int=9999,
zoom_in:bool=False,
):
"""
Zoomvideogenerationfunction

Parameters:
pipe(OVStableDiffusionInpaintingPipeline):inpaintingpipeline.
prompt(strorList[str]):Thepromptorpromptstoguidetheimagegeneration.
negative_prompt(strorList[str]):Thenegativepromptorpromptstoguidetheimagegeneration.
guidance_scale(float,*optional*,defaultsto7.5):
GuidancescaleasdefinedinClassifier-FreeDiffusionGuidance(https://arxiv.org/abs/2207.12598).
guidance_scaleisdefinedas`w`ofequation2.
Higherguidancescaleencouragestogenerateimagesthatarecloselylinkedtothetextprompt,
usuallyattheexpenseoflowerimagequality.
num_inference_steps(int,*optional*,defaultsto50):Thenumberofdenoisingstepsforeachframe.Moredenoisingstepsusuallyleadtoahigherqualityimageattheexpenseofslowerinference.
num_frames(int,*optional*,20):numberframesforvideo.
mask_width(int,*optional*,128):sizeofbordermaskforinpaintingoneachstep.
zoom_in(bool,*optional*,False):zoommodeZoomInorZoomOut.
Returns:
output_path(str):Pathwheregeneratedvideoloacated.
"""

height=512
width=height

current_image=PIL.Image.new(mode="RGBA",size=(height,width))
mask_image=np.array(current_image)[:,:,3]
mask_image=PIL.Image.fromarray(255-mask_image).convert("RGB")
current_image=current_image.convert("RGB")
init_images=pipe(
prompt=prompt,
negative_prompt=negative_prompt,
image=current_image,
guidance_scale=guidance_scale,
mask_image=mask_image,
num_inference_steps=num_inference_steps,
).images

image_grid(init_images,rows=1,cols=1)

num_outpainting_steps=num_frames
num_interpol_frames=30

current_image=init_images[0]
all_frames=[]
all_frames.append(current_image)
foriintrange(
num_outpainting_steps,
desc=f"Generating{num_outpainting_steps}additionalimages...",
):
prev_image_fix=current_image

prev_image=shrink_and_paste_on_blank(current_image,mask_width)

current_image=prev_image

#createmask(blackimagewithwhitemask_widthwidthedges)
mask_image=np.array(current_image)[:,:,3]
mask_image=PIL.Image.fromarray(255-mask_image).convert("RGB")

#inpaintingstep
current_image=current_image.convert("RGB")
images=pipe(
prompt=prompt,
negative_prompt=negative_prompt,
image=current_image,
guidance_scale=guidance_scale,
mask_image=mask_image,
num_inference_steps=num_inference_steps,
).images
current_image=images[0]
current_image.paste(prev_image,mask=prev_image)

#interpolationstepsbewteen2inpaintedimages(=sequentialzoomandcrop)
forjinrange(num_interpol_frames-1):
interpol_image=current_image
interpol_width=round((1-(1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2)
interpol_image=interpol_image.crop(
(
interpol_width,
interpol_width,
width-interpol_width,
height-interpol_width,
)
)

interpol_image=interpol_image.resize((height,width))

#pastethehigherresolutionpreviousimageinthemiddletoavoiddropinqualitycausedbyzooming
interpol_width2=round((1-(height-2*mask_width)/(height-2*interpol_width))/2*height)
prev_image_fix_crop=shrink_and_paste_on_blank(prev_image_fix,interpol_width2)
interpol_image.paste(prev_image_fix_crop,mask=prev_image_fix_crop)
all_frames.append(interpol_image)
all_frames.append(current_image)

video_file_name=f"infinite_zoom_{'in'ifzoom_inelse'out'}"
fps=30
save_path=video_file_name+".mp4"
write_video(save_path,all_frames,fps,reversed_order=zoom_in)
returnsave_path

..code::ipython3

defshrink_and_paste_on_blank(current_image:PIL.Image.Image,mask_width:int):
"""
Decreasessizeofcurrent_imagebymask_widthpixelsfromeachside,
thenaddsamask_widthwidthtransparentframe,
sothattheimagethefunctionreturnsisthesamesizeastheinput.

Parameters:
current_image(PIL.Image):inputimagetotransform
mask_width(int):widthinpixelstoshrinkfromeachside
Returns:
prev_image(PIL.Image):resizedimagewithextendedborders
"""

height=current_image.height
width=current_image.width

#shrinkdownbymask_width
prev_image=current_image.resize((height-2*mask_width,width-2*mask_width))
prev_image=prev_image.convert("RGBA")
prev_image=np.array(prev_image)

#createblanknon-transparentimage
blank_image=np.array(current_image.convert("RGBA"))*0
blank_image[:,:,3]=1

#pasteshrinkedontoblank
blank_image[mask_width:height-mask_width,mask_width:width-mask_width,:]=prev_image
prev_image=PIL.Image.fromarray(blank_image)

returnprev_image


defimage_grid(imgs:List[PIL.Image.Image],rows:int,cols:int):
"""
Insertimagestogrid

Parameters:
imgs(List[PIL.Image.Image]):listofimagesformakinggrid
rows(int):numberofrowsingrid
cols(int):numberofcolumnsingrid
Returns:
grid(PIL.Image):imagewithinputimagescollage
"""
assertlen(imgs)==rows*cols

w,h=imgs[0].size
grid=PIL.Image.new("RGB",size=(cols*w,rows*h))

fori,imginenumerate(imgs):
grid.paste(img,box=(i%cols*w,i//cols*h))
returngrid


defwrite_video(
file_path:str,
frames:List[PIL.Image.Image],
fps:float,
reversed_order:bool=True,
gif:bool=True,
):
"""
Writesframestoanmp4videofileandoptionalytogif

Parameters:
file_path(str):Pathtooutputvideo,mustendwith.mp4
frames(ListofPIL.Image):listofframes
fps(float):Desiredframerate
reversed_order(bool):iforderofimagestobereversed(default=True)
gif(bool):saveframestogifformat(default=True)
Returns:
None
"""
ifreversed_order:
frames.reverse()

w,h=frames[0].size
fourcc=cv2.VideoWriter_fourcc("m","p","4","v")
writer=cv2.VideoWriter(file_path,fourcc,fps,(w,h))

forframeinframes:
np_frame=np.array(frame.convert("RGB"))
cv_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
writer.write(cv_frame)

writer.release()
ifgif:
frames[0].save(
file_path.replace(".mp4",".gif"),
save_all=True,
append_images=frames[1:],
duratiobn=len(frames)/fps,
loop=0,
)

RunInfiniteZoomvideogeneration
----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defgenerate(
prompt,
negative_prompt,
seed,
steps,
frames,
edge_size,
zoom_in,
progress=gr.Progress(track_tqdm=True),
):
np.random.seed(seed)
video_path=generate_video(
ov_pipe,
prompt,
negative_prompt,
num_inference_steps=steps,
num_frames=frames,
mask_width=edge_size,
zoom_in=zoom_in,
)
np.random.seed(None)

returnvideo_path.replace(".mp4",".gif")


gr.close_all()
demo=gr.Interface(
generate,
[
gr.Textbox(
"valleyintheAlpsatsunset,epicvista,beautifullandscape,4k,8k",
label="Prompt",
),
gr.Textbox("lurry,badart,blurred,text,watermark",label="Negativeprompt"),
gr.Slider(value=9999,label="Seed",step=1,maximum=10000000),
gr.Slider(value=20,label="Steps",minimum=1,maximum=50),
gr.Slider(value=3,label="Frames",minimum=1,maximum=50),
gr.Slider(value=128,label="Edgesize",minimum=32,maximum=256),
gr.Checkbox(label="Zoomin"),
],
"image",
)

try:
demo.queue().launch()
exceptException:
demo.queue().launch(share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

StableDiffusionv2.1usingOpenVINOTorchDynamobackend
========================================================

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

Thisnotebookdemonstrateshowtorunstablediffusionmodelusing
`Diffusers<https://huggingface.co/docs/diffusers/index>`__libraryand
`OpenVINOTorchDynamo
backend<https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html>`__
forText-to-ImageandImage-to-Imagegenerationtasks.

Notebookcontainsthefollowingsteps:

1.CreatepipelinewithPyTorchmodels.
2.AddOpenVINOoptimizationusingOpenVINOTorchDynamobackend.
3.RunStableDiffusionpipelinewithOpenVINO.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`StableDiffusionwithDiffusers
library<#stable-diffusion-with-diffusers-library>`__
-`OpenVINOTorchDynamobackend<#openvino-torchdynamo-backend>`__

-`RunImagegeneration<#run-image-generation>`__

-`Interactivedemo<#interactive-demo>`__
-`SupportforAutomatic1111StableDiffusion
WebUI<#support-for-automatic1111-stable-diffusion-webui>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.2"transformersdiffusers"gradio>=4.19"ipywidgets--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2024.1.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importgradioasgr
importrandom
importtorch
importtime

fromdiffusersimportStableDiffusionPipeline,StableDiffusionImg2ImgPipeline
importipywidgetsaswidgets


..parsed-literal::

2024-07-1304:01:03.234781:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:01:03.269549:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:01:03.796151:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


StableDiffusionwithDiffuserslibrary
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

ToworkwithStableDiffusionv2.1,wewilluseHuggingFaceDiffusers
library.ToexperimentwithStableDiffusionmodels,Diffusersexposes
the
`StableDiffusionPipeline<https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
and
`StableDiffusionImg2ImgPipeline<https://huggingface.co/docs/diffusers/using-diffusers/img2img>`__
similartotheother`Diffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Thecodebelowdemonstrateshowtocreatethe
``StableDiffusionPipeline``using``stable-diffusion-2-1-base``model:

..code::ipython3

model_id="stabilityai/stable-diffusion-2-1-base"

#Pipelinefortext-to-imagegeneration
pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32)



..parsed-literal::

Loadingpipelinecomponents...:0%||0/6[00:00<?,?it/s]


..parsed-literal::

TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.


OpenVINOTorchDynamobackend
----------------------------

`backtotop⬆️<#table-of-contents>`__

The`OpenVINOTorchDynamo
backend<https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html>`__
letsyouenable`OpenVINO<https://docs.openvino.ai/2024/home.html>`__
supportforPyTorchmodelswithminimalchangestotheoriginalPyTorch
script.ItspeedsupPyTorchcodebyJIT-compilingitintooptimized
kernels.Bydefault,Torchcoderunsineager-mode,butwiththeuseof
torch.compileitgoesthroughthefollowingsteps:1.Graphacquisition
-themodelisrewrittenasblocksofsubgraphsthatareeither:-
compiledbyTorchDynamoand“flattened”,-fallingbacktothe
eager-mode,duetounsupportedPythonconstructs(likecontrol-flow
code).2.Graphlowering-allPyTorchoperationsaredecomposedinto
theirconstituentkernelsspecifictothechosenbackend.3.Graph
compilation-thekernelscalltheircorrespondinglow-level
device-specificoperations.

Selectdeviceforinferenceandenableordisablesavingtheoptimized
modelfilestoaharddrive,afterthefirstapplicationrun.Thismakes
themavailableforthefollowingapplicationexecutions,reducingthe
first-inferencelatency.Readmoreaboutavailable`Environment
Variables
options<https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#options>`__

..code::ipython3

importopenvinoasov

core=ov.Core()
device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

model_caching=widgets.Dropdown(
options=[True,False],
value=True,
description="Modelcaching:",
disabled=False,
)

model_caching




..parsed-literal::

Dropdown(description='Modelcaching:',options=(True,False),value=True)



Touse`torch.compile()
method<https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__,
youjustneedtoaddanimportstatementanddefinetheOpenVINO
backend:

..code::ipython3

#thisimportisrequiredtoactivatetheopenvinobackendfortorchdynamo
importopenvino.torch#noqa:F401

pipe.unet=torch.compile(
pipe.unet,
backend="openvino",
options={"device":device.value,"model_caching":model_caching.value},
)

**Note**:Readmoreaboutavailable`OpenVINO
backends<https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#how-to-use>`__

RunImagegeneration
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

prompt="aphotoofanastronautridingahorseonmars"
image=pipe(prompt).images[0]
image



..parsed-literal::

0%||0/50[00:00<?,?it/s]




..image::stable-diffusion-torchdynamo-backend-with-output_files/stable-diffusion-torchdynamo-backend-with-output_14_1.png



Interactivedemo
================

`backtotop⬆️<#table-of-contents>`__

Nowyoucanstartthedemo,choosetheinferencemode,defineprompts
(andinputimageforImage-to-Imagegeneration)andruninference
pipeline.Optionally,youcanalsochangesomeinputparameters.

..code::ipython3

time_stamps=[]


defcallback(iter,t,latents):
time_stamps.append(time.time())


deferror_str(error,title="Error"):
return(
f"""####{title}
{error}"""
iferror
else""
)


defon_mode_change(mode):
returngr.update(visible=mode==modes["img2img"]),gr.update(visible=mode==modes["txt2img"])


definference(
inf_mode,
prompt,
guidance=7.5,
steps=25,
width=768,
height=768,
seed=-1,
img=None,
strength=0.5,
neg_prompt="",
):
ifseed==-1:
seed=random.randint(0,10000000)
generator=torch.Generator().manual_seed(seed)
res=None

globaltime_stamps,pipe
time_stamps=[]
try:
ifinf_mode==modes["txt2img"]:
iftype(pipe).__name__!="StableDiffusionPipeline":
pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
pipe.unet=torch.compile(pipe.unet,backend="openvino")
res=pipe(
prompt,
negative_prompt=neg_prompt,
num_inference_steps=int(steps),
guidance_scale=guidance,
width=width,
height=height,
generator=generator,
callback=callback,
callback_steps=1,
).images
elifinf_mode==modes["img2img"]:
ifimgisNone:
return(
None,
None,
gr.update(
visible=True,
value=error_str("ImageisrequiredforImagetoImagemode"),
),
)
iftype(pipe).__name__!="StableDiffusionImg2ImgPipeline":
pipe=StableDiffusionImg2ImgPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
pipe.unet=torch.compile(pipe.unet,backend="openvino")
res=pipe(
prompt,
negative_prompt=neg_prompt,
image=img,
num_inference_steps=int(steps),
strength=strength,
guidance_scale=guidance,
generator=generator,
callback=callback,
callback_steps=1,
).images
exceptExceptionase:
returnNone,None,gr.update(visible=True,value=error_str(e))

warmup_duration=time_stamps[1]-time_stamps[0]
generation_rate=(steps-1)/(time_stamps[-1]-time_stamps[1])
res_info="Warmuptime:"+str(round(warmup_duration,2))+"secs"
ifgeneration_rate>=1.0:
res_info=res_info+",Performance:"+str(round(generation_rate,2))+"it/s"
else:
res_info=res_info+",Performance:"+str(round(1/generation_rate,2))+"s/it"

return(
res,
gr.update(visible=True,value=res_info),
gr.update(visible=False,value=None),
)


modes={
"txt2img":"TexttoImage",
"img2img":"ImagetoImage",
}

withgr.Blocks(css="style.css")asdemo:
gr.HTML(
f"""
Modelused:{model_id}
"""
)
withgr.Row():
withgr.Column(scale=60):
withgr.Group():
prompt=gr.Textbox(
"aphotographofanastronautridingahorse",
label="Prompt",
max_lines=2,
)
neg_prompt=gr.Textbox(
"frames,borderline,text,character,duplicate,error,outofframe,watermark,lowquality,ugly,deformed,blur",
label="Negativeprompt",
)
res_img=gr.Gallery(label="Generatedimages",show_label=False)
error_output=gr.Markdown(visible=False)

withgr.Column(scale=40):
generate=gr.Button(value="Generate")

withgr.Group():
inf_mode=gr.Dropdown(list(modes.values()),label="InferenceMode",value=modes["txt2img"])

withgr.Column(visible=False)asi2i:
image=gr.Image(label="Image",height=128,type="pil")
strength=gr.Slider(
label="Transformationstrength",
minimum=0,
maximum=1,
step=0.01,
value=0.5,
)

withgr.Group():
withgr.Row()astxt2i:
width=gr.Slider(label="Width",value=512,minimum=64,maximum=1024,step=8)
height=gr.Slider(label="Height",value=512,minimum=64,maximum=1024,step=8)

withgr.Group():
withgr.Row():
steps=gr.Slider(label="Steps",value=20,minimum=1,maximum=50,step=1)
guidance=gr.Slider(label="Guidancescale",value=7.5,maximum=15)

seed=gr.Slider(-1,10000000,label="Seed(-1=random)",value=-1,step=1)

res_info=gr.Markdown(visible=False)

inf_mode.change(on_mode_change,inputs=[inf_mode],outputs=[i2i,txt2i],queue=False)

inputs=[
inf_mode,
prompt,
guidance,
steps,
width,
height,
seed,
image,
strength,
neg_prompt,
]

outputs=[res_img,res_info,error_output]
prompt.submit(inference,inputs=inputs,outputs=outputs)
generate.click(inference,inputs=inputs,outputs=outputs)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


SupportforAutomatic1111StableDiffusionWebUI
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Automatic1111StableDiffusionWebUIisanopen-sourcerepositorythat
hostsabrowser-basedinterfacefortheStableDiffusionbasedimage
generation.Itallowsuserstocreaterealisticandcreativeimagesfrom
textprompts.StableDiffusionWebUIissupportedonIntelCPUs,Intel
integratedGPUs,andInteldiscreteGPUsbyleveragingOpenVINO
torch.compilecapability.Detailedinstructionsareavailable
in\`StableDiffusionWebUI
repository<https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon>`__.

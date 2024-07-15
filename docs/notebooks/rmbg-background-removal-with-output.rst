BackgroundremovalwithRMBGv1.4andOpenVINO
==============================================

Backgroundmattingistheprocessofaccuratelyestimatingthe
foregroundobjectinimagesandvideos.Itisaveryimportanttechnique
inimageandvideoeditingapplications,particularlyinfilmproduction
forcreatingvisualeffects.Incaseofimagesegmentation,wesegment
theimageintoforegroundandbackgroundbylabelingthepixels.Image
segmentationgeneratesabinaryimage,inwhichapixeleitherbelongs
toforegroundorbackground.However,ImageMattingisdifferentfrom
theimagesegmentation,whereinsomepixelsmaybelongtoforegroundas
wellasbackground,suchpixelsarecalledpartialormixedpixels.In
ordertofullyseparatetheforegroundfromthebackgroundinanimage,
accurateestimationofthealphavaluesforpartialormixedpixelsis
necessary.

RMBGv1.4isbackgroundremovalmodel,designedtoeffectivelyseparate
foregroundfrombackgroundinarangeofcategoriesandimagetypes.
Thismodelhasbeentrainedonacarefullyselecteddataset,which
includes:generalstockimages,e-commerce,gaming,andadvertising
content,makingitsuitableforcommercialusecasespoweringenterprise
contentcreationatscale.Theaccuracy,efficiency,andversatility
currentlyrivalleadingsource-availablemodels.

Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/briaai/RMBG-1.4>`__.

Inthistutorialweconsiderhowtoconvertandrunthismodelusing
OpenVINO.####Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchmodel<#load-pytorch-model>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__
-`ConvertModeltoOpenVINOIntermediateRepresentation
format<#convert-model-to-openvino-intermediate-representation-format>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__
-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

installrequireddependencies

..code::ipython3

%pipinstall-qtorchtorchvisionpillowhuggingface_hub"openvino>=2024.0.0"matplotlib"gradio>=4.15""transformers>=4.39.1"tqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


DownloadmodelcodefromHuggingFacehub

..code::ipython3

fromhuggingface_hubimporthf_hub_download
frompathlibimportPath

repo_id="briaai/RMBG-1.4"

download_files=["utilities.py","example_input.jpg"]

forfile_for_downloadingindownload_files:
ifnotPath(file_for_downloading).exists():
hf_hub_download(repo_id=repo_id,filename=file_for_downloading,local_dir=".")



..parsed-literal::

utilities.py:0%||0.00/980[00:00<?,?B/s]



..parsed-literal::

example_input.jpg:0%||0.00/327k[00:00<?,?B/s]


LoadPyTorchmodel
------------------

`backtotop⬆️<#table-of-contents>`__

ForloadingmodelusingPyTorch,weshoulduse
``AutoModelForImageSegmentation.from_pretrained``method.Modelweights
willbedownloadedautomaticallyduringfirstmodelusage.Please,note,
itmaytakesometime.

..code::ipython3

fromtransformersimportAutoModelForImageSegmentation

net=AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(


RunPyTorchmodelinference
---------------------------

`backtotop⬆️<#table-of-contents>`__

``preprocess_image``functionisresponsibleforpreparinginputdatain
model-specificformat.``postprocess_image``functionisresponsiblefor
postprocessingmodeloutput.Afterpostprocessing,generatedbackground
maskcanbeinsertedintooriginalimageasalpha-channel.

..code::ipython3

importtorch
fromPILimportImage
fromutilitiesimportpreprocess_image,postprocess_image
importnumpyasnp
frommatplotlibimportpyplotasplt


defvisualize_result(orig_img:Image,mask:Image,result_img:Image):
"""
Helperforresultsvisualization

parameters:
orig_img(Image):inputimage
mask(Image):backgroundmask
result_img(Image)outputimage
returns:
plt.Figure:plotwith3imagesforvisualization
"""
titles=["Original","BackgroundMask","Withoutbackground"]
im_w,im_h=orig_img.size
is_horizontal=im_h<=im_w
figsize=(20,20)
num_images=3
fig,axs=plt.subplots(
num_imagesifis_horizontalelse1,
1ifis_horizontalelsenum_images,
figsize=figsize,
sharex="all",
sharey="all",
)
fig.patch.set_facecolor("white")
list_axes=list(axs.flat)
forainlist_axes:
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
list_axes[0].imshow(np.array(orig_img))
list_axes[1].imshow(np.array(mask),cmap="gray")
list_axes[0].set_title(titles[0],fontsize=15)
list_axes[1].set_title(titles[1],fontsize=15)
list_axes[2].imshow(np.array(result_img))
list_axes[2].set_title(titles[2],fontsize=15)

fig.subplots_adjust(wspace=0.01ifis_horizontalelse0.00,hspace=0.01ifis_horizontalelse0.1)
fig.tight_layout()
returnfig


im_path="./example_input.jpg"

#prepareinput
model_input_size=[1024,1024]
orig_im=np.array(Image.open(im_path))
orig_im_size=orig_im.shape[0:2]
image=preprocess_image(orig_im,model_input_size)

#inference
result=net(image)

#postprocess
result_image=postprocess_image(result[0][0],orig_im_size)

#saveresult
pil_im=Image.fromarray(result_image)
no_bg_image=Image.new("RGBA",pil_im.size,(0,0,0,0))
orig_image=Image.open(im_path)
no_bg_image.paste(orig_image,mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

visualize_result(orig_image,pil_im,no_bg_image);



..image::rmbg-background-removal-with-output_files/rmbg-background-removal-with-output_8_0.png


ConvertModeltoOpenVINOIntermediateRepresentationformat
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversiontoOpenVINOIntermediate
Representation(IR).`OpenVINOmodelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
shouldbeusedforthesepurposes.``ov.convert_model``functionaccepts
originalPyTorchmodelinstanceandexampleinputfortracingand
returns``ov.Model``representingthismodelinOpenVINOframework.
Convertedmodelcanbeusedforsavingondiskusing``ov.save_model``
functionordirectlyloadingondeviceusing``core.complie_model``.

..code::ipython3

importopenvinoasov

ov_model_path=Path("rmbg-1.4.xml")

ifnotov_model_path.exists():
ov_model=ov.convert_model(net,example_input=image,input=[1,3,*model_input_size])
ov.save_model(ov_model,ov_model_path)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(


..parsed-literal::

['x']


RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Afterfinishingconversion,wecancompileconvertedmodelandrunit
usingOpenVINOonspecifieddevice.Forselectioninferencedevice,
pleaseusedropdownlistbelow:

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



Let’srunmodelonthesameimagethatweusedbeforeforlaunching
PyTorchmodel.OpenVINOmodelinputandoutputisfullycompatiblewith
originalpre-andpostprocessingsteps,itmeansthatwecanreusethem.

..code::ipython3

ov_compiled_model=core.compile_model(ov_model_path,device.value)

result=ov_compiled_model(image)[0]

#postprocess
result_image=postprocess_image(torch.from_numpy(result),orig_im_size)

#saveresult
pil_im=Image.fromarray(result_image)
no_bg_image=Image.new("RGBA",pil_im.size,(0,0,0,0))
orig_image=Image.open(im_path)
no_bg_image.paste(orig_image,mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

visualize_result(orig_image,pil_im,no_bg_image);



..image::rmbg-background-removal-with-output_files/rmbg-background-removal-with-output_14_0.png


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


title="#RMBGbackgroundremovalwithOpenVINO"


defget_background_mask(model,image):
returnmodel(image)[0]


withgr.Blocks()asdemo:
gr.Markdown(title)

withgr.Row():
input_image=gr.Image(label="InputImage",type="numpy")
background_image=gr.Image(label="BackgroundremovalImage")
submit=gr.Button("Submit")

defon_submit(image):
original_image=image.copy()

h,w=image.shape[:2]
image=preprocess_image(original_image,model_input_size)

mask=get_background_mask(ov_compiled_model,image)
result_image=postprocess_image(torch.from_numpy(mask),(h,w))
pil_im=Image.fromarray(result_image)
orig_img=Image.fromarray(original_image)
no_bg_image=Image.new("RGBA",pil_im.size,(0,0,0,0))
no_bg_image.paste(orig_img,mask=pil_im)

returnno_bg_image

submit.click(on_submit,inputs=[input_image],outputs=[background_image])
examples=gr.Examples(
examples=["./example_input.jpg"],
inputs=[input_image],
outputs=[background_image],
fn=on_submit,
cache_examples=False,
)


if__name__=="__main__":
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


ImageBackgroundRemovalwithU^2-NetandOpenVINO™
===================================================

Thisnotebookdemonstratesbackgroundremovalinimagesusing
U\:math:`^2`-NetandOpenVINO.

FormoreinformationaboutU\:math:`^2`-Net,includingsourcecodeand
testdata,seethe`GitHub
page<https://github.com/xuebinqin/U-2-Net>`__andtheresearchpaper:
`U^2-Net:GoingDeeperwithNestedU-StructureforSalientObject
Detection<https://arxiv.org/pdf/2005.09007.pdf>`__.

ThePyTorchU\:math:`^2`-NetmodelisconvertedtoOpenVINOIRformat.
Themodelsourceisavailable
`here<https://github.com/xuebinqin/U-2-Net>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Installrequirements<#install-requirements>`__
-`ImportthePyTorchLibraryand
U\:math:`^2`-Net<#import-the-pytorch-library-and-u2-net>`__
-`Settings<#settings>`__
-`LoadtheU\:math:`^2`-NetModel<#load-the-u2-net-model>`__

-`ConvertPyTorchU\:math:`^2`-NetmodeltoOpenVINO
IR<#convert-pytorch-u2-net-model-to-openvino-ir>`__
-`LoadandPre-ProcessInput
Image<#load-and-pre-process-input-image>`__
-`Selectinferencedevice<#select-inference-device>`__
-`DoInferenceonOpenVINOIR
Model<#do-inference-on-openvino-ir-model>`__
-`VisualizeResults<#visualize-results>`__

-`AddaBackgroundImage<#add-a-background-image>`__

-`References<#references>`__

Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Installrequirements
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"torch>=2.1"opencv-python
%pipinstall-q"gdown<4.6.4"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


ImportthePyTorchLibraryandU\:math:`^2`-Net
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos
importtime
fromcollectionsimportnamedtuple
frompathlibimportPath

importcv2
importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov
importtorch
fromIPython.displayimportHTML,FileLink,display

..code::ipython3

#Importlocalmodules
importrequests

ifnotPath("./notebook_utils.py").exists():
#Fetch`notebook_utils`module

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportload_image,download_file

ifnotPath("./model/u2net.py").exists():
download_file(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/vision-background-removal/model/u2net.py",directory="model"
)
frommodel.u2netimportU2NET,U2NETP

Settings
~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThistutorialsupportsusingtheoriginalU\:math:`^2`-Netsalient
objectdetectionmodel,aswellasthesmallerU2NETPversion.Twosets
ofweightsaresupportedfortheoriginalmodel:salientobject
detectionandhumansegmentation.

..code::ipython3

model_config=namedtuple("ModelConfig",["name","url","model","model_args"])

u2net_lite=model_config(
name="u2net_lite",
url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
model=U2NETP,
model_args=(),
)
u2net=model_config(
name="u2net",
url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
model=U2NET,
model_args=(3,1),
)
u2net_human_seg=model_config(
name="u2net_human_seg",
url="https://drive.google.com/uc?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP",
model=U2NET,
model_args=(3,1),
)

#Setu2net_modeltooneofthethreeconfigurationslistedabove.
u2net_model=u2net_lite

..code::ipython3

#Thefilenamesofthedownloadedandconvertedmodels.
MODEL_DIR="model"
model_path=Path(MODEL_DIR)/u2net_model.name/Path(u2net_model.name).with_suffix(".pth")

LoadtheU\:math:`^2`-NetModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheU\:math:`^2`-Nethumansegmentationmodelweightsarestoredon
GoogleDrive.Theywillbedownloadediftheyarenotpresentyet.The
nextcellloadsthemodelandthepre-trainedweights.

..code::ipython3

ifnotmodel_path.exists():
importgdown

os.makedirs(name=model_path.parent,exist_ok=True)
print("Startdownloadingmodelweightsfile...")
withopen(model_path,"wb")asmodel_file:
gdown.download(url=u2net_model.url,output=model_file)
print(f"Modelweightshavebeendownloadedto{model_path}")


..parsed-literal::

Startdownloadingmodelweightsfile...


..parsed-literal::

Downloading...
From:https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD
To:<_io.BufferedWritername='model/u2net_lite/u2net_lite.pth'>
100%|██████████|4.68M/4.68M[00:00<00:00,34.0MB/s]

..parsed-literal::

Modelweightshavebeendownloadedtomodel/u2net_lite/u2net_lite.pth


..parsed-literal::




..code::ipython3

#Loadthemodel.
net=u2net_model.model(*u2net_model.model_args)
net.eval()

#Loadtheweights.
print(f"Loadingmodelweightsfrom:'{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path,map_location="cpu"))


..parsed-literal::

Loadingmodelweightsfrom:'model/u2net_lite/u2net_lite.pth'




..parsed-literal::

<Allkeysmatchedsuccessfully>



ConvertPyTorchU\:math:`^2`-NetmodeltoOpenVINOIR
------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

WeusemodelconversionPythonAPItoconvertthePytorchmodelto
OpenVINOIRformat.Executingthefollowingcommandmaytakeawhile.

..code::ipython3

model_ir=ov.convert_model(net,example_input=torch.zeros((1,3,512,512)),input=([1,3,512,512]))


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/nn/functional.py:3782:UserWarning:nn.functional.upsampleisdeprecated.Usenn.functional.interpolateinstead.
warnings.warn("nn.functional.upsampleisdeprecated.Usenn.functional.interpolateinstead.")


..parsed-literal::

['x']


LoadandPre-ProcessInputImage
--------------------------------

`backtotop⬆️<#table-of-contents>`__

WhileOpenCVreadsimagesin``BGR``format,theOpenVINOIRmodel
expectsimagesin``RGB``.Therefore,converttheimagesto``RGB``,
resizethemto``512x512``,andtransposethedimensionstotheformat
theOpenVINOIRmodelexpects.

Weaddthemeanvaluestotheimagetensorandscaletheinputwiththe
standarddeviation.Itiscalledtheinputdatanormalizationbefore
propagatingitthroughthenetwork.Themeanandstandarddeviation
valuescanbefoundinthe
`dataloader<https://github.com/xuebinqin/U-2-Net/blob/master/data_loader.py>`__
fileinthe`U^2-Net
repository<https://github.com/xuebinqin/U-2-Net/>`__andmultipliedby
255tosupportimageswithpixelvaluesfrom0-255.

..code::ipython3

IMAGE_URI="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg"

input_mean=np.array([123.675,116.28,103.53]).reshape(1,3,1,1)
input_scale=np.array([58.395,57.12,57.375]).reshape(1,3,1,1)

image=cv2.cvtColor(
src=load_image(IMAGE_URI),
code=cv2.COLOR_BGR2RGB,
)

resized_image=cv2.resize(src=image,dsize=(512,512))
#Converttheimageshapetoashapeandadatatypeexpectedbythenetwork
#forOpenVINOIRmodel:(1,3,512,512).
input_image=np.expand_dims(np.transpose(resized_image,(2,0,1)),0)

input_image=(input_image-input_mean)/input_scale

Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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



DoInferenceonOpenVINOIRModel
---------------------------------

`backtotop⬆️<#table-of-contents>`__

LoadtheOpenVINOIRmodeltoOpenVINORuntimeanddoinference.

..code::ipython3

core=ov.Core()
#LoadthenetworktoOpenVINORuntime.
compiled_model_ir=core.compile_model(model=model_ir,device_name=device.value)
#Getthenamesofinputandoutputlayers.
input_layer_ir=compiled_model_ir.input(0)
output_layer_ir=compiled_model_ir.output(0)

#Doinferenceontheinputimage.
start_time=time.perf_counter()
result=compiled_model_ir([input_image])[output_layer_ir]
end_time=time.perf_counter()
print(f"Inferencefinished.Inferencetime:{end_time-start_time:.3f}seconds,"f"FPS:{1/(end_time-start_time):.2f}.")


..parsed-literal::

Inferencefinished.Inferencetime:0.119seconds,FPS:8.43.


VisualizeResults
-----------------

`backtotop⬆️<#table-of-contents>`__

Showtheoriginalimage,thesegmentationresult,andtheoriginalimage
withthebackgroundremoved.

..code::ipython3

#Resizethenetworkresulttotheimageshapeandroundthevalues
#to0(background)and1(foreground).
#Thenetworkresulthas(1,1,512,512)shape.The`np.squeeze`functionconvertsthisto(512,512).
resized_result=np.rint(cv2.resize(src=np.squeeze(result),dsize=(image.shape[1],image.shape[0]))).astype(np.uint8)

#Createacopyoftheimageandsetallbackgroundvaluesto255(white).
bg_removed_result=image.copy()
bg_removed_result[resized_result==0]=255

fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(20,7))
ax[0].imshow(image)
ax[1].imshow(resized_result,cmap="gray")
ax[2].imshow(bg_removed_result)
forainax:
a.axis("off")



..image::vision-background-removal-with-output_files/vision-background-removal-with-output_22_0.png


AddaBackgroundImage
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthesegmentationresult,allforegroundpixelshaveavalueof1,all
backgroundpixelsavalueof0.Replacethebackgroundimageasfollows:

-Loadanew``background_image``.
-Resizetheimagetothesamesizeastheoriginalimage.
-In``background_image``,setallthepixels,wheretheresized
segmentationresulthasavalueof1-theforegroundpixelsinthe
originalimage-to0.
-Add``bg_removed_result``fromthepreviousstep-thepartofthe
originalimagethatonlycontainsforegroundpixels-to
``background_image``.

..code::ipython3

BACKGROUND_FILE="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"
OUTPUT_DIR="output"

os.makedirs(name=OUTPUT_DIR,exist_ok=True)

background_image=cv2.cvtColor(src=load_image(BACKGROUND_FILE),code=cv2.COLOR_BGR2RGB)
background_image=cv2.resize(src=background_image,dsize=(image.shape[1],image.shape[0]))

#Setalltheforegroundpixelsfromtheresultto0
#inthebackgroundimageandaddtheimagewiththebackgroundremoved.
background_image[resized_result==1]=0
new_image=background_image+bg_removed_result

#Savethegeneratedimage.
new_image_path=Path(f"{OUTPUT_DIR}/{Path(IMAGE_URI).stem}-{Path(BACKGROUND_FILE).stem}.jpg")
cv2.imwrite(filename=str(new_image_path),img=cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR))

#Displaytheoriginalimageandtheimagewiththenewbackgroundsidebyside
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(18,7))
ax[0].imshow(image)
ax[1].imshow(new_image)
forainax:
a.axis("off")
plt.show()

#Createalinktodownloadtheimage.
image_link=FileLink(new_image_path)
image_link.html_link_str="<ahref='%s'download>%s</a>"
display(
HTML(
f"Thegeneratedimage<code>{new_image_path.name}</code>issavedin"
f"thedirectory<code>{new_image_path.parent}</code>.Youcanalso"
"downloadtheimagebyclickingonthislink:"
f"{image_link._repr_html_()}"
)
)



..image::vision-background-removal-with-output_files/vision-background-removal-with-output_24_0.png



..raw::html

Thegeneratedimage<code>coco_hollywood-wall.jpg</code>issavedinthedirectory<code>output</code>.Youcanalsodownloadtheimagebyclickingonthislink:output/coco_hollywood-wall.jpg<br>


References
----------

`backtotop⬆️<#table-of-contents>`__

-`PIPinstall
openvino-dev<https://github.com/openvinotoolkit/openvino/blob/releases/2023/2/docs/install_guides/pypi-openvino-dev.md>`__
-`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
-`U^2-Net<https://github.com/xuebinqin/U-2-Net>`__
-U^2-Netresearchpaper:`U^2-Net:GoingDeeperwithNested
U-StructureforSalientObject
Detection<https://arxiv.org/pdf/2005.09007.pdf>`__

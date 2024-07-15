ConvertofTensorFlowHubmodelstoOpenVINOIntermediateRepresentation(IR)
=============================================================================

|Colab||Binder|

Thistutorialdemonstratesstep-by-stepinstructionsonhowtoconvert
modelsloadedfromTensorFlowHubusingOpenVINORuntime.

`TensorFlowHub<https://tfhub.dev/>`__isalibraryandonlineplatform
developedbyGooglethatsimplifiesmachinelearningmodelreuseand
sharing.Itservesasarepositoryofpre-trainedmodels,embeddings,
andreusablecomponents,allowingresearchersanddeveloperstoaccess
andintegratestate-of-the-artmachinelearningmodelsintotheirown
projectswithease.TensorFlowHubprovidesadiverserangeofmodels
forvarioustaskslikeimageclassification,textembedding,andmore.
ItstreamlinestheprocessofincorporatingthesemodelsintoTensorFlow
workflows,fosteringcollaborationandacceleratingthedevelopmentof
AIapplications.Thiscentralizedhubenhancesmodelaccessibilityand
promotestherapidadvancementofmachinelearningcapabilitiesacross
thecommunity.

Youhavetheflexibilitytorunthistutorialnotebookinitsentirety
orselectivelyexecutespecificsections,aseachsectionoperates
independently.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Installrequiredpackages<#install-required-packages>`__
-`Imageclassification<#image-classification>`__

-`Importlibraries<#import-libraries>`__
-`Downloadtheclassifier<#download-the-classifier>`__
-`Downloadasingleimagetotrythemodel
on<#download-a-single-image-to-try-the-model-on>`__
-`ConvertmodeltoOpenVINOIR<#convert-model-to-openvino-ir>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Inference<#inference>`__

-`Imagestyletransfer<#image-style-transfer>`__

-`Installrequiredpackages<#install-required-packages>`__
-`Loadthemodel<#load-the-model>`__
-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Inference<#inference>`__

..|Colab|image::https://colab.research.google.com/assets/colab-badge.svg
:target:https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/tensorflow-hub/tensorflow-hub.ipynb
..|Binder|image::https://mybinder.org/badge_logo.svg
:target:https://mybinder.org/v2/gh/eaidova/openvino_notebooks_binder.git/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fopenvinotoolkit%252Fopenvino_notebooks%26urlpath%3Dtree%252Fopenvino_notebooks%252Fnotebooks%2Ftensorflow-hub%2Ftensorflow-hub.ipynb

Installrequiredpackages
-------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-qpillownumpy
%pipinstall-q"openvino>=2023.2.0""opencv-python"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'andpython_version<='3.8'"
%pipinstall-qtf_kerastensorflow_hub


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imageclassification
--------------------

`backtotop⬆️<#table-of-contents>`__

Wewillusethe`MobileNet_v2<https://arxiv.org/abs/1704.04861>`__
imageclassificationmodelfrom`TensorFlowHub<https://tfhub.dev/>`__.

MobileNetV2isacompactandefficientdeeplearningarchitecture
designedformobileandembeddeddevices,developedbyGoogle
researchers.ItbuildsonthesuccessoftheoriginalMobileNetby
introducingimprovementsinbothspeedandaccuracy.MobileNetV2employs
astreamlinedarchitecturewithinvertedresidualblocks,makingit
highlyefficientforreal-timeapplicationswhileminimizing
computationalresources.Thisnetworkexcelsintaskslikeimage
classification,objectdetection,andimagesegmentation,offeringa
balancebetweenmodelsizeandperformance.MobileNetV2hasbecomea
popularchoiceforon-deviceAIapplications,enablingfasterandmore
efficientdeeplearninginferenceonsmartphonesandedgedevices.

Moreinformationaboutmodelcanbefoundon`ModelpageonTensorFlow
Hub<https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5>`__

Importlibraries
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath
importos
importrequests

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_USE_LEGACY_KERAS"]="1"
os.environ["TFHUB_CACHE_DIR"]=str(Path("./tfhub_modules").resolve())

importtensorflow_hubashub
importtensorflowastf
importPIL
importnumpyasnp
importmatplotlib.pyplotasplt

importopenvinoasov

tf.get_logger().setLevel("ERROR")

..code::ipython3

IMAGE_SHAPE=(224,224)
IMAGE_URL,IMAGE_PATH=(
"https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
"data/grace_hopper.jpg",
)
MODEL_URL,MODEL_PATH=(
"https://www.kaggle.com/models/google/mobilenet-v1/frameworks/tensorFlow2/variations/100-224-classification/versions/2",
"models/mobilenet_v2_100_224.xml",
)

Downloadtheclassifier
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__SelectaMobileNetV2
pre-trainedmodel`fromTensorFlow
Hub<https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5>`__
andwrapitasaKeraslayerwith``hub.KerasLayer``.

..code::ipython3

model=hub.KerasLayer(MODEL_URL,input_shape=IMAGE_SHAPE+(3,))


..parsed-literal::

2024-07-1304:05:09.169100:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1304:05:09.169274:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


Downloadasingleimagetotrythemodelon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Theinput``images``are
expectedtohavecolorvaluesintherange[0,1],followingthe`common
imageinput
conventions<https://www.tensorflow.org/hub/common_signatures/images#input>`__.
Forthismodel,thesizeoftheinputimagesisfixedto``height``x
``width``=224x224pixels.

..code::ipython3

Path(IMAGE_PATH).parent.mkdir(parents=True,exist_ok=True)

r=requests.get(IMAGE_URL)
withPath(IMAGE_PATH).open("wb")asf:
f.write(r.content)
grace_hopper=PIL.Image.open(IMAGE_PATH).resize(IMAGE_SHAPE)
grace_hopper




..image::tensorflow-hub-with-output_files/tensorflow-hub-with-output_11_0.png



Normalizetheimageto[0,1]range.

..code::ipython3

grace_hopper=np.array(grace_hopper)/255.0
grace_hopper.shape




..parsed-literal::

(224,224,3)



ConvertmodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WewillconverttheloadedmodeltoOpenVINOIRusing
``ov.convert_model``function.Wepassthemodelobjecttoit,no
additionalargumentsrequired.Then,wesavethemodeltodiskusing
``ov.save_model``function.

..code::ipython3

ifnotPath(MODEL_PATH).exists():
converted_model=ov.convert_model(model)
ov.save_model(converted_model,MODEL_PATH)

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

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



..code::ipython3

compiled_model=core.compile_model(MODEL_PATH,device_name=device.value)

Inference
~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Addabatchdimension(with``np.newaxis``)andpasstheimagetothe
model:

..code::ipython3

output=compiled_model(grace_hopper[np.newaxis,...])[0]
output.shape




..parsed-literal::

(1,1001)



Theresultisa1001-elementvectoroflogits,ratingtheprobabilityof
eachclassfortheimage.

ThetopclassIDcanbefoundwith``np.argmax``:

..code::ipython3

predicted_class=np.argmax(output[0],axis=-1)
predicted_class




..parsed-literal::

653



Takethe``predicted_class``ID(suchas``653``)andfetchtheImageNet
datasetlabelstodecodethepredictions:

..code::ipython3

labels_path=tf.keras.utils.get_file(
"ImageNetLabels.txt",
"https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
)
imagenet_labels=np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis("off")
predicted_class_name=imagenet_labels[predicted_class]
_=plt.title("Prediction:"+predicted_class_name.title())



..image::tensorflow-hub-with-output_files/tensorflow-hub-with-output_26_0.png


Imagestyletransfer
--------------------

`backtotop⬆️<#table-of-contents>`__

Wewilluse`arbitraryimagestylization
model<https://arxiv.org/abs/1705.06830>`__from`TensorFlow
Hub<https://tfhub.dev>`__.

Themodelcontainsconditionalinstancenormalization(CIN)layers

TheCINnetworkconsistsoftwomaincomponents:afeatureextractorand
astylizationmodule.Thefeatureextractorextractsasetoffeatures
fromthecontentimage.Thestylizationmodulethenusesthesefeatures
togenerateastylizedimage.

Thestylizationmoduleisastackofconvolutionallayers.Each
convolutionallayerisfollowedbyaCINlayer.TheCINlayertakesthe
featuresfromthepreviouslayerandtheCINparametersfromthestyle
imageasinputandproducesanewsetoffeaturesasoutput.

Theoutputofthestylizationmoduleisastylizedimage.Thestylized
imagehasthesamecontentastheoriginalcontentimage,butthestyle
hasbeentransferredfromthestyleimage.

TheCINnetworkisabletostylizeimagesinrealtimebecauseitis
veryefficient.

Moremodelinformationcanbefoundon`ModelpageonTensorFlow
Hub<https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2>`__.

..code::ipython3

importos

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_USE_LEGACY_KERAS"]="1"
os.environ["TFHUB_CACHE_DIR"]=str(Path("./tfhub_modules").resolve())
frompathlibimportPath

importopenvinoasov

importtensorflow_hubashub
importtensorflowastf
importcv2
importnumpyasnp
importmatplotlib.pyplotasplt

..code::ipython3

CONTENT_IMAGE_URL="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/525babb8-1289-45f8-a3a5-e248f74dfb24"
CONTENT_IMAGE_PATH="./data/YellowLabradorLooking_new.jpg"

STYLE_IMAGE_URL="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/c212233d-9a33-4979-b8f9-2a94a529026e"
STYLE_IMAGE_PATH="./data/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"

MODEL_URL="https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/tensorFlow1/variations/256/versions/2"
MODEL_PATH="./models/arbitrary-image-stylization-v1-256.xml"

Loadthemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WeloadthemodelfromTensorFlowHubusing``hub.KerasLayer``.Since
themodelhasmultipleinputs(contentimageandstyleimage),weneed
tobuilditbycallingwithplaceholdersandwrapin``tf.keras.Model``
function.

..code::ipython3

inputs={
"placeholder":tf.keras.layers.Input(shape=(None,None,3)),
"placeholder_1":tf.keras.layers.Input(shape=(None,None,3)),
}
model=hub.KerasLayer(MODEL_URL,signature="serving_default",signature_outputs_as_dict=True)#definethesignaturetoallowpassinginputsasadictionary
outputs=model(inputs)
model=tf.keras.Model(inputs=inputs,outputs=outputs)

ConvertthemodeltoOpenVINOIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WeconverttheloadedmodeltoOpenVINOIRusing``ov.convert_model``
function.Wepassourmodeltothefunction,noadditionalarguments
needed.Afterconverting,wesavethemodeltodiskusing
``ov.save_model``function.

..code::ipython3

ifnotPath(MODEL_PATH).exists():
Path(MODEL_PATH).parent.mkdir(parents=True,exist_ok=True)
converted_model=ov.convert_model(model)
ov.save_model(converted_model,MODEL_PATH)

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

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



..code::ipython3

compiled_model=core.compile_model(MODEL_PATH,device_name=device.value)

Inference
~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotPath(STYLE_IMAGE_PATH).exists():
r=requests.get(STYLE_IMAGE_URL)
withopen(STYLE_IMAGE_PATH,"wb")asf:
f.write(r.content)
ifnotPath(CONTENT_IMAGE_PATH).exists():
r=requests.get(CONTENT_IMAGE_URL)
withopen(CONTENT_IMAGE_PATH,"wb")asf:
f.write(r.content)


defload_image(dst):
image=cv2.imread(dst)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#ConvertimagecolortoRGBspace
image=image/255#Normalizeto[0,1]interval
image=image.astype(np.float32)
returnimage

..code::ipython3

content_image=load_image(CONTENT_IMAGE_PATH)
style_image=load_image(STYLE_IMAGE_PATH)
style_image=cv2.resize(style_image,(256,256))#modelwastrainedon256x256images

..code::ipython3

result=compiled_model([content_image[np.newaxis,...],style_image[np.newaxis,...]])[0]

..code::ipython3

title2img={
"Sourceimage":content_image,
"Referencestyle":style_image,
"Result":result[0],
}
plt.figure(figsize=(12,12))
fori,(title,img)inenumerate(title2img.items()):
ax=plt.subplot(1,3,i+1)
ax.set_title(title)
plt.imshow(img)
plt.axis("off")



..image::tensorflow-hub-with-output_files/tensorflow-hub-with-output_43_0.png


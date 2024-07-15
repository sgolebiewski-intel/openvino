OptimizePreprocessing
======================

Wheninputdatadoesnotfitthemodelinputtensorperfectly,
additionaloperations/stepsareneededtotransformthedatatothe
formatexpectedbythemodel.Thistutorialdemonstrateshowitcouldbe
performedwithPreprocessingAPI.PreprocessingAPIisaneasy-to-use
instrument,thatenablesintegrationofpreprocessingstepsintoan
executiongraphandperformingitonaselecteddevice,whichcan
improvedeviceutilization.FormoreinformationaboutPreprocessing
API,seethis
`overview<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html#>`__
and
`details<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details.html>`__

Thistutorialincludefollowingsteps:

-Downloadingthemodel.
-SetuppreprocessingwithPreprocessingAPI,loadingthemodeland
inferencewithoriginalimage.
-Fittingimagetothemodelinputtypeandinferencewithprepared
image.
-Comparingresultsononepicture.
-Comparingperformance.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Settings<#settings>`__
-`Imports<#imports>`__

-`Setupimageanddevice<#setup-image-and-device>`__
-`Downloadingthemodel<#downloading-the-model>`__
-`Createcore<#create-core>`__
-`Checktheoriginalparametersof
image<#check-the-original-parameters-of-image>`__

-`SetuppreprocessingstepswithPreprocessingAPIandperform
inference<#setup-preprocessing-steps-with-preprocessing-api-and-perform-inference>`__

-`ConvertmodeltoOpenVINOIRwithmodelconversion
API<#convert-model-to-openvino-ir-with-model-conversion-api>`__
-`CreatePrePostProcessor
Object<#create-prepostprocessor-object>`__
-`DeclareUser’sDataFormat<#declare-users-data-format>`__
-`DeclaringModelLayout<#declaring-model-layout>`__
-`PreprocessingSteps<#preprocessing-steps>`__
-`IntegratingStepsintoa
Model<#integrating-steps-into-a-model>`__

-`Loadmodelandperform
inference<#load-model-and-perform-inference>`__
-`Fitimagemanuallyandperform
inference<#fit-image-manually-and-perform-inference>`__

-`Loadthemodel<#load-the-model>`__
-`Loadimageandfitittomodel
input<#load-image-and-fit-it-to-model-input>`__
-`Performinference<#perform-inference>`__

-`Compareresults<#compare-results>`__

-`Compareresultsononeimage<#compare-results-on-one-image>`__
-`Compareperformance<#compare-performance>`__

Settings
--------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm
ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version<='3.8'"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtime
importos
frompathlibimportPath

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_USE_LEGACY_KERAS"]="1"

importcv2
importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov
importtensorflowastf

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

Setupimageanddevice
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_path=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)
image_path=str(image_path)



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]


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



Downloadingthemodel
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thistutorialusesthe
`InceptionResNetV2<https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__.
TheInceptionResNetV2modelisthesecondofthe
`Inception<https://github.com/tensorflow/tpu/tree/master/models/experimental/inception>`__
familyofmodelsdesignedtoperformimageclassification.Likeother
Inceptionmodels,InceptionResNetV2hasbeenpre-trainedonthe
`ImageNet<https://image-net.org/>`__dataset.Formoredetailsabout
thisfamilyofmodels,seethe`research
paper<https://arxiv.org/abs/1602.07261>`__.

Loadthemodelbyusing`tf.keras.applications
api<https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__
andsaveittothedisk.

..code::ipython3

model_name="InceptionResNetV2"

model_dir=Path("model")
model_dir.mkdir(exist_ok=True)

model_path=model_dir/model_name

model=tf.keras.applications.InceptionV3()
model.save(model_path)


..parsed-literal::

2024-07-1301:20:31.789841:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1301:20:31.790029:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


..parsed-literal::

WARNING:tensorflow:Compiledtheloadedmodel,butthecompiledmetricshaveyettobebuilt.`model.compile_metrics`willbeemptyuntilyoutrainorevaluatethemodel.


..parsed-literal::

WARNING:absl:Founduntracedfunctionssuchas_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_opwhilesaving(showing5of94).Thesefunctionswillnotbedirectlycallableafterloading.


..parsed-literal::

INFO:tensorflow:Assetswrittento:model/InceptionResNetV2/assets


..parsed-literal::

INFO:tensorflow:Assetswrittento:model/InceptionResNetV2/assets


Createcore
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()

Checktheoriginalparametersofimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

image=cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
print(f"Theoriginalshapeoftheimageis{image.shape}")
print(f"Theoriginaldatatypeoftheimageis{image.dtype}")


..parsed-literal::

Theoriginalshapeoftheimageis(577,800,3)
Theoriginaldatatypeoftheimageisuint8



..image::optimize-preprocessing-with-output_files/optimize-preprocessing-with-output_14_1.png


SetuppreprocessingstepswithPreprocessingAPIandperforminference
----------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Intuitively,preprocessingAPIconsistsofthefollowingparts:

-Tensor-declaresuserdataformat,likeshape,layout,precision,
colorformatfromactualuser’sdata.
-Steps-describessequenceofpreprocessingstepswhichneedtobe
appliedtouserdata.
-Model-specifiesmodeldataformat.Usually,precisionandshapeare
alreadyknownformodel,onlyadditionalinformation,likelayoutcan
bespecified.

Graphmodificationsofamodelshallbeperformedafterthemodelis
readfromadriveandbeforeitisloadedontheactualdevice.

Pre-processingsupportfollowingoperations(please,seemoredetails
`here<https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__dev__exec__model.html#_CPPv3N2ov10preprocess15PreProcessStepsD0Ev>`__)

-Mean/ScaleNormalization
-ConvertingPrecision
-Convertinglayout(transposing)
-ResizingImage
-ColorConversion
-CustomOperations

ConvertmodeltoOpenVINOIRwithmodelconversionAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Theoptionsforpreprocessingarenotrequired.

..code::ipython3

ir_path=model_dir/"ir_model"/f"{model_name}.xml"

ppp_model=None

ifir_path.exists():
ppp_model=core.read_model(model=ir_path)
print(f"ModelinOpenVINOformatalreadyexists:{ir_path}")
else:
ppp_model=ov.convert_model(model_path,input=[1,299,299,3])
ov.save_model(ppp_model,str(ir_path))

Create``PrePostProcessor``Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The
`PrePostProcessor()<https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1preprocess_1_1_pre_post_processor.html>`__
classenablesspecifyingthepreprocessingandpostprocessingstepsfor
amodel.

..code::ipython3

fromopenvino.preprocessimportPrePostProcessor

ppp=PrePostProcessor(ppp_model)

DeclareUser’sDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Toaddressparticularinputofamodel/preprocessor,usethe
``PrePostProcessor.input(input_name)``method.Ifthemodelhasonlyone
input,thensimple``PrePostProcessor.input()``willgetareferenceto
pre-processingbuilderforthisinput(atensor,thesteps,amodel).In
general,whenamodelhasmultipleinputs/outputs,eachonecanbe
addressedbyatensornameorbyitsindex.Bydefault,information
aboutuser’sinputtensorwillbeinitializedtosamedata
(type/shape/etc)asmodel’sinputparameter.Userapplicationcan
overrideparticularparametersaccordingtoapplication’sdata.Referto
thefollowing
`page<https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__dev__exec__model.html#_CPPv3N2ov10preprocess9InputInfo6tensorEv>`__
formoreinformationaboutparametersforoverriding.

Belowisallthespecifiedinputinformation:

-Precisionis``U8``(unsigned8-bitinteger).
-Sizeisnon-fixed,setupofonedeterminedshapesizecanbedone
with``.set_shape([1,577,800,3])``
-Layoutis``“NHWC”``.Itmeans,forexample:height=577,width=800,
channels=3.

Theheightandwidtharenecessaryforresizing,andchannelsareneeded
formean/scalenormalization.

..code::ipython3

#setupformantofdata
ppp.input().tensor().set_element_type(ov.Type.u8).set_spatial_dynamic_shape().set_layout(ov.Layout("NHWC"))




..parsed-literal::

<openvino._pyopenvino.preprocess.InputTensorInfoat0x7f19306c9630>



DeclaringModelLayout
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Modelinputalreadyhasinformationaboutprecisionandshape.
PreprocessingAPIisnotintendedtomodifythis.Theonlythingthat
maybespecifiedisinputdata
`layout<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview.html>`__.

..code::ipython3

input_layer_ir=next(iter(ppp_model.inputs))
print(f"Theinputshapeofthemodelis{input_layer_ir.shape}")

ppp.input().model().set_layout(ov.Layout("NHWC"))


..parsed-literal::

Theinputshapeofthemodelis[1,299,299,3]




..parsed-literal::

<openvino._pyopenvino.preprocess.InputModelInfoat0x7f1ac05d62f0>



PreprocessingSteps
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,thesequenceofpreprocessingstepscanbedefined.Formore
informationaboutpreprocessingsteps,see
`here<https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html>`__.

Performthefollowing:

-Convert``U8``to``FP32``precision.
-Resizetoheight/widthofamodel.Beawarethatifamodelaccepts
dynamicsize,forexample,``{?,3,?,?}``resizewillnotknowhow
toresizethepicture.Therefore,inthiscase,targetheight/width
shouldbespecified.Formoredetails,seealsothe
`PreProcessSteps.resize()<https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html#openvino.preprocess.PreProcessSteps.resize>`__.
-Subtractmeanfromeachchannel.
-Divideeachpixeldatatoappropriatescalevalue.

Thereisnoneedtospecifyconversionlayout.Iflayoutsaredifferent,
thensuchconversionwillbeaddedexplicitly.

..code::ipython3

fromopenvino.preprocessimportResizeAlgorithm

ppp.input().preprocess().convert_element_type(ov.Type.f32).resize(ResizeAlgorithm.RESIZE_LINEAR).mean([127.5,127.5,127.5]).scale([127.5,127.5,127.5])




..parsed-literal::

<openvino._pyopenvino.preprocess.PreProcessStepsat0x7f19306c3eb0>



IntegratingStepsintoaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Oncethepreprocessingstepshavebeenfinished,themodelcanbe
finallybuilt.Itispossibletodisplay``PrePostProcessor``
configurationfordebuggingpurposes.

..code::ipython3

print(f"Dumppreprocessor:{ppp}")
model_with_preprocess=ppp.build()


..parsed-literal::

Dumppreprocessor:Input"input_1":
User'sinputtensor:[1,?,?,3],[N,H,W,C],u8
Model'sexpectedtensor:[1,299,299,3],[N,H,W,C],f32
Pre-processingsteps(4):
converttype(f32):([1,?,?,3],[N,H,W,C],u8)->([1,?,?,3],[N,H,W,C],f32)
resizetomodelwidth/height:([1,?,?,3],[N,H,W,C],f32)->([1,299,299,3],[N,H,W,C],f32)
mean(127.5,127.5,127.5):([1,299,299,3],[N,H,W,C],f32)->([1,299,299,3],[N,H,W,C],f32)
scale(127.5,127.5,127.5):([1,299,299,3],[N,H,W,C],f32)->([1,299,299,3],[N,H,W,C],f32)



Loadmodelandperforminference
--------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defprepare_image_api_preprocess(image_path,model=None):
image=cv2.imread(image_path)
input_tensor=np.expand_dims(image,0)
returninput_tensor


compiled_model_with_preprocess_api=core.compile_model(model=ppp_model,device_name=device.value)

ppp_output_layer=compiled_model_with_preprocess_api.output(0)

ppp_input_tensor=prepare_image_api_preprocess(image_path)
results=compiled_model_with_preprocess_api(ppp_input_tensor)[ppp_output_layer][0]

Fitimagemanuallyandperforminference
----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Loadthemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

model=core.read_model(model=ir_path)
compiled_model=core.compile_model(model=model,device_name=device.value)

Loadimageandfitittomodelinput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defmanual_image_preprocessing(path_to_image,compiled_model):
input_layer_ir=next(iter(compiled_model.inputs))

#N,H,W,C=batchsize,height,width,numberofchannels
N,H,W,C=input_layer_ir.shape

#loadimage,imagewillberesizedtomodelinputsizeandconvertedtoRGB
img=tf.keras.preprocessing.image.load_img(image_path,target_size=(H,W),color_mode="rgb")

x=tf.keras.preprocessing.image.img_to_array(img)
x=np.expand_dims(x,axis=0)

#willscaleinputpixelsbetween-1and1
input_tensor=tf.keras.applications.inception_resnet_v2.preprocess_input(x)

returninput_tensor


input_tensor=manual_image_preprocessing(image_path,compiled_model)
print(f"Theshapeoftheimageis{input_tensor.shape}")
print(f"Thedatatypeoftheimageis{input_tensor.dtype}")


..parsed-literal::

Theshapeoftheimageis(1,299,299,3)
Thedatatypeoftheimageisfloat32


Performinference
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

output_layer=compiled_model.output(0)

result=compiled_model(input_tensor)[output_layer]

Compareresults
---------------

`backtotop⬆️<#table-of-contents>`__

Compareresultsononeimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defcheck_results(input_tensor,compiled_model,imagenet_classes):
output_layer=compiled_model.output(0)

results=compiled_model(input_tensor)[output_layer][0]

top_indices=np.argsort(results)[-5:][::-1]
top_softmax=results[top_indices]

forindex,softmax_probabilityinzip(top_indices,top_softmax):
print(f"{imagenet_classes[index]},{softmax_probability:.5f}")

returntop_indices,top_softmax


#Converttheinferenceresulttoaclassname.
imagenet_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
directory="data",
)
imagenet_classes=imagenet_filename.read_text().splitlines()
imagenet_classes=["background"]+imagenet_classes

#getresultforinferencewithpreprocessingapi
print("ResultofinferencewithPreprocessingAPI:")
res=check_results(ppp_input_tensor,compiled_model_with_preprocess_api,imagenet_classes)

print("\n")

#getresultforinferencewiththemanualpreparingoftheimage
print("Resultofinferencewithmanualimagesetup:")
res=check_results(input_tensor,compiled_model,imagenet_classes)



..parsed-literal::

data/imagenet_2012.txt:0%||0.00/30.9k[00:00<?,?B/s]


..parsed-literal::

ResultofinferencewithPreprocessingAPI:
n02099601goldenretriever,0.80560
n02098413Lhasa,Lhasaapso,0.10039
n02108915Frenchbulldog,0.01915
n02111129Leonberg,0.00825
n02097047miniatureschnauzer,0.00294


Resultofinferencewithmanualimagesetup:
n02098413Lhasa,Lhasaapso,0.76843
n02099601goldenretriever,0.19322
n02111129Leonberg,0.00720
n02097047miniatureschnauzer,0.00287
n02100877Irishsetter,redsetter,0.00115


Compareperformance
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defcheck_performance(compiled_model,preprocessing_function=None):
num_images=1000

start=time.perf_counter()

for_inrange(num_images):
input_tensor=preprocessing_function(image_path,compiled_model)
compiled_model(input_tensor)

end=time.perf_counter()
time_ir=end-start

returntime_ir,num_images


time_ir,num_images=check_performance(compiled_model,manual_image_preprocessing)
print(f"IRmodelinOpenVINORuntime/CPUwithmanualimagepreprocessing:{time_ir/num_images:.4f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}")

time_ir,num_images=check_performance(compiled_model_with_preprocess_api,prepare_image_api_preprocess)
print(f"IRmodelinOpenVINORuntime/CPUwithpreprocessingAPI:{time_ir/num_images:.4f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}")


..parsed-literal::

IRmodelinOpenVINORuntime/CPUwithmanualimagepreprocessing:0.0153secondsperimage,FPS:65.49
IRmodelinOpenVINORuntime/CPUwithpreprocessingAPI:0.0138secondsperimage,FPS:72.21


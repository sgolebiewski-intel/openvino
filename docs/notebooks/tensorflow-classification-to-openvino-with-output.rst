ConvertaTensorFlowModeltoOpenVINO™
=======================================

ThisshorttutorialshowshowtoconvertaTensorFlow
`MobileNetV3<https://docs.openvino.ai/2024/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
imageclassificationmodeltoOpenVINO`Intermediate
Representation<https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINOIR)format,using`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
AftercreatingtheOpenVINOIR,loadthemodelin`OpenVINO
Runtime<https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
anddoinferencewithasampleimage.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Settings<#settings>`__
-`Downloadmodel<#download-model>`__
-`ConvertaModeltoOpenVINOIR
Format<#convert-a-model-to-openvino-ir-format>`__

-`ConvertaTensorFlowModeltoOpenVINOIR
Format<#convert-a-tensorflow-model-to-openvino-ir-format>`__

-`TestInferenceontheConverted
Model<#test-inference-on-the-converted-model>`__

-`LoadtheModel<#load-the-model>`__

-`Selectinferencedevice<#select-inference-device>`__

-`GetModelInformation<#get-model-information>`__
-`LoadanImage<#load-an-image>`__
-`DoInference<#do-inference>`__

-`Timing<#timing>`__

..code::ipython3

importplatform

#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0""opencv-python"
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
%pipinstall-qtf_kerastensorflow_hubtqdm


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


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos
importtime
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

Settings
--------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Thepathsofthesourceandconvertedmodels.
model_dir=Path("model")
model_dir.mkdir(exist_ok=True)

model_path=Path("model/v3-small_224_1.0_float")

ir_path=Path("model/v3-small_224_1.0_float.xml")

Downloadmodel
--------------

`backtotop⬆️<#table-of-contents>`__

Loadmodelusing`tf.keras.applications
api<https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small>`__
andsaveittothedisk.

..code::ipython3

model=tf.keras.applications.MobileNetV3Small()
model.save(model_path)


..parsed-literal::

WARNING:tensorflow:`input_shape`isundefinedornon-square,or`rows`isnot224.Weightsforinputshape(224,224)willbeloadedasthedefault.


..parsed-literal::

2024-07-1304:04:30.355686:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1304:04:30.355861:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


..parsed-literal::

WARNING:tensorflow:Compiledtheloadedmodel,butthecompiledmetricshaveyettobebuilt.`model.compile_metrics`willbeemptyuntilyoutrainorevaluatethemodel.


..parsed-literal::

WARNING:absl:Founduntracedfunctionssuchas_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_op,_jit_compiled_convolution_opwhilesaving(showing5of54).Thesefunctionswillnotbedirectlycallableafterloading.


..parsed-literal::

INFO:tensorflow:Assetswrittento:model/v3-small_224_1.0_float/assets


..parsed-literal::

INFO:tensorflow:Assetswrittento:model/v3-small_224_1.0_float/assets


ConvertaModeltoOpenVINOIRFormat
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

ConvertaTensorFlowModeltoOpenVINOIRFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

UsethemodelconversionPythonAPItoconverttheTensorFlowmodelto
OpenVINOIR.The``ov.convert_model``functionacceptpathtosaved
modeldirectoryandreturnsOpenVINOModelclassinstancewhich
representsthismodel.Obtainedmodelisreadytouseandtobeloaded
onadeviceusing``ov.compile_model``orcanbesavedonadiskusing
the``ov.save_model``function.Seethe
`tutorial<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html>`__
formoreinformationaboutusingmodelconversionAPIwithTensorFlow
models.

..code::ipython3

#RunmodelconversionAPIiftheIRmodelfiledoesnotexist
ifnotir_path.exists():
print("ExportingTensorFlowmodeltoIR...Thismaytakeafewminutes.")
ov_model=ov.convert_model(model_path,input=[[1,224,224,3]])
ov.save_model(ov_model,ir_path)
else:
print(f"IRmodel{ir_path}alreadyexists.")


..parsed-literal::

ExportingTensorFlowmodeltoIR...Thismaytakeafewminutes.


TestInferenceontheConvertedModel
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

LoadtheModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()
model=core.read_model(ir_path)

Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

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

compiled_model=core.compile_model(model=model,device_name=device.value)

GetModelInformation
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

input_key=compiled_model.input(0)
output_key=compiled_model.output(0)
network_input_shape=input_key.shape

LoadanImage
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Loadanimage,resizeit,andconvertittotheinputshapeofthe
network.

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)

#TheMobileNetnetworkexpectsimagesinRGBformat.
image=cv2.cvtColor(cv2.imread(filename=str(image_filename)),code=cv2.COLOR_BGR2RGB)

#Resizetheimagetothenetworkinputshape.
resized_image=cv2.resize(src=image,dsize=(224,224))

#Transposetheimagetothenetworkinputshape.
input_image=np.expand_dims(resized_image,0)

plt.imshow(image);



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]



..image::tensorflow-classification-to-openvino-with-output_files/tensorflow-classification-to-openvino-with-output_19_1.png


DoInference
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

result=compiled_model(input_image)[output_key]

result_index=np.argmax(result)

..code::ipython3

#Downloadthedatasetsfromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
directory="data",
)

#Converttheinferenceresulttoaclassname.
imagenet_classes=image_filename.read_text().splitlines()

imagenet_classes[result_index]



..parsed-literal::

data/imagenet_2012.txt:0%||0.00/30.9k[00:00<?,?B/s]




..parsed-literal::

'n02099267flat-coatedretriever'



Timing
------

`backtotop⬆️<#table-of-contents>`__

Measurethetimeittakestodoinferenceonthousandimages.Thisgives
anindicationofperformance.Formoreaccuratebenchmarking,usethe
`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
inOpenVINO.Notethatmanyoptimizationsarepossibletoimprovethe
performance.

..code::ipython3

num_images=1000

start=time.perf_counter()

for_inrange(num_images):
compiled_model([input_image])

end=time.perf_counter()
time_ir=end-start

print(f"IRmodelinOpenVINORuntime/CPU:{time_ir/num_images:.4f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}")


..parsed-literal::

IRmodelinOpenVINORuntime/CPU:0.0011secondsperimage,FPS:933.99


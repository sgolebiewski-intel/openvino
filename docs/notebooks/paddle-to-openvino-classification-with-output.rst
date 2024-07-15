ConvertaPaddlePaddleModeltoOpenVINO™IR
============================================

ThisnotebookshowshowtoconvertaMobileNetV3modelfrom
`PaddleHub<https://github.com/PaddlePaddle/PaddleHub>`__,pre-trained
onthe`ImageNet<https://www.image-net.org>`__dataset,toOpenVINOIR.
Italsoshowshowtoperformclassificationinferenceonasampleimage,
using`OpenVINO
Runtime<https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
andcomparestheresultsofthe
`PaddlePaddle<https://github.com/PaddlePaddle/Paddle>`__modelwiththe
IRmodel.

Sourceofthe
`model<https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v3_large_imagenet_ssld&en_category=ImageClassification>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Imports<#imports>`__
-`Settings<#settings>`__

-`ShowInferenceonPaddlePaddle
Model<#show-inference-on-paddlepaddle-model>`__
-`ConverttheModeltoOpenVINOIR
Format<#convert-the-model-to-openvino-ir-format>`__
-`Selectinferencedevice<#select-inference-device>`__
-`ShowInferenceonOpenVINO
Model<#show-inference-on-openvino-model>`__
-`TimingandComparison<#timing-and-comparison>`__
-`Selectinferencedevice<#select-inference-device>`__
-`References<#references>`__

Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

ifplatform.system()=="Windows":
%pipinstall-q"paddlepaddle>=2.5.1,<2.6.0"
else:
%pipinstall-q"paddlepaddle>=2.5.1"
%pipinstall-q"paddleclas>=2.5.2"--no-deps
%pipinstall-q"prettytable""ujson""visualdl>=2.5.3""faiss-cpu>=1.7.1"Pillowtqdm
#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
paddleclas2.5.2requireseasydict,whichisnotinstalled.
paddleclas2.5.2requiresgast==0.3.3,butyouhavegast0.4.0whichisincompatible.
paddleclas2.5.2requiresopencv-python==4.6.0.66,butyouhaveopencv-python4.10.0.84whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

ifplatform.system()=="Linux":
!wgethttp://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
!sudodpkg-ilibssl1.1_1.1.1f-1ubuntu2.19_amd64.deb


..parsed-literal::

--2024-07-1301:22:03--http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
Resolvingproxy-dmz.intel.com(proxy-dmz.intel.com)...10.241.208.166
Connectingtoproxy-dmz.intel.com(proxy-dmz.intel.com)|10.241.208.166|:911...connected.
Proxyrequestsent,awaitingresponse...404NotFound
2024-07-1301:22:03ERROR404:NotFound.

dpkg:error:cannotaccessarchive'libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb':Nosuchfileordirectory


..code::ipython3

importtime
importtarfile
frompathlibimportPath

importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov
frompaddleclasimportPaddleClas
fromPILimportImage

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file


..parsed-literal::

2024-07-1301:22:05INFO:LoadingfaisswithAVX512support.
2024-07-1301:22:05INFO:SuccessfullyloadedfaisswithAVX512support.


Settings
~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Set``IMAGE_FILENAME``tothefilenameofanimagetouse.Set
``MODEL_NAME``tothePaddlePaddlemodeltodownloadfromPaddleHub.
``MODEL_NAME``willalsobethebasenamefortheIRmodel.Thenotebook
istestedwiththe
`MobileNetV3_large_x1_0<https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/models/Mobile_en.md>`__
model.Othermodelsmayusedifferentpreprocessingmethodsand
thereforerequiresomemodificationtogetthesameresultsonthe
originalandconvertedmodel.

Firstofall,weneedtodownloadandunpackmodelfiles.Thefirsttime
yourunthisnotebook,thePaddlePaddlemodelisdownloadedfrom
PaddleHub.Thismaytakeawhile.

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
img=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_close.png",
directory="data",
)

IMAGE_FILENAME=img.as_posix()

MODEL_NAME="MobileNetV3_large_x1_0"
MODEL_DIR=Path("model")
ifnotMODEL_DIR.exists():
MODEL_DIR.mkdir()
MODEL_URL="https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar".format(MODEL_NAME)
download_file(MODEL_URL,directory=MODEL_DIR)
file=tarfile.open(MODEL_DIR/"{}_infer.tar".format(MODEL_NAME))
res=file.extractall(MODEL_DIR)
ifnotres:
print(f'ModelExtractedto"./{MODEL_DIR}".')
else:
print("ErrorExtractingthemodel.Pleasecheckthenetwork.")



..parsed-literal::

data/coco_close.png:0%||0.00/133k[00:00<?,?B/s]



..parsed-literal::

model/MobileNetV3_large_x1_0_infer.tar:0%||0.00/19.5M[00:00<?,?B/s]


..parsed-literal::

ModelExtractedto"./model".


ShowInferenceonPaddlePaddleModel
------------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthenextcell,weloadthemodel,loadanddisplayanimage,do
inferenceonthatimage,andthenshowthetopthreepredictionresults.

..code::ipython3

classifier=PaddleClas(inference_model_dir=MODEL_DIR/"{}_infer".format(MODEL_NAME))
result=next(classifier.predict(IMAGE_FILENAME))
class_names=result[0]["label_names"]
scores=result[0]["scores"]
image=Image.open(IMAGE_FILENAME)
plt.imshow(image)
forclass_name,softmax_probabilityinzip(class_names,scores):
print(f"{class_name},{softmax_probability:.5f}")


..parsed-literal::

[2024/07/1301:22:31]ppclsWARNING:ThecurrentrunningenvironmentdoesnotsupporttheuseofGPU.CPUhasbeenusedinstead.
Labradorretriever,0.75138
Germanshort-hairedpointer,0.02373
GreatDane,0.01848
Rottweiler,0.01435
flat-coatedretriever,0.01144



..image::paddle-to-openvino-classification-with-output_files/paddle-to-openvino-classification-with-output_8_1.png


``classifier.predict()``takesanimagefilename,readstheimage,
preprocessestheinput,thenreturnstheclasslabelsandscoresofthe
image.Preprocessingtheimageisdonebehindthescenes.The
classificationmodelreturnsanarraywithfloatingpointvaluesfor
eachofthe1000ImageNetclasses.Thehigherthevalue,themore
confidentthenetworkisthattheclassnumbercorrespondingtothat
value(theindexofthatvalueinthenetworkoutputarray)istheclass
numberfortheimage.

ToseePaddlePaddle’simplementationfortheclassificationfunctionand
forloadingandpreprocessingdata,uncommentthenexttwocells.

..code::ipython3

#classifier??

..code::ipython3

#classifier.get_config()

The``classifier.get_config()``moduleshowsthepreprocessing
configurationforthemodel.Itshouldshowthatimagesarenormalized,
resizedandcropped,andthattheBGRimageisconvertedtoRGBbefore
propagatingitthroughthenetwork.Inthenextcell,wegetthe
``classifier.predictror.preprocess_ops``propertythatreturnslistof
preprocessingoperationstodoinferenceontheOpenVINOIRmodelusing
thesamemethod.

..code::ipython3

preprocess_ops=classifier.predictor.preprocess_ops


defprocess_image(image):
foropinpreprocess_ops:
image=op(image)
returnimage

Itisusefultoshowtheoutputofthe``process_image()``function,to
seetheeffectofcroppingandresizing.Becauseofthenormalization,
thecolorswilllookstrange,and``matplotlib``willwarnabout
clippingvalues.

..code::ipython3

pil_image=Image.open(IMAGE_FILENAME)
processed_image=process_image(np.array(pil_image))
print(f"Processedimageshape:{processed_image.shape}")
#Processedimageisin(C,H,W)format,convertto(H,W,C)toshowtheimage
plt.imshow(np.transpose(processed_image,(1,2,0)))


..parsed-literal::

2024-07-1301:22:31WARNING:ClippinginputdatatothevalidrangeforimshowwithRGBdata([0..1]forfloatsor[0..255]forintegers).


..parsed-literal::

Processedimageshape:(3,224,224)




..parsed-literal::

<matplotlib.image.AxesImageat0x7fa888475730>




..image::paddle-to-openvino-classification-with-output_files/paddle-to-openvino-classification-with-output_15_3.png


Todecodethelabelspredictedbythemodeltonamesofclasses,weneed
tohaveamappingbetweenthem.Themodelconfigcontainsinformation
about``class_id_map_file``,whichstoressuchmapping.Thecodebelow
showshowtoparsethemappingintoadictionarytousewiththe
OpenVINOmodel.

..code::ipython3

class_id_map_file=classifier.get_config()["PostProcess"]["Topk"]["class_id_map_file"]
class_id_map={}
withopen(class_id_map_file,"r")asfin:
lines=fin.readlines()
forlineinlines:
partition=line.split("\n")[0].partition("")
class_id_map[int(partition[0])]=str(partition[-1])

ConverttheModeltoOpenVINOIRFormat
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

CalltheOpenVINOModelConversionAPItoconvertthePaddlePaddlemodel
toOpenVINOIR,withFP32precision.``ov.convert_model``function
acceptpathtoPaddlePaddlemodelandreturnsOpenVINOModelclass
instancewhichrepresentsthismodel.Obtainedmodelisreadytouseand
loadingondeviceusing``ov.compile_model``orcanbesavedondisk
using``ov.save_model``function.Seethe`ModelConversion
Guide<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
formoreinformationabouttheModelConversionAPI.

..code::ipython3

model_xml=Path(MODEL_NAME).with_suffix(".xml")
ifnotmodel_xml.exists():
ov_model=ov.convert_model("model/MobileNetV3_large_x1_0_infer/inference.pdmodel")
ov.save_model(ov_model,str(model_xml))
else:
print(f"{model_xml}alreadyexists.")

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



ShowInferenceonOpenVINOModel
--------------------------------

`backtotop⬆️<#table-of-contents>`__

LoadtheIRmodel,getmodelinformation,loadtheimage,doinference,
converttheinferencetoameaningfulresult,andshowtheoutput.See
the`OpenVINORuntimeAPI
Notebook<openvino-api-with-output.html>`__formoreinformation.

..code::ipython3

#LoadOpenVINORuntimeandOpenVINOIRmodel
core=ov.Core()
model=core.read_model(model_xml)
compiled_model=core.compile_model(model=model,device_name=device.value)

#Getmodeloutput
output_layer=compiled_model.output(0)

#Read,show,andpreprocessinputimage
#Seethe"ShowInferenceonPaddlePaddleModel"sectionforsourceofprocess_image
image=Image.open(IMAGE_FILENAME)
plt.imshow(image)
input_image=process_image(np.array(image))[None,]

#Doinference
ov_result=compiled_model([input_image])[output_layer][0]

#findthetopthreevalues
top_indices=np.argsort(ov_result)[-3:][::-1]
top_scores=ov_result[top_indices]

#Converttheinferenceresultstoclassnames,usingthesamelabelsasthePaddlePaddleclassifier
forindex,softmax_probabilityinzip(top_indices,top_scores):
print(f"{class_id_map[index]},{softmax_probability:.5f}")


..parsed-literal::

Labradorretriever,0.74909
Germanshort-hairedpointer,0.02368
GreatDane,0.01873



..image::paddle-to-openvino-classification-with-output_files/paddle-to-openvino-classification-with-output_23_1.png


TimingandComparison
---------------------

`backtotop⬆️<#table-of-contents>`__

Measurethetimeittakestodoinferenceonfiftyimagesandcompare
theresult.Thetiminginformationgivesanindicationofperformance.
Forafaircomparison,weincludethetimeittakestoprocessthe
image.Formoreaccuratebenchmarking,usethe`OpenVINObenchmark
tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
Notethatmanyoptimizationsarepossibletoimprovetheperformance.

..code::ipython3

num_images=50

image=Image.open(fp=IMAGE_FILENAME)

..code::ipython3

#Showdeviceinformation
core=ov.Core()
devices=core.available_devices

fordevice_nameindevices:
device_full_name=core.get_property(device_name,"FULL_DEVICE_NAME")
print(f"{device_name}:{device_full_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)i9-10920XCPU@3.50GHz


..code::ipython3

#ShowinferencespeedonPaddlePaddlemodel
start=time.perf_counter()
for_inrange(num_images):
result=next(classifier.predict(np.array(image)))
end=time.perf_counter()
time_ir=end-start
print(f"PaddlePaddlemodelonCPU:{time_ir/num_images:.4f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}\n")
print("PaddlePaddleresult:")
class_names=result[0]["label_names"]
scores=result[0]["scores"]
forclass_name,softmax_probabilityinzip(class_names,scores):
print(f"{class_name},{softmax_probability:.5f}")
plt.imshow(image);


..parsed-literal::

PaddlePaddlemodelonCPU:0.0076secondsperimage,FPS:131.17

PaddlePaddleresult:
Labradorretriever,0.75138
Germanshort-hairedpointer,0.02373
GreatDane,0.01848
Rottweiler,0.01435
flat-coatedretriever,0.01144



..image::paddle-to-openvino-classification-with-output_files/paddle-to-openvino-classification-with-output_27_1.png


Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#ShowinferencespeedonOpenVINOIRmodel
compiled_model=core.compile_model(model=model,device_name=device.value)
output_layer=compiled_model.output(0)


start=time.perf_counter()
input_image=process_image(np.array(image))[None,]
for_inrange(num_images):
ie_result=compiled_model([input_image])[output_layer][0]
top_indices=np.argsort(ie_result)[-5:][::-1]
top_softmax=ie_result[top_indices]

end=time.perf_counter()
time_ir=end-start

print(f"OpenVINOIRmodelinOpenVINORuntime({device.value}):{time_ir/num_images:.4f}"f"secondsperimage,FPS:{num_images/time_ir:.2f}")
print()
print("OpenVINOresult:")
forindex,softmax_probabilityinzip(top_indices,top_softmax):
print(f"{class_id_map[index]},{softmax_probability:.5f}")
plt.imshow(image);


..parsed-literal::

OpenVINOIRmodelinOpenVINORuntime(AUTO):0.0029secondsperimage,FPS:348.21

OpenVINOresult:
Labradorretriever,0.74909
Germanshort-hairedpointer,0.02368
GreatDane,0.01873
Rottweiler,0.01448
flat-coatedretriever,0.01153



..image::paddle-to-openvino-classification-with-output_files/paddle-to-openvino-classification-with-output_30_1.png


References
----------

`backtotop⬆️<#table-of-contents>`__

-`PaddleClas<https://github.com/PaddlePaddle/PaddleClas>`__
-`OpenVINOPaddlePaddle
support<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-paddle.html>`__

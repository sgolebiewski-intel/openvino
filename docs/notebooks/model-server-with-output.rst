HelloModelServer
==================

IntroductiontoOpenVINO™ModelServer(OVMS).

WhatisModelServing?
----------------------

Amodelserverhostsmodelsandmakesthemaccessibletosoftware
componentsoverstandardnetworkprotocols.Aclientsendsarequestto
themodelserver,whichperformsinferenceandsendsaresponsebackto
theclient.Modelservingoffersmanyadvantagesforefficientmodel
deployment:

-Remoteinferenceenablesusinglightweightclientswithonlythe
necessaryfunctionstoperformAPIcallstoedgeorcloud
deployments.
-Applicationsareindependentofthemodelframework,hardwaredevice,
andinfrastructure.
-ClientapplicationsinanyprogramminglanguagethatsupportsRESTor
gRPCcallscanbeusedtoruninferenceremotelyonthemodelserver.
-Clientsrequirefewerupdatessinceclientlibrarieschangevery
rarely.
-Modeltopologyandweightsarenotexposeddirectlytoclient
applications,makingiteasiertocontrolaccesstothemodel.
-Idealarchitectureformicroservices-basedapplicationsand
deploymentsincloudenvironments–includingKubernetesand
OpenShiftclusters.
-Efficientresourceutilizationwithhorizontalandverticalinference
scaling.

..figure::https://user-images.githubusercontent.com/91237924/215658773-4720df00-3b95-4a84-85a2-40f06138e914.png
:alt:ovms_diagram

ovms_diagram

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`ServingwithOpenVINOModel
Server<#serving-with-openvino-model-server>`__
-`Step1:PrepareDocker<#step-1-prepare-docker>`__
-`Step2:PreparingaModel
Repository<#step-2-preparing-a-model-repository>`__
-`Step3:StarttheModelServer
Container<#step-3-start-the-model-server-container>`__
-`Step4:PreparetheExampleClient
Components<#step-4-prepare-the-example-client-components>`__

-`Prerequisites<#prerequisites>`__
-`Imports<#imports>`__
-`RequestModelStatus<#request-model-status>`__
-`RequestModelMetadata<#request-model-metadata>`__
-`Loadinputimage<#load-input-image>`__
-`RequestPredictiononaNumpy
Array<#request-prediction-on-a-numpy-array>`__
-`Visualization<#visualization>`__

-`References<#references>`__

ServingwithOpenVINOModelServer
----------------------------------

`backtotop⬆️<#table-of-contents>`__OpenVINOModelServer(OVMS)is
ahigh-performancesystemforservingmodels.ImplementedinC++for
scalabilityandoptimizedfordeploymentonIntelarchitectures,the
modelserverusesthesamearchitectureandAPIasTensorFlowServing
andKServewhileapplyingOpenVINOforinferenceexecution.Inference
serviceisprovidedviagRPCorRESTAPI,makingdeployingnew
algorithmsandAIexperimentseasy.

..figure::https://user-images.githubusercontent.com/91237924/215658767-0e0fc221-aed0-4db1-9a82-6be55f244dba.png
:alt:ovms_high_level

ovms_high_level

ToquicklystartusingOpenVINO™ModelServer,followthesesteps:

Step1:PrepareDocker
----------------------

`backtotop⬆️<#table-of-contents>`__Install`Docker
Engine<https://docs.docker.com/engine/install/>`__,includingits
`post-installation<https://docs.docker.com/engine/install/linux-postinstall/>`__
steps,onyourdevelopmentsystem.Toverifyinstallation,testit,
usingthefollowingcommand.Whenitisready,itwilldisplayatest
imageandamessage.

..code::ipython3

!dockerrunhello-world


..parsed-literal::


HellofromDocker!
Thismessageshowsthatyourinstallationappearstobeworkingcorrectly.

Togeneratethismessage,Dockertookthefollowingsteps:
1.TheDockerclientcontactedtheDockerdaemon.
2.TheDockerdaemonpulledthe"hello-world"imagefromtheDockerHub.
(amd64)
3.TheDockerdaemoncreatedanewcontainerfromthatimagewhichrunsthe
executablethatproducestheoutputyouarecurrentlyreading.
4.TheDockerdaemonstreamedthatoutputtotheDockerclient,whichsentit
toyourterminal.

Totrysomethingmoreambitious,youcanrunanUbuntucontainerwith:
$dockerrun-itubuntubash

Shareimages,automateworkflows,andmorewithafreeDockerID:
https://hub.docker.com/

Formoreexamplesandideas,visit:
https://docs.docker.com/get-started/



Step2:PreparingaModelRepository
------------------------------------

`backtotop⬆️<#table-of-contents>`__Themodelsneedtobeplaced
andmountedinaparticulardirectorystructureandaccordingtothe
followingrules:

::

treemodels/
models/
├──model1
│├──1
││├──ir_model.bin
││└──ir_model.xml
│└──2
│├──ir_model.bin
│└──ir_model.xml
├──model2
│└──1
│├──ir_model.bin
│├──ir_model.xml
│└──mapping_config.json
├──model3
│└──1
│└──model.onnx
├──model4
│└──1
│├──model.pdiparams
│└──model.pdmodel
└──model5
└──1
└──TF_fronzen_model.pb

-Eachmodelshouldbestoredinadedicateddirectory,forexample,
model1andmodel2.

-Eachmodeldirectoryshouldincludeasub-folderforeachofits
versions(1,2,etc).Theversionsandtheirfoldernamesshouldbe
positiveintegervalues.

-Notethatinexecution,theversionsareenabledaccordingtoa
pre-definedversionpolicy.Iftheclientdoesnotspecifythe
versionnumberinparameters,bydefault,thelatestversionis
served.

-Everyversionfoldermustincludemodelfiles,thatis,``.bin``and
``.xml``forOpenVINOIR,``.onnx``forONNX,``.pdiparams``and
``.pdmodel``forPaddlePaddle,and``.pb``forTensorFlow.Thefile
namecanbearbitrary.

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

..code::ipython3

importos

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

dedicated_dir="models"
model_name="detection"
model_version="1"

MODEL_DIR=f"{dedicated_dir}/{model_name}/{model_version}"
XML_PATH="horizontal-text-detection-0001.xml"
BIN_PATH="horizontal-text-detection-0001.bin"
os.makedirs(MODEL_DIR,exist_ok=True)
model_xml_url=(
"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
)
model_bin_url=(
"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"
)

download_file(model_xml_url,XML_PATH,MODEL_DIR)
download_file(model_bin_url,BIN_PATH,MODEL_DIR)



..parsed-literal::

models/detection/1/horizontal-text-detection-0001.xml:0%||0.00/680k[00:00<?,?B/s]



..parsed-literal::

models/detection/1/horizontal-text-detection-0001.bin:0%||0.00/7.39M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/home/ethan/intel/openvino_notebooks/notebooks/model-server/models/detection/1/horizontal-text-detection-0001.bin')



Step3:StarttheModelServerContainer
----------------------------------------

`backtotop⬆️<#table-of-contents>`__Pullandstartthecontainer:

Searchingforanavailableservingportinlocal.

..code::ipython3

importsocket

sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind(("localhost",0))
sock.listen(1)
port=sock.getsockname()[1]
sock.close()
print(f"Port{port}isavailable")

os.environ["port"]=str(port)


..parsed-literal::

Port39801isavailable


..code::ipython3

!dockerrun-d--rm--name="ovms"-v$(pwd)/models:/models-p$port:9000openvino/model_server:latest--model_path/models/detection/--model_namedetection--port9000


..parsed-literal::

64aa9391ba019b3ef26ae3010e5605e38d0a12e3f93bf74b3afb938f39b86ad2


CheckwhethertheOVMScontainerisrunningnormally:

..code::ipython3

!dockerps|grepovms


..parsed-literal::

64aa9391ba01openvino/model_server:latest"/ovms/bin/ovms--mo…"29secondsagoUp28seconds0.0.0.0:37581->9000/tcp,:::37581->9000/tcpovms


TherequiredModelServerparametersarelistedbelow.Foradditional
configurationoptions,seethe`ModelServerParameters
section<https://docs.openvino.ai/2024/ovms_docs_parameters.html>`__.

..raw::html

<tableclass="table">

..raw::html

<colgroup>

..raw::html

<colstyle="width:20%"/>

..raw::html

<colstyle="width:80%"/>

..raw::html

</colgroup>

..raw::html

<tbody>

..raw::html

<trclass="row-odd">

..raw::html

<td>

..raw::html

<p>

–rm

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

removethecontainerwhenexitingtheDockercontainer

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-even">

..raw::html

<td>

..raw::html

<p>

-d

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

runsthecontainerinthebackground

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-odd">

..raw::html

<td>

..raw::html

<p>

-v

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

defineshowtomountthemodelfolderintheDockercontainer

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-even">

..raw::html

<td>

..raw::html

<p>

-p

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

exposesthemodelservingportoutsidetheDockercontainer

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-odd">

..raw::html

<td>

..raw::html

<p>

openvino/model_server:latest

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

representstheimagename;theOVMSbinaryistheDockerentry
point

..container::line

variesbytagandbuildprocess-seetags:
https://hub.docker.com/r/openvino/model_server/tags/forafull
taglist.

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-even">

..raw::html

<td>

..raw::html

<p>

–model_path

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

modellocation,whichcanbe:

..container::line

aDockercontainerpaththatismountedduringstart-up

..container::line

aGoogleCloudStoragepathgs://<bucket>/<model_path>

..container::line

anAWSS3paths3://<bucket>/<model_path>

..container::line

anAzureblobpathaz://<container>/<model_path>

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-odd">

..raw::html

<td>

..raw::html

<p>

–model_name

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

thenameofthemodelinthemodel_path

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-even">

..raw::html

<td>

..raw::html

<p>

–port

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

thegRPCserverport

..raw::html

</td>

..raw::html

</tr>

..raw::html

<trclass="row-odd">

..raw::html

<td>

..raw::html

<p>

–rest_port

..raw::html

</p>

..raw::html

</td>

..raw::html

<td>

..container::line-block

..container::line

theRESTserverport

..raw::html

</td>

..raw::html

</tr>

..raw::html

</tbody>

..raw::html

</table>

Iftheservingportisalreadyinuse,pleaseswitchittoanother
availableportonyoursystem.Forexample:\``-p9020:9000``

Step4:PreparetheExampleClientComponents
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__OpenVINOModelServerexposes
twosetsofAPIs:onecompatiblewith``TensorFlowServing``andanother
one,with``KServeAPI``,forinference.BothAPIsworkon``gRPC``and
``REST``\interfaces.SupportingtwosetsofAPIsmakesOpenVINOModel
Servereasiertoplugintoexistingsystemsthealreadyleverageoneof
theseAPIsforinference.Thisexamplewilldemonstratehowtowritea
TensorFlowServingAPIclientforobjectdetection.

Prerequisites
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Installnecessarypackages.

..code::ipython3

%pipinstall-qovmsclient


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcv2
importnumpyasnp
importmatplotlib.pyplotasplt
fromovmsclientimportmake_grpc_client

RequestModelStatus
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

address="localhost:"+str(port)

#Bindthegrpcaddresstotheclientobject
client=make_grpc_client(address)
model_status=client.get_model_status(model_name=model_name)
print(model_status)


..parsed-literal::

{1:{'state':'AVAILABLE','error_code':0,'error_message':'OK'}}


RequestModelMetadata
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

model_metadata=client.get_model_metadata(model_name=model_name)
print(model_metadata)


..parsed-literal::

{'model_version':1,'inputs':{'image':{'shape':[1,3,704,704],'dtype':'DT_FLOAT'}},'outputs':{'boxes':{'shape':[-1,5],'dtype':'DT_FLOAT'},'labels':{'shape':[-1],'dtype':'DT_INT64'}}}


Loadinputimage
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg",
directory="data",
)

#TextdetectionmodelsexpectanimageinBGRformat.
image=cv2.imread(str(image_filename))
fp_image=image.astype("float32")

#Resizetheimagetomeetnetworkexpectedinputsizes.
input_shape=model_metadata["inputs"]["image"]["shape"]
height,width=input_shape[2],input_shape[3]
resized_image=cv2.resize(fp_image,(height,width))

#Reshapetothenetworkinputshape.
input_image=np.expand_dims(resized_image.transpose(2,0,1),0)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))



..parsed-literal::

data/intel_rnb.jpg:0%||0.00/288k[00:00<?,?B/s]




..parsed-literal::

<matplotlib.image.AxesImageat0x7f254faeec50>




..image::model-server-with-output_files/model-server-with-output_23_2.png


RequestPredictiononaNumpyArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

inputs={"image":input_image}

#Runinferenceonmodelserverandreceivetheresultdata
boxes=client.predict(inputs=inputs,model_name=model_name)["boxes"]

#Removezeroonlyboxes.
boxes=boxes[~np.all(boxes==0,axis=1)]
print(boxes)


..parsed-literal::

[[4.0075238e+028.1240105e+015.6262683e+021.3609659e+025.3646392e-01]
[2.6150497e+026.8225861e+013.8433078e+021.2111545e+024.7504124e-01]
[6.1611401e+022.8000638e+026.6605963e+023.1116574e+024.5030469e-01]
[2.0762566e+026.2619057e+012.3446707e+021.0711832e+023.7426147e-01]
[5.1753296e+025.5611102e+025.4918005e+025.8740009e+023.2477754e-01]
[2.2038467e+014.5390991e+011.8856328e+021.0215196e+022.9959568e-01]]


Visualization
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Foreachdetection,thedescriptionisinthe[x_min,y_min,x_max,y_max,conf]format:
#TheimagepassedhereisinBGRformatwithchangedwidthandheight.Todisplayitincolorsexpectedbymatplotlib,usecvtColorfunction
defconvert_result_to_image(bgr_image,resized_image,boxes,threshold=0.3,conf_labels=True):
#Definecolorsforboxesanddescriptions.
colors={"red":(255,0,0),"green":(0,255,0)}

#Fetchtheimageshapestocalculatearatio.
(real_y,real_x),(resized_y,resized_x)=(
bgr_image.shape[:2],
resized_image.shape[:2],
)
ratio_x,ratio_y=real_x/resized_x,real_y/resized_y

#ConvertthebaseimagefromBGRtoRGBformat.
rgb_image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

#Iteratethroughnon-zeroboxes.
forboxinboxes:
#Pickaconfidencefactorfromthelastplaceinanarray.
conf=box[-1]
ifconf>threshold:
#Convertfloattointandmultiplycornerpositionofeachboxbyxandyratio.
#Iftheboundingboxisfoundatthetopoftheimage,
#positiontheupperboxbarlittlelowertomakeitvisibleontheimage.
(x_min,y_min,x_max,y_max)=[
(int(max(corner_position*ratio_y,10))ifidx%2elseint(corner_position*ratio_x))foridx,corner_positioninenumerate(box[:-1])
]

#Drawaboxbasedontheposition,parametersinrectanglefunctionare:image,start_point,end_point,color,thickness.
rgb_image=cv2.rectangle(rgb_image,(x_min,y_min),(x_max,y_max),colors["green"],3)

#Addtexttotheimagebasedonpositionandconfidence.
#Parametersintextfunctionare:image,text,bottom-left_corner_textfield,font,font_scale,color,thickness,line_type.
ifconf_labels:
rgb_image=cv2.putText(
rgb_image,
f"{conf:.2f}",
(x_min,y_min-10),
cv2.FONT_HERSHEY_SIMPLEX,
0.8,
colors["red"],
1,
cv2.LINE_AA,
)

returnrgb_image

..code::ipython3

plt.figure(figsize=(10,6))
plt.axis("off")
plt.imshow(convert_result_to_image(image,resized_image,boxes,conf_labels=False))




..parsed-literal::

<matplotlib.image.AxesImageat0x7f25490829b0>




..image::model-server-with-output_files/model-server-with-output_28_1.png


Tostopandremovethemodelservercontainer,youcanusethefollowing
command:

..code::ipython3

!dockerstopovms


..parsed-literal::

ovms


References
----------

`backtotop⬆️<#table-of-contents>`__

1.`OpenVINO™ModelServer
documentation<https://docs.openvino.ai/2024/ovms_what_is_openvino_model_server.html>`__
2.`OpenVINO™ModelServerGitHub
repository<https://github.com/openvinotoolkit/model_server/>`__

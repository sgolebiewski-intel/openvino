Text-to-VideoretrievalwithS3DMIL-NCEandOpenVINO
=====================================================

Thistutorialbasedon`theTensorFlow
tutorial<https://www.tensorflow.org/hub/tutorials/text_to_video_retrieval_with_s3d_milnce>`__
thatdemonstrateshowtousethe`S3D
MIL-NCE<https://tfhub.dev/deepmind/mil-nce/s3d/1>`__modelfrom
TensorFlowHubtodotext-to-videoretrievaltofindthemostsimilar
videosforagiventextquery.

MIL-NCEinheritsfromMultipleInstanceLearning(MIL)andNoise
ContrastiveEstimation(NCE).Themethodiscapableofaddressing
visuallymisalignednarrationsfromuncuratedinstructionalvideos.Two
modelvariationsareavailablewithdifferent3DCNNbackbones:I3Dand
S3D.InthistutorialweuseS3Dvariation.Moredetailsaboutthe
trainingandthemodelcanbefoundin`End-to-EndLearningofVisual
RepresentationsfromUncuratedInstructional
Videos<https://arxiv.org/abs/1912.06430>`__paper.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimizeS3DMIL-NCEmodelwithOpenVINO.Anadditionalpart
demonstrateshowtorunquantizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__tospeedupthe
inference.

Thetutorialconsistsofthefollowingsteps:

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Theoriginalinference<#the-original-inference>`__
-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__
-`Compilingmodels<#compiling-models>`__
-`Inference<#inference>`__
-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__

-`Preparedataset<#prepare-dataset>`__
-`Performmodelquantization<#perform-model-quantization>`__

-`Runquantizedmodelinference<#run-quantized-model-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-Uqpip
%pipinstall--upgrade--preopenvino-tokenizersopenvino--extra-index-url"https://storage.openvinotoolkit.org/simple/wheels/nightly"
%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'andpython_version<='3.8'"

%pipinstall-qtensorflow_hubtf_kerasnumpy"opencv-python""nncf>=2.10.0"
ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Lookinginindexes:https://pypi.org/simple,https://storage.openvinotoolkit.org/simple/wheels/nightly
Requirementalreadysatisfied:openvino-tokenizersin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(2024.4.0.0.dev20240712)
Requirementalreadysatisfied:openvinoin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(2024.4.0.dev20240712)
Requirementalreadysatisfied:numpy<2.0.0,>=1.16.6in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino)(1.23.5)
Requirementalreadysatisfied:openvino-telemetry>=2023.2.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino)(2024.1.0)
Requirementalreadysatisfied:packagingin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino)(24.1)
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importos
frompathlibimportPath

importtensorflowastf
importtensorflow_hubashub

importnumpyasnp
importcv2
fromIPythonimportdisplay
importmath

os.environ["TFHUB_CACHE_DIR"]=str(Path("./tfhub_modules").resolve())


..parsed-literal::

2024-07-1302:43:13.726530:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1302:43:13.762325:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1302:43:14.360751:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


Downloadthemodel

..code::ipython3

hub_handle="https://www.kaggle.com/models/deepmind/mil-nce/TensorFlow1/s3d/1"
hub_model=hub.load(hub_handle)


..parsed-literal::

2024-07-1302:43:22.100111:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1302:43:22.100148:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168]retrievingCUDAdiagnosticinformationforhost:iotg-dev-workstation-07
2024-07-1302:43:22.100152:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175]hostname:iotg-dev-workstation-07
2024-07-1302:43:22.100286:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199]libcudareportedversionis:470.223.2
2024-07-1302:43:22.100302:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203]kernelreportedversionis:470.182.3
2024-07-1302:43:22.100306:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


Themodelhas2signatures,oneforgeneratingvideoembeddingsandone
forgeneratingtextembeddings.Wewillusetheseembeddingtofindthe
nearestneighborsintheembeddingspaceasintheoriginaltutorial.
Belowwewilldefineauxiliaryfunctions

..code::ipython3

defgenerate_embeddings(model,input_frames,input_words):
"""Generateembeddingsfromthemodelfromvideoframesandinputwords."""
#Input_framesmustbenormalizedin[0,1]andoftheshapeBatchxTxHxWx3
vision_output=model.signatures["video"](tf.constant(tf.cast(input_frames,dtype=tf.float32)))
text_output=model.signatures["text"](tf.constant(input_words))

returnvision_output["video_embedding"],text_output["text_embedding"]

..code::ipython3

#@titleDefinevideoloadingandvisualizationfunctions{display-mode:"form"}


#UtilitiestoopenvideofilesusingCV2
defcrop_center_square(frame):
y,x=frame.shape[0:2]
min_dim=min(y,x)
start_x=(x//2)-(min_dim//2)
start_y=(y//2)-(min_dim//2)
returnframe[start_y:start_y+min_dim,start_x:start_x+min_dim]


defload_video(video_url,max_frames=32,resize=(224,224)):
path=tf.keras.utils.get_file(os.path.basename(video_url)[-128:],video_url)
cap=cv2.VideoCapture(path)
frames=[]
try:
whileTrue:
ret,frame=cap.read()
ifnotret:
break
frame=crop_center_square(frame)
frame=cv2.resize(frame,resize)
frame=frame[:,:,[2,1,0]]
frames.append(frame)

iflen(frames)==max_frames:
break
finally:
cap.release()
frames=np.array(frames)
iflen(frames)<max_frames:
n_repeat=int(math.ceil(max_frames/float(len(frames))))
frames=frames.repeat(n_repeat,axis=0)
frames=frames[:max_frames]
returnframes/255.0


defdisplay_video(urls):
html="<table>"
html+="<tr><th>Video1</th><th>Video2</th><th>Video3</th></tr><tr>"
forurlinurls:
html+="<td>"
html+='<imgsrc="{}"height="224">'.format(url)
html+="</td>"
html+="</tr></table>"
returndisplay.HTML(html)


defdisplay_query_and_results_video(query,urls,scores):
"""Displayatextqueryandthetopresultvideosandscores."""
sorted_ix=np.argsort(-scores)
html=""
html+="<h2>Inputquery:<i>{}</i></h2><div>".format(query)
html+="Results:<div>"
html+="<table>"
html+="<tr><th>Rank#1,Score:{:.2f}</th>".format(scores[sorted_ix[0]])
html+="<th>Rank#2,Score:{:.2f}</th>".format(scores[sorted_ix[1]])
html+="<th>Rank#3,Score:{:.2f}</th></tr><tr>".format(scores[sorted_ix[2]])
fori,idxinenumerate(sorted_ix):
url=urls[sorted_ix[i]]
html+="<td>"
html+='<imgsrc="{}"height="224">'.format(url)
html+="</td>"
html+="</tr></table>"

returnhtml

..code::ipython3

#@titleLoadexamplevideosanddefinetextqueries{display-mode:"form"}

video_1_url="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"#@param{type:"string"}
video_2_url="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"#@param{type:"string"}
video_3_url="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"#@param{type:"string"}

video_1=load_video(video_1_url)
video_2=load_video(video_2_url)
video_3=load_video(video_3_url)
all_videos=[video_1,video_2,video_3]

query_1_video="waterfall"#@param{type:"string"}
query_2_video="playingguitar"#@param{type:"string"}
query_3_video="cardrifting"#@param{type:"string"}
all_queries_video=[query_1_video,query_2_video,query_3_video]
all_videos_urls=[video_1_url,video_2_url,video_3_url]
display_video(all_videos_urls)




..raw::html

<table><tr><th>Video1</th><th>Video2</th><th>Video3</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td></tr></table>



Theoriginalinference
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Preparevideoinputs.
videos_np=np.stack(all_videos,axis=0)

#Preparetextinput.
words_np=np.array(all_queries_video)

#Generatethevideoandtextembeddings.
video_embd,text_embd=generate_embeddings(hub_model,videos_np,words_np)

#Scoresbetweenvideoandtextiscomputedbydotproducts.
all_scores=np.dot(text_embd,tf.transpose(video_embd))

..code::ipython3

#Displayresults.
html=""
fori,wordsinenumerate(words_np):
html+=display_query_and_results_video(words,all_videos_urls,all_scores[i,:])
html+="<br>"
display.HTML(html)




..raw::html

<h2>Inputquery:<i>waterfall</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:4.71</th><th>Rank#2,Score:-1.63</th><th>Rank#3,Score:-4.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>playingguitar</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:6.50</th><th>Rank#2,Score:-1.79</th><th>Rank#3,Score:-2.67</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>cardrifting</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:8.78</th><th>Rank#2,Score:-1.07</th><th>Rank#3,Score:-2.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br>



ConvertthemodeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__OpenVINOsupportsTensorFlow
modelsviaconversionintoIntermediateRepresentation(IR)format.We
needtoprovideamodelobject,inputdataformodeltracingto
``ov.convert_model``functiontoobtainOpenVINO``ov.Model``object
instance.Modelcanbesavedondiskfornextdeploymentusing
``ov.save_model``function.

..code::ipython3

importopenvino_tokenizers#NOQANeedtoimportconversionandoperationextensions
importopenvinoasov

model_path=hub.resolve(hub_handle)
#inferonrandomdata
images_data=np.random.rand(3,32,224,224,3).astype(np.float32)
words_data=np.array(["Firstsentence","Secondone","Abracadabra"],dtype=str)

ov_model=ov.convert_model(model_path,input=[("words",[3]),("images",[3,32,224,224,3])])

Compilingmodels
----------------

`backtotop⬆️<#table-of-contents>`__

OnlyCPUissupportedforthismodelduetostringsasinput.

..code::ipython3

core=ov.Core()

compiled_model=core.compile_model(ov_model,device_name="CPU")

Inference
---------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Redefine`generate_embeddings`functiontomakeitpossibletousethecompileIRmodel.
defgenerate_embeddings(model,input_frames,input_words):
"""Generateembeddingsfromthemodelfromvideoframesandinputwords."""
#Input_framesmustbenormalizedin[0,1]andoftheshapeBatchxTxHxWx3
output=compiled_model({"words":input_words,"images":tf.cast(input_frames,dtype=tf.float32)})

returnoutput["video_embedding"],output["text_embedding"]

..code::ipython3

#Generatethevideoandtextembeddings.
video_embd,text_embd=generate_embeddings(compiled_model,videos_np,words_np)

#Scoresbetweenvideoandtextiscomputedbydotproducts.
all_scores=np.dot(text_embd,tf.transpose(video_embd))

..code::ipython3

#Displayresults.
html=""
fori,wordsinenumerate(words_np):
html+=display_query_and_results_video(words,all_videos_urls,all_scores[i,:])
html+="<br>"
display.HTML(html)




..raw::html

<h2>Inputquery:<i>waterfall</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:4.71</th><th>Rank#2,Score:-1.63</th><th>Rank#3,Score:-4.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>playingguitar</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:6.50</th><th>Rank#2,Score:-1.79</th><th>Rank#3,Score:-2.67</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>cardrifting</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:8.78</th><th>Rank#2,Score:-1.07</th><th>Rank#3,Score:-2.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br>



OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline).Theoptimization
processcontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeanOpenVINOIRmodel,usingthe``ov.save_model``function.

Preparedataset
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thismodeldoesn’trequireabigdatasetforcalibration.Wewilluse
onlyexamplevideosforthispurpose.NNCFprovides``nncf.Dataset``
wrapperforusingnativeframeworkdataloadersinquantizationpipeline.
Additionally,wespecifytransformfunctionthatwillberesponsiblefor
preparinginputdatainmodelexpectedformat.

..code::ipython3

importnncf

dataset=nncf.Dataset(((words_np,tf.cast(videos_np,dtype=tf.float32)),))


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Performmodelquantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscopeetc.)canbeprovided.

..code::ipython3

MODEL_DIR=Path("model/")
MODEL_DIR.mkdir(exist_ok=True)

quantized_model_path=MODEL_DIR/"quantized_model.xml"


ifnotquantized_model_path.exists():
quantized_model=nncf.quantize(model=ov_model,calibration_dataset=dataset,model_type=nncf.ModelType.TRANSFORMER)
ov.save_model(quantized_model,quantized_model_path)



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:39ignorednodeswerefoundbynamesintheNNCFGraph



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



Runquantizedmodelinference
-----------------------------

`backtotop⬆️<#table-of-contents>`__

Therearenochangesinmodelusageafterapplyingquantization.Let’s
checkthemodelworkonthepreviouslyusedexample.

..code::ipython3

int8_model=core.compile_model(quantized_model_path,device_name="CPU")

..code::ipython3

#Generatethevideoandtextembeddings.
video_embd,text_embd=generate_embeddings(int8_model,videos_np,words_np)

#Scoresbetweenvideoandtextiscomputedbydotproducts.
all_scores=np.dot(text_embd,tf.transpose(video_embd))

..code::ipython3

#Displayresults.
html=""
fori,wordsinenumerate(words_np):
html+=display_query_and_results_video(words,all_videos_urls,all_scores[i,:])
html+="<br>"
display.HTML(html)




..raw::html

<h2>Inputquery:<i>waterfall</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:4.71</th><th>Rank#2,Score:-1.63</th><th>Rank#3,Score:-4.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>playingguitar</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:6.50</th><th>Rank#2,Score:-1.79</th><th>Rank#3,Score:-2.67</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td></tr></table><br><h2>Inputquery:<i>cardrifting</i></h2><div>Results:<div><table><tr><th>Rank#1,Score:8.78</th><th>Rank#2,Score:-1.07</th><th>Rank#3,Score:-2.17</th></tr><tr><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif"height="224"></td><td><imgsrc="https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif"height="224"></td></tr></table><br>



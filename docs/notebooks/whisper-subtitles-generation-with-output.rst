VideoSubtitleGenerationusingWhisperandOpenVINO™
=====================================================

`Whisper<https://openai.com/blog/whisper/>`__isanautomaticspeech
recognition(ASR)systemtrainedon680,000hoursofmultilingualand
multitasksuperviseddatacollectedfromtheweb.Itisamulti-task
modelthatcanperformmultilingualspeechrecognitionaswellasspeech
translationandlanguageidentification.

..figure::https://user-images.githubusercontent.com/29454499/204536347-28976978-9a07-416c-acff-fc1214bbfbe0.svg
:alt:asr-training-data-desktop.svg

asr-training-data-desktop.svg

Youcanfindmoreinformationaboutthismodelinthe`research
paper<https://cdn.openai.com/papers/whisper.pdf>`__,`OpenAI
blog<https://openai.com/blog/whisper/>`__,`model
card<https://github.com/openai/whisper/blob/main/model-card.md>`__and
GitHub`repository<https://github.com/openai/whisper>`__.

Inthisnotebook,wewilluseWhisperwithOpenVINOtogenerate
subtitlesinasamplevideo.Additionally,wewilluse
`NNCF<https://github.com/openvinotoolkit/nncf>`__improvingmodel
performancebyINT8quantization.Notebookcontainsthefollowingsteps:
1.Downloadthemodel.2.InstantiatethePyTorchmodelpipeline.3.
ConvertmodeltoOpenVINOIR,usingmodelconversionAPI.4.Runthe
WhisperpipelinewithOpenVINOmodels.5.QuantizetheOpenVINOmodel
withNNCF.6.Checkquantizedmodelresultforthedemovideo.7.
Comparemodelsize,performanceandaccuracyofFP32andquantizedINT8
models.8.LaunchInteractivedemoforvideosubtitlesgeneration.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Instantiatemodel<#instantiate-model>`__

-`ConvertmodeltoOpenVINOIntermediateRepresentation(IR)
format.<#convert-model-to-openvino-intermediate-representation-ir-format->`__

-`Prepareinferencepipeline<#prepare-inference-pipeline>`__

-`Selectinferencedevice<#select-inference-device>`__

-`Runvideotranscription
pipeline<#run-video-transcription-pipeline>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`QuantizeWhisperencoderanddecoder
models<#quantize-whisper-encoder-and-decoder-models>`__
-`Runquantizedmodelinference<#run-quantized-model-inference>`__
-`Compareperformanceandaccuracyoftheoriginalandquantized
models<#compare-performance-and-accuracy-of-the-original-and-quantized-models>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installdependencies.

..code::ipython3

%pipinstall-q"openvino>=2024.1.0""nncf>=2.10.0"
%pipinstall-q"python-ffmpeg<=1.0.16"moviepytransformersonnx"git+https://github.com/huggingface/optimum-intel.git""peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/garywu007/pytube.git"soundfilelibrosajiwer
%pipinstall-q"gradio>=4.19"

Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

WhisperisaTransformerbasedencoder-decodermodel,alsoreferredto
asasequence-to-sequencemodel.Itmapsasequenceofaudiospectrogram
featurestoasequenceoftexttokens.First,therawaudioinputsare
convertedtoalog-Melspectrogrambyactionofthefeatureextractor.
Then,theTransformerencoderencodesthespectrogramtoformasequence
ofencoderhiddenstates.Finally,thedecoderautoregressivelypredicts
texttokens,conditionalonboththeprevioustokensandtheencoder
hiddenstates.

Youcanseethemodelarchitectureinthediagrambelow:

..figure::https://user-images.githubusercontent.com/29454499/204536571-8f6d8d77-5fbd-4c6d-8e29-14e734837860.svg
:alt:whisper_architecture.svg

whisper_architecture.svg

Thereareseveralmodelsofdifferentsizesandcapabilitiestrainedby
theauthorsofthemodel.Inthistutorial,wewillusethe``tiny``
model,butthesameactionsarealsoapplicabletoothermodelsfrom
Whisperfamily.

..code::ipython3

importipywidgetsaswidgets

MODELS=[
"openai/whisper-large-v3",
"openai/whisper-large-v2",
"openai/whisper-large",
"openai/whisper-medium",
"openai/whisper-small",
"openai/whisper-base",
"openai/whisper-tiny",
]

model_id=widgets.Dropdown(
options=list(MODELS),
value="openai/whisper-tiny",
description="Model:",
disabled=False,
)

model_id




..parsed-literal::

Dropdown(description='Model:',index=6,options=('openai/whisper-large-v3','openai/whisper-large-v2','openai…



ConvertmodeltoOpenVINOIntermediateRepresentation(IR)formatusingOptimum-Intel.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheHuggingFaceOptimumAPIisahigh-levelAPIthatenablesusto
convertandquantizemodelsfromtheHuggingFaceTransformerslibrary
totheOpenVINO™IRformat.Formoredetails,refertothe`HuggingFace
Optimum
documentation<https://huggingface.co/docs/optimum/intel/inference>`__.

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.TheOptimumInferencemodelsareAPIcompatiblewithHugging
FaceTransformersmodels.Thismeanswejustneedtoreplacethe
``AutoModelForXxx``classwiththecorresponding``OVModelForXxx``
class.

Belowisanexampleofthewhisper-tinymodel

..code::diff

-fromtransformersimportAutoModelForSpeechSeq2Seq
+fromoptimum.intel.openvinoimportOVModelForSpeechSeq2Seq
fromtransformersimportAutoTokenizer,pipeline

model_id="openai/whisper-tiny"
-model=AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
+model=OVModelForSpeechSeq2Seq.from_pretrained(model_id,export=True)

Modelclassinitializationstartswithcallingthe``from_pretrained``
method.WhendownloadingandconvertingtheTransformersmodel,the
parameter``export=True``shouldbeadded.Wecansavetheconverted
modelforthenextusagewiththe``save_pretrained``method.
Alternatively,modelconversioncanbeperformedusingOptimum-CLI
interface.YoucanfindmoredetailsaboutOptimum-IntelandOptimumCLI
usageinthis`tutorial<hugging-face-hub-with-output.html>`__.
Thecommandbellowillustrateshowtoconvertwhisperusingoptimumcli.

..code::ipython3

frompathlibimportPath

model_dir=model_id.value.split("/")[-1]

ifnotPath(model_dir).exists():
!optimum-cliexportopenvino-m{model_id.value}{model_dir}--weight-formatfp16

Prepareinferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Theimagebelowillustratesthepipelineofvideotranscribingusingthe
Whispermodel.

..figure::https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png
:alt:whisper_pipeline.png

whisper_pipeline.png

Preprocessingandpost-processingareimportantinthismodeluse.
``transformers.AutoProcessor``classusedforinitialization
``WhisperProcessor``isresponsibleforpreparingaudioinputdatafor
thePyTorchmodel,convertingittoMel-spectrogramanddecoding
predictedoutputtoken_idsintostringusingtokenizer.Tokenizersand
ProcessorsaredistributedwithmodelsalsocompatiblewiththeOpenVINO
model.

LiketheoriginalPyTorchmodel,theOpenVINOmodelisalsocompatible
withHuggingFace
`pipeline<https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline>`__
interfacefor``automatic-speech-recognition``.Pipelinecanbeusedfor
longaudiotranscription.Distil-Whisperusesachunkedalgorithmto
transcribelong-formaudiofiles.Inpractice,thischunkedlong-form
algorithmis9xfasterthanthesequentialalgorithmproposedbyOpenAI
intheWhisperpaper.Toenablechunking,passthechunk_length_s
parametertothepipeline.ForDistil-Whisper,achunklengthof15
secondsisoptimal.Toactivatebatching,passtheargumentbatch_size.

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importopenvinoasov

core=ov.Core()

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

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



..code::ipython3

fromoptimum.intel.openvinoimportOVModelForSpeechSeq2Seq
fromtransformersimportAutoProcessor,pipeline

ov_model=OVModelForSpeechSeq2Seq.from_pretrained(model_dir,device=device.value)

processor=AutoProcessor.from_pretrained(model_dir)

pipe=pipeline(
"automatic-speech-recognition",
model=ov_model,
chunk_length_s=30,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor,
)


..parsed-literal::

2024-06-1009:43:58.190233:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-06-1009:43:58.192258:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-06-1009:43:58.228701:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-06-1009:43:58.903562:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.0.1+cu118withCUDA1108(youhave2.3.0+cu121)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
CompilingtheencodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


Runvideotranscriptionpipeline
--------------------------------

`backtotop⬆️<#table-of-contents>`__

Now,wearereadytostarttranscription.WeselectavideofromYouTube
thatwewanttotranscribe.Bepatient,asdownloadingthevideomay
takesometime.

..code::ipython3

importipywidgetsaswidgets

VIDEO_LINK="https://youtu.be/kgL5LBM-hFI"
link=widgets.Text(
value=VIDEO_LINK,
placeholder="Typelinkforvideo",
description="Video:",
disabled=False,
)

link




..parsed-literal::

Text(value='https://youtu.be/kgL5LBM-hFI',description='Video:',placeholder='Typelinkforvideo')



..code::ipython3

frompathlibimportPath
frompytubeimportYouTube

print(f"Downloadingvideo{link.value}started")

output_file=Path("downloaded_video.mp4")
yt=YouTube(link.value)
yt.streams.get_highest_resolution().download(filename=output_file)
print(f"Videosavedto{output_file}")


..parsed-literal::

Downloadingvideohttps://youtu.be/kgL5LBM-hFIstarted
Videosavedtodownloaded_video.mp4


Selectthetaskforthemodel:

-**transcribe**-generateaudiotranscriptioninthesourcelanguage
(automaticallydetected).
-**translate**-generateaudiotranscriptionwithtranslationto
Englishlanguage.

..code::ipython3

task=widgets.Select(
options=["transcribe","translate"],
value="translate",
description="Selecttask:",
disabled=False,
)
task




..parsed-literal::

Select(description='Selecttask:',index=1,options=('transcribe','translate'),value='translate')



..code::ipython3

frommoviepy.editorimportVideoFileClip
fromtransformers.pipelines.audio_utilsimportffmpeg_read


defget_audio(video_file):
"""
Extractaudiosignalfromagivenvideofile,thenconvertittofloat,
thenmono-channelformatandresampleittotheexpectedsamplerate

Parameters:
video_file:pathtoinputvideofile
Returns:
resampled_audio:mono-channelfloataudiosignalwith16000Hzsamplerate
extractedfromvideo
duration:durationofvideofragmentinseconds
"""
input_video=VideoFileClip(str(video_file))
duration=input_video.duration
audio_file=video_file.stem+".wav"
input_video.audio.write_audiofile(audio_file,verbose=False,logger=None)
withopen(audio_file,"rb")asf:
inputs=f.read()
audio=ffmpeg_read(inputs,pipe.feature_extractor.sampling_rate)
return{"raw":audio,"sampling_rate":pipe.feature_extractor.sampling_rate},duration

..code::ipython3

inputs,duration=get_audio(output_file)

transcription=pipe(inputs,generate_kwargs={"task":task.value},return_timestamps=True)["chunks"]

..code::ipython3

importmath


defformat_timestamp(seconds:float):
"""
formattimeinsrt-fileexpectedformat
"""
assertseconds>=0,"non-negativetimestampexpected"
milliseconds=round(seconds*1000.0)

hours=milliseconds//3_600_000
milliseconds-=hours*3_600_000

minutes=milliseconds//60_000
milliseconds-=minutes*60_000

seconds=milliseconds//1_000
milliseconds-=seconds*1_000

return(f"{hours}:"ifhours>0else"00:")+f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


defprepare_srt(transcription,filter_duration=None):
"""
Formattranscriptionintosrtfileformat
"""
segment_lines=[]
foridx,segmentinenumerate(transcription):
#forthecasewherethemodelcouldnotpredictanendingtimestamp,whichcanhappenifaudioiscutoffinthemiddleofaword.
ifsegment["timestamp"][1]isNone:
segment["timestamp"]=(segment["timestamp"][0],filter_duration)

iffilter_durationisnotNoneand(segment["timestamp"][0]>=math.floor(filter_duration)orsegment["timestamp"][1]>math.ceil(filter_duration)+1):
break
segment_lines.append(str(idx+1)+"\n")
time_start=format_timestamp(segment["timestamp"][0])
time_end=format_timestamp(segment["timestamp"][1])
time_str=f"{time_start}-->{time_end}\n"
segment_lines.append(time_str)
segment_lines.append(segment["text"]+"\n\n")
returnsegment_lines

"Theresultswillbesavedinthe``downloaded_video.srt``file.SRTis
oneofthemostpopularformatsforstoringsubtitlesandiscompatible
withmanymodernvideoplayers.Thisfilecanbeusedtoembed
transcriptionintovideosduringplaybackorbyinjectingthemdirectly
intovideofilesusing``ffmpeg``.

..code::ipython3

srt_lines=prepare_srt(transcription,filter_duration=duration)
#savetranscription
withoutput_file.with_suffix(".srt").open("w")asf:
f.writelines(srt_lines)

Nowletusseetheresults.

..code::ipython3

widgets.Video.from_file(output_file,loop=False,width=800,height=800)




..parsed-literal::

Video(value=b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42\x00\x00:'moov\x00\x00\x00lmvhd...",height='800…



..code::ipython3

print("".join(srt_lines))


..parsed-literal::

1
00:00:00,000-->00:00:05,000
Oh,what'sthat?

2
00:00:05,000-->00:00:08,000
Oh,wow.

3
00:00:08,000-->00:00:10,000
Hello,humans.

4
00:00:13,000-->00:00:15,000
Focusonme.

5
00:00:15,000-->00:00:17,000
Focusontheguard.

6
00:00:17,000-->00:00:20,000
Don'ttellanyonewhatyou'reseeinginhere.

7
00:00:22,000-->00:00:24,000
Haveyouseenwhat'sinthere?

8
00:00:24,000-->00:00:25,000
Theyhaveintel.

9
00:00:25,000-->00:00:27,000
Thisiswhereitallchanges.




Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingthequantizationlayersintothe
modelgraphandthenusingasubsetofthetrainingdatasetto
initializetheparametersoftheseadditionalquantizationlayers.The
frameworkisdesignedsothatmodificationstoyouroriginaltraining
codeareminor.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize``toobtainquantizedencoderanddecodermodels.
3.Serializethe``INT8``modelusing``openvino.save_model``function.

..

**Note**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

PleaseselectbelowwhetheryouwouldliketorunWhisperquantization.

..code::ipython3

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

ov_quantized_model=None

%load_extskip_kernel_extension

Preparecalibrationdatasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Firststepistopreparecalibrationdatasetsforquantization.Sincewe
quantizewhisperencoderanddecoderseparately,weneedtopreparea
calibrationdatasetforeachofthemodels.Weimportan
``InferRequestWrapper``classthatwillinterceptmodelinputsand
collectthemtoalist.Thenwerunmodelinferenceonsomesmallamount
ofaudiosamples.Generally,increasingthecalibrationdatasetsize
improvesquantizationquality.

..code::ipython3

%%skipnot$to_quantize.value

fromitertoolsimportislice
fromoptimum.intel.openvino.quantizationimportInferRequestWrapper


defcollect_calibration_dataset(ov_model:OVModelForSpeechSeq2Seq,calibration_dataset_size:int):
#Overwritemodelrequestproperties,savingtheoriginalonesforrestoringlater
encoder_calibration_data=[]
decoder_calibration_data=[]
ov_model.encoder.request=InferRequestWrapper(ov_model.encoder.request,encoder_calibration_data,apply_caching=True)
ov_model.decoder_with_past.request=InferRequestWrapper(ov_model.decoder_with_past.request,
decoder_calibration_data,
apply_caching=True)

pipe=pipeline(
"automatic-speech-recognition",
model=ov_model,
chunk_length_s=30,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor)
try:
calibration_dataset=dataset=load_dataset("openslr/librispeech_asr","clean",split="validation",streaming=True,trust_remote_code=True)
forsampleintqdm(islice(calibration_dataset,calibration_dataset_size),desc="Collectingcalibrationdata",
total=calibration_dataset_size):
pipe(sample["audio"],generate_kwargs={"task":task.value},return_timestamps=True)
finally:
ov_model.encoder.request=ov_model.encoder.request.request
ov_model.decoder_with_past.request=ov_model.decoder_with_past.request.request

returnencoder_calibration_data,decoder_calibration_data

QuantizeWhisperencoderanddecodermodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Belowwerunthe``quantize``functionwhichcalls``nncf.quantize``on
Whisperencoderanddecoder-with-pastmodels.Wedon’tquantize
first-step-decoderbecauseitsshareinwholeinferencetimeis
negligible.

..code::ipython3

%%skipnot$to_quantize.value

importgc
importshutil
importnncf
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm

defextract_input_features(sample):
input_features=processor(
sample["audio"]["array"],
sampling_rate=sample["audio"]["sampling_rate"],
return_tensors="pt",
).input_features
returninput_features



CALIBRATION_DATASET_SIZE=50
quantized_model_path=Path(f"{model_dir}_quantized")


defquantize(ov_model:OVModelForSpeechSeq2Seq,calibration_dataset_size:int):
ifnotquantized_model_path.exists():
encoder_calibration_data,decoder_calibration_data=collect_calibration_dataset(
ov_model,calibration_dataset_size
)
print("Quantizingencoder")
quantized_encoder=nncf.quantize(
ov_model.encoder.model,
nncf.Dataset(encoder_calibration_data),
subset_size=len(encoder_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
#SmoothQuantalgorithmreducesactivationquantizationerror;optimalalphavaluewasobtainedthroughgridsearch
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
)
ov.save_model(quantized_encoder,quantized_model_path/"openvino_encoder_model.xml")
delquantized_encoder
delencoder_calibration_data
gc.collect()

print("Quantizingdecoderwithpast")
quantized_decoder_with_past=nncf.quantize(
ov_model.decoder_with_past.model,
nncf.Dataset(decoder_calibration_data),
subset_size=len(decoder_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
#SmoothQuantalgorithmreducesactivationquantizationerror;optimalalphavaluewasobtainedthroughgridsearch
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.96)
)
ov.save_model(quantized_decoder_with_past,quantized_model_path/"openvino_decoder_with_past_model.xml")
delquantized_decoder_with_past
deldecoder_calibration_data
gc.collect()

#Copytheconfigfileandthefirst-step-decodermanually
model_path=Path(model_dir)
shutil.copy(model_path/"config.json",quantized_model_path/"config.json")
shutil.copy(model_path/"generation_config.json",quantized_model_path/"generation_config.json")
shutil.copy(model_path/"openvino_decoder_model.xml",quantized_model_path/"openvino_decoder_model.xml")
shutil.copy(model_path/"openvino_decoder_model.bin",quantized_model_path/"openvino_decoder_model.bin")

quantized_ov_model=OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path,compile=False)
quantized_ov_model.to(device.value)
quantized_ov_model.compile()
returnquantized_ov_model


ov_quantized_model=quantize(ov_model,CALIBRATION_DATASET_SIZE)



..parsed-literal::

Collectingcalibrationdata:0%||0/50[00:00<?,?it/s]



..parsed-literal::

Output()


..parsed-literal::

Quantizingencoder



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

INFO:nncf:12ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:16ignorednodeswerefoundbynameintheNNCFGraph



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

Output()


..parsed-literal::

Quantizingdecoderwithpast



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

INFO:nncf:24ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:24ignorednodeswerefoundbynameintheNNCFGraph



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

CompilingtheencodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...


Runquantizedmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Let’scomparethetranscriptionresultsfororiginalandquantized
models.

..code::ipython3

ifov_quantized_modelisnotNone:
int8_pipe=pipeline(
"automatic-speech-recognition",
model=ov_quantized_model,
chunk_length_s=30,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor,
)
inputs,duration=get_audio(output_file)
transcription=int8_pipe(inputs,generate_kwargs={"task":task.value},return_timestamps=True)["chunks"]
srt_lines=prepare_srt(transcription,filter_duration=duration)
print("".join(srt_lines))
widgets.Video.from_file(output_file,loop=False,width=800,height=800)


..parsed-literal::

1
00:00:00,000-->00:00:05,000
What'sthat?

2
00:00:05,000-->00:00:07,000
Oh,wow.

3
00:00:09,000-->00:00:11,000
Hellohumans.

4
00:00:14,000-->00:00:15,000
Focusonme.

5
00:00:15,000-->00:00:16,000
Focusontheguard.

6
00:00:18,000-->00:00:20,000
Don'ttellanyonewhatyou'reseeninhere.

7
00:00:22,000-->00:00:24,000
Haveyouseenwhat'sinthere?

8
00:00:24,000-->00:00:25,000
Theyhaveintel.

9
00:00:25,000-->00:00:27,000
Thisiswhereitallchanges.




Compareperformanceandaccuracyoftheoriginalandquantizedmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Finally,wecompareoriginalandquantizedWhispermodelsfromaccuracy
andperformancestand-points.

Tomeasureaccuracy,weuse``1-WER``asametric,whereWERstands
forWordErrorRate.

Whenmeasuringinferencetime,wedoitseparatelyforencoderand
decoder-with-pastmodelforwards,andforthewholemodelinferencetoo.

..code::ipython3

%%skipnot$to_quantize.value

importtime
fromcontextlibimportcontextmanager
fromjiwerimportwer,wer_standardize


TEST_DATASET_SIZE=50
MEASURE_TIME=False

@contextmanager
deftime_measurement():
globalMEASURE_TIME
try:
MEASURE_TIME=True
yield
finally:
MEASURE_TIME=False

deftime_fn(obj,fn_name,time_list):
original_fn=getattr(obj,fn_name)

defwrapper(*args,**kwargs):
ifnotMEASURE_TIME:
returnoriginal_fn(*args,**kwargs)
start_time=time.perf_counter()
result=original_fn(*args,**kwargs)
end_time=time.perf_counter()
time_list.append(end_time-start_time)
returnresult

setattr(obj,fn_name,wrapper)

defcalculate_transcription_time_and_accuracy(ov_model,test_samples):
encoder_infer_times=[]
decoder_with_past_infer_times=[]
whole_infer_times=[]
time_fn(ov_model,"generate",whole_infer_times)
time_fn(ov_model.encoder,"forward",encoder_infer_times)
time_fn(ov_model.decoder_with_past,"forward",decoder_with_past_infer_times)

ground_truths=[]
predictions=[]
fordata_itemintqdm(test_samples,desc="Measuringperformanceandaccuracy"):
input_features=extract_input_features(data_item)

withtime_measurement():
predicted_ids=ov_model.generate(input_features)
transcription=processor.batch_decode(predicted_ids,skip_special_tokens=True)

ground_truths.append(data_item["text"])
predictions.append(transcription[0])

word_accuracy=(1-wer(ground_truths,predictions,reference_transform=wer_standardize,
hypothesis_transform=wer_standardize))*100
mean_whole_infer_time=sum(whole_infer_times)
mean_encoder_infer_time=sum(encoder_infer_times)
mean_decoder_with_time_infer_time=sum(decoder_with_past_infer_times)
returnword_accuracy,(mean_whole_infer_time,mean_encoder_infer_time,mean_decoder_with_time_infer_time)

test_dataset=load_dataset("openslr/librispeech_asr","clean",split="validation",streaming=True,trust_remote_code=True)
test_dataset=test_dataset.shuffle(seed=42).take(TEST_DATASET_SIZE)
test_samples=[sampleforsampleintest_dataset]

accuracy_original,times_original=calculate_transcription_time_and_accuracy(ov_model,test_samples)
accuracy_quantized,times_quantized=calculate_transcription_time_and_accuracy(ov_quantized_model,test_samples)
print(f"Encoderperformancespeedup:{times_original[1]/times_quantized[1]:.3f}")
print(f"Decoderwithpastperformancespeedup:{times_original[2]/times_quantized[2]:.3f}")
print(f"Wholepipelineperformancespeedup:{times_original[0]/times_quantized[0]:.3f}")
print(f"Whispertranscriptionwordaccuracy.Originalmodel:{accuracy_original:.2f}%.Quantizedmodel:{accuracy_quantized:.2f}%.")
print(f"Accuracydrop:{accuracy_original-accuracy_quantized:.2f}%.")



..parsed-literal::

Measuringperformanceandaccuracy:0%||0/50[00:00<?,?it/s]



..parsed-literal::

Measuringperformanceandaccuracy:0%||0/50[00:00<?,?it/s]


..parsed-literal::

Encoderperformancespeedup:1.352
Decoderwithpastperformancespeedup:1.342
Wholepipelineperformancespeedup:1.350
Whispertranscriptionwordaccuracy.Originalmodel:81.67%.Quantizedmodel:83.67%.
Accuracydrop:-1.99%.


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


deftranscribe(url,task,use_int8):
output_file=Path("downloaded_video.mp4")
yt=YouTube(url)
yt.streams.get_highest_resolution().download(filename=output_file)
inputs,duration=get_audio(output_file)
m_pipe=int8_pipeifuse_int8elsepipe
transcription=m_pipe(inputs,generate_kwargs={"task":task.lower()},return_timestamps=True)["chunks"]
srt_lines=prepare_srt(transcription,duration)
withoutput_file.with_suffix(".srt").open("w")asf:
f.writelines(srt_lines)
return[str(output_file),str(output_file.with_suffix(".srt"))]


demo=gr.Interface(
transcribe,
[
gr.Textbox(label="YouTubeURL"),
gr.Radio(["Transcribe","Translate"],value="Transcribe"),
gr.Checkbox(value=ov_quantized_modelisnotNone,visible=ov_quantized_modelisnotNone,label="UseINT8"),
],
"video",
examples=[["https://youtu.be/kgL5LBM-hFI","Transcribe"]],
allow_flagging="never",
)
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


..parsed-literal::

Keyboardinterruptioninmainthread...closingserver.


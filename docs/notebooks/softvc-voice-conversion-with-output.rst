SoftVCVITSSingingVoiceConversionandOpenVINO™
==================================================

Thistutorialisbasedon`SoftVCVITSSingingVoiceConversion
project<https://github.com/svc-develop-team/so-vits-svc>`__.The
purposeofthisprojectwastoenabledeveloperstohavetheirbeloved
animecharactersperformsingingtasks.Thedevelopers’intentionwasto
focussolelyonfictionalcharactersandavoidanyinvolvementofreal
individuals,anythingrelatedtorealindividualsdeviatesfromthe
developer’soriginalintention.

ThesingingvoiceconversionmodelusesSoftVCcontentencoderto
extractspeechfeaturesfromthesourceaudio.Thesefeaturevectorsare
directlyfedinto`VITS<https://github.com/jaywalnut310/vits>`__
withouttheneedforconversiontoatext-basedintermediate
representation.Asaresult,thepitchandintonationsoftheoriginal
audioarepreserved.

Inthistutorialwewillusethebasemodelflow.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Usetheoriginalmodeltorunan
inference<#use-the-original-model-to-run-an-inference>`__
-`ConverttoOpenVINOIRmodel<#convert-to-openvino-ir-model>`__
-`RuntheOpenVINOmodel<#run-the-openvino-model>`__
-`Interactiveinference<#interactive-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-qU"pip<24.1"#tofixfairseqinstallproblem
%pipinstall-q"openvino>=2023.2.0"
!gitclonehttps://github.com/svc-develop-team/so-vits-svc-b4.1-Stable
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cputqdmlibrosa"torch>=2.1.0""torchaudio>=2.1.0"faiss-cpu"gradio>=4.19""numpy>=1.23.5""fairseq==0.12.2"praat-parselmouth

Downloadpretrainedmodelsandconfigs.Weusearecommendedencoder
`ContentVec<https://arxiv.org/abs/2204.09224>`__andmodelsfrom`a
collectionofso-vits-svc-4.0modelsmadebythePonyPreservation
Project<https://huggingface.co/therealvul/so-vits-svc-4.0>`__for
example.Youcanchooseanyotherpretrainedmodelfromthisoranother
projector`prepareyour
own<https://github.com/svc-develop-team/so-vits-svc#%EF%B8%8F-training>`__.

..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

#ContentVec
download_file(
"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
"checkpoint_best_legacy_500.pt",
directory="so-vits-svc/pretrain/",
)

#pretrainedmodelsandconfigsfromacollectionofso-vits-svc-4.0models.Youcanuseothermodels.
download_file(
"https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/kmeans_10000.pt",
"kmeans_10000.pt",
directory="so-vits-svc/logs/44k/",
)
download_file(
"https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/config.json",
"config.json",
directory="so-vits-svc/configs/",
)
download_file(
"https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/G_30400.pth",
"G_30400.pth",
directory="so-vits-svc/logs/44k/",
)
download_file(
"https://huggingface.co/therealvul/so-vits-svc-4.0/resolve/main/Rainbow%20Dash%20(singing)/D_30400.pth",
"D_30400.pth",
directory="so-vits-svc/logs/44k/",
)

#awavsample
download_file(
"https://huggingface.co/datasets/santifiorino/spinetta/resolve/main/spinetta/000.wav",
"000.wav",
directory="so-vits-svc/raw/",
)

Usetheoriginalmodeltorunaninference
------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Changedirectoryto``so-vits-svc``inpurposenottobrakeinternal
relativepaths.

..code::ipython3

%cdso-vits-svc

DefinetheSovitsModel.

..code::ipython3

frominference.infer_toolimportSvc

model=Svc("logs/44k/G_30400.pth","configs/config.json",device="cpu")

Define``kwargs``andmakeaninference.

..code::ipython3

kwargs={
"raw_audio_path":"raw/000.wav",#pathtoasourceaudio
"spk":"RainbowDash(singing)",#speakerIDinwhichthesourceaudioshouldbeconverted.
"tran":0,
"slice_db":-40,
"cluster_infer_ratio":0,
"auto_predict_f0":False,
"noice_scale":0.4,
}

audio=model.slice_inference(**kwargs)

Andletcomparetheoriginalaudiowiththeresult.

..code::ipython3

importIPython.displayasipd

#original
ipd.Audio("raw/000.wav",rate=model.target_sample)

..code::ipython3

#result
ipd.Audio(audio,rate=model.target_sample)

ConverttoOpenVINOIRmodel
----------------------------

`backtotop⬆️<#table-of-contents>`__

ModelcomponentsarePyTorchmodules,thatcanbeconvertedwith
``ov.convert_model``functiondirectly.Wealsouse``ov.save_model``
functiontoserializetheresultofconversion.``Svc``isnotamodel,
itrunsmodelinferenceinside.Inbasescenarioonly``SynthesizerTrn``
named``net_g_ms``isused.Itisenoughtoconvertonlythismodeland
weshouldre-assign``forward``methodon``infer``methodforthis
purpose.

``SynthesizerTrn``usesseveralmodelsinsideit’sflow,
i.e. \``TextEncoder``,``Generator``,``ResidualCouplingBlock``,etc.,
butinourcaseOpenVINOallowstoconvertwholepipelinebyonestep
withoutneedtolookinside.

..code::ipython3

importopenvinoasov
importtorch
frompathlibimportPath


dummy_c=torch.randn(1,256,813)
dummy_f0=torch.randn(1,813)
dummy_uv=torch.ones(1,813)
dummy_g=torch.tensor([[0]])
model.net_g_ms.forward=model.net_g_ms.infer

net_g_kwargs={
"c":dummy_c,
"f0":dummy_f0,
"uv":dummy_uv,
"g":dummy_g,
"noice_scale":torch.tensor(0.35),#needtowrapnumericandbooleanvaluesforconversion
"seed":torch.tensor(52468),
"predict_f0":torch.tensor(False),
"vol":torch.tensor(0),
}
core=ov.Core()


net_g_model_xml_path=Path("models/ov_net_g_model.xml")

ifnotnet_g_model_xml_path.exists():
converted_model=ov.convert_model(model.net_g_ms,example_input=net_g_kwargs)
net_g_model_xml_path.parent.mkdir(parents=True,exist_ok=True)
ov.save_model(converted_model,net_g_model_xml_path)

RuntheOpenVINOmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

SelectadevicefromdropdownlistforrunninginferenceusingOpenVINO.

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device

Weshouldcreateawrapperfor``net_g_ms``modeltokeepit’s
interface.Thenreplace``net_g_ms``originalmodelbytheconvertedIR
model.Weuse``ov.compile_model``tomakeitreadytouseforloading
onadevice.

..code::ipython3

classNetGModelWrapper:
def__init__(self,net_g_model_xml_path):
super().__init__()
self.net_g_model=core.compile_model(net_g_model_xml_path,device.value)

definfer(self,c,*,f0,uv,g,noice_scale=0.35,seed=52468,predict_f0=False,vol=None):
ifvolisNone:#Noneisnotallowedasaninput
results=self.net_g_model((c,f0,uv,g,noice_scale,seed,predict_f0))
else:
results=self.net_g_model((c,f0,uv,g,noice_scale,seed,predict_f0,vol))

returntorch.from_numpy(results[0]),torch.from_numpy(results[1])


model.net_g_ms=NetGModelWrapper(net_g_model_xml_path)
audio=model.slice_inference(**kwargs)

Checkresult.Isitidenticaltothatcreatedbytheoriginalmodel.

..code::ipython3

importIPython.displayasipd

ipd.Audio(audio,rate=model.target_sample)

Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


src_audio=gr.Audio(label="SourceAudio",type="filepath")
output_audio=gr.Audio(label="OutputAudio",type="numpy")

title="SoftVCVITSSingingVoiceConversionwithGradio"
description=f'GradioDemoforSoftVCVITSSingingVoiceConversionandOpenVINO™.Uploadasourceaudio,thenclickthe"Submit"buttontoinference.Audiosamplerateshouldbe{model.target_sample}'


definfer(src_audio,tran,slice_db,noice_scale):
kwargs["raw_audio_path"]=src_audio
kwargs["tran"]=tran
kwargs["slice_db"]=slice_db
kwargs["noice_scale"]=noice_scale

audio=model.slice_inference(**kwargs)

returnmodel.target_sample,audio


demo=gr.Interface(
infer,
[
src_audio,
gr.Slider(-100,100,value=0,label="Pitchshift",step=1),
gr.Slider(
-80,
-20,
value=-30,
label="Slicedb",
step=10,
info="Thedefaultis-30,noisyaudiocanbe-30,drysoundcanbe-50topreservebreathing.",
),
gr.Slider(
0,
1,
value=0.4,
label="Noisescale",
step=0.1,
info="Noiselevelwillaffectpronunciationandsoundquality,whichismoremetaphysical",
),
],
output_audio,
title=title,
description=description,
examples=[["raw/000.wav",0,-30,0.4,False]],
)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/

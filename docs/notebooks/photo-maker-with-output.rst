Text-to-imagegenerationusingPhotoMakerandOpenVINO
======================================================

PhotoMakerisanefficientpersonalizedtext-to-imagegenerationmethod,
whichmainlyencodesanarbitrarynumberofinputIDimagesintoastack
IDembeddingforpreservingIDinformation.Suchanembedding,serving
asaunifiedIDrepresentation,cannotonlyencapsulatethe
characteristicsofthesameinputIDcomprehensively,butalso
accommodatethecharacteristicsofdifferentIDsforsubsequent
integration.Thispavesthewayformoreintriguingandpractically
valuableapplications.Userscaninputoneorafewfacephotos,along
withatextprompt,toreceiveacustomizedphotoorpainting(no
trainingrequired!).Additionally,thismodelcanbeadaptedtoanybase
modelbasedon``SDXL``orusedinconjunctionwithother``LoRA``
modules.MoredetailsaboutPhotoMakercanbefoundinthe`technical
report<https://arxiv.org/pdf/2312.04461.pdf>`__.

ThisnotebookexploreshowtospeedupPhotoMakerpipelineusing
OpenVINO.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`PhotoMakerpipeline
introduction<#photomaker-pipeline-introduction>`__
-`Prerequisites<#prerequisites>`__
-`Loadoriginalpipelineandpreparemodelsfor
conversion<#load-original-pipeline-and-prepare-models-for-conversion>`__
-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__

-`IDEncoder<#id-encoder>`__
-`TextEncoder<#text-encoder>`__
-`U-Net<#u-net>`__
-`VAEDecoder<#vae-decoder>`__

-`PrepareInferencepipeline<#prepare-inference-pipeline>`__

-`SelectinferencedeviceforStableDiffusion
pipeline<#select-inference-device-for-stable-diffusion-pipeline>`__
-`CompilemodelsandcreatetheirWrappersfor
inference<#compile-models-and-create-their-wrappers-for-inference>`__

-`RunningText-to-ImageGenerationwith
OpenVINO<#running-text-to-image-generation-with-openvino>`__
-`InteractiveDemo<#interactive-demo>`__

PhotoMakerpipelineintroduction
--------------------------------

`backtotop⬆️<#table-of-contents>`__

FortheproposedPhotoMaker,wefirstobtainthetextembeddingand
imageembeddingsfrom``textencoder(s)``and``image(ID)encoder``,
respectively.Then,weextractthefusedembeddingbymergingthe
correspondingclassembedding(e.g.,manandwoman)andeachimage
embedding.Next,weconcatenateallfusedembeddingsalongthelength
dimensiontoformthestackedIDembedding.Finally,wefeedthestacked
IDembeddingtoallcross-attentionlayersforadaptivelymergingtheID
contentinthe``diffusionmodel``.Notethatalthoughweuseimagesof
thesameIDwiththemaskedbackgroundduringtraining,wecandirectly
inputimagesofdifferentIDswithoutbackgrounddistortiontocreatea
newIDduringinference.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

ClonePhotoMakerrepository

..code::ipython3

frompathlibimportPath

ifnotPath("PhotoMaker").exists():
!gitclonehttps://github.com/TencentARC/PhotoMaker.git


..parsed-literal::

Cloninginto'PhotoMaker'...
remote:Enumeratingobjects:236,done.[K
remote:Countingobjects:100%(145/145),done.[K
remote:Compressingobjects:100%(96/96),done.[K
remote:Total236(delta114),reused68(delta49),pack-reused91[K
Receivingobjects:100%(236/236),9.31MiB|28.72MiB/s,done.
Resolvingdeltas:100%(120/120),done.


Installrequiredpackages

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
transformers"torch>=2.1""diffusers>=0.26""gradio>=4.19""openvino>=2024.0.0"torchvision"peft==0.6.2""nncf>=2.9.0""protobuf==3.20.3"


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
descript-audiotools0.7.2requiresprotobuf<3.20,>=3.9.2,butyouhaveprotobuf3.20.3whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


PreparePyTorchmodels

..code::ipython3

adapter_id="TencentARC/PhotoMaker"
base_model_id="SG161222/RealVisXL_V3.0"

TEXT_ENCODER_OV_PATH=Path("model/text_encoder.xml")
TEXT_ENCODER_2_OV_PATH=Path("model/text_encoder_2.xml")
UNET_OV_PATH=Path("model/unet.xml")
ID_ENCODER_OV_PATH=Path("model/id_encoder.xml")
VAE_DECODER_OV_PATH=Path("model/vae_decoder.xml")

Loadoriginalpipelineandpreparemodelsforconversion
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ForexportingeachPyTorchmodel,wewilldownloadthe``IDencoder``
weight,``LoRa``weightfromHuggingFacehub,thenusingthe
``PhotoMakerStableDiffusionXLPipeline``objectfromrepositoryof
PhotoMakertogeneratetheoriginalPhotoMakerpipeline.

..code::ipython3

importtorch
importnumpyasnp
importos
fromPILimportImage
frompathlibimportPath
fromPhotoMaker.photomaker.modelimportPhotoMakerIDEncoder
fromPhotoMaker.photomaker.pipelineimportPhotoMakerStableDiffusionXLPipeline
fromdiffusersimportEulerDiscreteScheduler
importgc

trigger_word="img"


defload_original_pytorch_pipeline_components(photomaker_path:str,base_model_id:str):
#Loadbasemodel
pipe=PhotoMakerStableDiffusionXLPipeline.from_pretrained(base_model_id,use_safetensors=True).to("cpu")

#LoadPhotoMakercheckpoint
pipe.load_photomaker_adapter(
os.path.dirname(photomaker_path),
subfolder="",
weight_name=os.path.basename(photomaker_path),
trigger_word=trigger_word,
)
pipe.scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()
gc.collect()
returnpipe


..parsed-literal::

2024-07-1301:28:21.659532:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:28:21.694860:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:28:22.366293:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..code::ipython3

fromhuggingface_hubimporthf_hub_download

photomaker_path=hf_hub_download(repo_id=adapter_id,filename="photomaker-v1.bin",repo_type="model")

pipe=load_original_pytorch_pipeline_components(photomaker_path,base_model_id)



..parsed-literal::

Loadingpipelinecomponents...:0%||0/7[00:00<?,?it/s]


..parsed-literal::

TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.


..parsed-literal::

LoadingPhotoMakercomponents[1]id_encoderfrom[/opt/home/k8sworker/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/d7ec3fc17290263135825194aeb3bc456da67cc5]...
LoadingPhotoMakercomponents[2]lora_weightsfrom[/opt/home/k8sworker/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/d7ec3fc17290263135825194aeb3bc456da67cc5]


ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodels
conversiondirectly.Weneedtoprovideamodelobject,inputdatafor
modeltracingto``ov.convert_model``functiontoobtainOpenVINO
``ov.Model``objectinstance.Modelcanbesavedondiskfornext
deploymentusing``ov.save_model``function.

Thepipelineconsistsoffiveimportantparts:

-IDEncoderforgeneratingimageembeddingstoconditionbyimage
annotation.
-TextEncodersforcreatingtextembeddingstogenerateanimagefrom
atextprompt.
-Unetforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Forreducingmemoryconsumption,weightscompressionoptimizationcanbe
appliedusing`NNCF<https://github.com/openvinotoolkit/nncf>`__.Weight
compressionaimstoreducethememoryfootprintofmodels,whichrequire
extensivememorytostoretheweightsduringinference,canbenefitfrom
weightcompressioninthefollowingways:

-enablingtheinferenceofexceptionallylargemodelsthatcannotbe
accommodatedinthememoryofthedevice;

-improvingtheinferenceperformanceofthemodelsbyreducingthe
latencyofthememoryaccesswhencomputingtheoperationswith
weights,forexample,Linearlayers.

`NeuralNetworkCompressionFramework
(NNCF)<https://github.com/openvinotoolkit/nncf>`__provides4-bit/
8-bitmixedweightquantizationasacompressionmethod.Themain
differencebetweenweightscompressionandfullmodelquantization
(post-trainingquantization)isthatactivationsremainfloating-point
inthecaseofweightscompressionwhichleadstoabetteraccuracy.

``nncf.compress_weights``functioncanbeusedforperformingweights
compression.ThefunctionacceptsanOpenVINOmodelandother
compressionparameters.

Moredetailsaboutweightscompressioncanbefoundin`OpenVINO
documentation<https://docs.openvino.ai/2023.3/weight_compression.html>`__.

..code::ipython3

importopenvinoasov
importnncf


defflattenize_inputs(inputs):
"""
Helperfunctionforresolvenestedinputstructure(e.g.listsortuplesoftensors)
"""
flatten_inputs=[]
forinput_dataininputs:
ifinput_dataisNone:
continue
ifisinstance(input_data,(list,tuple)):
flatten_inputs.extend(flattenize_inputs(input_data))
else:
flatten_inputs.append(input_data)
returnflatten_inputs


dtype_mapping={
torch.float32:ov.Type.f32,
torch.float64:ov.Type.f64,
torch.int32:ov.Type.i32,
torch.int64:ov.Type.i64,
torch.bool:ov.Type.boolean,
}


defprepare_input_info(input_dict):
"""
Helperfunctionforpreparinginputinfo(shapesanddatatypes)forconversionbasedonexampleinputs
"""
flatten_inputs=flattenize_inputs(input_dict.values())
input_info=[]
forinput_datainflatten_inputs:
updated_shape=list(input_data.shape)
ifinput_data.ndim==5:
updated_shape[1]=-1
input_info.append((dtype_mapping[input_data.dtype],updated_shape))
returninput_info


defconvert(model:torch.nn.Module,xml_path:str,example_input,input_info):
"""
HelperfunctionforconvertingPyTorchmodeltoOpenVINOIR
"""
xml_path=Path(xml_path)
ifnotxml_path.exists():
xml_path.parent.mkdir(parents=True,exist_ok=True)
withtorch.no_grad():
ov_model=ov.convert_model(model,example_input=example_input,input=input_info)
ov_model=nncf.compress_weights(ov_model)
ov.save_model(ov_model,xml_path)

delov_model
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


IDEncoder
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

PhotoMakermergedimageencoderandfusemoduletocreateanIDEncoder.
Itwillusedtogenerateimageembeddingstoupdatetextencoder’s
output(textembeddings)whichwillbetheinputforU-Netmodel.

..code::ipython3

id_encoder=pipe.id_encoder
id_encoder.eval()


defcreate_bool_tensor(*size):
new_tensor=torch.zeros((size),dtype=torch.bool)
returnnew_tensor


inputs={
"id_pixel_values":torch.randn((1,1,3,224,224)),
"prompt_embeds":torch.randn((1,77,2048)),
"class_tokens_mask":create_bool_tensor(1,77),
}

input_info=prepare_input_info(inputs)

convert(id_encoder,ID_ENCODER_OV_PATH,inputs,input_info)

delid_encoder
gc.collect()


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:279:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz*self.num_heads,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:319:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/photo-maker/PhotoMaker/photomaker/model.py:84:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
assertclass_tokens_mask.sum()==stacked_id_embeds.shape[0],f"{class_tokens_mask.sum()}!={stacked_id_embeds.shape[0]}"


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(151/151)│100%(151/151)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>





..parsed-literal::

15594



TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetext-encoderisresponsiblefortransformingtheinputprompt,for
example,“aphotoofanastronautridingahorse”intoanembedding
spacethatcanbeunderstoodbytheU-Net.Itisusuallyasimple
transformer-basedencoderthatmapsasequenceofinputtokenstoa
sequenceoflatenttextembeddings.

..code::ipython3

text_encoder=pipe.text_encoder
text_encoder.eval()
text_encoder_2=pipe.text_encoder_2
text_encoder_2.eval()

text_encoder.config.output_hidden_states=True
text_encoder.config.return_dict=False
text_encoder_2.config.output_hidden_states=True
text_encoder_2.config.return_dict=False

inputs={"input_ids":torch.ones((1,77),dtype=torch.long)}

input_info=prepare_input_info(inputs)

convert(text_encoder,TEXT_ENCODER_OV_PATH,inputs,input_info)
convert(text_encoder_2,TEXT_ENCODER_2_OV_PATH,inputs,input_info)

deltext_encoder
deltext_encoder_2
gc.collect()


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifinput_shape[-1]>1orself.sliding_windowisnotNone:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifpast_key_values_length>0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:287:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_attention_mask.size()!=(bsz,1,tgt_len,src_len):


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(73/73)│100%(73/73)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(194/194)│100%(194/194)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>





..parsed-literal::

32811



U-Net
~~~~~

`backtotop⬆️<#table-of-contents>`__

TheprocessofU-Netmodelconversionremainsthesame,likefor
originalStableDiffusionXLmodel.

..code::ipython3

unet=pipe.unet
unet.eval()


classUnetWrapper(torch.nn.Module):
def__init__(self,unet):
super().__init__()
self.unet=unet

defforward(
self,
sample=None,
timestep=None,
encoder_hidden_states=None,
text_embeds=None,
time_ids=None,
):
returnself.unet.forward(
sample,
timestep,
encoder_hidden_states,
added_cond_kwargs={"text_embeds":text_embeds,"time_ids":time_ids},
)


inputs={
"sample":torch.rand([2,4,128,128],dtype=torch.float32),
"timestep":torch.from_numpy(np.array(1,dtype=float)),
"encoder_hidden_states":torch.rand([2,77,2048],dtype=torch.float32),
"text_embeds":torch.rand([2,1280],dtype=torch.float32),
"time_ids":torch.rand([2,6],dtype=torch.float32),
}

input_info=prepare_input_info(inputs)

w_unet=UnetWrapper(unet)
convert(w_unet,UNET_OV_PATH,inputs,input_info)

delw_unet,unet
gc.collect()


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_2d_condition.py:1103:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifdim%default_overall_up_factor!=0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:136:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:145:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifhidden_states.shape[0]>=64:


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(794/794)│100%(794/794)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>





..parsed-literal::

101843



VAEDecoder
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheVAEmodelhastwoparts,anencoderandadecoder.Theencoderis
usedtoconverttheimageintoalowdimensionallatentrepresentation,
whichwillserveastheinputtotheU-Netmodel.Thedecoder,
conversely,transformsthelatentrepresentationbackintoanimage.

WhenrunningText-to-Imagepipeline,wewillseethatweonlyneedthe
VAEdecoder.

..code::ipython3

vae_decoder=pipe.vae
vae_decoder.eval()


classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae_decoder):
super().__init__()
self.vae=vae_decoder

defforward(self,latents):
returnself.vae.decode(latents)


w_vae_decoder=VAEDecoderWrapper(vae_decoder)
inputs=torch.zeros((1,4,128,128))

convert(w_vae_decoder,VAE_DECODER_OV_PATH,inputs,input_info=[1,4,128,128])

delw_vae_decoder,vae_decoder
gc.collect()


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│100%(40/40)│100%(40/40)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>





..parsed-literal::

5992



PrepareInferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Inthisexample,wewillreuse``PhotoMakerStableDiffusionXLPipeline``
pipelinetogeneratetheimagewithOpenVINO,soeachmodel’sobjectin
thispipelineshouldbereplacedwithnewOpenVINOmodelobject.

SelectinferencedeviceforStableDiffusionpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importipywidgetsaswidgets

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



CompilemodelsandcreatetheirWrappersforinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ToaccessoriginalPhotoMakerworkflow,wehavetocreateanewwrapper
foreachOpenVINOcompiledmodel.Formatchingoriginalpipeline,part
ofOpenVINOmodelwrapper’sattributesshouldbereusedfromoriginal
modelobjectsandinferenceoutputmustbeconvertedfromnumpyto
``torch.tensor``.

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

compiled_id_encoder=core.compile_model(ID_ENCODER_OV_PATH,device.value)
compiled_unet=core.compile_model(UNET_OV_PATH,device.value)
compiled_text_encoder=core.compile_model(TEXT_ENCODER_OV_PATH,device.value)
compiled_text_encoder_2=core.compile_model(TEXT_ENCODER_2_OV_PATH,device.value)
compiled_vae_decoder=core.compile_model(VAE_DECODER_OV_PATH,device.value)

..code::ipython3

fromcollectionsimportnamedtuple


classOVIDEncoderWrapper(PhotoMakerIDEncoder):
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,id_encoder,orig_id_encoder):
super().__init__()
self.id_encoder=id_encoder
self.modules=orig_id_encoder.modules#accessedintheoriginalworkflow
self.config=orig_id_encoder.config#accessedintheoriginalworkflow

def__call__(
self,
*args,
):
id_pixel_values,prompt_embeds,class_tokens_mask=args
inputs={
"id_pixel_values":id_pixel_values,
"prompt_embeds":prompt_embeds,
"class_tokens_mask":class_tokens_mask,
}
output=self.id_encoder(inputs)[0]
returntorch.from_numpy(output)

..code::ipython3

classOVTextEncoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,text_encoder,orig_text_encoder):
self.text_encoder=text_encoder
self.modules=orig_text_encoder.modules#accessedintheoriginalworkflow
self.config=orig_text_encoder.config#accessedintheoriginalworkflow

def__call__(self,input_ids,**kwargs):
inputs={"input_ids":input_ids}
output=self.text_encoder(inputs)

hidden_states=[]
hidden_states_len=len(output)
foriinrange(1,hidden_states_len):
hidden_states.append(torch.from_numpy(output[i]))

BaseModelOutputWithPooling=namedtuple("BaseModelOutputWithPooling","last_hidden_statehidden_states")
output=BaseModelOutputWithPooling(torch.from_numpy(output[0]),hidden_states)
returnoutput

..code::ipython3

classOVUnetWrapper:
def__init__(self,unet,unet_orig):
self.unet=unet
self.config=unet_orig.config#accessedintheoriginalworkflow
self.add_embedding=unet_orig.add_embedding#accessedintheoriginalworkflow

def__call__(self,*args,**kwargs):
latent_model_input,t=args
inputs={
"sample":latent_model_input,
"timestep":t,
"encoder_hidden_states":kwargs["encoder_hidden_states"],
"text_embeds":kwargs["added_cond_kwargs"]["text_embeds"],
"time_ids":kwargs["added_cond_kwargs"]["time_ids"],
}

output=self.unet(inputs)

return[torch.from_numpy(output[0])]

..code::ipython3

classOVVAEDecoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,vae,vae_orig):
self.vae=vae
self.config=vae_orig.config#accessedintheoriginalworkflow

defdecode(self,latents,return_dict=False):
output=self.vae(latents)[0]
output=torch.from_numpy(output)

return[output]

ReplacethePyTorchmodelobjectsinoriginalpipelinewithOpenVINO
models

..code::ipython3

pipe.id_encoder=OVIDEncoderWrapper(compiled_id_encoder,pipe.id_encoder)
pipe.unet=OVUnetWrapper(compiled_unet,pipe.unet)
pipe.text_encoder=OVTextEncoderWrapper(compiled_text_encoder,pipe.text_encoder)
pipe.text_encoder_2=OVTextEncoderWrapper(compiled_text_encoder_2,pipe.text_encoder_2)
pipe.vae=OVVAEDecoderWrapper(compiled_vae_decoder,pipe.vae)

RunningText-to-ImageGenerationwithOpenVINO
----------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromdiffusers.utilsimportload_image

prompt="sci-fi,closeupportraitphotoofamanimginIronmansuit,face"
negative_prompt="(asymmetry,worstquality,lowquality,illustration,3d,2d,painting,cartoons,sketch),openmouth"
generator=torch.Generator("cpu").manual_seed(42)

input_id_images=[]
original_image=load_image("./PhotoMaker/examples/newton_man/newton_0.jpg")
input_id_images.append(original_image)

##Parametersetting
num_steps=20
style_strength_ratio=20
start_merge_step=int(float(style_strength_ratio)/100*num_steps)
ifstart_merge_step>30:
start_merge_step=30

images=pipe(
prompt=prompt,
input_id_images=input_id_images,
negative_prompt=negative_prompt,
num_images_per_prompt=1,
num_inference_steps=num_steps,
start_merge_step=start_merge_step,
generator=generator,
).images



..parsed-literal::

0%||0/20[00:00<?,?it/s]


..code::ipython3

importmatplotlib.pyplotasplt


defvisualize_results(orig_img:Image.Image,output_img:Image.Image):
"""
Helperfunctionforposeestimationresultsvisualization

Parameters:
orig_img(Image.Image):originalimage
output_img(Image.Image):processedimagewithPhotoMaker
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigure
"""
orig_img=orig_img.resize(output_img.size)
orig_title="Originalimage"
output_title="Outputimage"
im_w,im_h=orig_img.size
is_horizontal=im_h<im_w
fig,axs=plt.subplots(
2ifis_horizontalelse1,
1ifis_horizontalelse2,
sharex="all",
sharey="all",
)
fig.suptitle(f"Prompt:'{prompt}'",fontweight="bold")
fig.patch.set_facecolor("white")
list_axes=list(axs.flat)
forainlist_axes:
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
list_axes[0].imshow(np.array(orig_img))
list_axes[1].imshow(np.array(output_img))
list_axes[0].set_title(orig_title,fontsize=15)
list_axes[1].set_title(output_title,fontsize=15)
fig.subplots_adjust(wspace=0.01ifis_horizontalelse0.00,hspace=0.01ifis_horizontalelse0.1)
fig.tight_layout()
returnfig


fig=visualize_results(original_image,images[0])



..image::photo-maker-with-output_files/photo-maker-with-output_33_0.png


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defgenerate_from_text(text_promt,input_image,neg_prompt,seed,num_steps,style_strength_ratio):
"""
Helperfunctionforgeneratingresultimagefromprompttext

Parameters:
text_promt(String):positiveprompt
input_image(Image.Image):originalimage
neg_prompt(String):negativeprompt
seed(Int):seedforrandomgeneratorstateinitialization
num_steps(Int):numberofsamplingsteps
style_strength_ratio(Int):thepercentageofstepwhenmergingtheIDembeddingtotextembedding

Returns:
result(Image.Image):generationresult
"""
start_merge_step=int(float(style_strength_ratio)/100*num_steps)
ifstart_merge_step>30:
start_merge_step=30
result=pipe(
text_promt,
input_id_images=input_image,
negative_prompt=neg_prompt,
num_inference_steps=num_steps,
num_images_per_prompt=1,
start_merge_step=start_merge_step,
generator=torch.Generator().manual_seed(seed),
height=1024,
width=1024,
).images[0]

returnresult


withgr.Blocks()asdemo:
withgr.Column():
withgr.Row():
input_image=gr.Image(label="Yourimage",sources=["upload"],type="pil")
output_image=gr.Image(label="GeneratedImages",type="pil")
positive_input=gr.Textbox(label=f"Textprompt,Triggerwordsis'{trigger_word}'")
neg_input=gr.Textbox(label="Negativeprompt")
withgr.Row():
seed_input=gr.Slider(0,10_000_000,value=42,label="Seed")
steps_input=gr.Slider(label="Steps",value=10,minimum=5,maximum=50,step=1)
style_strength_ratio_input=gr.Slider(label="Stylestrengthratio",value=20,minimum=5,maximum=100,step=5)
btn=gr.Button()
btn.click(
generate_from_text,
[
positive_input,
input_image,
neg_input,
seed_input,
steps_input,
style_strength_ratio_input,
],
output_image,
)
gr.Examples(
[
[prompt,negative_prompt],
[
"AwomanimgwearingaChristmashat",
negative_prompt,
],
[
"Amanimginahelmetandvestridingamotorcycle",
negative_prompt,
],
[
"photoofamiddle-agedmanimgsittingonaplushleathercouch,andwatchingtelevisionshow",
negative_prompt,
],
[
"photoofaskilleddoctorimginapristinewhitelabcoatenjoyingadeliciousmealinasophisticateddiningroom",
negative_prompt,
],
[
"photoofsupermanimgflyingthroughavibrantsunsetsky,withhiscapebillowinginthewind",
negative_prompt,
],
],
[positive_input,neg_input],
)


demo.queue().launch()
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>




..parsed-literal::





..code::ipython3

demo.close()


..parsed-literal::

Closingserverrunningonport:7860


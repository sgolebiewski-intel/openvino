TextGenerationviaSpeculativeSampling,KVCaching,andOpenVINO™
===================================================================

Asmodelsizesgrow,GenerativeAIimplementationsrequiresignificant
inferenceresources.Thisnotonlyincreasesthecostpergeneration
fromaprompt,butalsoincreasesthepowerconsumptionusedtoserve
suchrequests.

Inferenceoptimizationsfortextgenerationareessentialforreducing
costsandpowerconsumption.Whenoptimizingtheinferenceprocess,the
amountoftimeandenergyrequiredtogeneratetextcanbesignificantly
reduced.Thiscanleadtocostsavingsintermsofhardwareand
software,aswellasreducedpowerconsumption.Additionally,inference
optimizationscanhelpimprovetheaccuracyoftextgenerationaswell
asthespeedatwhichitcanbegenerated.Thiscanleadtoanimproved
userexperienceandincreasedefficiencyintext-generationtasks.In
summary,inferenceoptimizationsfortextgenerationareessentialto
reducecostsandpowerconsumption,whilealsoimprovingtheaccuracy
andspeedoftextgeneration.

Anothernecessaryconditionisthattheoptimizationsarecompatible
witheachother.Thatis,implementingacertainoptimizationshouldnot
precludeotheroptimizations.Thereareseverallevelsofoptimizations
thatcanprovidesignificantspeedupwithout“bumpingintoeachother”
inawaythatwillcompromiseoverallefficiency.

Fordetailsonthismethod,pleaserefertothepaperbyChenetal,
http://arxiv.org/abs/2302.01318.Additionally,there’saninteresting
proofofcorrectnessofspeculativesampling(showingthattheoriginal
distributionispreserved)byLeviathanetal,
http://arxiv.org/abs/2211.17192

OurblogarticledescribingthisimplementationwithOpenVinois
availableatopenvino.ai

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__

-`Selectinferencedevice<#select-inference-device>`__

-`CreateautoregressiveandspeculativeformsofsamplingwithKV
Cache
support<#create-autoregressive-and-speculative-forms-of-sampling-with-kv-cache-support>`__

-`Setupimports<#setup-imports>`__
-`Prepareautoregressive
sampling<#prepare-autoregressive-sampling>`__
-`Preparespeculativesampling<#prepare-speculative-sampling>`__

-`Maingenerationfunction<#main-generation-function>`__

-`DownloadandConvertModel<#download-and-convert-model>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

First,weshouldinstallthe`HuggingFace
Optimum<https://huggingface.co/docs/optimum/installation>`__library
acceleratedbyOpenVINOintegration.TheHuggingFaceOptimumIntelAPI
isahigh-levelAPIthatenablesustoconvertandquantizemodelsfrom
theHuggingFaceTransformerslibrarytotheOpenVINO™IRformat.For
moredetails,refertothe`HuggingFaceOptimumIntel
documentation<https://huggingface.co/docs/optimum/intel/inference>`__.

Wewillalsoneedtoinstalltransformers(HuggingFace)andsomeother
usefulmodules.

..code::ipython3

%pipinstall-Uqpip
%pipuninstall-q-yoptimumoptimum-intel
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q--upgradetransformers"torch>=2.1""gradio>=4.19"accelerateonnxipywidgets"peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git"

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Selectthedevicefromdropdownlistforrunninginferenceusing
OpenVINO.

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','AUTO'),value='CPU')



CreateautoregressiveandspeculativeformsofsamplingwithKVCachesupport
-----------------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Textgenerationisoftendoneinanautoregressivefashion.Wewillall
supportaKVcache(akaPastValueCache)inthecode.Notethatweare
usinggreedysampling.Wedonotadjustothertextgenerationparameters
(e.g. temperature)sokeepthisillustrationofspeculativesamplingas
simpleandunderstandableaspossible.

Setupimports
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtime
importnumpyasnp
importgradioasgr

Prepareautoregressivesampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defautoregressive_sampling_with_pkv(input,model,N=30):
input_ids,attention_mask=input.input_ids,input.attention_mask
seq_len=input_ids.shape[-1]
position_ids=np.arange(0,seq_len,dtype=np.int64).reshape([-1,seq_len])

#inallsubsequentinferenceswefeedtokensonebyone,
#butforthefirstonewefeedthewholeencodedprompt
request=model.create_infer_request()
request.infer((input_ids,attention_mask,position_ids,np.array([0])))
next_token=np.argmax(request.results["logits"][:,-1]).reshape([1])

all_tokens=[]
all_tokens.extend(input_ids[0])
all_tokens.append(next_token[0])

whileseq_len<N:
input_ids=next_token.reshape([1,1])
attention_mask=np.concatenate((attention_mask,np.array([1]).reshape([1,1])),axis=1)
position_ids=np.array([attention_mask.shape[1]]).reshape([1,1])

request.infer((input_ids,attention_mask,position_ids,np.array([0])))
next_token=np.argmax(request.results["logits"][:,-1])
all_tokens.append(next_token)
seq_len+=1

returnall_tokens

Preparespeculativesampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

-Step1:Withspeculativesampling,wefirstgenerateKsamplesfrom
thedraftmodel(inanautoregressivemanner).
-Step2:Thesearenowcandidatestoexamineusingthemainmodel
(step2)usingabatchsizeofK.
-Step3:WegothrougheachKpredictedtokens,andiftokensdiffer,
westopandkeepthelasttokenpredictedbythemainmodel.
-Step4:WeupdateKV-cachedroppingkeys&valuesfordiffering
tokensandrepeatStep1.

..code::ipython3

defupdate_state(request,seq_len):
forstateinrequest.query_state():
old_seq_len=state.state.shape[2]
ifseq_len>=old_seq_len:
continue
#Aftertheinferencerequest,key/valueshaveshape[BATCH_SIZE,seq_len+K,vocab_size].
#Incrementthesequencelengthbythenumberofmatchedtokens,and
#trimtheKVcachetomatchthenewsequencelength.
state.state=ov.Tensor(state.state.data[:,:,:seq_len])


defspeculative_sampling_with_pkv(input,draft_model,main_model,K,N=30,**kwargs):
input_ids,attention_mask=input.input_ids,input.attention_mask
#seq_lennumberofkey/valuesornumberofalreadyprocessedinputtokens
seq_len=input_ids.shape[-1]
position_ids=np.arange(0,seq_len,dtype=np.int64).reshape([-1,seq_len])

draft_request=draft_model.create_infer_request()
draft_request.infer((input_ids,attention_mask,position_ids,np.array([0])))

main_request=main_model.create_infer_request()
main_request.infer((input_ids,attention_mask,position_ids,np.array([0])))
first_token=np.argmax(main_request.results["logits"][:,-1]).reshape([1])

all_tokens=[]
all_tokens.extend(input_ids[0])
all_tokens.append(first_token[0])

accum_draft_tokens=[]
whileseq_len<N:
next_token=first_token
foriinrange(K):
input_ids=next_token.reshape([1,1])
attention_mask=np.concatenate((attention_mask,np.array([1]).reshape([1,1])),axis=1)
position_ids=np.array([attention_mask.shape[1]]).reshape([1,1])

draft_request.infer((input_ids,attention_mask,position_ids,np.array([0])))
next_token=np.argmax(draft_request.results["logits"][:,-1])
accum_draft_tokens.append(next_token)

#mainmodelwillgivealsoKouttokens
#feedthesamefirsttokentothemainmodelanddonotgivethelasttokengeneratedbythedraft
input_ids=np.concatenate((first_token.reshape([1]),accum_draft_tokens[:-1])).reshape([1,-1])
attention_mask=np.ones((1,seq_len+K))
position_ids=np.arange(seq_len,seq_len+K,dtype=np.int64).reshape([1,-1])

main_request.infer((input_ids,attention_mask,position_ids,np.array([0])))
next_tokens=np.argmax(main_request.results["logits"],axis=-1)[0]

#ifdisagreesfromtheverybegginingthencontextwillbeexpandedonlyforoneelement
#allelementsmatchthencontextwillbeexpandedtoKelements
fordisagree_idx,(t1,t2)inenumerate(zip(accum_draft_tokens,next_tokens)):
ift1!=t2:
break

first_token=next_tokens[disagree_idx]
all_tokens.extend(next_tokens[:disagree_idx+1])
seq_len+=disagree_idx+1

#cutkey/valuesdependingonthepositionwheredisagreementstarts
update_state(draft_request,seq_len)
update_state(main_request,seq_len)

attention_mask=np.ones((1,seq_len))
accum_draft_tokens=[]
all_tokens.extend(accum_draft_tokens)
returnall_tokens

Maingenerationfunction
------------------------

`backtotop⬆️<#table-of-contents>`__

DownloadandConvertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.Forspeculativedecodingweneedtomanuallyupdatestates,
thereforewewillusedirectlyopenvinoinferenceapi,andoptimumonly
formodelconversion.>TodownloadLlama-2-7b-chat-hf,youwillneedto
acceptlicenseagreement.Youmustbearegistereduserin🤗Hugging
FaceHub.PleasevisitHuggingFacemodel
`card<https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
carefullyreadtermsofusageandclickacceptbutton.Youwillneedto
useanaccesstokenforthecodebelowtorun.Formoreinformationon
accesstokens,refertothissectionofthedocumentation.

..code::ipython3

frompathlibimportPath

main_model_id="meta-llama/Llama-2-7b-chat-hf"
main_model_path=Path("Llama-2-7b-chat-hf")
draft_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
draft_model_path=Path("TinyLlama-1.1B-Chat-v1.0")

fromtransformersimportAutoTokenizer

main_tokenizer=AutoTokenizer.from_pretrained(main_model_id)
draft_tokenizer=AutoTokenizer.from_pretrained(draft_model_id)

..code::ipython3

#Inorderforspeculativesamplingtowork,bothmainanddrafttokenizersshouldbethesame.
token_test_txt="texttoensuretokenizersworkthesame,asof2024"
tokens_1=draft_tokenizer(token_test_txt,return_tensors="pt").input_ids
tokens_2=main_tokenizer(token_test_txt,return_tensors="pt").input_ids

assertall((tokens_1-tokens_2)[0]==0)

..code::ipython3

ifnotmain_model_path.exists():
!optimum-cliexportopenvino--model$main_model_id--weight-formatfp16$main_model_path
ifnotdraft_model_path.exists():
!optimum-cliexportopenvino--model$draft_model_id--weight-formatfp16$draft_model_path

InferdirectlyusingOpenVINOInferencePipeline

..code::ipython3

core=ov.Core()
draft_ov_model=core.read_model(draft_model_path/"openvino_model.xml")
draft_model=core.compile_model(draft_ov_model,device_name="CPU")

main_ov_model=core.read_model(main_model_path/"openvino_model.xml")
main_model=core.compile_model(main_ov_model,device_name="CPU")

..code::ipython3

defmain(
prompt:str,
n_tokens_to_generate:int=75,
K:int=5,
seed:int=5555,
):
#seednumpyrng
np.random.seed(seed)
tokenized=main_tokenizer(prompt,return_tensors="pt")

defrun_autoregressive_sampling_fn(decode_fn,tokenized,**kwargs):
start=time.perf_counter()
output_ids=decode_fn(tokenized,**kwargs)
text=main_tokenizer.decode(output_ids,skip_special_tokens=True)
elapsed_time=time.perf_counter()-start
returntext,elapsed_time

defrun_speculative_sampling_fn(decode_fn,input_ids,**kwargs):
start=time.perf_counter()
output_ids=decode_fn(input_ids,**kwargs)
text=main_tokenizer.decode(output_ids,skip_special_tokens=True)
elapsed_time=time.perf_counter()-start
returntext,elapsed_time

autoregressive_text,autoregressive_time=run_autoregressive_sampling_fn(
autoregressive_sampling_with_pkv,
tokenized,
model=main_model,
N=n_tokens_to_generate,
)

speculative_text,speculative_time=run_speculative_sampling_fn(
speculative_sampling_with_pkv,
tokenized,
main_model=main_model,
draft_model=draft_model,
N=n_tokens_to_generate,
K=K,
)

#Formatresultsforoutputingradio
out="\n"+"AutoregressiveDecode"+"\n"+"---------------------"+"\n"
out=out+f"Time={autoregressive_time:.2f}s"+"\n"+f"Text={autoregressive_text}"+"\n"
out=out+"\n"+"SpeculativeDecode"+"\n"+"------------------"+"\n"
out=out+f"Time={speculative_time:.2f}s"+"\n"+f"Text={speculative_text}"
returnout

..code::ipython3

res=main("AlanTuringwasa",n_tokens_to_generate=100)
print(res)


..parsed-literal::

2024-04-1710:21:41.642283:Itensorflow/core/util/port.cc:111]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-04-1710:21:41.644834:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-1710:21:41.677055:Etensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342]UnabletoregistercuDNNfactory:AttemptingtoregisterfactoryforplugincuDNNwhenonehasalreadybeenregistered
2024-04-1710:21:41.677093:Etensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609]UnabletoregistercuFFTfactory:AttemptingtoregisterfactoryforplugincuFFTwhenonehasalreadybeenregistered
2024-04-1710:21:41.677119:Etensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518]UnabletoregistercuBLASfactory:AttemptingtoregisterfactoryforplugincuBLASwhenonehasalreadybeenregistered
2024-04-1710:21:41.683198:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-04-1710:21:41.683977:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-04-1710:21:42.477656:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::


AutoregressiveDecode
---------------------
Time=44.39s
Text=AlanTuringwasaBritishmathematician,computerscientist,andcodebreakerwhoplayedapivotalroleincrackingtheGermanEnigmacodeduringWorldWarII.Hewasalsoapioneerinthefieldofartificialintelligenceandmadesignificantcontributionstothedevelopmentofcomputerscience.

TuringwasbornonJune23,1912,inLondon,England.HewaseducatedatCambridgeUniversity,whereheearnedadegreeinmathematicsin

SpeculativeDecode
------------------
Time=22.96s
Text=AlanTuringwasaBritishmathematician,computerscientist,andcodebreakerwhoplayedapivotalroleincrackingtheGermanEnigmacodeduringWorldWarII.Hewasalsoapioneerinthefieldofartificialintelligenceandmadesignificantcontributionstothedevelopmentofcomputerscience.

TuringwasbornonJune23,1912,inLondon,England.HewaseducatedatCambridgeUniversity,whereheearnedadegreeinmathematicsin1


..code::ipython3

withgr.Blocks()asdemo:
gr.Markdown(
f"""
#SpeculativeSamplingDemo
##TheoutputwillshowacomparisonofAutoregressiveSamplingvsSpeculativeSampling
-MainModel:{main_model_id}
-DraftModel:{draft_model_id}
-K=5
"""
)
withgr.Row():
inp=gr.Textbox(
"AlanTuringwasa",
placeholder="THISCANNOTBEEMPTY",
label="InputPrompt",
)
out=gr.Textbox(label="Output")
btn=gr.Button("Run")
btn.click(fn=main,inputs=inp,outputs=out)

demo.launch()

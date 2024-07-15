LLM-poweredchatbotusingStable-Zephyr-3bandOpenVINO
=======================================================

Intherapidlyevolvingworldofartificialintelligence(AI),chatbots
havebecomepowerfultoolsforbusinessestoenhancecustomer
interactionsandstreamlineoperations.LargeLanguageModels(LLMs)are
artificialintelligencesystemsthatcanunderstandandgeneratehuman
language.Theyusedeeplearningalgorithmsandmassiveamountsofdata
tolearnthenuancesoflanguageandproducecoherentandrelevant
responses.Whileadecentintent-basedchatbotcananswerbasic,
one-touchinquirieslikeordermanagement,FAQs,andpolicyquestions,
LLMchatbotscantacklemorecomplex,multi-touchquestions.LLMenables
chatbotstoprovidesupportinaconversationalmanner,similartohow
humansdo,throughcontextualmemory.Leveragingthecapabilitiesof
LanguageModels,chatbotsarebecomingincreasinglyintelligent,capable
ofunderstandingandrespondingtohumanlanguagewithremarkable
accuracy.

``StableZephyr3B``isa3billionparametermodelthatdemonstrated
outstandingresultsonmanyLLMevaluationbenchmarksoutperformingmany
popularmodelsinrelativelysmallsize.Inspiredby`HugginFaceH4’s
Zephyr7B<https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__
trainingpipelinethismodelwastrainedonamixofpubliclyavailable
datasets,syntheticdatasetsusing`DirectPreferenceOptimization
(DPO)<https://arxiv.org/abs/2305.18290>`__,evaluationforthismodel
basedon`MTBench<https://tatsu-lab.github.io/alpaca_eval/>`__and
`AlpacaBenchmark<https://tatsu-lab.github.io/alpaca_eval/>`__.More
detailsaboutmodelcanbefoundin`model
card<https://huggingface.co/stabilityai/stablelm-zephyr-3b>`__

Inthistutorial,weconsiderhowtooptimizeandrunthismodelusing
theOpenVINOtoolkit.Fortheconvenienceoftheconversionstepand
modelperformanceevaluation,wewilluse
`llm_bench<https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python>`__
tool,whichprovidesaunifiedapproachtoestimateperformanceforLLM.
ItisbasedonpipelinesprovidedbyOptimum-Intelandallowsto
estimateperformanceforPytorchandOpenVINOmodelsusingalmostthe
samecode.Wealsodemonstratehowtomakemodelstateful,thatprovides
opportunityforprocessingmodelcachestate.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`ConvertmodeltoOpenVINOIntermediateRepresentation(IR)and
compressmodelweightstoINT4using
NNCF<#convert-model-to-openvino-intermediate-representation-ir-and-compress-model-weights-to-int4-using-nncf>`__
-`Applystatefultransformationforautomatichandlingmodel
state<#apply-stateful-transformation-for-automatic-handling-model-state>`__
-`Selectdeviceforinference<#select-device-for-inference>`__
-`Estimatemodelperformance<#estimate-model-performance>`__
-`UsingmodelwithOptimumIntel<#using-model-with-optimum-intel>`__
-`Interactivechatbotdemo<#interactive-chatbot-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Forstartingwork,weshouldinstallrequiredpackagesfirst

..code::ipython3

frompathlibimportPath
importsys


genai_llm_bench=Path("openvino.genai/llm_bench/python")

ifnotgenai_llm_bench.exists():
!gitclonehttps://github.com/openvinotoolkit/openvino.genai.git

sys.path.append(str(genai_llm_bench))

..code::ipython3

%pipinstall-q"transformers>=4.38.2"
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu-r./openvino.genai/llm_bench/python/requirements.txt
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q"gradio>=4.19"

ConvertmodeltoOpenVINOIntermediateRepresentation(IR)andcompressmodelweightstoINT4usingNNCF
--------------------------------------------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

llm_benchprovidesconversionscriptforconvertingLLMSintoOpenVINO
IRformatcompatiblewithOptimum-Intel.Italsoallowstocompress
modelweightsintoINT8orINT4precisionwith
`NNCF<https://github.com/openvinotoolkit/nncf>`__.Forenablingweights
compressioninINT4weshoulduse``--compress_weights4BIT_DEFAULT``
argument.TheWeightsCompressionalgorithmisaimedatcompressingthe
weightsofthemodelsandcanbeusedtooptimizethemodelfootprint
andperformanceoflargemodelswherethesizeofweightsisrelatively
largerthanthesizeofactivations,forexample,LargeLanguageModels
(LLM).ComparedtoINT8compression,INT4compressionimproves
performanceevenmorebutintroducesaminordropinpredictionquality.

Applystatefultransformationforautomatichandlingmodelstate
----------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

StableZephyrisadecoder-onlytransformermodelandgeneratestext
tokenbytokeninanautoregressivefashion.Sincetheoutputsideis
auto-regressive,anoutputtokenhiddenstateremainsthesameonce
computedforeveryfurthergenerationstep.Therefore,recomputingit
everytimeyouwanttogenerateanewtokenseemswasteful.Tooptimize
thegenerationprocessandusememorymoreefficiently,HuggingFace
transformersAPIprovidesamechanismforcachingmodelstateexternally
using``use_cache=True``parameterand``past_key_values``argumentin
inputsandoutputs.Withthecache,themodelsavesthehiddenstate
onceithasbeencomputed.Themodelonlycomputestheoneforthemost
recentlygeneratedoutputtokenateachtimestep,re-usingthesaved
onesforhiddentokens.Thisreducesthegenerationcomplexityfrom
:math:`O(n^3)`to:math:`O(n^2)`foratransformermodel.Withthis
option,themodelgetsthepreviousstep’shiddenstates(cached
attentionkeysandvalues)asinputandadditionallyprovideshidden
statesforthecurrentstepasoutput.Itmeansforallnextiterations,
itisenoughtoprovideonlyanewtokenobtainedfromthepreviousstep
andcachedkeyvaluestogetthenexttokenprediction.

WithincreasingmodelsizelikeinmodernLLMs,wealsocannotean
increaseinthenumberofattentionblocksandsizepastkeyvalues
tensorsrespectively.Thestrategyforhandlingcachestateasmodel
inputsandoutputsintheinferencecyclemaybecomeabottleneckfor
memory-boundedsystems,especiallywithprocessinglonginputsequences,
forexampleinachatbotscenario.OpenVINOsuggestsatransformation
thatremovesinputsandcorrespondingoutputswithcachetensorsfrom
themodelkeepingcachehandlinglogicinsidethemodel.Hidingthe
cacheenablesstoringandupdatingthecachevaluesinamore
device-friendlyrepresentation.Ithelpstoreducememoryconsumption
andadditionallyoptimizemodelperformance.

llm_benchconvertmodelinstatefulformatbydefault,ifyouwant
disablethisbehavioryoucanspecify``--disable_stateful``flagfor
that

..code::ipython3

stateful_model_path=Path("stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")

convert_script=genai_llm_bench/"convert.py"

ifnot(stateful_model_path/"openvino_model.xml").exists():
!python$convert_script--model_idstabilityai/stable-zephyr-3b--precisionFP16--compress_weights4BIT_DEFAULT--outputstable-zephyr-3b-stateful--force_convert


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
2024-03-0513:50:49.184866:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-0513:50:49.186797:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:50:49.223416:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:50:49.223832:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-03-0513:50:49.887707:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.1.0+cu121withCUDA1201(youhave2.2.0+cpu)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
WARNING:nncf:NNCFprovidesbestresultswithtorch==2.2.1,whilecurrenttorchversionis2.2.0+cpu.Ifyouencounterissues,considerswitchingtotorch==2.2.1
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34:UserWarning:TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
warn("TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport."
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so:undefinedsymbol:cadam32bit_grad_fp32
[INFO]openvinoruntimeversion:2024.1.0-14645-e6dc0865128
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
[INFO]ModelconversiontoFP16willbeskippedasfoundconvertedmodelstable-zephyr-3b-stateful/pytorch/dldt/FP16/openvino_model.xml.Ifitisnotexpectedbehaviour,pleaseremovepreviouslyconvertedmodeloruse--force_convertoption
[INFO]Compressmodelweightsto4BIT_DEFAULT
[INFO]Compressionoptions:
[INFO]{'mode':<CompressWeightsMode.INT4_SYM:'int4_sym'>,'group_size':128}
INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|9%(2/226)|0%(0/224)|
+--------------+---------------------------+-----------------------------------+
|4|91%(224/226)|100%(224/224)|
+--------------+---------------------------+-----------------------------------+
[2KApplyingWeightCompression━━━━━━━━━━━━━━━━━━━100%226/226•0:01:29•0:00:00;0;104;181m0:00:01181m0:00:05


Selectdeviceforinference
---------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices,
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1'),value='CPU')



Estimatemodelperformance
--------------------------

`backtotop⬆️<#table-of-contents>`__

openvino.genai/llm_bench/python/benchmark.pyscriptallowto
estimatetextgenerationpipelineinferenceonspecificinputprompt
withgivennumberofmaximumgeneratedtokens.

..code::ipython3

benchmark_script=genai_llm_bench/"benchmark.py"

!python$benchmark_script-m$stateful_model_path-ic512-p"Tellmestoryaboutcats"-d$device.value


..parsed-literal::

/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.1.0+cu121withCUDA1201(youhave2.2.0+cpu)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
2024-03-0513:52:39.048911:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-0513:52:39.050779:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:52:39.088178:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:52:39.088623:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-03-0513:52:39.754578:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34:UserWarning:TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
warn("TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport."
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so:undefinedsymbol:cadam32bit_grad_fp32
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
[INFO]==SUCCESSFOUND==:use_case:text_gen,model_type:stable-zephyr-3b-stateful
[INFO]OVConfig={'PERFORMANCE_HINT':'LATENCY','CACHE_DIR':'','NUM_STREAMS':'1'}
[INFO]OPENVINO_TORCH_BACKEND_DEVICE=CPU
[INFO]Modelpath=stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT,openvinoruntimeversion:2024.1.0-14645-e6dc0865128
CompilingthemodeltoCPU...
[INFO]Frompretrainedtime:3.21s
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
[INFO]Numbeams:1,benchmarkingiternums(excludewarm-up):0,promptnums:1
[INFO][warm-up]Inputtext:Tellmestoryaboutcats
Setting`pad_token_id`to`eos_token_id`:0foropen-endgeneration.
[INFO][warm-up]Inputtokensize:5,Outputsize:336,Infercount:512,TokenizationTime:2.23ms,DetokenizationTime:0.51ms,GenerationTime:23.79s,Latency:70.80ms/token
[INFO][warm-up]Firsttokenlatency:837.58ms/token,othertokenslatency:68.43ms/token,lenoftokens:336
[INFO][warm-up]Firstinferlatency:836.44ms/infer,otherinferslatency:67.89ms/infer,inferencecount:336
[INFO][warm-up]ResultMD5:['601aa0958ff0e0f9b844a9e6d186fbd9']
[INFO][warm-up]Generated:Tellmestoryaboutcatsanddogs.
Onceuponatime,inasmallvillage,therelivedayounggirlnamedLily.Shehadtwopets,acatnamedMittensandadognamedMax.Mittenswasabeautifulblackcatwithgreeneyes,andMaxwasabiglovablegoldenretrieverwithawaggingtail.
Onesunnyday,Lilydecidedtotakeherpetsforawalkinthenearbyforest.Astheywerewalking,theyheardaloudbarkingsound.Suddenly,agroupofdogsappearedfromthebushes,ledbyabigbrowndogwithafriendlysmile.
Lilywasscaredatfirst,butMaxquicklyjumpedinfrontofherandgrowledatthedogs.ThebigbrowndogintroducedhimselfasRockyandexplainedthatheandhisfriendswerejustoutforawalktoo.
LilyandRockybecamefastfriends,andtheyoftenwentonwalkstogether.MaxandRockygotalongwelltoo,andtheywouldplaytogetherintheforest.
Oneday,whileLilywasatschool,MittensandMaxdecidedtoexploretheforestandstumbleduponagroupofstraycats.Thecatswerehungryandscared,soMittensandMaxdecidedtohelpthembygivingthemsomefood.
ThecatsweregratefulandthankedMittensandMaxfortheirkindness.TheyevenallowedMittenstoclimbontheirbacksandenjoythesun.
Fromthatdayon,MittensandMaxbecameknownasthevillage'scatanddogheroes.Theywerealwaystheretohelptheirfurryfriendsinneed.
Andso,Lilylearnedthatsometimesthebestfriendsaretheonesthatsharethesameloveforpets.<|endoftext|>


Comparewithmodelwithoutstate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

stateless_model_path=Path("stable-zephyr-3b-stateless/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")

ifnot(stateless_model_path/"openvino_model.xml").exists():
!python$convert_script--model_idstabilityai/stable-zephyr-3b--precisionFP16--compress_weights4BIT_DEFAULT--outputstable-zephyr-3b-stateless--force_convert--disable-stateful


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
2024-03-0513:53:12.727472:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-0513:53:12.729379:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:53:12.765262:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:53:12.765680:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-03-0513:53:13.414451:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.1.0+cu121withCUDA1201(youhave2.2.0+cpu)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
WARNING:nncf:NNCFprovidesbestresultswithtorch==2.2.1,whilecurrenttorchversionis2.2.0+cpu.Ifyouencounterissues,considerswitchingtotorch==2.2.1
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34:UserWarning:TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
warn("TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport."
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so:undefinedsymbol:cadam32bit_grad_fp32
[INFO]openvinoruntimeversion:2024.1.0-14645-e6dc0865128
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
Usingtheexportvariantdefault.Availablevariantsare:
-default:ThedefaultONNXvariant.
UsingframeworkPyTorch:2.2.0+cpu
Overriding1configurationitem(s)
	-use_cache->True
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/modeling_utils.py:4193:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:114:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
if(input_shape[-1]>1orself.sliding_windowisnotNone)andself.is_causal:
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/optimum/exporters/onnx/model_patcher.py:299:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifpast_key_values_length>0:
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:97:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifseq_len>self.max_seq_len_cached:
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:341:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz,self.num_heads,q_len,kv_seq_len):
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:348:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.size()!=(bsz,1,q_len,kv_seq_len):
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:360:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz,self.num_heads,q_len,self.head_dim):
[INFO]Compressmodelweightsto4BIT_DEFAULT
[INFO]Compressionoptions:
[INFO]{'mode':<CompressWeightsMode.INT4_SYM:'int4_sym'>,'group_size':128}
INFO:nncf:Statisticsofthebitwidthdistribution:
+--------------+---------------------------+-----------------------------------+
|Numbits(N)|%allparameters(layers)|%ratio-definingparameters|
|||(layers)|
+==============+===========================+===================================+
|8|9%(2/226)|0%(0/224)|
+--------------+---------------------------+-----------------------------------+
|4|91%(224/226)|100%(224/224)|
+--------------+---------------------------+-----------------------------------+
[2KApplyingWeightCompression━━━━━━━━━━━━━━━━━━━100%226/226•0:01:29•0:00:00;0;104;181m0:00:01181m0:00:05


..code::ipython3

!python$benchmark_script-m$stateless_model_path-ic512-p"Tellmestoryaboutcats"-d$device.value


..parsed-literal::

/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
WARNING[XFORMERS]:xFormerscan'tloadC++/CUDAextensions.xFormerswasbuiltfor:
PyTorch2.1.0+cu121withCUDA1201(youhave2.2.0+cpu)
Python3.8.18(youhave3.8.10)
Pleasereinstallxformers(seehttps://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficientattention,SwiGLU,sparseandmorewon'tbeavailable.
SetXFORMERS_MORE_DETAILS=1formoredetails
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
2024-03-0513:55:27.540258:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-0513:55:27.542166:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:55:27.578718:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-03-0513:55:27.579116:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-03-0513:55:28.229026:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34:UserWarning:TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.
warn("TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport."
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so:undefinedsymbol:cadam32bit_grad_fp32
/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
[INFO]==SUCCESSFOUND==:use_case:text_gen,model_type:stable-zephyr-3b-stateless
[INFO]OVConfig={'PERFORMANCE_HINT':'LATENCY','CACHE_DIR':'','NUM_STREAMS':'1'}
[INFO]OPENVINO_TORCH_BACKEND_DEVICE=CPU
[INFO]Modelpath=stable-zephyr-3b-stateless/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT,openvinoruntimeversion:2024.1.0-14645-e6dc0865128
Providedmodeldoesnotcontainstate.Itmayleadtosub-optimalperformance.PleasereexportmodelwithupdatedOpenVINOversion>=2023.3.0callingthe`from_pretrained`methodwithoriginalmodeland`export=True`parameter
CompilingthemodeltoCPU...
[INFO]Frompretrainedtime:3.15s
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.
[INFO]Numbeams:1,benchmarkingiternums(excludewarm-up):0,promptnums:1
[INFO][warm-up]Inputtext:Tellmestoryaboutcats
Setting`pad_token_id`to`eos_token_id`:0foropen-endgeneration.
[INFO][warm-up]Inputtokensize:5,Outputsize:336,Infercount:512,TokenizationTime:2.02ms,DetokenizationTime:0.51ms,GenerationTime:18.59s,Latency:55.32ms/token
[INFO][warm-up]Firsttokenlatency:990.01ms/token,othertokenslatency:52.47ms/token,lenoftokens:336
[INFO][warm-up]Firstinferlatency:989.00ms/infer,otherinferslatency:51.98ms/infer,inferencecount:336
[INFO][warm-up]ResultMD5:['601aa0958ff0e0f9b844a9e6d186fbd9']
[INFO][warm-up]Generated:Tellmestoryaboutcatsanddogs.
Onceuponatime,inasmallvillage,therelivedayounggirlnamedLily.Shehadtwopets,acatnamedMittensandadognamedMax.Mittenswasabeautifulblackcatwithgreeneyes,andMaxwasabiglovablegoldenretrieverwithawaggingtail.
Onesunnyday,Lilydecidedtotakeherpetsforawalkinthenearbyforest.Astheywerewalking,theyheardaloudbarkingsound.Suddenly,agroupofdogsappearedfromthebushes,ledbyabigbrowndogwithafriendlysmile.
Lilywasscaredatfirst,butMaxquicklyjumpedinfrontofherandgrowledatthedogs.ThebigbrowndogintroducedhimselfasRockyandexplainedthatheandhisfriendswerejustoutforawalktoo.
LilyandRockybecamefastfriends,andtheyoftenwentonwalkstogether.MaxandRockygotalongwelltoo,andtheywouldplaytogetherintheforest.
Oneday,whileLilywasatschool,MittensandMaxdecidedtoexploretheforestandstumbleduponagroupofstraycats.Thecatswerehungryandscared,soMittensandMaxdecidedtohelpthembygivingthemsomefood.
ThecatsweregratefulandthankedMittensandMaxfortheirkindness.TheyevenallowedMittenstoclimbontheirbacksandenjoythesun.
Fromthatdayon,MittensandMaxbecameknownasthevillage'scatanddogheroes.Theywerealwaystheretohelptheirfurryfriendsinneed.
Andso,Lilylearnedthatsometimesthebestfriendsaretheonesthatsharethesameloveforpets.<|endoftext|>


UsingmodelwithOptimumIntel
------------------------------

`backtotop⬆️<#table-of-contents>`__

RunningmodelwithOptimum-IntelAPIrequiredfollowingsteps:1.
registernormalizedconfigformodel2.createinstanceof
``OVModelForCausalLM``classusing``from_pretrained``method.

Themodeltextgenerationinterfaceremainswithoutchanges,thetext
generationprocessstartedwithrunning``ov_model.generate``methodand
passingtextencodedbythetokenizerasinput.Thismethodreturnsa
sequenceofgeneratedtokenidsthatshouldbedecodedusingatokenizer

..code::ipython3

fromoptimum.intel.openvinoimportOVModelForCausalLM
fromtransformersimportAutoConfig

ov_model=OVModelForCausalLM.from_pretrained(
stateful_model_path,
config=AutoConfig.from_pretrained(stateful_model_path,trust_remote_code=True),
device=device.value,
)

Interactivechatbotdemo
------------------------

`backtotop⬆️<#table-of-contents>`__

|Now,ourmodelreadytouse.Let’sseeitinaction.Wewilluse
Gradiointerfaceforinteractionwithmodel.Puttextmessageinto
``Chatmessagebox``andclick``Submit``buttonforstarting
conversation.Thereareseveralparametersthatcancontroltext
generationquality:\*``Temperature``isaparameterusedtocontrol
thelevelofcreativityinAI-generatedtext.Byadjustingthe
``temperature``,youcaninfluencetheAImodel’sprobability
distribution,makingthetextmorefocusedordiverse.
|Considerthefollowingexample:TheAImodelhastocompletethe
sentence“Thecatis\____.”withthefollowingtokenprobabilities:

::

playing:0.5
sleeping:0.25
eating:0.15
driving:0.05
flying:0.05

-**Lowtemperature**(e.g.,0.2):TheAImodelbecomesmorefocusedanddeterministic,choosingtokenswiththehighestprobability,suchas"playing."
-**Mediumtemperature**(e.g.,1.0):TheAImodelmaintainsabalancebetweencreativityandfocus,selectingtokensbasedontheirprobabilitieswithoutsignificantbias,suchas"playing,""sleeping,"or"eating."
-**Hightemperature**(e.g.,2.0):TheAImodelbecomesmoreadventurous,increasingthechancesofselectinglesslikelytokens,suchas"driving"and"flying."

-``Top-p``,alsoknownasnucleussampling,isaparameterusedto
controltherangeoftokensconsideredbytheAImodelbasedontheir
cumulativeprobability.Byadjustingthe``top-p``value,youcan
influencetheAImodel’stokenselection,makingitmorefocusedor
diverse.Usingthesameexamplewiththecat,considerthefollowing
top_psettings:

-**Lowtop_p**(e.g.,0.5):TheAImodelconsidersonlytokenswith
thehighestcumulativeprobability,suchas“playing.”
-**Mediumtop_p**(e.g.,0.8):TheAImodelconsiderstokenswitha
highercumulativeprobability,suchas“playing,”“sleeping,”and
“eating.”
-**Hightop_p**(e.g.,1.0):TheAImodelconsidersalltokens,
includingthosewithlowerprobabilities,suchas“driving”and
“flying.”

-``Top-k``isananotherpopularsamplingstrategy.Incomparisonwith
Top-P,whichchoosesfromthesmallestpossiblesetofwordswhose
cumulativeprobabilityexceedstheprobabilityP,inTop-KsamplingK
mostlikelynextwordsarefilteredandtheprobabilitymassis
redistributedamongonlythoseKnextwords.Inourexamplewithcat,
ifk=3,thenonly“playing”,“sleeping”and“eating”willbetaken
intoaccountaspossiblenextword.
-``RepetitionPenalty``Thisparametercanhelppenalizetokensbased
onhowfrequentlytheyoccurinthetext,includingtheinputprompt.
Atokenthathasalreadyappearedfivetimesispenalizedmore
heavilythanatokenthathasappearedonlyonetime.Avalueof1
meansthatthereisnopenaltyandvalueslargerthan1discourage
repeatedtokens.

Youcanmodifythemin``Advancedgenerationoptions``section.

..code::ipython3

importtorch
fromthreadingimportEvent,Thread
fromuuidimportuuid4
fromtypingimportList,Tuple
importgradioasgr
fromtransformersimport(
AutoTokenizer,
StoppingCriteria,
StoppingCriteriaList,
TextIteratorStreamer,
)

model_name="stable-zephyr-3b"

tok=AutoTokenizer.from_pretrained(stateful_model_path)

DEFAULT_SYSTEM_PROMPT="""\
Youareahelpful,respectfulandhonestassistant.Alwaysanswerashelpfullyaspossible,whilebeingsafe.Youranswersshouldnotincludeanyharmful,unethical,racist,sexist,toxic,dangerous,orillegalcontent.Pleaseensurethatyourresponsesaresociallyunbiasedandpositiveinnature.
Ifaquestiondoesnotmakeanysenseorisnotfactuallycoherent,explainwhyinsteadofansweringsomethingnotcorrect.Ifyoudon'tknowtheanswertoaquestion,pleasedon'tsharefalseinformation.\
"""

model_configuration={
"start_message":f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|endoftext|>",
"history_template":"<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}<|endoftext|>",
"current_message_template":"<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}",
}
history_template=model_configuration["history_template"]
current_message_template=model_configuration["current_message_template"]
start_message=model_configuration["start_message"]
stop_tokens=model_configuration.get("stop_tokens")
tokenizer_kwargs=model_configuration.get("tokenizer_kwargs",{})

examples=[
["Hellothere!Howareyoudoing?"],
["WhatisOpenVINO?"],
["Whoareyou?"],
["CanyouexplaintomebrieflywhatisPythonprogramminglanguage?"],
["ExplaintheplotofCinderellainasentence."],
["Whataresomecommonmistakestoavoidwhenwritingcode?"],
["Writea100-wordblogposton“BenefitsofArtificialIntelligenceandOpenVINO“"],
]

max_new_tokens=256


classStopOnTokens(StoppingCriteria):
def__init__(self,token_ids):
self.token_ids=token_ids

def__call__(self,input_ids:torch.LongTensor,scores:torch.FloatTensor,**kwargs)->bool:
forstop_idinself.token_ids:
ifinput_ids[0][-1]==stop_id:
returnTrue
returnFalse


ifstop_tokensisnotNone:
ifisinstance(stop_tokens[0],str):
stop_tokens=tok.convert_tokens_to_ids(stop_tokens)

stop_tokens=[StopOnTokens(stop_tokens)]


defdefault_partial_text_processor(partial_text:str,new_text:str):
"""
helperforupdatingpartiallygeneratedanswer,usedbyde

Params:
partial_text:textbufferforstoringprevioslygeneratedtext
new_text:textupdateforthecurrentstep
Returns:
updatedtextstring

"""
partial_text+=new_text
returnpartial_text


text_processor=model_configuration.get("partial_text_processor",default_partial_text_processor)


defconvert_history_to_text(history:List[Tuple[str,str]]):
"""
functionforconversionhistorystoredaslistpairsofuserandassistantmessagestostringaccordingtomodelexpectedconversationtemplate
Params:
history:dialoguehistory
Returns:
historyintextformat
"""
text=start_message+"".join(["".join([history_template.format(num=round,user=item[0],assistant=item[1])])forround,iteminenumerate(history[:-1])])
text+="".join(
[
"".join(
[
current_message_template.format(
num=len(history)+1,
user=history[-1][0],
assistant=history[-1][1],
)
]
)
]
)
returntext


defuser(message,history):
"""
callbackfunctionforupdatingusermessagesininterfaceonsubmitbuttonclick

Params:
message:currentmessage
history:conversationhistory
Returns:
None
"""
#Appendtheuser'smessagetotheconversationhistory
return"",history+[[message,""]]


defbot(history,temperature,top_p,top_k,repetition_penalty,conversation_id):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory
temperature:parameterforcontrolthelevelofcreativityinAI-generatedtext.
Byadjustingthe`temperature`,youcaninfluencetheAImodel'sprobabilitydistribution,makingthetextmorefocusedordiverse.
top_p:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability.
top_k:parameterforcontroltherangeoftokensconsideredbytheAImodelbasedontheircumulativeprobability,selectingnumberoftokenswithhighestprobability.
repetition_penalty:parameterforpenalizingtokensbasedonhowfrequentlytheyoccurinthetext.
conversation_id:uniqueconversationidentifier.

"""

#Constructtheinputmessagestringforthemodelbyconcatenatingthecurrentsystemmessageandconversationhistory
messages=convert_history_to_text(history)

#Tokenizethemessagesstring
input_ids=tok(messages,return_tensors="pt",**tokenizer_kwargs).input_ids
ifinput_ids.shape[1]>2000:
history=[history[-1]]
messages=convert_history_to_text(history)
input_ids=tok(messages,return_tensors="pt",**tokenizer_kwargs).input_ids
streamer=TextIteratorStreamer(tok,timeout=30.0,skip_prompt=True,skip_special_tokens=True)
generate_kwargs=dict(
input_ids=input_ids,
max_new_tokens=max_new_tokens,
temperature=temperature,
do_sample=temperature>0.0,
top_p=top_p,
top_k=top_k,
repetition_penalty=repetition_penalty,
streamer=streamer,
)
ifstop_tokensisnotNone:
generate_kwargs["stopping_criteria"]=StoppingCriteriaList(stop_tokens)

stream_complete=Event()

defgenerate_and_signal_complete():
"""
genrationfunctionforsinglethread
"""
globalstart_time
ov_model.generate(**generate_kwargs)
stream_complete.set()

t1=Thread(target=generate_and_signal_complete)
t1.start()

#Initializeanemptystringtostorethegeneratedtext
partial_text=""
fornew_textinstreamer:
partial_text=text_processor(partial_text,new_text)
history[-1][1]=partial_text
yieldhistory


defget_uuid():
"""
universaluniqueidentifierforthread
"""
returnstr(uuid4())


withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asdemo:
conversation_id=gr.State(get_uuid)
gr.Markdown(f"""<h1><center>OpenVINO{model_name}Chatbot</center></h1>""")
chatbot=gr.Chatbot(height=500)
withgr.Row():
withgr.Column():
msg=gr.Textbox(
label="ChatMessageBox",
placeholder="ChatMessageBox",
show_label=False,
container=False,
)
withgr.Column():
withgr.Row():
submit=gr.Button("Submit")
stop=gr.Button("Stop")
clear=gr.Button("Clear")
withgr.Row():
withgr.Accordion("AdvancedOptions:",open=False):
withgr.Row():
withgr.Column():
withgr.Row():
temperature=gr.Slider(
label="Temperature",
value=0.1,
minimum=0.0,
maximum=1.0,
step=0.1,
interactive=True,
info="Highervaluesproducemorediverseoutputs",
)
withgr.Column():
withgr.Row():
top_p=gr.Slider(
label="Top-p(nucleussampling)",
value=1.0,
minimum=0.0,
maximum=1,
step=0.01,
interactive=True,
info=(
"Samplefromthesmallestpossiblesetoftokenswhosecumulativeprobability"
"exceedstop_p.Setto1todisableandsamplefromalltokens."
),
)
withgr.Column():
withgr.Row():
top_k=gr.Slider(
label="Top-k",
value=50,
minimum=0.0,
maximum=200,
step=1,
interactive=True,
info="Samplefromashortlistoftop-ktokens—0todisableandsamplefromalltokens.",
)
withgr.Column():
withgr.Row():
repetition_penalty=gr.Slider(
label="RepetitionPenalty",
value=1.1,
minimum=1.0,
maximum=2.0,
step=0.1,
interactive=True,
info="Penalizerepetition—1.0todisable.",
)
gr.Examples(examples,inputs=msg,label="Clickonanyexampleandpressthe'Submit'button")

submit_event=msg.submit(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
temperature,
top_p,
top_k,
repetition_penalty,
conversation_id,
],
outputs=chatbot,
queue=True,
)
submit_click_event=submit.click(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
temperature,
top_p,
top_k,
repetition_penalty,
conversation_id,
],
outputs=chatbot,
queue=True,
)
stop.click(
fn=None,
inputs=None,
outputs=None,
cancels=[submit_event,submit_click_event],
queue=False,
)
clear.click(lambda:None,None,chatbot,queue=False)

demo.queue(max_size=2)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#ifyouhaveanyissuetolaunchonyourplatform,youcanpassshare=Truetolaunchmethod:
#demo.launch(share=True)
#itcreatesapubliclyshareablelinkfortheinterface.Readmoreinthedocs:https://gradio.app/docs/
demo.launch(share=True)

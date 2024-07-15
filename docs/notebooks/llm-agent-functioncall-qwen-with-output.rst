CreateFunction-callingAgentusingOpenVINOandQwen-Agent
===========================================================

LLMarelimitedtotheknowledgeonwhichtheyhavebeentrainedandthe
additionalknowledgeprovidedascontext,asaresult,ifausefulpiece
ofinformationismissingtheprovidedknowledge,themodelcannot“go
around”andtrytofinditinothersources.Thisisthereasonwhywe
needtointroducetheconceptofAgents.

Thecoreideaofagentsistousealanguagemodeltochooseasequence
ofactionstotake.Inagents,alanguagemodelisusedasareasoning
enginetodeterminewhichactionstotakeandinwhichorder.Agentscan
beseenasapplicationspoweredbyLLMsandintegratedwithasetof
toolslikesearchengines,databases,websites,andsoon.Withinan
agent,theLLMisthereasoningenginethat,basedontheuserinput,is
abletoplanandexecuteasetofactionsthatareneededtofulfillthe
request.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/22fa5396-8381-400f-a78f-97e25d57d807
:alt:agent

agent

`Qwen-Agent<https://github.com/QwenLM/Qwen-Agent>`__isaframeworkfor
developingLLMapplicationsbasedontheinstructionfollowing,tool
usage,planning,andmemorycapabilitiesofQwen.Italsocomeswith
exampleapplicationssuchasBrowserAssistant,CodeInterpreter,and
CustomAssistant.

ThisnotebookexploreshowtocreateaFunctioncallingAgentstepby
stepusingOpenVINOandQwen-Agent.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`CreateaFunctioncalling
agent<#create-a-function-calling-agent>`__

-`Createfunctions<#create-functions>`__
-`Downloadmodel<#download-model>`__
-`Selectinferencedevicefor
LLM<#select-inference-device-for-llm>`__
-`CreateLLMforQwen-Agent<#create-llm-for-qwen-agent>`__
-`CreateFunction-calling
pipeline<#create-function-calling-pipeline>`__

-`InteractiveDemo<#interactive-demo>`__

-`Createtools<#create-tools>`__
-`CreateAIagentdemowithQwen-AgentandGradio
UI<#create-ai-agent-demo-with-qwen-agent-and-gradio-ui>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipinstall-Uqpip
%pipuninstall-q-yoptimumoptimum-intel
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"git+https://github.com/huggingface/optimum-intel.git"\
"git+https://github.com/openvinotoolkit/nncf.git"\
"torch>=2.1"\
"datasets"\
"accelerate"\
"qwen-agent>=0.0.6""transformers>=4.38.1""gradio==4.21.0","modelscope-studio>=0.4.0""langchain>=0.2.3""langchain-community>=0.2.4""wikipedia"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
WARNING:typer0.12.3doesnotprovidetheextra'all'
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importrequests
fromPILimportImage

openvino_logo="openvino_log.png"
url="https://cdn-avatars.huggingface.co/v1/production/uploads/1671615670447-6346651be2dcb5422bcd13dd.png"

image=Image.open(requests.get(url,stream=True).raw)
image.save(openvino_logo)

CreateaFunctioncallingagent
-------------------------------

`backtotop⬆️<#table-of-contents>`__

Functioncallingallowsamodeltodetectwhenoneormoretoolsshould
becalledandrespondwiththeinputsthatshouldbepassedtothose
tools.InanAPIcall,youcandescribetoolsandhavethemodel
intelligentlychoosetooutputastructuredobjectlikeJSONcontaining
argumentstocallthesetools.ThegoaloftoolsAPIsistomore
reliablyreturnvalidandusefultoolcallsthanwhatcanbedoneusing
agenerictextcompletionorchatAPI.

Wecantakeadvantageofthisstructuredoutput,combinedwiththefact
thatyoucanbindmultipletoolstoatoolcallingchatmodelandallow
themodeltochoosewhichonetocall,tocreateanagentthat
repeatedlycallstoolsandreceivesresultsuntilaqueryisresolved.

Createafunction
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,weneedtocreateaexamplefunction/toolforgettingthe
informationofcurrentweather.

..code::ipython3

importjson


defget_current_weather(location,unit="fahrenheit"):
"""Getthecurrentweatherinagivenlocation"""
if"tokyo"inlocation.lower():
returnjson.dumps({"location":"Tokyo","temperature":"10","unit":"celsius"})
elif"sanfrancisco"inlocation.lower():
returnjson.dumps({"location":"SanFrancisco","temperature":"72","unit":"fahrenheit"})
elif"paris"inlocation.lower():
returnjson.dumps({"location":"Paris","temperature":"22","unit":"celsius"})
else:
returnjson.dumps({"location":location,"temperature":"unknown"})

Wrapthefunction’snameanddescriptionintoajsonlist,anditwill
helpLLMtofindoutwhichfunctionshouldbecalledforcurrenttask.

..code::ipython3

functions=[
{
"name":"get_current_weather",
"description":"Getthecurrentweatherinagivenlocation",
"parameters":{
"type":"object",
"properties":{
"location":{
"type":"string",
"description":"Thecityandstate,e.g.SanFrancisco,CA",
},
"unit":{"type":"string","enum":["celsius","fahrenheit"]},
},
"required":["location"],
},
}
]

Downloadmodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LargeLanguageModels(LLMs)areacorecomponentofAgent.Inthis
example,wewilldemonstratehowtocreateaOpenVINOLLMmodelin
Qwen-Agentframework.SinceQwen2cansupportfunctioncallingduring
textgeneration,weselect``Qwen/Qwen2-7B-Instruct``asLLMinagent
pipeline.

-**Qwen/Qwen2-7B-Instruct**-Qwen2isthenewseriesofQwenlarge
languagemodels.Comparedwiththestate-of-the-artopensource
languagemodels,includingthepreviousreleasedQwen1.5,Qwen2has
generallysurpassedmostopensourcemodelsanddemonstrated
competitivenessagainstproprietarymodelsacrossaseriesof
benchmarkstargetingforlanguageunderstanding,languagegeneration,
multilingualcapability,coding,mathematics,reasoning,etc.`Model
Card<https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__

TorunLLMlocally,wehavetodownloadthemodelinthefirststep.It
ispossibleto`exportyour
model<https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
totheOpenVINOIRformatwiththeCLI,andloadthemodelfromlocal
folder.

..code::ipython3

frompathlibimportPath

model_id="Qwen/Qwen2-7B-Instruct"
model_path="Qwen2-7B-Instruct-ov"

ifnotPath(model_path).exists():
!optimum-cliexportopenvino--model{model_id}--tasktext-generation-with-past--trust-remote-code--weight-formatint4--ratio0.72{model_path}

SelectinferencedeviceforLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov
importipywidgetsaswidgets

core=ov.Core()

support_devices=core.available_devices
if"NPU"insupport_devices:
support_devices.remove("NPU")

device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



CreateLLMforQwen-Agent
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINOhasbeenintegratedintothe``Qwen-Agent``framework.Youcan
usefollowingmethodtocreateaOpenVINObasedLLMfora``Qwen-Agent``
pipeline.

..code::ipython3

fromqwen_agent.llmimportget_chat_model

ov_config={"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":""}
llm_cfg={
"ov_model_dir":model_path,
"model_type":"openvino",
"device":device.value,
"ov_config":ov_config,
#(Optional)LLMhyperparametersforgeneration:
"generate_cfg":{"top_p":0.8},
}
llm=get_chat_model(llm_cfg)


..parsed-literal::

CompilingthemodeltoCPU...
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


Youcangetadditionalinferencespeedimprovementwith`Dynamic
QuantizationofactivationsandKV-cachequantizationon
CPU<https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html#enabling-openvino-runtime-optimizations>`__.
Theseoptionscanbeenabledwith``ov_config``asfollows:

..code::ipython3

ov_config={
"KV_CACHE_PRECISION":"u8",
"DYNAMIC_QUANTIZATION_GROUP_SIZE":"32",
"PERFORMANCE_HINT":"LATENCY",
"NUM_STREAMS":"1",
"CACHE_DIR":"",
}

CreateFunction-callingpipeline
--------------------------------

`backtotop⬆️<#table-of-contents>`__

AfterdefiningthefunctionsandLLM,wecanbuildtheagentpipeline
withcapabilityoffunctioncalling.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/3170ca30-23af-4a1a-a655-1d0d67df2ded
:alt:functioncalling

functioncalling

TheworkflowofQwen2functioncallingconsistsofseveralsteps:

1.Role``user``sendingtherequest.
2.Checkifthemodelwantedtocallafunction,andcallthefunction
ifneeded
3.Gettheobservationfrom``function``\’sresults.
4.Consolidatetheobservationintofinalresponseof``assistant``.

Atypicalmulti-turndialoguestructureisasfollows:

-**Query**:
``{'role':'user','content':'createapictureofcutecat'},``

-**Functioncalling**:
``{'role':'assistant','content':'','function_call':{'name':'my_image_gen','arguments':'{"prompt":"acutecat"}'}},``

-**Observation**:
``{'role':'function','content':'{"image_url":"https://image.pollinations.ai/prompt/a%20cute%20cat"}','name':'my_image_gen'}``

-**FinalResponse**:
``{'role':'assistant','content':"Hereistheimageofacutecatbasedonyourdescription:\n\n![](https://image.pollinations.ai/prompt/a%20cute%20cat)."}``

..code::ipython3

print("#Userquestion:")
messages=[{"role":"user","content":"What'stheweatherlikeinSanFrancisco?"}]
print(messages)

print("#AssistantResponse1:")
responses=[]

#Step1:Role`user`sendingtherequest
responses=llm.chat(
messages=messages,
functions=functions,
stream=False,
)
print(responses)

messages.extend(responses)

#Step2:checkifthemodelwantedtocallafunction,andcallthefunctionifneeded
last_response=messages[-1]
iflast_response.get("function_call",None):
available_functions={
"get_current_weather":get_current_weather,
}#onlyonefunctioninthisexample,butyoucanhavemultiple
function_name=last_response["function_call"]["name"]
function_to_call=available_functions[function_name]
function_args=json.loads(last_response["function_call"]["arguments"])
function_response=function_to_call(
location=function_args.get("location"),
)
print("#FunctionResponse:")
print(function_response)

#Step3:Gettheobservationfrom`function`'sresults
messages.append(
{
"role":"function",
"name":function_name,
"content":function_response,
}
)

print("#AssistantResponse2:")
#Step4:Consolidatetheobservationfromfunctionintofinalresponse
responses=llm.chat(
messages=messages,
functions=functions,
stream=False,
)
print(responses)


..parsed-literal::

#Userquestion:
[{'role':'user','content':"What'stheweatherlikeinSanFrancisco?"}]
#AssistantResponse1:
[{'role':'assistant','content':'','function_call':{'name':'get_current_weather','arguments':'{"location":"SanFrancisco,CA"}'}}]
#FunctionResponse:
{"location":"SanFrancisco","temperature":"72","unit":"fahrenheit"}
#AssistantResponse2:
[{'role':'assistant','content':'ThecurrentweatherinSanFranciscois72degreesFahrenheit.'}]


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Let’screateainteractiveagentusing
`Gradio<https://www.gradio.app/>`__.

Createtools
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Qwen-Agentprovidesamechanismfor`registering
tools<https://github.com/QwenLM/Qwen-Agent/blob/main/docs/tool.md>`__.
Forexample,toregisteryourownimagegenerationtool:

-Specifythetool’sname,description,andparameters.Notethatthe
stringpassedto``@register_tool('my_image_gen')``isautomatically
addedasthe``.name``attributeoftheclassandwillserveasthe
uniqueidentifierforthetool.
-Implementthe``call(...)``function.

Inthisnotebook,wewillcreate3toolsasexamples:-
**image_generation**:AIpainting(imagegeneration)service,inputtext
description,andreturntheimageURLdrawnbasedontextinformation.-
**get_current_weather**:Getthecurrentweatherinagivencityname.-
**wikipedia**:AwrapperaroundWikipedia.Usefulforwhenyouneedto
answergeneralquestionsaboutpeople,places,companies,facts,
historicalevents,orothersubjects.

..code::ipython3

importurllib.parse
importjson5
importrequests
fromqwen_agent.tools.baseimportBaseTool,register_tool


@register_tool("image_generation")
classImageGeneration(BaseTool):
description="AIpainting(imagegeneration)service,inputtextdescription,andreturntheimageURLdrawnbasedontextinformation."
parameters=[{"name":"prompt","type":"string","description":"Detaileddescriptionofthedesiredimagecontent,inEnglish","required":True}]

defcall(self,params:str,**kwargs)->str:
prompt=json5.loads(params)["prompt"]
prompt=urllib.parse.quote(prompt)
returnjson5.dumps({"image_url":f"https://image.pollinations.ai/prompt/{prompt}"},ensure_ascii=False)


@register_tool("get_current_weather")
classGetCurrentWeather(BaseTool):
description="Getthecurrentweatherinagivencityname."
parameters=[{"name":"city_name","type":"string","description":"Thecityandstate,e.g.SanFrancisco,CA","required":True}]

defcall(self,params:str,**kwargs)->str:
#`params`aretheargumentsgeneratedbytheLLMagent.
city_name=json5.loads(params)["city_name"]
key_selection={
"current_condition":[
"temp_C",
"FeelsLikeC",
"humidity",
"weatherDesc",
"observation_time",
],
}
resp=requests.get(f"https://wttr.in/{city_name}?format=j1")
resp.raise_for_status()
resp=resp.json()
ret={k:{_v:resp[k][0][_v]for_vinv}fork,vinkey_selection.items()}
returnstr(ret)


@register_tool("wikipedia")
classWikipedia(BaseTool):
description="AwrapperaroundWikipedia.Usefulforwhenyouneedtoanswergeneralquestionsaboutpeople,places,companies,facts,historicalevents,orothersubjects."
parameters=[{"name":"query","type":"string","description":"Querytolookuponwikipedia","required":True}]

defcall(self,params:str,**kwargs)->str:
#`params`aretheargumentsgeneratedbytheLLMagent.
fromlangchain.toolsimportWikipediaQueryRun
fromlangchain_community.utilitiesimportWikipediaAPIWrapper

query=json5.loads(params)["query"]
wikipedia=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=1000))
resutlt=wikipedia.run(query)
returnstr(resutlt)

..code::ipython3

tools=["image_generation","get_current_weather","wikipedia"]

CreateAIagentdemowithQwen-AgentandGradioUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheAgentclassservesasahigher-levelinterfaceforQwen-Agent,where
anAgentobjectintegratestheinterfacesfortoolcallsandLLM(Large
LanguageModel).TheAgentreceivesalistofmessagesasinputand
producesageneratorthatyieldsalistofmessages,effectively
providingastreamofoutputmessages.

Qwen-AgentoffersagenericAgentclass:the``Assistant``class,which,
whendirectlyinstantiated,canhandlethemajorityofSingle-Agent
tasks.Features:

-Itsupportsrole-playing.
-Itprovidesautomaticplanningandtoolcallsabilities.
-RAG(Retrieval-AugmentedGeneration):Itacceptsdocumentsinput,and
canuseanintegratedRAGstrategytoparsethedocuments.

..code::ipython3

fromqwen_agent.agentsimportAssistant
fromqwen_agent.guiimportWebUI

bot=Assistant(llm=llm_cfg,function_list=tools,name="OpenVINOAgent")


..parsed-literal::

CompilingthemodeltoCPU...
Specialtokenshavebeenaddedinthevocabulary,makesuretheassociatedwordembeddingsarefine-tunedortrained.


..code::ipython3

fromtypingimportList
fromqwen_agent.llm.schemaimportCONTENT,ROLE,USER,Message
fromqwen_agent.gui.utilsimportconvert_history_to_chatbot
fromqwen_agent.gui.gradioimportgr,mgr


classOpenVINOUI(WebUI):
defrequest_cancel(self):
self.agent_list[0].llm.ov_model.request.cancel()

defclear_history(self):
return[]

defadd_text(self,_input,_chatbot,_history):
_history.append(
{
ROLE:USER,
CONTENT:[{"text":_input}],
}
)
_chatbot.append([_input,None])
yieldgr.update(interactive=False,value=None),_chatbot,_history

defrun(
self,
messages:List[Message]=None,
share:bool=False,
server_name:str=None,
server_port:int=None,
**kwargs,
):
self.run_kwargs=kwargs

withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asself.demo:
gr.Markdown("""<h1><center>OpenVINOQwenAgent</center></h1>""")
history=gr.State([])

withgr.Row():
withgr.Column(scale=4):
chatbot=mgr.Chatbot(
value=convert_history_to_chatbot(messages=messages),
avatar_images=[
self.user_config,
self.agent_config_list,
],
height=900,
avatar_image_width=80,
flushing=False,
show_copy_button=True,
)
withgr.Column():
input=gr.Textbox(
label="ChatMessageBox",
placeholder="ChatMessageBox",
show_label=False,
container=False,
)
withgr.Column():
withgr.Row():
submit=gr.Button("Submit",variant="primary")
stop=gr.Button("Stop")
clear=gr.Button("Clear")
withgr.Column(scale=1):
agent_interactive=self.agent_list[0]
capabilities=[keyforkeyinagent_interactive.function_map.keys()]
gr.CheckboxGroup(
label="Tools",
value=capabilities,
choices=capabilities,
interactive=False,
)
withgr.Row():
gr.Examples(self.prompt_suggestions,inputs=[input],label="Clickonanyexampleandpressthe'Submit'button")

input_promise=submit.click(
fn=self.add_text,
inputs=[input,chatbot,history],
outputs=[input,chatbot,history],
queue=False,
)
input_promise=input_promise.then(
self.agent_run,
[chatbot,history],
[chatbot,history],
)
input_promise.then(self.flushed,None,[input])
stop.click(
fn=self.request_cancel,
inputs=None,
outputs=None,
cancels=[input_promise],
queue=False,
)
clear.click(lambda:None,None,chatbot,queue=False).then(self.clear_history,None,history)

self.demo.load(None)

self.demo.launch(share=share,server_name=server_name,server_port=server_port)


chatbot_config={
"prompt.suggestions":[
"BasedoncurrentweatherinLondon,showmeapictureofBigBen",
"WhatisOpenVINO?",
"Createanimageofpinkcat",
"WhatistheweatherlikeinNewYorknow?",
"HowmanypeopleliveinCanada?",
],
"agent.avatar":openvino_logo,
"input.placeholder":"Pleaseinputyourrequesthere",
}

demo=OpenVINOUI(
bot,
chatbot_config=chatbot_config,
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.run(server_name='yourservername',server_port='serverportinint')
try:
demo.run()
exceptException:
demo.run(share=True)

..code::ipython3

#demo.demo.close()

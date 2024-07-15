Cross-lingualBooksAlignmentwithTransformersandOpenVINO™
=============================================================

Cross-lingualtextalignmentisthetaskofmatchingsentencesinapair
oftextsthataretranslationsofeachother.Inthisnotebook,you’ll
learnhowtouseadeeplearningmodeltocreateaparallelbookin
EnglishandGerman

Thismethodhelpsyoulearnlanguagesbutalsoprovidesparalleltexts
thatcanbeusedtotrainmachinetranslationmodels.Thisis
particularlyusefulifoneofthelanguagesislow-resourceoryoudon’t
haveenoughdatatotrainafull-fledgedtranslationmodel.

Thenotebookshowshowtoacceleratethemostcomputationallyexpensive
partofthepipeline-gettingvectorsfromsentences-usingthe
OpenVINO™framework.

Pipeline
--------

Thenotebookguidesyouthroughtheentireprocessofcreatinga
parallelbook:fromobtainingrawtextstobuildingavisualizationof
alignedsentences.Hereisthepipelinediagram:

|image0|

Visualizingtheresultallowsyoutoidentifyareasforimprovementin
thepipelinesteps,asindicatedinthediagram.

Prerequisites
-------------

-``requests``-forgettingbooks
-``pysbd``-forsplittingsentences
-``transformers[torch]``and``openvino_dev``-forgettingsentence
embeddings
-``seaborn``-foralignmentmatrixvisualization
-``ipywidgets``-fordisplayingHTMLandJSoutputinthenotebook

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`GetBooks<#get-books>`__
-`CleanText<#clean-text>`__
-`SplitText<#split-text>`__
-`GetSentenceEmbeddings<#get-sentence-embeddings>`__

-`OptimizetheModelwith
OpenVINO<#optimize-the-model-with-openvino>`__

-`CalculateSentenceAlignment<#calculate-sentence-alignment>`__
-`PostprocessSentenceAlignment<#postprocess-sentence-alignment>`__
-`VisualizeSentenceAlignment<#visualize-sentence-alignment>`__
-`SpeedupEmbeddings
Computation<#speed-up-embeddings-computation>`__

..|image0|image::https://user-images.githubusercontent.com/51917466/254582697-18f3ab38-e264-4b2c-a088-8e54b855c1b2.png

..code::ipython3

importplatform

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpurequestspysbdtransformers"torch>=2.1""openvino>=2023.1.0"seabornipywidgets

GetBooks
---------

`backtotop⬆️<#table-of-contents>`__

Thefirststepistogetthebooksthatwewillbeworkingwith.For
thisnotebook,wewilluseEnglishandGermanversionsofAnnaKarenina
byLeoTolstoy.Thetextscanbeobtainedfromthe`ProjectGutenberg
site<https://www.gutenberg.org/>`__.Sincecopyrightlawsarecomplex
anddifferfromcountrytocountry,checkthebook’slegalavailability
inyourcountry.RefertotheProjectGutenbergPermissions,Licensing
andotherCommonRequests
`page<https://www.gutenberg.org/policy/permission.html>`__formore
information.

FindthebooksonProjectGutenberg`search
page<https://www.gutenberg.org/ebooks/>`__andgettheIDofeachbook.
Togetthetexts,wewillpasstheIDstothe
`Gutendex<http://gutendex.com/>`__API.

..code::ipython3

importrequests


defget_book_by_id(book_id:int,gutendex_url:str="https://gutendex.com/")->str:
book_metadata_url=gutendex_url+"/books/"+str(book_id)
request=requests.get(book_metadata_url,timeout=30)
request.raise_for_status()

book_metadata=request.json()
text_format_key="text/plain"
text_plain=[kforkinbook_metadata["formats"]ifk.startswith(text_format_key)]
book_url=book_metadata["formats"][text_plain[0]]
returnrequests.get(book_url).text


en_book_id=1399
de_book_id=44956

anna_karenina_en=get_book_by_id(en_book_id)
anna_karenina_de=get_book_by_id(de_book_id)

Let’scheckthatwegottherightbooksbyshowingapartofthetexts:

..code::ipython3

print(anna_karenina_en[:1500])


..parsed-literal::

TheProjectGutenbergeBookofAnnaKarenina

ThisebookisfortheuseofanyoneanywhereintheUnitedStatesand
mostotherpartsoftheworldatnocostandwithalmostnorestrictions
whatsoever.Youmaycopyit,giveitawayorre-useitundertheterms
oftheProjectGutenbergLicenseincludedwiththisebookoronline
atwww.gutenberg.org.IfyouarenotlocatedintheUnitedStates,
youwillhavetocheckthelawsofthecountrywhereyouarelocated
beforeusingthiseBook.

Title:AnnaKarenina


Author:grafLeoTolstoy

Translator:ConstanceGarnett

Releasedate:July1,1998[eBook#1399]
Mostrecentlyupdated:April9,2023

Language:English



***STARTOFTHEPROJECTGUTENBERGEBOOKANNAKARENINA***
[Illustration]




ANNAKARENINA

byLeoTolstoy

TranslatedbyConstanceGarnett

Contents


PARTONE
PARTTWO
PARTTHREE
PARTFOUR
PARTFIVE
PARTSIX
PARTSEVEN
PARTEIGHT




PARTONE

Chapter1


Happyfamiliesareallalike;everyunhappyfamilyisunhappyinits
ownway.

EverythingwasinconfusionintheOblonskys’house.Thewifehad
discoveredthatthehusbandwascarryingonanintriguewithaFrench
girl,whohadbeenagovernessintheirfamily,andshehadannounced
toherhusbandthatshecouldnotgoonlivinginthesamehousewith
him.Thispositionofaffairshadnowlastedthreedays,andnotonly
thehusbandandwifethemselves,butalltheme


whichinarawformatlookslikethis:

..code::ipython3

anna_karenina_en[:1500]




..parsed-literal::

'\ufeffTheProjectGutenbergeBookofAnnaKarenina\r\n\r\nThisebookisfortheuseofanyoneanywhereintheUnitedStatesand\r\nmostotherpartsoftheworldatnocostandwithalmostnorestrictions\r\nwhatsoever.Youmaycopyit,giveitawayorre-useitundertheterms\r\noftheProjectGutenbergLicenseincludedwiththisebookoronline\r\natwww.gutenberg.org.IfyouarenotlocatedintheUnitedStates,\r\nyouwillhavetocheckthelawsofthecountrywhereyouarelocated\r\nbeforeusingthiseBook.\r\n\r\nTitle:AnnaKarenina\r\n\r\n\r\nAuthor:grafLeoTolstoy\r\n\r\nTranslator:ConstanceGarnett\r\n\r\nReleasedate:July1,1998[eBook#1399]\r\nMostrecentlyupdated:April9,2023\r\n\r\nLanguage:English\r\n\r\n\r\n\r\n\*\*\*STARTOFTHEPROJECTGUTENBERGEBOOKANNAKARENINA\*\*\*\r\n[Illustration]\r\n\r\n\r\n\r\n\r\nANNAKARENINA\r\n\r\nbyLeoTolstoy\r\n\r\nTranslatedbyConstanceGarnett\r\n\r\nContents\r\n\r\n\r\nPARTONE\r\nPARTTWO\r\nPARTTHREE\r\nPARTFOUR\r\nPARTFIVE\r\nPARTSIX\r\nPARTSEVEN\r\nPARTEIGHT\r\n\r\n\r\n\r\n\r\nPARTONE\r\n\r\nChapter1\r\n\r\n\r\nHappyfamiliesareallalike;everyunhappyfamilyisunhappyinits\r\nownway.\r\n\r\nEverythingwasinconfusionintheOblonskys’house.Thewifehad\r\ndiscoveredthatthehusbandwascarryingonanintriguewithaFrench\r\ngirl,whohadbeenagovernessintheirfamily,andshehadannounced\r\ntoherhusbandthatshecouldnotgoonlivinginthesamehousewith\r\nhim.Thispositionofaffairshadnowlastedthreedays,andnotonly\r\nthehusbandandwifethemselves,butalltheme'



..code::ipython3

anna_karenina_de[:1500]




..parsed-literal::

'TheProjectGutenbergEBookofAnnaKarenina,1.Band,byLeoN.Tolstoi\r\n\r\nThiseBookisfortheuseofanyoneanywhereatnocostandwith\r\nalmostnorestrictionswhatsoever.Youmaycopyit,giveitawayor\r\nre-useitunderthetermsoftheProjectGutenbergLicenseincluded\r\nwiththiseBookoronlineatwww.gutenberg.org\r\n\r\n\r\nTitle:AnnaKarenina,1.Band\r\n\r\nAuthor:LeoN.Tolstoi\r\n\r\nReleaseDate:February18,2014[EBook#44956]\r\n\r\nLanguage:German\r\n\r\nCharactersetencoding:ISO-8859-1\r\n\r\n\*\*\*STARTOFTHISPROJECTGUTENBERGEBOOKANNAKARENINA,1.BAND\*\*\*\r\n\r\n\r\n\r\n\r\nProducedbyNorbertH.Langkau,JensNordmannandthe\r\nOnlineDistributedProofreadingTeamathttp://www.pgdp.net\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\nAnnaKarenina.\r\n\r\n\r\nRomanausdemRussischen\r\n\r\ndes\r\n\r\nGrafenLeoN.Tolstoi.\r\n\r\n\r\n\r\nNachdersiebentenAuflageübersetzt\r\n\r\nvon\r\n\r\nHansMoser.\r\n\r\n\r\nErsterBand.\r\n\r\n\r\n\r\nLeipzig\r\n\r\nDruckundVerlagvonPhilippReclamjun.\r\n\r\n*****\r\n\r\n\r\n\r\n\r\nErsterTeil.\r\n\r\n»DieRacheistmein,ichwillvergelten.«\r\n\r\n1.\r\n\r\n\r\nAlleglücklichenFamiliensindeinanderähnlich;jedeunglücklich'



CleanText
----------

`backtotop⬆️<#table-of-contents>`__

Thedownloadedbooksmaycontainserviceinformationbeforeandafter
themaintext.Thetextmighthavedifferentformattingstylesand
markup,forexample,phrasesfromadifferentlanguageenclosedin
underscoresforpotentialemphasisoritalicization:

Yes,Alabinwasgivingadinneronglasstables,andthetablessang,
\*Ilmiotesoro*—not*Ilmiotesoro*\though,butsomethingbetter,
andthereweresomesortoflittledecantersonthetable,andthey
werewomen,too,”heremembered.

Thenextstagesofthepipelinewillbedifficulttocompletewithout
cleaningandnormalizingthetext.Sinceformattingmaydiffer,manual
workisrequiredatthisstage.Forexample,themaincontentinthe
Germanversionisenclosedin``*****``,so
itissafetoremoveeverythingbeforethefirstoccurrenceandafter
thelastoccurrenceoftheseasterisks.

**Hint**:Therearetext-cleaninglibrariesthatcleanupcommon
flaws.Ifthesourceofthetextisknown,youcanlookforalibrary
designedforthatsource,forexample
`gutenberg_cleaner<https://github.com/kiasar/gutenberg_cleaner>`__.
Theselibrariescanreducemanualworkandevenautomatethe
process.process.

..code::ipython3

importre
fromcontextlibimportcontextmanager
fromtqdm.autoimporttqdm


start_pattern_en=r"\nPARTONE"
anna_karenina_en=re.split(start_pattern_en,anna_karenina_en)[1].strip()

end_pattern_en="***ENDOFTHEPROJECTGUTENBERGEBOOKANNAKARENINA***"
anna_karenina_en=anna_karenina_en.split(end_pattern_en)[0].strip()

..code::ipython3

start_pattern_de="*****"
anna_karenina_de=anna_karenina_de.split(start_pattern_de,maxsplit=1)[1].strip()
anna_karenina_de=anna_karenina_de.rsplit(start_pattern_de,maxsplit=1)[0].strip()

..code::ipython3

anna_karenina_en=anna_karenina_en.replace("\r\n","\n")
anna_karenina_de=anna_karenina_de.replace("\r\n","\n")

Forthisnotebook,wewillworkonlywiththefirstchapter.

..code::ipython3

chapter_pattern_en=r"Chapter\d?\d"
chapter_1_en=re.split(chapter_pattern_en,anna_karenina_en)[1].strip()

..code::ipython3

chapter_pattern_de=r"\d?\d.\n\n"
chapter_1_de=re.split(chapter_pattern_de,anna_karenina_de)[1].strip()

Let’scutitoutanddefinesomecleaningfunctions.

..code::ipython3

defremove_single_newline(text:str)->str:
returnre.sub(r"\n(?!\n)","",text)


defunify_quotes(text:str)->str:
returnre.sub(r"['\"»«“”]",'"',text)


defremove_markup(text:str)->str:
text=text.replace(">=","").replace("=<","")
returnre.sub(r"_\w|\w_","",text)

Combinethecleaningfunctionsintoasinglepipeline.The``tqdm``
libraryisusedtotracktheexecutionprogress.Defineacontext
managertooptionallydisabletheprogressindicatorsiftheyarenot
needed.

..code::ipython3

disable_tqdm=False


@contextmanager
defdisable_tqdm_context():
globaldisable_tqdm
disable_tqdm=True
yield
disable_tqdm=False


defclean_text(text:str)->str:
text_cleaning_pipeline=[
remove_single_newline,
unify_quotes,
remove_markup,
]
progress_bar=tqdm(text_cleaning_pipeline,disable=disable_tqdm)
forclean_funcinprogress_bar:
progress_bar.set_postfix_str(clean_func.__name__)
text=clean_func(text)
returntext


chapter_1_en=clean_text(chapter_1_en)
chapter_1_de=clean_text(chapter_1_de)



..parsed-literal::

0%||0/3[00:00<?,?it/s]



..parsed-literal::

0%||0/3[00:00<?,?it/s]


SplitText
----------

`backtotop⬆️<#table-of-contents>`__

Dividingtextintosentencesisachallengingtaskintextprocessing.
Theproblemiscalled`sentenceboundary
disambiguation<https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation>`__,
whichcanbesolvedusingheuristicsormachinelearningmodels.This
notebookusesa``Segmenter``fromthe``pysbd``library,whichis
initializedwithan`ISOlanguage
code<https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`__,asthe
rulesforsplittingtextintosentencesmayvaryfordifferent
languages.

**Hint**:The``book_metadata``obtainedfromtheGutendexcontains
thelanguagecodeaswell,enablingautomationofthispartofthe
pipeline.

..code::ipython3

importpysbd


splitter_en=pysbd.Segmenter(language="en",clean=True)
splitter_de=pysbd.Segmenter(language="de",clean=True)


sentences_en=splitter_en.segment(chapter_1_en)
sentences_de=splitter_de.segment(chapter_1_de)

len(sentences_en),len(sentences_de)




..parsed-literal::

(32,34)



GetSentenceEmbeddings
-----------------------

`backtotop⬆️<#table-of-contents>`__

Thenextstepistotransformsentencesintovectorrepresentations.
Transformerencodermodels,likeBERT,providehigh-qualityembeddings
butcanbeslow.Additionally,themodelshouldsupportbothchosen
languages.Trainingseparatemodelsforeachlanguagepaircanbe
expensive,sotherearemanymodelspre-trainedonmultiplelanguages
simultaneously,forexample:

-`multilingual-MiniLM<https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>`__
-`distiluse-base-multilingual-cased<https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2>`__
-`bert-base-multilingual-uncased<https://huggingface.co/bert-base-multilingual-uncased>`__
-`LaBSE<https://huggingface.co/rasa/LaBSE>`__

LaBSEstandsfor`Language-agnosticBERTSentence
Embedding<https://arxiv.org/pdf/2007.01852.pdf>`__andsupports109+
languages.IthasthesamearchitectureastheBERTmodelbuthasbeen
trainedonadifferenttask:toproduceidenticalembeddingsfor
translationpairs.

|image0|

ThismakesLaBSEagreatchoiceforourtaskanditcanbereusedfor
differentlanguagepairsstillproducinggoodresults.

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/627d3a39-7076-479f-a7b1-392f49a0b83e

..code::ipython3

fromtypingimportList,Union,Dict
fromtransformersimportAutoTokenizer,AutoModel,BertModel
importnumpyasnp
importtorch
fromopenvino.runtimeimportCompiledModelasOVModel
importopenvinoasov


model_id="rasa/LaBSE"
pt_model=AutoModel.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)

Themodelhastwooutputs:``last_hidden_state``and``pooler_output``.
Forgeneratingembeddings,youcanuseeitherthefirstvectorfromthe
``last_hidden_state``,whichcorrespondstothespecial``[CLS]``token,
ortheentirevectorfromthesecondinput.Usually,thesecondoption
isused,butwewillbeusingthefirstoptionasitalsoworkswellfor
ourtask.Fillfreetoexperimentwithdifferentoutputstofindthe
bestfit.

..code::ipython3

defget_embeddings(
sentences:List[str],
embedding_model:Union[BertModel,OVModel],
)->np.ndarray:
ifisinstance(embedding_model,OVModel):
embeddings=[embedding_model(tokenizer(sent,return_tensors="np").data)["last_hidden_state"][0][0]forsentintqdm(sentences,disable=disable_tqdm)]
returnnp.vstack(embeddings)
else:
embeddings=[embedding_model(**tokenizer(sent,return_tensors="pt"))["last_hidden_state"][0][0]forsentintqdm(sentences,disable=disable_tqdm)]
returntorch.vstack(embeddings)


embeddings_en_pt=get_embeddings(sentences_en,pt_model)
embeddings_de_pt=get_embeddings(sentences_de,pt_model)



..parsed-literal::

0%||0/32[00:00<?,?it/s]



..parsed-literal::

0%||0/34[00:00<?,?it/s]


OptimizetheModelwithOpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheLaBSEmodelisquitelargeandcanbeslowtoinferonsome
hardware,solet’soptimizeitwithOpenVINO.`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/conversion-parameters.html>`__
acceptsthePyTorch/Transformersmodelobjectandadditionalinformation
aboutmodelinputs.An``example_input``isneededtotracethemodel
executiongraph,asPyTorchconstructsitdynamicallyduringinference.
Theconvertedmodelmustbecompiledforthetargetdeviceusingthe
``Core``objectbeforeitcanbeused.

Forstartingwork,weshouldselectdeviceforinferencefirst:

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

..code::ipython3

#3inputswithdynamicaxis[batch_size,sequence_length]andtypeint64
inputs_info=[([-1,-1],ov.Type.i64)]*3
ov_model=ov.convert_model(
pt_model,
example_input=tokenizer("test",return_tensors="pt").data,
input=inputs_info,
)

core=ov.Core()
compiled_model=core.compile_model(ov_model,device.value)

embeddings_en=get_embeddings(sentences_en,compiled_model)
embeddings_de=get_embeddings(sentences_de,compiled_model)



..parsed-literal::

0%||0/32[00:00<?,?it/s]



..parsed-literal::

0%||0/34[00:00<?,?it/s]


OnanIntelCorei9-10980XECPU,thePyTorchmodelprocessed40-43
sentencespersecond.AfteroptimizationwithOpenVINO,theprocessing
speedincreasedto56-60sentencespersecond.Thisisabout40%
performanceboostwithjustafewlinesofcode.Let’scheckifthe
modelpredictionsremainwithinanacceptabletolerance:

..code::ipython3

np.all(np.isclose(embeddings_en,embeddings_en_pt.detach().numpy(),atol=1e-3))




..parsed-literal::

True



CalculateSentenceAlignment
----------------------------

`backtotop⬆️<#table-of-contents>`__

Withtheembeddingmatricesfromthepreviousstep,wecancalculatethe
alignment:1.Calculatesentencesimilaritybetweeneachpairof
sentences.1.Transformthevaluesinthesimilaritymatrixrowsand
columnstoaspecifiedrange,forexample``[-1,1]``.1.Comparethe
valueswithathresholdtogetbooleanmatriceswith0and1.1.
Sentencepairsthathave1inbothmatricesshouldbealignedaccording
tothemodel.

Wevisualizetheresultingmatrixandalsomakesurethattheresultof
theconvertedmodelisthesameastheoriginalone.

..code::ipython3

importseabornassns
importmatplotlib.pyplotasplt


sns.set_style("whitegrid")


deftransform(x):
x=x-np.mean(x)
returnx/np.var(x)


defcalculate_alignment_matrix(first:np.ndarray,second:np.ndarray,threshold:float=1e-3)->np.ndarray:
similarity=first@second.T#1
similarity_en_to_de=np.apply_along_axis(transform,-1,similarity)#2
similarity_de_to_en=np.apply_along_axis(transform,-2,similarity)#2

both_one=(similarity_en_to_de>threshold)*(similarity_de_to_en>threshold)#3and4
returnboth_one


threshold=0.028

alignment_matrix=calculate_alignment_matrix(embeddings_en,embeddings_de,threshold)
alignment_matrix_pt=calculate_alignment_matrix(
embeddings_en_pt.detach().numpy(),
embeddings_de_pt.detach().numpy(),
threshold,
)

graph,axis=plt.subplots(1,2,figsize=(10,5),sharey=True)

formatrix,ax,titleinzip((alignment_matrix,alignment_matrix_pt),axis,("OpenVINO","PyTorch")):
plot=sns.heatmap(matrix,cbar=False,square=True,ax=ax)
plot.set_title(f"SentenceAlignmentMatrix{title}")
plot.set_xlabel("German")
iftitle=="OpenVINO":
plot.set_ylabel("English")

graph.tight_layout()



..image::cross-lingual-books-alignment-with-output_files/cross-lingual-books-alignment-with-output_32_0.png


Aftervisualizingandcomparingthealignmentmatrices,let’stransform
themintoadictionarytomakeitmoreconvenienttoworkwithalignment
inPython.DictionarykeyswillbeEnglishsentencenumbersandvalues
willbelistsofGermansentencenumbers.

..code::ipython3

defmake_alignment(alignment_matrix:np.ndarray)->Dict[int,List[int]]:
aligned={idx:[]foridx,sentinenumerate(sentences_en)}
foren_idx,de_idxinzip(*np.nonzero(alignment_matrix)):
aligned[en_idx].append(de_idx)
returnaligned


aligned=make_alignment(alignment_matrix)
aligned




..parsed-literal::

{0:[0],
1:[2],
2:[3],
3:[4],
4:[5],
5:[6],
6:[7],
7:[8],
8:[9,10],
9:[11],
10:[13,14],
11:[15],
12:[16],
13:[17],
14:[],
15:[18],
16:[19],
17:[20],
18:[21],
19:[23],
20:[24],
21:[25],
22:[26],
23:[],
24:[27],
25:[],
26:[28],
27:[29],
28:[30],
29:[31],
30:[32],
31:[33]}



PostprocessSentenceAlignment
------------------------------

`backtotop⬆️<#table-of-contents>`__

Thereareseveralgapsintheresultingalignment,suchasEnglish
sentence#14notmappingtoanyGermansentence.Herearesomepossible
reasonsforthis:

1.Therearenoequivalentsentencesintheotherbook,andinsuch
cases,themodelisworkingcorrectly.
2.Thesentencehasanequivalentsentenceinanotherlanguage,butthe
modelfailedtoidentifyit.The``threshold``mightbetoohigh,or
themodelisnotsensitiveenough.Toaddressthis,lowerthe
``threshold``valueortryadifferentmodel.
3.Thesentencehasanequivalenttextpartinanotherlanguage,meaning
thateitherthesentencesplittersaretoofineortoocoarse.Try
tuningthetextcleaningandsplittingstepstofixthisissue.
4.Combinationof2and3,whereboththemodel’ssensitivityandtext
preparationstepsneedadjustments.

Anothersolutiontoaddressthisissueisbyapplyingheuristics.Asyou
cansee,Englishsentence13correspondstoGerman17,and15to18.
Mostlikely,Englishsentence14ispartofeitherGermansentence17or
18.Bycomparingthesimilarityusingthemodel,youcanchoosethemost
suitablealignment.

VisualizeSentenceAlignment
----------------------------

`backtotop⬆️<#table-of-contents>`__

Toevaluatethefinalalignmentandchoosethebestwaytoimprovethe
resultsofthepipeline,wewillcreateaninteractivetablewithHTML
andJS.

..code::ipython3

fromIPython.displayimportdisplay,HTML
fromitertoolsimportzip_longest
fromioimportStringIO


defcreate_interactive_table(list1:List[str],list2:List[str],mapping:Dict[int,List[int]])->str:
definverse_mapping(mapping):
inverse_map={idx:[]foridxinrange(len(list2))}

forkey,valuesinmapping.items():
forvalueinvalues:
inverse_map[value].append(key)

returninverse_map

inversed_mapping=inverse_mapping(mapping)

table_html=StringIO()
table_html.write('<tableid="mappings-table"><tr><th>SentencesEN</th><th>SentencesDE</th></tr>')
fori,(first,second)inenumerate(zip_longest(list1,list2)):
table_html.write("<tr>")
ifi<len(list1):
table_html.write(f'<tdid="list1-{i}">{first}</td>')
else:
table_html.write("<td></td>")
ifi<len(list2):
table_html.write(f'<tdid="list2-{i}">{second}</td>')
else:
table_html.write("<td></td>")
table_html.write("</tr>")

table_html.write("</table>")

hover_script=(
"""
<scripttype="module">
consthighlightColor='#0054AE';
consttextColor='white'
constmappings={
'list1':"""
+str(mapping)
+""",
'list2':"""
+str(inversed_mapping)
+"""
};

consttable=document.getElementById('mappings-table');
lethighlightedIds=[];

table.addEventListener('mouseover',({target})=>{
if(target.tagName!=='TD'||!target.id){
return;
}

const[listName,listId]=target.id.split('-');
constmappedIds=mappings[listName]?.[listId]?.map((id)=>`${listName==='list1'?'list2':'list1'}-${id}`)||[];
constidsToHighlight=[target.id,...mappedIds];

setBackgroud(idsToHighlight,highlightColor,textColor);
highlightedIds=idsToHighlight;
});

table.addEventListener('mouseout',()=>setBackgroud(highlightedIds,''));

functionsetBackgroud(ids,color,text_color="unset"){
ids.forEach((id)=>{
document.getElementById(id).style.backgroundColor=color;
document.getElementById(id).style.color=text_color
});
}
</script>
"""
)
table_html.write(hover_script)
returntable_html.getvalue()

..code::ipython3

html_code=create_interactive_table(sentences_en,sentences_de,aligned)
display(HTML(html_code))



..raw::html

<tableid="mappings-table"><tr><th>SentencesEN</th><th>SentencesDE</th></tr><tr><tdid="list1-0">Happyfamiliesareallalike;everyunhappyfamilyisunhappyinitsownway.</td><tdid="list2-0">AlleglücklichenFamiliensindeinanderähnlich;jedeunglücklicheFamilieistaufhrWeiseunglücklich.</td></tr><tr><tdid="list1-1">EverythingwasinconfusionintheOblonskys’house.</td><tdid="list2-1">--</td></tr><tr><tdid="list1-2">ThewifehaddiscoveredthatthehusbandwascarryingonanintriguewithaFrenchgirl,whohadbeenagovernessintheirfamily,andshehadannouncedtoherhusbandthatshecouldnotgoonlivinginthesamehousewithhim.</td><tdid="list2-2">ImHausederOblonskiyherrschteallgemeineVerwirrung.</td></tr><tr><tdid="list1-3">Thispositionofaffairshadnowlastedthreedays,andnotonlythehusbandandwifethemselves,butallthemembersoftheirfamilyandhousehold,werepainfullyconsciousofit.</td><tdid="list2-3">DieDamedesHauseshatteinErfahrunggebracht,daßihrGattemitderimHausegewesenenfranzösischenGouvernanteeinVerhältnisunterhalten,undihmerklärt,siekönnefürderhinnichtmehrmitihmuntereinemDachebleiben.</td></tr><tr><tdid="list1-4">Everypersoninthehousefeltthattherewasnosenseintheirlivingtogether,andthatthestraypeoplebroughttogetherbychanceinanyinnhadmoreincommonwithoneanotherthanthey,themembersofthefamilyandhouseholdoftheOblonskys.</td><tdid="list2-4">DieseSituationwährtebereitsseitdreiTagenundsiewurdenichtalleinvondenbeidenEhegattenselbst,neinauchvonallenFamilienmitgliedernunddemPersonalaufsPeinlichsteempfunden.</td></tr><tr><tdid="list1-5">Thewifedidnotleaveherownroom,thehusbandhadnotbeenathomeforthreedays.</td><tdid="list2-5">Sieallefühlten,daßinihremZusammenlebenkeinhöhererGedankemehrliege,daßdieLeute,welcheaufjederPoststationsichzufälligträfen,nochengerzueinandergehörten,alssie,dieGliederderFamilieselbst,unddasimHausegeboreneundaufgewachseneGesindederOblonskiy.</td></tr><tr><tdid="list1-6">Thechildrenranwildalloverthehouse;theEnglishgovernessquarreledwiththehousekeeper,andwrotetoafriendaskinghertolookoutforanewsituationforher;theman-cookhadwalkedoffthedaybeforejustatdinnertime;thekitchen-maid,andthecoachmanhadgivenwarning.</td><tdid="list2-6">DieHerrindesHausesverließihreGemächernicht,derGebieterwarschonseitdreiTagenabwesend.</td></tr><tr><tdid="list1-7">Threedaysafterthequarrel,PrinceStepanArkadyevitchOblonsky—Stiva,ashewascalledinthefashionableworld—wokeupathisusualhour,thatis,ateighto’clockinthemorning,notinhiswife’sbedroom,butontheleather-coveredsofainhisstudy.</td><tdid="list2-7">DieKinderliefenwieverwaistimganzenHauseumher,dieEngländerinschaltaufdieWirtschafterinundschriebaneineFreundin,diesemöchteihreineneueStellungverschaffen,derKochhattebereitsseitgesternumdieMittagszeitdasHausverlassenunddieKöchin,sowiederKutscherhattenihreRechnungeneingereicht.</td></tr><tr><tdid="list1-8">Heturnedoverhisstout,well-cared-forpersononthespringysofa,asthoughhewouldsinkintoalongsleepagain;hevigorouslyembracedthepillowontheothersideandburiedhisfaceinit;butallatoncehejumpedup,satuponthesofa,andopenedhiseyes.</td><tdid="list2-8">AmdrittenTagenachderSceneerwachtederFürstStefanArkadjewitschOblonskiy--StiwahießerinderWelt--umdiegewöhnlicheStunde,dasheißtumachtUhrmorgens,abernichtimSchlafzimmerseinerGattin,sonderninseinemKabinettaufdemSaffiandiwan.</td></tr><tr><tdid="list1-9">"Yes,yes,howwasitnow?"hethought,goingoverhisdream.</td><tdid="list2-9">ErwandteseinenvollenverweichlichtenLeibaufdenSprungfederndesDiwans,alswünscheernochweiterzuschlafen,währendervonderandernSeiteinnigeinKissenumfaßteundandieWangedrückte.</td></tr><tr><tdid="list1-10">"Now,howwasit?Tobesure!AlabinwasgivingadinneratDarmstadt;no,notDarmstadt,butsomethingAmerican.Yes,butthen,DarmstadtwasinAmerica.Yes,Alabinwasgivingadinneronglasstables,andthetablessang,lmiotesor—notlmiotesorthough,butsomethingbetter,andthereweresomesortoflittledecantersonthetable,andtheywerewomen,too,"heremembered.</td><tdid="list2-10">Plötzlichabersprangerempor,setztesichaufrechtundöffnetedieAugen.</td></tr><tr><tdid="list1-11">StepanArkadyevitch’seyestwinkledgaily,andheponderedwithasmile.</td><tdid="list2-11">"Ja,ja,wiewardochdas?"sanner,überseinemTraumgrübelnd.</td></tr><tr><tdid="list1-12">"Yes,itwasnice,verynice.Therewasagreatdealmorethatwasdelightful,onlythere’snoputtingitintowords,orevenexpressingitinone’sthoughtsawake."</td><tdid="list2-12">"Wiewardochdas?</td></tr><tr><tdid="list1-13">Andnoticingagleamoflightpeepinginbesideoneofthesergecurtains,hecheerfullydroppedhisfeetovertheedgeofthesofa,andfeltaboutwiththemforhisslippers,apresentonhislastbirthday,workedforhimbyhiswifeongold-coloredmorocco.</td><tdid="list2-13">Richtig;AlabingabeinDinerinDarmstadt;nein,nichtinDarmstadt,eswarsoetwasAmerikanischesdabei.</td></tr><tr><tdid="list1-14">And,ashehaddoneeverydayforthelastnineyears,hestretchedouthishand,withoutgettingup,towardstheplacewherehisdressing-gownalwayshunginhisbedroom.</td><tdid="list2-14">DiesesDarmstadtwaraberinAmerika,ja,undAlabingabdasEssenaufgläsernenTischen,ja,unddieTischesangen:Ilmiotesoro--odernichtso,eswaretwasBesseres,undgewissekleineKaraffen,wieFrauenzimmeraussehend,"--fielihmein.</td></tr><tr><tdid="list1-15">Andthereuponhesuddenlyrememberedthathewasnotsleepinginhiswife’sroom,butinhisstudy,andwhy:thesmilevanishedfromhisface,heknittedhisbrows.</td><tdid="list2-15">DieAugenStefanArkadjewitschsblitztenheiter,ersannundlächelte.</td></tr><tr><tdid="list1-16">"Ah,ah,ah!Oo!..."hemuttered,recallingeverythingthathadhappened.</td><tdid="list2-16">"Ja,eswarhübsch,sehrhübsch.EsgabvielAusgezeichnetesdabei,wasmanmitWortennichtschildernkönnteundinGedankennichtausdrücken."</td></tr><tr><tdid="list1-17">Andagaineverydetailofhisquarrelwithhiswifewaspresenttohisimagination,allthehopelessnessofhisposition,andworstofall,hisownfault.</td><tdid="list2-17">ErbemerkteeinenLichtstreif,dersichvonderSeitedurchdiebaumwollenenStoresgestohlenhatteundschnelltelustigmitdenFüßenvomSofa,ummitihnendievonseinerGattinihmimvorigenJahrzumGeburtstagverehrtengold-undsaffiangesticktenPantoffelnzusuchen;währender,eineraltenneunjährigenGewohnheitfolgend,ohneaufzustehenmitderHandnachderStellefuhr,woindemSchlafzimmersonstseinMorgenrockzuhängenpflegte.</td></tr><tr><tdid="list1-18">"Yes,shewon’tforgiveme,andshecan’tforgiveme.Andthemostawfulthingaboutitisthatit’sallmyfault—allmyfault,thoughI’mnottoblame.That’sthepointofthewholesituation,"hereflected.</td><tdid="list2-18">HierbeierstkamerzurBesinnung;erentsannsichjähwieeskam,daßernichtimSchlafgemachseinerGattin,sondernindemKabinettschlief;dasLächelnverschwandvonseinenZügenunderrunzeltedieStirn.</td></tr><tr><tdid="list1-19">"Oh,oh,oh!"hekeptrepeatingindespair,asherememberedtheacutelypainfulsensationscausedhimbythisquarrel.</td><tdid="list2-19">"O,o,o,ach,"bracherjammerndaus,indemihmalleswiedereinfiel,wasvorgefallenwar.</td></tr><tr><tdid="list1-20">Mostunpleasantofallwasthefirstminutewhen,oncoming,happyandgood-humored,fromthetheater,withahugepearinhishandforhiswife,hehadnotfoundhiswifeinthedrawing-room,tohissurprisehadnotfoundherinthestudyeither,andsawheratlastinherbedroomwiththeunluckyletterthatrevealedeverythinginherhand.</td><tdid="list2-20">VorseinemInnernerstandenvonneuemalledieEinzelheitendesAuftrittsmitseinerFrau,erstanddieganzeMißlichkeitseinerLageund--wasihmampeinlichstenwar--seineeigeneSchuld.</td></tr><tr><tdid="list1-21">She,hisDolly,foreverfussingandworryingoverhouseholddetails,andlimitedinherideas,asheconsidered,wassittingperfectlystillwiththeletterinherhand,lookingathimwithanexpressionofhorror,despair,andindignation.</td><tdid="list2-21">"Jawohl,siewirdnichtverzeihen,siekannnichtverzeihen,undamSchrecklichstenist,daßdieSchuldanallemnurichselbsttrage--ichbinschuld--abernichtschuldig!</td></tr><tr><tdid="list1-22">"What’sthis?this?"sheasked,pointingtotheletter.</td><tdid="list2-22">UndhierinliegtdasganzeDrama,"dachteer,"oweh,oweh!"</td></tr><tr><tdid="list1-23">Andatthisrecollection,StepanArkadyevitch,asissooftenthecase,wasnotsomuchannoyedatthefactitselfasatthewayinwhichhehadmethiswife’swords.</td><tdid="list2-23">ErsprachvollerVerzweiflung,indemersichalledietiefenEindrückevergegenwärtigte,dieerinjenerSceneerhalten.</td></tr><tr><tdid="list1-24">Therehappenedtohimatthatinstantwhatdoeshappentopeoplewhentheyareunexpectedlycaughtinsomethingverydisgraceful.</td><tdid="list2-24">AmunerquicklichstenwarihmjeneersteMinutegewesen,daer,heiterundzufriedenausdemTheaterheimkehrend,eineungeheureBirnefürseineFrauinderHand,diesewederimSalonnochimKabinettfand,undsieendlichimSchlafzimmerantraf,jenenunglückseligenBrief,derallesentdeckte,indenHänden.</td></tr><tr><tdid="list1-25">Hedidnotsucceedinadaptinghisfacetothepositioninwhichhewasplacedtowardshiswifebythediscoveryofhisfault.</td><tdid="list2-25">Sie,dieerfürdieewigsorgende,ewigsichmühende,allgegenwärtigeDollygehalten,siesaßjetztregungslos,denBriefinderHand,mitdemAusdruckdesEntsetzens,derVerzweiflungundderWutihmentgegenblickend.</td></tr><tr><tdid="list1-26">Insteadofbeinghurt,denying,defendinghimself,beggingforgiveness,insteadofremainingindifferenteven—anythingwouldhavebeenbetterthanwhathediddo—hisfaceutterlyinvoluntarily(reflexspinalaction,reflectedStepanArkadyevitch,whowasfondofphysiology)—utterlyinvoluntarilyassumeditshabitual,good-humored,andthereforeidioticsmile.</td><tdid="list2-26">"Wasistdas?"frugsieihn,aufdasSchreibenweisend,undinderErinnerunghieranquälteihn,wiedasoftzugeschehenpflegt,nichtsowohlderVorfallselbst,alsdieArt,wieerihraufdieseWortegeantwortethatte.</td></tr><tr><tdid="list1-27">Thisidioticsmilehecouldnotforgivehimself.</td><tdid="list2-27">EsgingihmindiesemAugenblick,wiedenmeistenMenschen,wennsieunerwarteteineszuschmählichenVergehensüberführtwerden.</td></tr><tr><tdid="list1-28">Catchingsightofthatsmile,Dollyshudderedasthoughatphysicalpain,brokeoutwithhercharacteristicheatintoafloodofcruelwords,andrushedoutoftheroom.</td><tdid="list2-28">Erverstandnicht,seinGesichtderSituationanzupassen,inwelcheernachderEntdeckungseinerSchuldgeratenwar,undanstattdenGekränktenzuspielen,sichzuverteidigen,sichzurechtfertigenundumVerzeihungzubittenoderwenigstensgleichmütigzubleiben--allesdieswärenochbessergewesenalsdas,waserwirklichthat--verzogensichseineMienen("Gehirnreflexe"dachteStefanArkadjewitsch,alsLiebhabervonPhysiologie)unwillkürlichundplötzlichzuseinemgewohnten,gutmütigenunddaherziemlicheinfältigenLächeln.</td></tr><tr><tdid="list1-29">Sincethenshehadrefusedtoseeherhusband.</td><tdid="list2-29">DiesesdummeLächelnkonnteersichselbstnichtvergeben.</td></tr><tr><tdid="list1-30">"It’sthatidioticsmilethat’stoblameforitall,"thoughtStepanArkadyevitch.</td><tdid="list2-30">AlsDollyesgewahrthatte,erbebtesie,wievoneinemphysischenSchmerz,undergingsichdannmitderihreigenenLeidenschaftlichkeitineinemStrombittererWorte,woraufsiedasGemachverließ.</td></tr><tr><tdid="list1-31">"Butwhat’stobedone?What’stobedone?"hesaidtohimselfindespair,andfoundnoanswer.</td><tdid="list2-31">VondieserZeitanwolltesieihrenGattennichtmehrsehen.</td></tr><tr><td></td><tdid="list2-32">"AnallemistdasdummeLächelnschuld,"dachteStefanArkadjewitsch.</td></tr><tr><td></td><tdid="list2-33">"Aberwassollichthun,wassollichthun?"frugervollVerzweiflungsichselbst,ohneeineAntwortzufinden.</td></tr></table>
<scripttype="module">
consthighlightColor='#0054AE';
consttextColor='white'
constmappings={
'list1':{0:[0],1:[2],2:[3],3:[4],4:[5],5:[6],6:[7],7:[8],8:[9,10],9:[11],10:[13,14],11:[15],12:[16],13:[17],14:[],15:[18],16:[19],17:[20],18:[21],19:[23],20:[24],21:[25],22:[26],23:[],24:[27],25:[],26:[28],27:[29],28:[30],29:[31],30:[32],31:[33]},
'list2':{0:[0],1:[],2:[1],3:[2],4:[3],5:[4],6:[5],7:[6],8:[7],9:[8],10:[8],11:[9],12:[],13:[10],14:[10],15:[11],16:[12],17:[13],18:[15],19:[16],20:[17],21:[18],22:[],23:[19],24:[20],25:[21],26:[22],27:[24],28:[26],29:[27],30:[28],31:[29],32:[30],33:[31]}
};

consttable=document.getElementById('mappings-table');
lethighlightedIds=[];

table.addEventListener('mouseover',({target})=>{
if(target.tagName!=='TD'||!target.id){
return;
}

const[listName,listId]=target.id.split('-');
constmappedIds=mappings[listName]?.[listId]?.map((id)=>`${listName==='list1'?'list2':'list1'}-${id}`)||[];
constidsToHighlight=[target.id,...mappedIds];

setBackgroud(idsToHighlight,highlightColor,textColor);
highlightedIds=idsToHighlight;
});

table.addEventListener('mouseout',()=>setBackgroud(highlightedIds,''));

functionsetBackgroud(ids,color,text_color="unset"){
ids.forEach((id)=>{
document.getElementById(id).style.backgroundColor=color;
document.getElementById(id).style.color=text_color
});
}
</script>



YoucanseethatthepipelinedoesnotfullycleanuptheGermantext,
resultinginissueslikethesecondsentenceconsistingofonly``--``.
Onapositivenote,thesplitsentencesintheGermantranslationline
upcorrectlywiththesingleEnglishsentence.Overall,thepipeline
alreadyworkswell,butthereisstillroomforimprovement.

SavetheOpenVINOmodeltodiskforfutureuse:

..code::ipython3

fromopenvino.runtimeimportserialize


ov_model_path="ov_model/model.xml"
serialize(ov_model,ov_model_path)

Toreadthemodelfromdisk,usethe``read_model``methodofthe
``Core``object:

..code::ipython3

ov_model=core.read_model(ov_model_path)

SpeedupEmbeddingsComputation
-------------------------------

`backtotop⬆️<#table-of-contents>`__

Let’sseehowwecanspeedupthemostcomputationallycomplexpartof
thepipeline-gettingembeddings.Youmightwonderwhy,whenusing
OpenVINO,youneedtocompilethemodelafterreadingit.Therearetwo
mainreasonsforthis:1.Compatibilitywithdifferentdevices.The
modelcanbecompiledtorunona`specific
device<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html>`__,
likeCPU,GPUorGNA.Eachdevicemayworkwithdifferentdatatypes,
supportdifferentfeatures,andgainperformancebychangingtheneural
networkforaspecificcomputingmodel.WithOpenVINO,youdonotneed
tostoremultiplecopiesofthenetworkwithoptimizedfordifferent
hardware.AuniversalOpenVINOmodelrepresentationisenough.1.
Optimizationfordifferentscenarios.Forexample,onescenario
prioritizesminimizingthe*timebetweenstartingandfinishingmodel
inference*(`latency-oriented
optimization<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html>`__).
Inourcase,itismoreimportant*howmanytextspersecondthemodel
canprocess*(`throughput-oriented
optimization<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-throughput.html>`__).

Togetathroughput-optimizedmodel,passa`performance
hint<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html#performance-hints-latency-and-throughput>`__
asaconfigurationduringcompilation.ThenOpenVINOselectstheoptimal
parametersforexecutionontheavailablehardware.

..code::ipython3

fromtypingimportAny


compiled_throughput_hint=core.compile_model(
ov_model,
device_name=device.value,
config={"PERFORMANCE_HINT":"THROUGHPUT"},
)

Tofurtheroptimizehardwareutilization,let’schangetheinference
modefromsynchronous(Sync)toasynchronous(Async).Whilethe
synchronousAPImaybeeasiertostartwith,itis
`recommended<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html#prefer-openvino-async-api>`__
tousetheasynchronous(callbacks-based)APIinproductioncode.Itis
themostgeneralandscalablewaytoimplementflowcontrolforany
numberofrequests.

Toworkinasynchronousmode,youneedtodefinetwothings:

1.Instantiatean``AsyncInferQueue``,whichcanthenbepopulatedwith
inferencerequests.
2.Definea``callback``functionthatwillbecalledaftertheoutput
requesthasbeenexecutedanditsresultshavebeenprocessed.

Inadditiontothemodelinput,anydatarequiredforpost-processing
canbepassedtothequeue.Wecancreateazeroembeddingmatrixin
advanceandfillitinastheinferencerequestsareexecuted.

..code::ipython3

defget_embeddings_async(sentences:List[str],embedding_model:OVModel)->np.ndarray:
defcallback(infer_request:ov.InferRequest,user_data:List[Any])->None:
embeddings,idx,pbar=user_data
embedding=infer_request.get_output_tensor(0).data[0,0]
embeddings[idx]=embedding
pbar.update()

infer_queue=ov.AsyncInferQueue(embedding_model)
infer_queue.set_callback(callback)

embedding_dim=embedding_model.output(0).get_partial_shape().get_dimension(2).get_length()
embeddings=np.zeros((len(sentences),embedding_dim))

withtqdm(total=len(sentences),disable=disable_tqdm)aspbar:
foridx,sentinenumerate(sentences):
tokenized=tokenizer(sent,return_tensors="np").data

infer_queue.start_async(tokenized,[embeddings,idx,pbar])

infer_queue.wait_all()

returnembeddings

Let’scomparethemodelsandplottheresults.

**Note**:Togetamoreaccuratebenchmark,usethe`BenchmarkPython
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__

..code::ipython3

number_of_chars=15_000
more_sentences_en=splitter_en.segment(clean_text(anna_karenina_en[:number_of_chars]))
len(more_sentences_en)



..parsed-literal::

0%||0/3[00:00<?,?it/s]




..parsed-literal::

112



..code::ipython3

importpandasaspd
fromtimeimportperf_counter


benchmarks=[
(pt_model,get_embeddings,"PyTorch"),
(compiled_model,get_embeddings,"OpenVINO\nSync"),
(
compiled_throughput_hint,
get_embeddings_async,
"OpenVINO\nThroughputHint\nAsync",
),
]

number_of_sentences=100
benchmark_data=more_sentences_en[:min(number_of_sentences,len(more_sentences_en))]

benchmark_results={name:[]for*_,nameinbenchmarks}

benchmarks_iterator=tqdm(benchmarks,leave=False,disable=disable_tqdm)
formodel,func,nameinbenchmarks_iterator:
printable_name=name.replace("\n","")
benchmarks_iterator.set_description(f"Runbenchmarkfor{printable_name}model")
forrunintqdm(range(10+1),leave=False,desc="BenchmarkRuns:",disable=disable_tqdm):
withdisable_tqdm_context():
start=perf_counter()
func(benchmark_data,model)
end=perf_counter()
benchmark_results[name].append(len(benchmark_data)/(end-start))

benchmark_dataframe=pd.DataFrame(benchmark_results)[1:]



..parsed-literal::

0%||0/3[00:00<?,?it/s]



..parsed-literal::

BenchmarkRuns:0%||0/11[00:00<?,?it/s]



..parsed-literal::

BenchmarkRuns:0%||0/11[00:00<?,?it/s]



..parsed-literal::

BenchmarkRuns:0%||0/11[00:00<?,?it/s]


..code::ipython3

cpu_name=core.get_property("CPU","FULL_DEVICE_NAME")

plot=sns.barplot(benchmark_dataframe,errorbar="sd")
plot.set(ylabel="SentencesPerSecond",title=f"SentenceEmbeddingsBenchmark\n{cpu_name}")
perf_ratio=benchmark_dataframe.mean()/benchmark_dataframe.mean()[0]
plot.spines["right"].set_visible(False)
plot.spines["top"].set_visible(False)
plot.spines["left"].set_visible(False)



..image::cross-lingual-books-alignment-with-output_files/cross-lingual-books-alignment-with-output_49_0.png


OnanIntelCorei9-10980XECPU,theOpenVINOmodelprocessed45%more
sentencespersecondcomparedwiththeoriginalPyTorchmodel.Using
Asyncmodewiththroughputhint,weget×3.21(or221%)performance
boost.

Hereareusefullinkswithinformationaboutthetechniquesusedinthis
notebook:-`OpenVINOperformance
hints<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html>`__
-`OpenVINOAsync
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html#prefer-openvino-async-api>`__
-`Throughput
Optimizations<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-throughput.html>`__

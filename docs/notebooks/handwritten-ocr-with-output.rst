HandwrittenChineseandJapaneseOCRwithOpenVINO™
===================================================

Inthistutorial,weperformopticalcharacterrecognition(OCR)for
handwrittenChinese(simplified)andJapanese.AnOCRtutorialusingthe
Latinalphabetisavailablein`notebook
208<optical-character-recognition-with-output.html>`__.
Thismodeliscapableofprocessingonlyonelineofsymbolsatatime.

Themodelsusedinthisnotebookare
`handwritten-japanese-recognition-0001<https://docs.openvino.ai/2024/omz_models_model_handwritten_japanese_recognition_0001.html>`__
and
`handwritten-simplified-chinese-0001<https://docs.openvino.ai/2024/omz_models_model_handwritten_simplified_chinese_recognition_0001.html>`__.
Todecodemodeloutputsasreadabletext
`kondate_nakayosi<https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/kondate_nakayosi.txt>`__
and
`scut_ept<https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/scut_ept.txt>`__
charlistsareused.Bothmodelsareavailableon`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Settings<#settings>`__
-`SelectaLanguage<#select-a-language>`__
-`DownloadtheModel<#download-the-model>`__
-`LoadtheModelandExecute<#load-the-model-and-execute>`__
-`Selectinferencedevice<#select-inference-device>`__
-`FetchInformationAboutInputandOutput
Layers<#fetch-information-about-input-and-output-layers>`__
-`LoadanImage<#load-an-image>`__
-`VisualizeInputImage<#visualize-input-image>`__
-`PrepareCharlist<#prepare-charlist>`__
-`RunInference<#run-inference>`__
-`ProcesstheOutputData<#process-the-output-data>`__
-`PrinttheOutput<#print-the-output>`__

..code::ipython3

importplatform

#Installopenvino-devpackage
%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromcollectionsimportnamedtuple
fromitertoolsimportgroupby

importcv2
importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

Settings
--------

`backtotop⬆️<#table-of-contents>`__

Setupallconstantsandfoldersusedinthisnotebook

..code::ipython3

#Directorieswheredatawillbeplaced.
base_models_dir="models"
data_folder="data"
charlist_folder=f"{data_folder}/text"

#Precisionusedbythemodel.
precision="FP16"

Togroupfiles,youhavetodefinethecollection.Inthiscase,use
``namedtuple``.

..code::ipython3

Language=namedtuple(typename="Language",field_names=["model_name","charlist_name","demo_image_name"])
chinese_files=Language(
model_name="handwritten-simplified-chinese-recognition-0001",
charlist_name="chinese_charlist.txt",
demo_image_name="handwritten_chinese_test.jpg",
)
japanese_files=Language(
model_name="handwritten-japanese-recognition-0001",
charlist_name="japanese_charlist.txt",
demo_image_name="handwritten_japanese_test.png",
)

SelectaLanguage
-----------------

`backtotop⬆️<#table-of-contents>`__

Dependingonyourchoiceyouwillneedtochangealineofcodeinthe
cellbelow.

IfyouwanttoperformOCRonatextinJapanese,set
``language="japanese"``.ForChinese,set``language="chinese"``.

..code::ipython3

#Selectthelanguagebyusingeitherlanguage="chinese"orlanguage="japanese".
language="chinese"

languages={"chinese":chinese_files,"japanese":japanese_files}

selected_language=languages.get(language)

DownloadtheModel
------------------

`backtotop⬆️<#table-of-contents>`__

Inadditiontoimagesandcharlists,youneedtodownloadthemodel
file.Inthesectionsbelow,therearecellsfordownloadingeitherthe
ChineseorJapanesemodel.

Ifitisyourfirsttimerunningthenotebook,themodelwillbe
downloaded.Itmaytakeafewminutes.

Use``download_file``functionfromtheutilspackage,which
automaticallycreatesadirectorystructureanddownloadstheselected
modelfile.

..code::ipython3

path_to_model=download_file(
url=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{selected_language.model_name}/{precision}/{selected_language.model_name}.xml",
directory=base_models_dir,
)
_=download_file(
url=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{selected_language.model_name}/{precision}/{selected_language.model_name}.bin",
directory=base_models_dir,
)



..parsed-literal::

models/handwritten-simplified-chinese-recognition-0001.xml:0%||0.00/108k[00:00<?,?B/s]



..parsed-literal::

models/handwritten-simplified-chinese-recognition-0001.bin:0%||0.00/32.9M[00:00<?,?B/s]


LoadtheModelandExecute
--------------------------

`backtotop⬆️<#table-of-contents>`__

Whenallfilesaredownloadedandlanguageisselected,readandcompile
thenetworktoruninference.Thepathtothemodelisdefinedbasedon
theselectedlanguage.

..code::ipython3

core=ov.Core()
model=core.read_model(model=path_to_model)

Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

compiled_model=core.compile_model(model=model,device_name=device.value)

FetchInformationAboutInputandOutputLayers
-----------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Nowthatthemodelisloaded,fetchinformationabouttheinputand
outputlayers(shape).

..code::ipython3

recognition_output_layer=compiled_model.output(0)
recognition_input_layer=compiled_model.input(0)

LoadanImage
-------------

`backtotop⬆️<#table-of-contents>`__

Next,loadanimage.Themodelexpectsasingle-channelimageasinput,
sotheimageisreadingrayscale.

Afterloadingtheinputimage,getinformationtouseforcalculating
thescaleratiobetweenrequiredinputlayerheightandthecurrent
imageheight.Inthecellbelow,theimagewillberesizedandpaddedto
keeplettersproportionalandmeetinputshape.

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstoragebasedontheselectedmodel.
file_name=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/"+selected_language.demo_image_name,
directory=data_folder,
)

#Textdetectionmodelsexpectanimageingrayscaleformat.
#IMPORTANT!Thismodelenablesreadingonlyonelineattime.

#Readtheimage.
image=cv2.imread(filename=str(file_name),flags=cv2.IMREAD_GRAYSCALE)

#Fetchtheshape.
image_height,_=image.shape

#B,C,H,W=batchsize,numberofchannels,height,width.
_,_,H,W=recognition_input_layer.shape

#Calculatescaleratiobetweentheinputshapeheightandimageheighttoresizetheimage.
scale_ratio=H/image_height

#Resizetheimagetoexpectedinputsizes.
resized_image=cv2.resize(image,None,fx=scale_ratio,fy=scale_ratio,interpolation=cv2.INTER_AREA)

#Padtheimagetomatchinputsize,withoutchangingaspectratio.
resized_image=np.pad(resized_image,((0,0),(0,W-resized_image.shape[1])),mode="edge")

#Reshapetonetworkinputshape.
input_image=resized_image[None,None,:,:]



..parsed-literal::

data/handwritten_chinese_test.jpg:0%||0.00/42.1k[00:00<?,?B/s]


VisualizeInputImage
---------------------

`backtotop⬆️<#table-of-contents>`__

Afterpreprocessing,youcandisplaytheimage.

..code::ipython3

plt.figure(figsize=(20,1))
plt.axis("off")
plt.imshow(resized_image,cmap="gray",vmin=0,vmax=255);



..image::handwritten-ocr-with-output_files/handwritten-ocr-with-output_22_0.png


PrepareCharlist
----------------

`backtotop⬆️<#table-of-contents>`__

Themodelisloadedandtheimageisready.Theonlyelementleftisthe
charlist,whichisdownloaded.Youmustaddablanksymbolatthe
beginningofthecharlistbeforeusingit.Thisisexpectedforboththe
ChineseandJapanesemodels.

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstoragebasedontheselectedmodel.
used_charlist_file=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/"+selected_language.charlist_name,
directory=charlist_folder,
)



..parsed-literal::

data/text/chinese_charlist.txt:0%||0.00/15.8k[00:00<?,?B/s]


..code::ipython3

#Getadictionarytoencodetheoutput,basedonmodeldocumentation.
used_charlist=selected_language.charlist_name

#Withbothmodels,thereshouldbeblanksymboladdedatindex0ofeachcharlist.
blank_char="~"

withused_charlist_file.open(mode="r",encoding="utf-8")ascharlist:
letters=blank_char+"".join(line.strip()forlineincharlist)

RunInference
-------------

`backtotop⬆️<#table-of-contents>`__

Now,runinference.The``compiled_model()``functiontakesalistwith
input(s)inthesameorderasmodelinput(s).Then,fetchtheoutput
fromoutputtensors.

..code::ipython3

#Runinferenceonthemodel
predictions=compiled_model([input_image])[recognition_output_layer]

ProcesstheOutputData
-----------------------

`backtotop⬆️<#table-of-contents>`__

Theoutputofamodelisinthe``WxBxL``format,where:

-W-outputsequencelength
-B-batchsize
-L-confidencedistributionacrossthesupportedsymbolsinKondate
andNakayosi.

Togetamorehuman-readableformat,selectasymbolwiththehighest
probability.Whenyouholdalistofindexesthatarepredictedtohave
thehighestprobability,duetolimitationsin`CTC
Decoding<https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7>`__,
youwillremoveconcurrentsymbolsandthenremovetheblanks.

Finally,getthesymbolsfromcorrespondingindexesinthecharlist.

..code::ipython3

#Removeabatchdimension.
predictions=np.squeeze(predictions)

#Runthe`argmax`functiontopickthesymbolswiththehighestprobability.
predictions_indexes=np.argmax(predictions,axis=1)

..code::ipython3

#Usethe`groupby`functiontoremoveconcurrentletters,asrequiredbyCTCgreedydecoding.
output_text_indexes=list(groupby(predictions_indexes))

#Removegrouperobjects.
output_text_indexes,_=np.transpose(output_text_indexes,(1,0))

#Removeblanksymbols.
output_text_indexes=output_text_indexes[output_text_indexes!=0]

#Assignletterstoindexesfromtheoutputarray.
output_text=[letters[letter_index]forletter_indexinoutput_text_indexes]

PrinttheOutput
----------------

`backtotop⬆️<#table-of-contents>`__

Now,havingalistofletterspredictedbythemodel,youcandisplay
theimagewithpredictedtextprintedbelow.

..code::ipython3

plt.figure(figsize=(20,1))
plt.axis("off")
plt.imshow(resized_image,cmap="gray",vmin=0,vmax=255)

print("".join(output_text))


..parsed-literal::

人有悲欢离合，月有阴睛圆缺，此事古难全。



..image::handwritten-ocr-with-output_files/handwritten-ocr-with-output_32_1.png


PartSegmentationof3DPointCloudswithOpenVINO™
===================================================

Thisnotebookdemonstrateshowtoprocess`point
cloud<https://en.wikipedia.org/wiki/Point_cloud>`__dataandrun3D
PartSegmentationwithOpenVINO.Weusethe
`PointNet<https://arxiv.org/abs/1612.00593>`__pre-trainedmodelto
detecteachpartofachairandreturnitscategory.

PointNet
--------

PointNetwasproposedbyCharlesRuizhongtaiQi,aresearcherat
StanfordUniversityin2016:`PointNet:DeepLearningonPointSetsfor
3DClassificationand
Segmentation<https://arxiv.org/abs/1612.00593>`__.Themotivation
behindtheresearchistoclassifyandsegment3Drepresentationsof
images.Theyuseadatastructurecalledpointcloud,whichisasetof
pointsthatrepresentsa3Dshapeorobject.PointNetprovidesaunified
architectureforapplicationsrangingfromobjectclassification,part
segmentation,toscenesemanticparsing.Itishighlyefficientand
effective,showingstrongperformanceonparorevenbetterthanstate
oftheart.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`PreparetheModel<#prepare-the-model>`__
-`DataProcessingModule<#data-processing-module>`__
-`Visualizetheoriginal3Ddata<#visualize-the-original-3d-data>`__
-`Runinference<#run-inference>`__

-`Selectinferencedevice<#select-inference-device>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0""tqdm"

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

frompathlibimportPath
fromtypingimportUnion
importnumpyasnp
importmatplotlib.pyplotasplt
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

PreparetheModel
-----------------

`backtotop⬆️<#table-of-contents>`__

Downloadthepre-trainedPointNetONNXmodel.Thispre-trainedmodelis
providedby`axinc-ai<https://github.com/axinc-ai>`__,andyoucan
findmorepointcloudsexamples
`here<https://github.com/axinc-ai/ailia-models/tree/master/point_segmentation>`__.

..code::ipython3

#Setthedataandmodeldirectories,modelsourceURLandmodelfilename
MODEL_DIR=Path("model")
MODEL_DIR.mkdir(exist_ok=True)
download_file(
"https://storage.googleapis.com/ailia-models/pointnet_pytorch/chair_100.onnx",
directory=Path(MODEL_DIR),
show_progress=False,
)
onnx_model_path=MODEL_DIR/"chair_100.onnx"

ConverttheONNXmodeltoOpenVINOIR.AnOpenVINOIR(Intermediate
Representation)modelconsistsofan``.xml``file,containing
informationaboutnetworktopology,anda``.bin``file,containingthe
weightsandbiasesbinarydata.ModelconversionPythonAPIisusedfor
conversionofONNXmodeltoOpenVINOIR.The``ov.convert_model``Python
functionreturnsanOpenVINOmodelreadytoloadonadeviceandstart
makingpredictions.Wecansaveitonadiskfornextusagewith
``ov.save_model``.FormoreinformationaboutmodelconversionPython
API,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

ir_model_xml=onnx_model_path.with_suffix(".xml")

core=ov.Core()

ifnotir_model_xml.exists():
#ConvertmodeltoOpenVINOModel
model=ov.convert_model(onnx_model_path)
#SerializemodelinOpenVINOIRformatxml+bin
ov.save_model(model,ir_model_xml)
else:
#Readmodel
model=core.read_model(model=ir_model_xml)

DataProcessingModule
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defload_data(point_file:Union[str,Path]):
"""
Loadthepointclouddataandconvertittondarray

Parameters:
point_file:string,pathof.ptsdata
Returns:
point_set:pointcloundrepresentedinnp.arrayformat
"""

point_set=np.loadtxt(point_file).astype(np.float32)

#normailization
point_set=point_set-np.expand_dims(np.mean(point_set,axis=0),0)#center
dist=np.max(np.sqrt(np.sum(point_set**2,axis=1)),0)
point_set=point_set/dist#scale

returnpoint_set


defvisualize(point_set:np.ndarray):
"""
Createa3Dviewfordatavisualization

Parameters:
point_set:np.ndarray,thecoordinatedatainXYZformat
"""

fig=plt.figure(dpi=192,figsize=(4,4))
ax=fig.add_subplot(111,projection="3d")
X=point_set[:,0]
Y=point_set[:,2]
Z=point_set[:,1]

#Scaletheviewofeachaxistoadapttothecoordinatedatadistribution
max_range=np.array([X.max()-X.min(),Y.max()-Y.min(),Z.max()-Z.min()]).max()*0.5
mid_x=(X.max()+X.min())*0.5
mid_y=(Y.max()+Y.min())*0.5
mid_z=(Z.max()+Z.min())*0.5
ax.set_xlim(mid_x-max_range,mid_x+max_range)
ax.set_ylim(mid_y-max_range,mid_y+max_range)
ax.set_zlim(mid_z-max_range,mid_z+max_range)

plt.tick_params(labelsize=5)
ax.set_xlabel("X",fontsize=10)
ax.set_ylabel("Y",fontsize=10)
ax.set_zlabel("Z",fontsize=10)

returnax

Visualizetheoriginal3Ddata
------------------------------

`backtotop⬆️<#table-of-contents>`__

Thepointclouddatacanbedownloadedfrom
`ShapeNet<https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip>`__,
alarge-scaledatasetof3Dshapes.Here,weselectthe3Ddataofa
chairforexample.

..code::ipython3

#Downloaddatafromtheopenvino_notebooksstorage
point_data=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/pts/chair.pts",
directory="data",
)

points=load_data(str(point_data))
X=points[:,0]
Y=points[:,2]
Z=points[:,1]
ax=visualize(points)
ax.scatter3D(X,Y,Z,s=5,cmap="jet",marker="o",label="chair")
ax.set_title("3DVisualization")
plt.legend(loc="upperright",fontsize=8)
plt.show()



..parsed-literal::

data/chair.pts:0%||0.00/69.2k[00:00<?,?B/s]


..parsed-literal::

/tmp/ipykernel_113341/2434168836.py:12:UserWarning:Nodataforcolormappingprovidedvia'c'.Parameters'cmap'willbeignored
ax.scatter3D(X,Y,Z,s=5,cmap="jet",marker="o",label="chair")



..image::3D-segmentation-point-clouds-with-output_files/3D-segmentation-point-clouds-with-output_11_2.png


Runinference
-------------

`backtotop⬆️<#table-of-contents>`__

Runinferenceandvisualizetheresultsof3Dsegmentation.-Theinput
dataisapointcloudwith``1batchsize``\，\``3axisvalue``(x,y,
z)and``arbitrarynumberofpoints``(dynamicshape).-Theoutputdata
isamaskwith``1batchsize``and``4classificationconfidence``for
eachinputpoint.

..code::ipython3

#Partsofachair
classes=["back","seat","leg","arm"]

#Preprocesstheinputdata
point=points.transpose(1,0)
point=np.expand_dims(point,axis=0)

#Printinfoaboutmodelinputandoutputshape
print(f"inputshape:{model.input(0).partial_shape}")
print(f"outputshape:{model.output(0).partial_shape}")


..parsed-literal::

inputshape:[1,3,?]
outputshape:[1,?,4]


Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

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

#Inference
compiled_model=core.compile_model(model=model,device_name=device.value)
output_layer=compiled_model.output(0)
result=compiled_model([point])[output_layer]

#Findthelabelmapforallpointsofchairwithhighestconfidence
pred=np.argmax(result[0],axis=1)
ax=visualize(point)
fori,nameinenumerate([0,1,2,3]):
XCur=[]
YCur=[]
ZCur=[]
forj,nameCurinenumerate(pred):
ifname==nameCur:
XCur.append(X[j])
YCur.append(Y[j])
ZCur.append(Z[j])
XCur=np.array(XCur)
YCur=np.array(YCur)
ZCur=np.array(ZCur)

#addcurrentpointofthepart
ax.scatter(XCur,YCur,ZCur,s=5,cmap="jet",marker="o",label=classes[i])

ax.set_title("3DSegmentationVisualization")
plt.legend(loc="upperright",fontsize=8)
plt.show()


..parsed-literal::

/tmp/ipykernel_113341/2804603389.py:23:UserWarning:Nodataforcolormappingprovidedvia'c'.Parameters'cmap'willbeignored
ax.scatter(XCur,YCur,ZCur,s=5,cmap="jet",marker="o",label=classes[i])



..image::3D-segmentation-point-clouds-with-output_files/3D-segmentation-point-clouds-with-output_16_1.png


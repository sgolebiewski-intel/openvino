LiveHumanPoseEstimationwithOpenVINO™
=========================================

ThisnotebookdemonstratesliveposeestimationwithOpenVINO,usingthe
OpenPose
`human-pose-estimation-0001<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001>`__
modelfrom`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__.Finalpart
ofthisnotebookshowsliveinferenceresultsfromawebcam.
Additionally,youcanalsouploadavideofile.

**NOTE**:Touseawebcam,youmustrunthisJupyternotebookona
computerwithawebcam.Ifyourunonaserver,thewebcamwillnot
work.However,youcanstilldoinferenceonavideointhefinal
step.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Themodel<#the-model>`__

-`Downloadthemodel<#download-the-model>`__
-`Loadthemodel<#load-the-model>`__

-`Processing<#processing>`__

-`OpenPoseDecoder<#openpose-decoder>`__
-`ProcessResults<#process-results>`__
-`DrawPoseOverlays<#draw-pose-overlays>`__
-`MainProcessingFunction<#main-processing-function>`__

-`Run<#run>`__

-`RunLivePoseEstimation<#run-live-pose-estimation>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importtime
frompathlibimportPath

importcv2
importnumpyasnp
fromIPythonimportdisplay
fromnumpy.lib.stride_tricksimportas_strided
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
importnotebook_utilsasutils

Themodel
---------

`backtotop⬆️<#table-of-contents>`__

Downloadthemodel
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Usethe``download_file``,afunctionfromthe``notebook_utils``file.
Itautomaticallycreatesadirectorystructureanddownloadsthe
selectedmodel.

Ifyouwanttodownloadanothermodel,replacethenameofthemodeland
precisioninthecodebelow.

**NOTE**:Thismayrequireadifferentposedecoder.

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.
base_model_dir=Path("model")

#ThenameofthemodelfromOpenModelZoo.
model_name="human-pose-estimation-0001"
#Selectedprecision(FP32,FP16,FP16-INT8).
precision="FP16-INT8"

model_path=base_model_dir/"intel"/model_name/precision/f"{model_name}.xml"

ifnotmodel_path.exists():
model_url_dir=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
utils.download_file(model_url_dir+model_name+".xml",model_path.name,model_path.parent)
utils.download_file(
model_url_dir+model_name+".bin",
model_path.with_suffix(".bin").name,
model_path.parent,
)



..parsed-literal::

model/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml:0%||0.00/474k[0…



..parsed-literal::

model/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.bin:0%||0.00/4.03M[…


Loadthemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Downloadedmodelsarelocatedinafixedstructure,whichindicatesa
vendor,thenameofthemodelandaprecision.

Onlyafewlinesofcodearerequiredtorunthemodel.First,
initializeOpenVINORuntime.Then,readthenetworkarchitectureand
modelweightsfromthe``.bin``and``.xml``filestocompileitforthe
desireddevice.Selectdevicefromdropdownlistforrunninginference
usingOpenVINO.

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




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#InitializeOpenVINORuntime
core=ov.Core()
#Readthenetworkfromafile.
model=core.read_model(model_path)
#LettheAUTOdevicedecidewheretoloadthemodel(youcanuseCPU,GPUaswell).
compiled_model=core.compile_model(model=model,device_name=device.value,config={"PERFORMANCE_HINT":"LATENCY"})

#Gettheinputandoutputnamesofnodes.
input_layer=compiled_model.input(0)
output_layers=compiled_model.outputs

#Gettheinputsize.
height,width=list(input_layer.shape)[2:]

Inputlayerhasthenameoftheinputnodeandoutputlayerscontain
namesofoutputnodesofthenetwork.InthecaseofOpenPoseModel,
thereis1inputand2outputs:PAFsandkeypointsheatmap.

..code::ipython3

input_layer.any_name,[o.any_nameforoinoutput_layers]




..parsed-literal::

('data',['Mconv7_stage2_L1','Mconv7_stage2_L2'])



OpenPoseDecoder
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Totransformtherawresultsfromtheneuralnetworkintopose
estimations,youneedOpenPoseDecoder.Itisprovidedinthe`Open
Model
Zoo<https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/openvino/model_zoo/model_api/models/open_pose.py>`__
andcompatiblewiththe``human-pose-estimation-0001``model.

Ifyouchooseamodelotherthan``human-pose-estimation-0001``youwill
needanotherdecoder(forexample,``AssociativeEmbeddingDecoder``),
whichisavailableinthe`demos
section<https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/openvino/model_zoo/model_api/models/hpe_associative_embedding.py>`__
ofOpenModelZoo.

..code::ipython3

#codefromhttps://github.com/openvinotoolkit/open_model_zoo/blob/9296a3712069e688fe64ea02367466122c8e8a3b/demos/common/python/models/open_pose.py#L135
classOpenPoseDecoder:
BODY_PARTS_KPT_IDS=(
(1,2),
(1,5),
(2,3),
(3,4),
(5,6),
(6,7),
(1,8),
(8,9),
(9,10),
(1,11),
(11,12),
(12,13),
(1,0),
(0,14),
(14,16),
(0,15),
(15,17),
(2,16),
(5,17),
)
BODY_PARTS_PAF_IDS=(
12,
20,
14,
16,
22,
24,
0,
2,
4,
6,
8,
10,
28,
30,
34,
32,
36,
18,
26,
)

def__init__(
self,
num_joints=18,
skeleton=BODY_PARTS_KPT_IDS,
paf_indices=BODY_PARTS_PAF_IDS,
max_points=100,
score_threshold=0.1,
min_paf_alignment_score=0.05,
delta=0.5,
):
self.num_joints=num_joints
self.skeleton=skeleton
self.paf_indices=paf_indices
self.max_points=max_points
self.score_threshold=score_threshold
self.min_paf_alignment_score=min_paf_alignment_score
self.delta=delta

self.points_per_limb=10
self.grid=np.arange(self.points_per_limb,dtype=np.float32).reshape(1,-1,1)

def__call__(self,heatmaps,nms_heatmaps,pafs):
batch_size,_,h,w=heatmaps.shape
assertbatch_size==1,"Batchsizeof1onlysupported"

keypoints=self.extract_points(heatmaps,nms_heatmaps)
pafs=np.transpose(pafs,(0,2,3,1))

ifself.delta>0:
forkptsinkeypoints:
kpts[:,:2]+=self.delta
np.clip(kpts[:,0],0,w-1,out=kpts[:,0])
np.clip(kpts[:,1],0,h-1,out=kpts[:,1])

pose_entries,keypoints=self.group_keypoints(keypoints,pafs,pose_entry_size=self.num_joints+2)
poses,scores=self.convert_to_coco_format(pose_entries,keypoints)
iflen(poses)>0:
poses=np.asarray(poses,dtype=np.float32)
poses=poses.reshape((poses.shape[0],-1,3))
else:
poses=np.empty((0,17,3),dtype=np.float32)
scores=np.empty(0,dtype=np.float32)

returnposes,scores

defextract_points(self,heatmaps,nms_heatmaps):
batch_size,channels_num,h,w=heatmaps.shape
assertbatch_size==1,"Batchsizeof1onlysupported"
assertchannels_num>=self.num_joints

xs,ys,scores=self.top_k(nms_heatmaps)
masks=scores>self.score_threshold
all_keypoints=[]
keypoint_id=0
forkinrange(self.num_joints):
#Filterlow-scorepoints.
mask=masks[0,k]
x=xs[0,k][mask].ravel()
y=ys[0,k][mask].ravel()
score=scores[0,k][mask].ravel()
n=len(x)
ifn==0:
all_keypoints.append(np.empty((0,4),dtype=np.float32))
continue
#Applyquarteroffsettoimprovelocalizationaccuracy.
x,y=self.refine(heatmaps[0,k],x,y)
np.clip(x,0,w-1,out=x)
np.clip(y,0,h-1,out=y)
#Packresultingpoints.
keypoints=np.empty((n,4),dtype=np.float32)
keypoints[:,0]=x
keypoints[:,1]=y
keypoints[:,2]=score
keypoints[:,3]=np.arange(keypoint_id,keypoint_id+n)
keypoint_id+=n
all_keypoints.append(keypoints)
returnall_keypoints

deftop_k(self,heatmaps):
N,K,_,W=heatmaps.shape
heatmaps=heatmaps.reshape(N,K,-1)
#Getpositionswithtopscores.
ind=heatmaps.argpartition(-self.max_points,axis=2)[:,:,-self.max_points:]
scores=np.take_along_axis(heatmaps,ind,axis=2)
#Keeptopscoressorted.
subind=np.argsort(-scores,axis=2)
ind=np.take_along_axis(ind,subind,axis=2)
scores=np.take_along_axis(scores,subind,axis=2)
y,x=np.divmod(ind,W)
returnx,y,scores

@staticmethod
defrefine(heatmap,x,y):
h,w=heatmap.shape[-2:]
valid=np.logical_and(np.logical_and(x>0,x<w-1),np.logical_and(y>0,y<h-1))
xx=x[valid]
yy=y[valid]
dx=np.sign(heatmap[yy,xx+1]-heatmap[yy,xx-1],dtype=np.float32)*0.25
dy=np.sign(heatmap[yy+1,xx]-heatmap[yy-1,xx],dtype=np.float32)*0.25
x=x.astype(np.float32)
y=y.astype(np.float32)
x[valid]+=dx
y[valid]+=dy
returnx,y

@staticmethod
defis_disjoint(pose_a,pose_b):
pose_a=pose_a[:-2]
pose_b=pose_b[:-2]
returnnp.all(np.logical_or.reduce((pose_a==pose_b,pose_a<0,pose_b<0)))

defupdate_poses(
self,
kpt_a_id,
kpt_b_id,
all_keypoints,
connections,
pose_entries,
pose_entry_size,
):
forconnectioninconnections:
pose_a_idx=-1
pose_b_idx=-1
forj,poseinenumerate(pose_entries):
ifpose[kpt_a_id]==connection[0]:
pose_a_idx=j
ifpose[kpt_b_id]==connection[1]:
pose_b_idx=j
ifpose_a_idx<0andpose_b_idx<0:
#Createnewposeentry.
pose_entry=np.full(pose_entry_size,-1,dtype=np.float32)
pose_entry[kpt_a_id]=connection[0]
pose_entry[kpt_b_id]=connection[1]
pose_entry[-1]=2
pose_entry[-2]=np.sum(all_keypoints[connection[0:2],2])+connection[2]
pose_entries.append(pose_entry)
elifpose_a_idx>=0andpose_b_idx>=0andpose_a_idx!=pose_b_idx:
#Mergetwoposesaredisjointmergethem,otherwiseignoreconnection.
pose_a=pose_entries[pose_a_idx]
pose_b=pose_entries[pose_b_idx]
ifself.is_disjoint(pose_a,pose_b):
pose_a+=pose_b
pose_a[:-2]+=1
pose_a[-2]+=connection[2]
delpose_entries[pose_b_idx]
elifpose_a_idx>=0andpose_b_idx>=0:
#Adjustscoreofapose.
pose_entries[pose_a_idx][-2]+=connection[2]
elifpose_a_idx>=0:
#Addanewlimbintopose.
pose=pose_entries[pose_a_idx]
ifpose[kpt_b_id]<0:
pose[-2]+=all_keypoints[connection[1],2]
pose[kpt_b_id]=connection[1]
pose[-2]+=connection[2]
pose[-1]+=1
elifpose_b_idx>=0:
#Addanewlimbintopose.
pose=pose_entries[pose_b_idx]
ifpose[kpt_a_id]<0:
pose[-2]+=all_keypoints[connection[0],2]
pose[kpt_a_id]=connection[0]
pose[-2]+=connection[2]
pose[-1]+=1
returnpose_entries

@staticmethod
defconnections_nms(a_idx,b_idx,affinity_scores):
#Fromallretrievedconnectionsthatsharestarting/endingkeypointsleaveonlythetop-scoringones.
order=affinity_scores.argsort()[::-1]
affinity_scores=affinity_scores[order]
a_idx=a_idx[order]
b_idx=b_idx[order]
idx=[]
has_kpt_a=set()
has_kpt_b=set()
fort,(i,j)inenumerate(zip(a_idx,b_idx)):
ifinotinhas_kpt_aandjnotinhas_kpt_b:
idx.append(t)
has_kpt_a.add(i)
has_kpt_b.add(j)
idx=np.asarray(idx,dtype=np.int32)
returna_idx[idx],b_idx[idx],affinity_scores[idx]

defgroup_keypoints(self,all_keypoints_by_type,pafs,pose_entry_size=20):
all_keypoints=np.concatenate(all_keypoints_by_type,axis=0)
pose_entries=[]
#Foreverylimb.
forpart_id,paf_channelinenumerate(self.paf_indices):
kpt_a_id,kpt_b_id=self.skeleton[part_id]
kpts_a=all_keypoints_by_type[kpt_a_id]
kpts_b=all_keypoints_by_type[kpt_b_id]
n=len(kpts_a)
m=len(kpts_b)
ifn==0orm==0:
continue

#Getvectorsbetweenallpairsofkeypoints,i.e.candidatelimbvectors.
a=kpts_a[:,:2]
a=np.broadcast_to(a[None],(m,n,2))
b=kpts_b[:,:2]
vec_raw=(b[:,None,:]-a).reshape(-1,1,2)

#Samplepointsalongeverycandidatelimbvector.
steps=1/(self.points_per_limb-1)*vec_raw
points=steps*self.grid+a.reshape(-1,1,2)
points=points.round().astype(dtype=np.int32)
x=points[...,0].ravel()
y=points[...,1].ravel()

#Computeaffinityscorebetweencandidatelimbvectorsandpartaffinityfield.
part_pafs=pafs[0,:,:,paf_channel:paf_channel+2]
field=part_pafs[y,x].reshape(-1,self.points_per_limb,2)
vec_norm=np.linalg.norm(vec_raw,ord=2,axis=-1,keepdims=True)
vec=vec_raw/(vec_norm+1e-6)
affinity_scores=(field*vec).sum(-1).reshape(-1,self.points_per_limb)
valid_affinity_scores=affinity_scores>self.min_paf_alignment_score
valid_num=valid_affinity_scores.sum(1)
affinity_scores=(affinity_scores*valid_affinity_scores).sum(1)/(valid_num+1e-6)
success_ratio=valid_num/self.points_per_limb

#Getalistoflimbsaccordingtotheobtainedaffinityscore.
valid_limbs=np.where(np.logical_and(affinity_scores>0,success_ratio>0.8))[0]
iflen(valid_limbs)==0:
continue
b_idx,a_idx=np.divmod(valid_limbs,n)
affinity_scores=affinity_scores[valid_limbs]

#Suppressincompatibleconnections.
a_idx,b_idx,affinity_scores=self.connections_nms(a_idx,b_idx,affinity_scores)
connections=list(
zip(
kpts_a[a_idx,3].astype(np.int32),
kpts_b[b_idx,3].astype(np.int32),
affinity_scores,
)
)
iflen(connections)==0:
continue

#Updateposeswithnewconnections.
pose_entries=self.update_poses(
kpt_a_id,
kpt_b_id,
all_keypoints,
connections,
pose_entries,
pose_entry_size,
)

#Removeposeswithnotenoughpoints.
pose_entries=np.asarray(pose_entries,dtype=np.float32).reshape(-1,pose_entry_size)
pose_entries=pose_entries[pose_entries[:,-1]>=3]
returnpose_entries,all_keypoints

@staticmethod
defconvert_to_coco_format(pose_entries,all_keypoints):
num_joints=17
coco_keypoints=[]
scores=[]
forposeinpose_entries:
iflen(pose)==0:
continue
keypoints=np.zeros(num_joints*3)
reorder_map=[0,-1,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
person_score=pose[-2]
forkeypoint_id,target_idinzip(pose[:-2],reorder_map):
iftarget_id<0:
continue
cx,cy,score=0,0,0#keypointnotfound
ifkeypoint_id!=-1:
cx,cy,score=all_keypoints[int(keypoint_id),0:3]
keypoints[target_id*3+0]=cx
keypoints[target_id*3+1]=cy
keypoints[target_id*3+2]=score
coco_keypoints.append(keypoints)
scores.append(person_score*max(0,(pose[-1]-1)))#-1for'neck'
returnnp.asarray(coco_keypoints),np.asarray(scores)

Processing
----------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

decoder=OpenPoseDecoder()

ProcessResults
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Abunchofusefulfunctionstotransformresultsintoposes.

First,pooltheheatmap.Sincepoolingisnotavailableinnumpy,usea
simplemethodtodoitdirectlywithnumpy.Then,usenon-maximum
suppressiontogetthekeypointsfromtheheatmap.Afterthat,decode
posesbyusingthedecoder.Sincetheinputimageisbiggerthanthe
networkoutputs,youneedtomultiplyallposecoordinatesbyascaling
factor.

..code::ipython3

#2Dpoolinginnumpy(from:https://stackoverflow.com/a/54966908/1624463)
defpool2d(A,kernel_size,stride,padding,pool_mode="max"):
"""
2DPooling

Parameters:
A:input2Darray
kernel_size:int,thesizeofthewindow
stride:int,thestrideofthewindow
padding:int,implicitzeropaddingsonbothsidesoftheinput
pool_mode:string,'max'or'avg'
"""
#Padding
A=np.pad(A,padding,mode="constant")

#WindowviewofA
output_shape=(
(A.shape[0]-kernel_size)//stride+1,
(A.shape[1]-kernel_size)//stride+1,
)
kernel_size=(kernel_size,kernel_size)
A_w=as_strided(
A,
shape=output_shape+kernel_size,
strides=(stride*A.strides[0],stride*A.strides[1])+A.strides,
)
A_w=A_w.reshape(-1,*kernel_size)

#Returntheresultofpooling.
ifpool_mode=="max":
returnA_w.max(axis=(1,2)).reshape(output_shape)
elifpool_mode=="avg":
returnA_w.mean(axis=(1,2)).reshape(output_shape)


#nonmaximumsuppression
defheatmap_nms(heatmaps,pooled_heatmaps):
returnheatmaps*(heatmaps==pooled_heatmaps)


#Getposesfromresults.
defprocess_results(img,pafs,heatmaps):
#Thisprocessingcomesfrom
#https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
pooled_heatmaps=np.array([[pool2d(h,kernel_size=3,stride=1,padding=1,pool_mode="max")forhinheatmaps[0]]])
nms_heatmaps=heatmap_nms(heatmaps,pooled_heatmaps)

#Decodeposes.
poses,scores=decoder(heatmaps,nms_heatmaps,pafs)
output_shape=list(compiled_model.output(index=0).partial_shape)
output_scale=(
img.shape[1]/output_shape[3].get_length(),
img.shape[0]/output_shape[2].get_length(),
)
#Multiplycoordinatesbyascalingfactor.
poses[:,:,:2]*=output_scale
returnposes,scores

DrawPoseOverlays
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Drawposeoverlaysontheimagetovisualizeestimatedposes.Jointsare
drawnascirclesandlimbsaredrawnaslines.Thecodeisbasedonthe
`HumanPoseEstimation
Demo<https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/human_pose_estimation_demo/python>`__
fromOpenModelZoo.

..code::ipython3

colors=(
(255,0,0),
(255,0,255),
(170,0,255),
(255,0,85),
(255,0,170),
(85,255,0),
(255,170,0),
(0,255,0),
(255,255,0),
(0,255,85),
(170,255,0),
(0,85,255),
(0,255,170),
(0,0,255),
(0,255,255),
(85,0,255),
(0,170,255),
)

default_skeleton=(
(15,13),
(13,11),
(16,14),
(14,12),
(11,12),
(5,11),
(6,12),
(5,6),
(5,7),
(6,8),
(7,9),
(8,10),
(1,2),
(0,1),
(0,2),
(1,3),
(2,4),
(3,5),
(4,6),
)


defdraw_poses(img,poses,point_score_threshold,skeleton=default_skeleton):
ifposes.size==0:
returnimg

img_limbs=np.copy(img)
forposeinposes:
points=pose[:,:2].astype(np.int32)
points_scores=pose[:,2]
#Drawjoints.
fori,(p,v)inenumerate(zip(points,points_scores)):
ifv>point_score_threshold:
cv2.circle(img,tuple(p),1,colors[i],2)
#Drawlimbs.
fori,jinskeleton:
ifpoints_scores[i]>point_score_thresholdandpoints_scores[j]>point_score_threshold:
cv2.line(
img_limbs,
tuple(points[i]),
tuple(points[j]),
color=colors[j],
thickness=4,
)
cv2.addWeighted(img,0.4,img_limbs,0.6,0,dst=img)
returnimg

MainProcessingFunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Runposeestimationonthespecifiedsource.Eitherawebcamoravideo
file.

..code::ipython3

#Mainprocessingfunctiontorunposeestimation.
defrun_pose_estimation(source=0,flip=False,use_popup=False,skip_first_frames=0):
pafs_output_key=compiled_model.output("Mconv7_stage2_L1")
heatmaps_output_key=compiled_model.output("Mconv7_stage2_L2")
player=None
try:
#Createavideoplayertoplaywithtargetfps.
player=utils.VideoPlayer(source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
#Startcapturing.
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)

processing_times=collections.deque()

whileTrue:
#Grabtheframe.
frame=player.next()
ifframeisNone:
print("Sourceended")
break
#IftheframeislargerthanfullHD,reducesizetoimprovetheperformance.
scale=1280/max(frame.shape)
ifscale<1:
frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)

#Resizetheimageandchangedimstofitneuralnetworkinput.
#(seehttps://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
input_img=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
#Createabatchofimages(size=1).
input_img=input_img.transpose((2,0,1))[np.newaxis,...]

#Measureprocessingtime.
start_time=time.time()
#Getresults.
results=compiled_model([input_img])
stop_time=time.time()

pafs=results[pafs_output_key]
heatmaps=results[heatmaps_output_key]
#Getposesfromnetworkresults.
poses,scores=process_results(frame,pafs,heatmaps)

#Drawposesonaframe.
frame=draw_poses(frame,poses,0.1)

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()

_,f_width=frame.shape[:2]
#meanprocessingtime[ms]
processing_time=np.mean(processing_times)*1000
fps=1000/processing_time
cv2.putText(
frame,
f"Inferencetime:{processing_time:.1f}ms({fps:.1f}FPS)",
(20,40),
cv2.FONT_HERSHEY_COMPLEX,
f_width/1000,
(0,0,255),
1,
cv2.LINE_AA,
)

#Usethisworkaroundifthereisflickering.
ifuse_popup:
cv2.imshow(title,frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(".jpg",frame,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateanIPythonimage.
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook.
display.clear_output(wait=True)
display.display(i)
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifplayerisnotNone:
#Stopcapturing.
player.stop()
ifuse_popup:
cv2.destroyAllWindows()

Run
---

`backtotop⬆️<#table-of-contents>`__

RunLivePoseEstimation
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set``flip=True``when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,set
``use_popup=True``.

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
server(forexample,Binder),thewebcamwillnotwork.Popupmode
maynotworkifyourunthisnotebookonaremotecomputer(for
example,Binder).

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
willwork.Youcanskipfirst``N``framestofastforwardvideo.

Runtheposeestimation:

..code::ipython3

USE_WEBCAM=False
cam_id=0
video_file="https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true"
source=cam_idifUSE_WEBCAMelsevideo_file

additional_options={"skip_first_frames":500}ifnotUSE_WEBCAMelse{}
run_pose_estimation(source=source,flip=isinstance(source,int),use_popup=False,**additional_options)



..image::pose-estimation-with-output_files/pose-estimation-with-output_22_0.png


..parsed-literal::

Sourceended


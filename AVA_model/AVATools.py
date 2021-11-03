# ================================================================
#
#   File name   : AVATools.py
#   Author      : Thi TRAN
#   Created date: 2021-06-01
#   GitHub      : https://github.com/ThiTranWork/AVA_Project_02
#   Description : All tools for Detect, sort and count objects, detect and alert when traffic violations happen
#
# ===============================================================


from IPython.display import clear_output

import os
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir ("/workspace/YOLOV3_M")
import shutil

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *

import time
import csv
import math

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from clipwriter import ClipWriter



def MyTruncatedData (mydataframe, storeframesize,current_frameIndex):
    if mydataframe.empty==False:
        first_frame= mydataframe.iloc[0, mydataframe.columns.get_loc('frame_index')]
        if current_frameIndex >= (storeframesize+first_frame):
            mydataframe= mydataframe[mydataframe["frame_index"]>(current_frameIndex-storeframesize)]
    return mydataframe


def Zone_cacl(bounding_box, referenceZone):
    #caculate the zone where the object belongs to
    zone =(-1)* np.ones((bounding_box.shape[0], 1)) #-1 out of interest area; zone's dimension = nr of objects
    x= referenceZone[0]
    y= referenceZone[1]
    ref_1= np.transpose(referenceZone)

    for i in range (bounding_box.shape[0]):
        x_test=(bounding_box[i][0]+bounding_box[i][2])/2
        y_test=bounding_box[i][3]
        test= [x_test,y_test]
        if referenceZone.shape[1]==2:  # input is a line
            if (x[0]!=x[1]):
                a = (y[:-1]-y[1:])/(x[:-1]-x[1:])
                b = y[:-1] - np.multiply(a,x[:-1])
                if ((x_test >= np.min(x)) & (x_test <= np.max(x))):
                    if (y_test > (a[0]*x_test+b[0])):
                        zone [i][0]=1
                    else:
                        zone [i][0]=0
            else:
                if ((y_test >= np.min(y)) & (y_test <= np.max(y))):
                    if (x_test > x[0]):
                        zone [i][0]=1
                    else:
                        zone [i][0]=0

        elif referenceZone.shape[1]>=3: # input is a polygon
            vector=test- ref_1
            vectorLength= np.linalg.norm(vector,axis=1)
            unit_vector_1= vector/vectorLength[:,None]
            unit_vector_2= np.roll(unit_vector_1, -1, axis=0)
            dot_product= np.sum(unit_vector_1*unit_vector_2,axis=1)
            angle = np.arccos(dot_product)
            total_angle= np.sum(angle)
            error =total_angle - 2*np.pi
            if abs(error)<1e-6:
                zone [i][0]=1 #object inside the polygon
            else: 
                zone [i][0]=0  #object outside the polygon

    return zone  

def ObjCount_process (mydataframe,output_path, current_image= [],\
                      class_interest=[],cropping=False,cross_line=[]):
    obj_counter=[0,0]
    if mydataframe.empty==False:
        df = mydataframe[mydataframe['class_object'].isin(class_interest)]
        for unique_tracking_id in df.tracking_id.unique():
            if (len(df[df.tracking_id==unique_tracking_id].frame_index.tolist())==2): #check only objects appears in 2 frames
                df_object= df.query ('tracking_id==@unique_tracking_id') # pairs of same ID, different frame
                bbox_veh= np.array(list(df_object.bbox)).astype(int)
                cross_zone= Zone_cacl(bbox_veh,cross_line)
                if (cross_zone==[[0],[1]]).all(): #cross from zone0 to zone 1
                    obj_counter[0]+=1
                    if cropping:
                        cropping_zone = np.where(bbox_veh[1]<0, 0, bbox_veh[1]) # assign 0 if the box is out of image size, take the lastest image
                        cropping_zone = np.where(bbox_veh[0]<0, 0, bbox_veh[0])
                        cropped_image = current_image[cropping_zone[1]:(cropping_zone[3]+17),cropping_zone[0]:cropping_zone[2],:]  
                        x = cv2.imwrite(os.path.join(output_path,\
                                                "CroppedImage_CarIndex{:.1f}.jpg".format(unique_tracking_id)),\
                                                cropped_image) 
                elif (cross_zone==[[1],[0]]).all(): #cross from zone1 to zone 0
                    obj_counter[1]+=1
                    
    return obj_counter

def UpdateStationaryStatus (mydataframe, current_frameIndex,  stationaryDist, class_interest=[], ):

    if mydataframe.empty==False:
        df = mydataframe[mydataframe['class_object'].isin(class_interest)]
        for unique_tracking_id in df.tracking_id.unique():
            if (len(df[df.tracking_id==unique_tracking_id].frame_index.tolist())==2): #check only objects appears in 2 frames
                df_object= df.query ('tracking_id==@unique_tracking_id')
                indexOb = df_object.index.tolist() # find the index of the object in two frames
                centroidPairs= np.array(list(df_object.centroid)).astype(int)
                dist= math.dist(centroidPairs[0,:],(centroidPairs[1,:]))
                if dist < stationaryDist :                
                    mydataframe.at[indexOb[1], 'stationaryTrack'] = 1+ mydataframe.at[indexOb[0], 'stationaryTrack']
                # else:
                #     mydataframe.at[indexOb[1], 'stationaryTrack']=0 # reset the stationary tracking
    return mydataframe



def initialization(video_output_path):
    #create directory
    output_path, _ = os.path.split(video_output_path)
    #clean up directory    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)
    return output_path




def Object_tracking(video_path, video_output_path, fps=20, show=False,  \
                     vehicle_counting=False, cropping_line=[], \
                     per_counting=False, person_crossing_line=[], \
                     lineCrossingDect=False, vehicle_crossing_line=[],\
                     PedXViolationDect=False, PedX=[],\
                     stationaryObjectAlarm=False, stationaryDist=1, timeTrackinginSecond=0.15):
    
    Yolo = Load_Yolo_model()
    TRACK_ONLY = ["person","car", "bus", "truck", "suitcase"]

    #global parameters for AVA
    mydataframe = pd.DataFrame()
    output_path = initialization(video_output_path)
    
    
    #initial parameters for tracking
    CLASSES=YOLO_COCO_CLASSES
    rectangle_colors=(255,0,0)
    input_size=416
    score_threshold=0.3
    iou_threshold=0.45
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initial parametersfor deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #initial parameters for AVA
    storeframesize=2 # store only the current and the previous frame
    counter_car=0
    counter_pp_Z01=0
    counter_pp_Z10=0

    
    #parameters to save violation clip
    lengthClip = 10 #clip length in seconds
    bufferSize = int(lengthClip/2*fps)
    bufferSize=100 #test for stationary objects


    #initialized for PedX violation
    consecFramesPedX=0
    PedXClip= ClipWriter(bufferSize)
    pedXTrigger=0 #count nr of violations

    #initialized for car line crossing violation
    lcvConsecFrames=0
    lcvClip= ClipWriter(bufferSize)
    vclTrigger=0 

    #initialized for object stationary tracking
    ostConsecFrames=0
    ostClip= ClipWriter(bufferSize)
    ostTrigger=0 

    if stationaryObjectAlarm:
        StationaryFile="{}/StationaryObjects.csv".format(output_path)
        stationaryObjects_info= ['frame_index', 'tracking_id', 'class_object', "bbox", "centroid", "timestamp", "stationaryTrack" ]
        ostAlertSave_df= pd.DataFrame(columns=stationaryObjects_info)
        ostAlertSave_df.to_csv(StationaryFile,mode='w', encoding='utf-8', index=False)


    objects_info = ['frame_index', 'tracking_id', 'class_object', "bbox", "centroid", "timestamp" ]
    fileName="{}/Objects.csv".format(output_path)
    new_dict =pd.DataFrame(columns=objects_info)
    new_dict.to_csv(fileName,mode='w', encoding='utf-8', index=False)

    
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fontSize=int(0.002*height)
    fps1 = int(vid.get(cv2.CAP_PROP_FPS))   # to read fps  
    codec = cv2.VideoWriter_fourcc(*'XVID')  # codec should be compatible to the video format, for the MP4

    if (fps ==0 | fps > 300): # to make sure the fps is read correctly
        out = cv2.VideoWriter(video_output_path, codec, fps, (width, height)) 
    else:
        out = cv2.VideoWriter(video_output_path, codec, fps1, (width, height))

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    

    countTest=0

    while True:
        success, frame = vid.read()
        print(".")

        # countTest+=1
        # if countTest<3500:
        #     continue
        
        if success:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        else:
            break

        timestamp = datetime.datetime.now()

        current_frameIndex = int(vid.get(cv2.CAP_PROP_POS_FRAMES)) #get the current frame index
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)        

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, x+width, y+height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(TRACK_ONLY) !=0 and NUM_CLASS[int(bbox[5])] in TRACK_ONLY or len(TRACK_ONLY) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        
        # Obtain info from the tracks
        tracked_bboxes = []
        new_dict =pd.DataFrame(columns=objects_info)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            centroid= np.array([(bbox[0]+bbox[2])/2,  (bbox[1]+bbox[3])/2]) #x,y
            stationaryTrack=0
            new_row = {'frame_index':current_frameIndex,'tracking_id':tracking_id, 'class_object':class_name,\
                        "bbox":bbox.tolist(), "centroid":centroid.tolist(), "timestamp":timestamp.strftime("%Y%m%d_%H%M%S"), "stationaryTrack":stationaryTrack} #info of every object in the frame
            new_dict=new_dict.append(new_row,ignore_index=True)
            mydataframe = mydataframe.append(new_row,ignore_index=True)

        


        mydataframe = MyTruncatedData (mydataframe,storeframesize,current_frameIndex) #limit the size of storing data to counting 
        
        # Store the whole data to the csv file
        new_dict.to_csv(fileName, mode='a', columns = objects_info, header=False, encoding='utf-8', index=False)
        

        image = original_frame
        image = cv2.putText(image, "Frame: {:.1f} ".format(current_frameIndex), (100, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (0, 102, 255), 2)

        ## Proceesing:

        if vehicle_counting:
            # Counting nr of vehicles + crop vehicle image 

            counter_car+= ObjCount_process (mydataframe, output_path,current_image= original_frame,\
                                        class_interest= ["car", "bus", "truck"],cropping=True,cross_line=cropping_line)[0]       
            # show/imprint nr of vehicles        
            start_point_car = tuple(cropping_line[:,0])
            end_point_car = tuple(cropping_line[:,1])
            org_car=tuple(cropping_line[:,0]) 
            image = cv2.putText(image, "NrCar: {:.1f} ".format(counter_car), org_car, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (0, 206, 0), 2)
            image = cv2.line(image, start_point_car, end_point_car, (0, 206, 0), 2) #imprint the cropping line in de video
        
        if per_counting:
            # Counting nr of pedestrians in two directions
            counter_pp_Z01 += ObjCount_process (mydataframe, output_path,current_image= original_frame,\
                                        class_interest= ["person"],cropping=False,cross_line=person_crossing_line)[0]     
            counter_pp_Z10 += ObjCount_process (mydataframe, output_path,current_image= original_frame,\
                                        class_interest= ["person"],cropping=False,cross_line=person_crossing_line)[1]
            
            #showing/imprint person crossing line+ counting nr in both directions           
            start_point_person = tuple(person_crossing_line[:,0])
            end_point_person = tuple(person_crossing_line[:,1])
            org_pp1= tuple(person_crossing_line[:,1] - [int(0.16*height), -int(0.008*height)])
            image = cv2.putText(image, "NrPer:{:.1f} ".format(counter_pp_Z01), org_pp1, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (76,0, 153), 2)
            image = cv2.line(image, start_point_person, end_point_person, (76,0, 153), 2) #imprint the person crossing line in de video
        
            start_point_person = tuple(person_crossing_line[:,0]+[int(0.03*height),0])
            end_point_person = tuple(person_crossing_line[:,1]+[int(0.03*height),0])
            org_pp2= tuple(person_crossing_line[:,1] +[int(0.03*height), int(0.008*height)])
            image = cv2.putText(image, "NrPer:{:.1f} ".format(counter_pp_Z10), org_pp2, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (128, 0, 0), 2)     
            image = cv2.line(image, start_point_person, end_point_person, (128, 0, 0), 2) #imprint the person crossing line in de video

        if lineCrossingDect:
            if ObjCount_process (mydataframe, output_path,current_image= original_frame,class_interest= ["car", "bus", "truck"],cropping=False,cross_line=vehicle_crossing_line)[0] ==1:       
                #reset counting frame
                lcvConsecFrames=0
                vclTrigger +=1
                # if we are not already recording, start recording
                if not lcvClip.recording:                
                    p = "{}/lcvClip{}.mp4".format(output_path,timestamp.strftime("%Y%m%d-%H%M%S"))
                    # p="{0}/test.mp4".format(output_path)
                    lcvClip.start(p, codec, fps)
                    print("lcvClip writing")
            # increment the number of consecutive frames 
            lcvConsecFrames += 1
            # update the key frame clip buffer
            lcvClip.update(image)
            # if reaching a threshold on consecutive number of frames with no action, stop recording the clip
            if lcvClip.recording and lcvConsecFrames == bufferSize:
                lcvClip.finish()

            #showing/imprint   
            vcl=np.transpose(vehicle_crossing_line)
            vcl_tup=map(tuple, vcl)
            vcl_tuples = tuple(vcl_tup)
            image = cv2.line(image, vcl_tuples[0], vcl_tuples[1], (128, 0, 0), 2) #imprint the crossing line in de video 
            image = cv2.putText(image, "Nr car crossing line Violation:{:.1f} ".format(vclTrigger), vcl_tuples[0], cv2.FONT_HERSHEY_COMPLEX_SMALL,fontSize, (150, 206, 105), 2)

        if PedXViolationDect:
            eventTrigger=False
            if mydataframe.empty==False:
                #check if person in the PedX
                df= mydataframe[(mydataframe.frame_index== current_frameIndex) & (mydataframe.class_object == 'person')]
                if df.empty==False:
                    bbox_ppl= np.array(list(df.bbox)).astype(int)
                    cross_zonet= Zone_cacl(bbox_ppl,PedX)
                    if np.sum(cross_zonet)==0:
                        PedXLock= 0
                    else:
                        PedXLock= 1
                else:
                    PedXLock= 0 #no person in the current frame
            
                #check car violation: -obtain subset of mydataframe, the ID unique in current frame
                #when consider violation: PedXLock= True, vehicle= in the PedX, in the previous frame, vehicle : outside the PedX
                vehdf = mydataframe[mydataframe['class_object'].isin(["car", "bus", "truck"])]
                if vehdf.empty==False:
                    bbox_veht= np.array(list(vehdf.bbox)).astype(int)
                    cross_zonet= Zone_cacl(bbox_veht,PedX)
                    vehdf1=vehdf.copy()
                    vehdf1.loc[:,"ZoneIn"]=cross_zonet #vehdf1 contains zone info
                    vehIDZoneIn= vehdf1.loc[(vehdf1["frame_index"] == current_frameIndex) & (vehdf1["ZoneIn"] == 1), "tracking_id"].values
                    if len(vehIDZoneIn)!=0:
                        #if there is a vehicle in the PedX
                        vehdf2 = vehdf1[vehdf1["tracking_id"].isin(vehIDZoneIn)] #extract out all the data of vehicles in PedX
                        print(vehdf2)
                        for unique_tracking_id in vehdf2.tracking_id.unique():
                            vehdf3= vehdf2.loc[(vehdf1["frame_index"] == (current_frameIndex-1))] #data of the previous frame
                            if unique_tracking_id in vehdf3.values:
                                zoneInPreviousFrame= int(vehdf3["ZoneIn"].values[0])
                                print(zoneInPreviousFrame)
                            else:
                                zoneInPreviousFrame=0
                                print("this vehicle does not appear in previous frame")
                            if ((PedXLock ==1)& (zoneInPreviousFrame==0)):
                                eventTrigger=True
                                print ("trigger")
                                #reset counting frame
                            else:
                                print ("do nothing")
            
                if eventTrigger:
                    pedXTrigger+=1        
                    consecFramesPedX=0
                    # if we are not already recording, start recording
                    if not PedXClip.recording:                
                        p = "{}/PedXClip{}.mp4".format(output_path,timestamp.strftime("%Y%m%d-%H%M%S"))
                        PedXClip.start(p, codec, fps)
                        print("PedXClip writing")

            # increment the number of consecutive frames 
            consecFramesPedX += 1
            # update the key frame clip buffer
            PedXClip.update(image)
            # if reaching a threshold on consecutive number of frames with no action, stop recording the clip
            if PedXClip.recording and consecFramesPedX == bufferSize:
                PedXClip.finish()
           
            #showing/imprint 
            image= ImprintShape (image, PedX) #show the zone boundary

            PedXt=np.transpose(PedX)
            PedXtm=np.append(PedXt, [PedXt[0,:]],axis=0)
            PedXt_tup = map(tuple, PedXtm)
            PedXt_tuples = tuple(PedXt_tup)
            print('test')
            image = cv2.putText(image, "Nr PedX Violation: {:.1f} ".format(pedXTrigger), PedXt_tuples[0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (0, 206, 105), 2)
            print('test2')

# image = cv2.putText(image, "Frame: {:.1f} ".format(current_frameIndex), (100, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontSize, (0, 102, 255), 2)(0, 206, 105)
        if stationaryObjectAlarm: 
            ost_eventTrigger = False
            ostAlert_df = pd.DataFrame()
            
            if mydataframe.empty==False:
                ost_df= UpdateStationaryStatus(mydataframe,current_frameIndex,stationaryDist, class_interest=["car", "bus", "truck", "person", "suicase"])
                frame_tracking= int(timeTrackinginSecond*fps) #min number of frames that the object stationary is alert 
                ostAlert_df = ost_df.loc[(ost_df['frame_index'] == current_frameIndex) & (ost_df['stationaryTrack'] == frame_tracking)]
                if ostAlert_df.empty ==False:
                    print("Trigger")
                    ost_eventTrigger=True
                
                if ost_eventTrigger:
                    ostTrigger +=1
                    ostConsecFrames=0
                    # if we are not already recording, start recording
                    if not ostClip.recording:                
                        ost = "{}/StationaryTrackingClip{}.mp4".format(output_path,timestamp.strftime("%Y%m%d-%H%M%S"))
                        ostClip.start(ost, codec, fps)
                        print("ostClip writing")
                    #save to output file
                    ostAlertSave_df = ost_df.loc[(ost_df['frame_index'] == current_frameIndex) & (ost_df['stationaryTrack'] >= frame_tracking)]
                    ostAlertSave_df.to_csv(StationaryFile, mode='a', columns = stationaryObjects_info, header=False, encoding='utf-8', index=False)                
                
            # increment the number of consecutive frames 
            ostConsecFrames += 1
            ostClip.update(image)
            
            # if reaching a threshold on consecutive number of frames with no action, stop recording the clip
            if ostClip.recording and ostConsecFrames == bufferSize:
                ostClip.finish()
                    
                    
        
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True) # draw detection on frame
        
        if video_output_path != '': # To write out video  
            out.write(image)

        
        if show:
            
            fig = plt.figure(figsize=(8, 8*height/width), dpi=160)        
            clear_output(wait=True)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            plt.imshow(image)
            plt.show()
   

                

    
    if PedXClip.recording: # in case of in the middle of recording a clip, wrap it up
	    PedXClip.finish()
    if lcvClip.recording: # in case of in the middle of recording a clip, wrap it up
	    lcvClip.finish() 
    if ostClip.recording: # in case of in the middle of recording a clip, wrap it up
	    ostClip.finish() 

def RefX_visualization (video_path, shapeCoord, grid_size= 100):
    # the reference zone for visualization on the image
    # shapeCoord is 2D array consisting x, y coordinate

    x= shapeCoord[0] 
    y= shapeCoord[1]
    
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    success, frame = vid.read()

    if success: 
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame[0:height:grid_size] = 1 #write grid
        original_frame[:,0:width:grid_size] = 1

        image= ImprintShape (original_frame, shapeCoord)
        
        fig = plt.figure(figsize=(16, 16*height/width), dpi=160)
        plt.imshow(original_frame)
        # plt.plot(x,y)

    else:
         print ('video invalid')

def ImprintShape (image, shapeCoord):          #showing/imprint on the image
    shapeCoordt=np.transpose(shapeCoord)
    shapeCoordtm=np.append(shapeCoordt, [shapeCoordt[0,:]],axis=0)
    shapeCoordt_tup = map(tuple, shapeCoordtm)
    shapeCoordt_tuples = tuple(shapeCoordt_tup)
    for pnr in range (0,len(shapeCoordtm)-1):
        image = cv2.line(image, shapeCoordt_tuples[pnr],shapeCoordt_tuples[pnr+1], (128, 0, 0), 2) #imprint the person crossing line in de video 



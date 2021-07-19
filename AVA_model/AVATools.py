# +
from IPython.display import clear_output

import os
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

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


mydataframe = pd.DataFrame()
Yolo = Load_Yolo_model()

# -

def MyTruncatedData (storeframesize,current_frameIndex):
    global mydataframe
    if mydataframe.empty==False:
        first_frame= mydataframe.iloc[0, mydataframe.columns.get_loc('frame_index')]
        if current_frameIndex >= (storeframesize+first_frame):
            mydataframe= mydataframe[mydataframe["frame_index"]>(current_frameIndex-storeframesize)]


## Caculate the zone where the object belongs to
def Zone_cacl(bounding_box, referenceZone):
    zone =(-1)* np.ones((bounding_box.shape[0], 1)) #-1 out of interest area
    for i in range (bounding_box.shape[0]):
        if referenceZone.shape[1]==2:  # input is a line
            x= referenceZone[0]
            y= referenceZone[1]
            ## x_test, y_test = the middle point of the ground line in bbox
            x_test=(bounding_box[i][0]+bounding_box[i][2])/2
            y_test=bounding_box[i][3]
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

    return zone  


def ObjCount_process (mydataframe,output_path, current_image= [],\
                      class_interest=[],cropping=False,cross_line=[]):
    obj_counter=0;
    if mydataframe.empty==False:
        df = mydataframe[mydataframe['class_object'].isin(class_interest)]
        for unique_tracking_id in df.tracking_id.unique():
            if (len(df[df.tracking_id==unique_tracking_id].frame_index.tolist())==2): #check only objects appears in 2 frames
                df_object= df.query ('tracking_id==@unique_tracking_id')
                bbox_veh= np.array(list(df_object.bbox)).astype(int)
                cross_zone= Zone_cacl(bbox_veh,cross_line)
                if ((cross_zone==[[1],[0]]).all() | (cross_zone==[[0],[1]]).all()):
                    obj_counter +=1
                    if cropping:
                        cropping_zone = np.where(bbox_veh[1]<0, 0, bbox_veh[1]) # assign 0 if the box is out of image size, take the lastest image
                        cropping_zone = np.where(bbox_veh[0]<0, 0, bbox_veh[0])

                        cropped_image = current_image[cropping_zone[1]:(cropping_zone[3]+17),cropping_zone[0]:cropping_zone[2],:]  

                        x = cv2.imwrite(os.path.join(output_path,\
                                                "CroppedImage_CarIndex{:.1f}.jpg".format(unique_tracking_id)),\
                                                cropped_image) 

    return obj_counter


def Object_tracking(video_path, video_output_path, fps, show=False,cropping_line=[]):
    
    Track_only = ["person","car", "bus", "truck"]

    ## global parameters for AVA
    global mydataframe
    
    ## create directory
    output_path, tail = os.path.split(video_output_path)
    
    print (output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)
    
    ## initial parameters for tracking
    CLASSES=YOLO_COCO_CLASSES
    rectangle_colors=(255,0,0)
    input_size=416
    score_threshold=0.3
    iou_threshold=0.45
    max_cosine_distance = 0.7
    nn_budget = None
    
    ## initial parametersfor deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    ## initial parameters for AVA
    storeframesize=2
    counter_car=0
    counter_pp_dir1=0
    counter_pp_dir2=0
    
    
    times, times_2 = [], []
    

    
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
        
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
        print('No input file')

    ## by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    codec = cv2.VideoWriter_fourcc(*'mp4v')  #codec should be compatible to the video format, for the MP4

    if (fps ==0 | fps > 300): # to make sure the fps is read correctly
        out = cv2.VideoWriter(video_output_path, codec, 30, (width, height)) 
    else:
        out = cv2.VideoWriter(video_output_path, codec, fps, (width, height))

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    
    while True:
        success, frame = vid.read()     
        
        if success:  
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        else:
            break

        current_frameIndex = int(vid.get(cv2.CAP_PROP_POS_FRAMES)) #get the current frame index
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        

        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        ## Extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        ## Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        ## Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        
        ## Obtain info from the tracks
        tracked_bboxes = []
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() #Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id #Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) #Structure data, that we could use it with our draw_bbox function
            
            new_row = {'frame_index':current_frameIndex,'tracking_id':tracking_id, 'class_object':class_name,\
                        "bbox":bbox.tolist()} #info of every object in the frame
            mydataframe = mydataframe.append(new_row,ignore_index=True)
                   
        MyTruncatedData (storeframesize,current_frameIndex) #limit the size of storing data to counting 
        
        ## Counting nr of vehicles + crop vehicle image 
        counter_car+= ObjCount_process (mydataframe, output_path,current_image= original_frame,\
                                        class_interest= ["car", "bus", "truck"],cropping=True,cross_line=cropping_line)        
        
        
        ## Draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
            
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]
        start_point = tuple(cropping_line[:,0])
        end_point = tuple(cropping_line[:,1])
        ms = sum(times)/len(times)*1000
        fps1 = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        image = cv2.putText(image, "Frame: {:.1f} ".format(current_frameIndex), (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (102, 102, 255), 2)
        image = cv2.putText(image, "NrCar: {:.1f} ".format(counter_car), (0, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)
        image = cv2.line(image, start_point, end_point, (0, 255, 0), 5) #imprint the cropping line in de video
        
        if video_output_path != '': out.write(image)
        if show:

            fig = plt.figure(figsize=(8, 8*height/width), dpi=160)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            clear_output(wait=True)
            plt.imshow(image)
            plt.plot(cropping_line[0],cropping_line[1]) #draw the cropping line
            plt.show()


def PedX_visualization(video_path, zone):
# PedX visualization on the image

    x= np.append(zone[0],zone[0][0]) #to make easy drawing
    y= np.append(zone[1],zone[1][0])

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    success, frame = vid.read()
    if success: 
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(8, 8*height/width), dpi=80)
        plt.imshow(original_frame)
        # define the check line
        x_middle = ((x[:-1]+x[1:])/2)
        y_middle = ((y[:-1]+y[1:])/2)
        plt.plot(x,y)
        plt.plot([x_middle[1],x_middle[3]],[y_middle[1],y_middle[3]])
        plt.text(250,300,'zone1')
        plt.text(600,300,'zone2')
        plt.text(500,180,'zone0')

    else:
         print ('video invalid')


def focal_line_visualization (video_path, cross_line):
# PedX visualization on the image

    x= cross_line[0] 
    y= cross_line[1]

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    success, frame = vid.read()
    if success: 
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(16, 16*height/width), dpi=160)
        plt.imshow(original_frame)
        plt.plot(x,y)

    else:
         print ('video invalid')

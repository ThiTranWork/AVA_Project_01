# +
# ================================================================
#
#   File name   : AVA_main.py
#   Author      : Thi TRAN
#   Created date: 2021-06-01
#   GitHub      : https://github.com/ThiTranWork/AVA_Project_02
#   Description : Main for Detect, sort and count objects, detect and alert when traffic violations happen
#
# ===============================================================

import AVATools
import clipwriter

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import cv2



# +
## part 1 - Settings and Calibration

import importlib


# input and output path

#video_input_path  = "/workspace/AVA_model/Hasselt_4.mp4"
#video_output_path = "/workspace/AVA_model/OutResults_Hasselt_4/Hasselt4_out.avi" 
#fps=10

# video_input_path  = "/workspace/AVA_model/Hasselt_4 _Full_10fps.mp4"
#video_output_path = "/workspace/AVA_model/OutResults_Hasselt_4/Hasselt_4 _Full_10fpsOut.avi" 
#fps=10


# ## test for the object stationary

video_input_path   = "/workspace/AVA_model/Hasselt_Stationary_Objects_anonymized.mkv"
video_output_path = "/workspace/AVA_model/OutResults_Hasselt_Stationary_Objects/Hasselt_SO_A_out.mp4" 
fps=30


## Test for person counting; vehicle counting, PedX violation, vehicle solid line violation
## Define position of a line: a line from left to right [(x1,x2), (y1,y2)]
## Define position of a zone: all points in series [(x1,x2,x3,...), (y1,y2, y3,...)]

# person_crossing_line = np.array([(0,1200), (950,400)]) # Hasselt_123  _1080 size;
# vehicle_cropping_line = np.array([(850,1600), (300,600)]) # Hasselt_123; 
# vehicle_crossing_line= vehicle_cropping_line  # Hasselt_123
# PedX= np.array([(180, 750,1450,530 ), (550, 350,600, 1100)])  # Hasselt_123

# person_crossing_line = np.array([(0,1600), (1100,600)]) # Hasselt_123 _full size;
# vehicle_cropping_line = np.array([(1000,2000), (400,780)]) # Hasselt_123; 
# vehicle_crossing_line= vehicle_cropping_line   # Hasselt_123
# PedX= np.array([(150, 1100,1900, 780), (750, 420,780, 1410)])  # Hasselt_123


person_crossing_line = np.array([(50,500), (300,200)]) # anonymous file 480size;
vehicle_cropping_line = np.array([(310,600), (150,260)]) # Hasselt_123; 
vehicle_crossing_line= vehicle_cropping_line  # Hasselt_123
PedX= np.array([(100, 500,950, 400), (360, 220,410, 710)])  # Hasselt_123

# person_crossing_line = np.array([(50,500), (300,200)]) # Hasselt_123 - 480size;
# vehicle_cropping_line = np.array([(310,600), (150,260)]) # Hasselt_123; 
# vehicle_crossing_line= vehicle_cropping_line  # Hasselt_123
# PedX= np.array([(52, 330,600, 220), (250, 140,260, 450)])  # Hasselt_123

## to visuallize the line or zone on the image:

AVATools.RefX_visualization (video_input_path, person_crossing_line,grid_size= 200 ) #to visualize the person_crossing_line
AVATools.RefX_visualization (video_input_path, PedX, grid_size= 200) #to visualize the PEDX 
AVATools.RefX_visualization (video_input_path, vehicle_cropping_line, grid_size= 200 ) 


# -
#turn on option "show" to view results
importlib.reload(AVATools)
AVATools.Object_tracking(video_input_path, video_output_path,fps=fps, show=False, savingOpt=True,\
                        vehicle_counting=True, cropping_line=vehicle_cropping_line , \
                        per_counting=False, person_crossing_line=person_crossing_line , \
                        lineCrossingDect=False, vehicle_crossing_line=vehicle_crossing_line ,\
                        PedXViolationDect=False, PedX=PedX,\
                        stationaryObjectAlarm=True, stationaryDist=15, timeTrackinginSecond=60)

fps


width
height




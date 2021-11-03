# +
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


## input and output path

# video_input_path  = "/workspace/AVA_model/Hasselt_4.mp4"
# video_output_path = "/workspace/AVA_model/OutResults_Hasselt_4/Hasselt4_out.mp4" 
# fps=20

## test for the object stationary

video_input_path   = "/workspace/AVA_model/Hasselt_Stationary_Objects_anonymized.mkv"
video_output_path = "/workspace/AVA_model/OutResults_Hasselt_Stationary_Objects/Hasselt_SO_A_out.mp4" 
fps=30


## Test for person counting; vehicle counting, PedX violation, vehicle solid line violation
## Define position of a line: a line from left to right [(x1,x2), (y1,y2)]
## Define position of a zone: all points in series [(x1,x2,x3,...), (y1,y2, y3,...)]

person_crossing_line = np.array([(0,1600), (1100,600)]) # Hasselt_123;
vehicle_cropping_line = np.array([(1000,2000), (400,780)]) # Hasselt_123; 
vehicle_crossing_line= np.array([(1300,2000), (300,780)])  # Hasselt_123
PedX= np.array([(150, 1100,1900, 780), (750, 420,780, 1410)])  # Hasselt_123


## to visuallize the line or zone on the image:

# AVATools.RefX_visualization (video_input_path, person_crossing_line,grid_size= 200 ) #to visualize the person_crossing_line
# AVATools.RefX_visualization (video_input_path, PedX, grid_size= 200) #to visualize the PEDX 



# -
#turn on option "show" to view results
importlib.reload(AVATools)
AVATools.Object_tracking(video_input_path, video_output_path,fps=fps, show=False,  \
                        vehicle_counting=False, cropping_line=vehicle_cropping_line , \
                        per_counting=False, person_crossing_line=person_crossing_line , \
                        lineCrossingDect=False, vehicle_crossing_line=vehicle_crossing_line ,\
                        PedXViolationDect=False, PedX=PedX,\
                        stationaryObjectAlarm=True, stationaryDist=15, timeTrackinginSecond=60)



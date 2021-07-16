# +
import AVATools

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import cv2



# +
## define the PedX
# PedX=np.array([(100,380,780,720), (340,220,250,400)]) # first row is x, second is y

# AVATools.PedX_visualization (video_input_path, PedX)
# -

def violation_process():
#     print(object_tracking.tracked_bboxes_history[-1])
    pass


# +
## part 1 - Detection

import importlib
importlib.reload(AVATools)

## input and output path
video_input_path   = "/workspace/AVA_model/Dubai_1.ts"
video_output_path = "/workspace/AVA_model/OutResults_Dubai_1/Dubai_1_out.mp4" 
fps=20

## define position of cropping: a line from left to right [(x1,x2), (y1,y2)]

# vehicle_cropping_line = np.array([(330,770), (200,260)]) 
## test with videos from Vitronics
vehicle_cropping_line = np.array([(330,770), (210,250)]) # Dubai_1,2
# vehicle_cropping_line = np.array([(430,1200), (450,200)]) # DMM_1-4
# vehicle_cropping_line = np.array([(650,1400), (280,500)]) # DMM_5
# vehicle_cropping_line = np.array([(900,1700), (400,340)]) # PACIDAL_1
# vehicle_cropping_line = np.array([(780,1600), (700,680)]) # PACIDAL_2

# AVATools.focal_line_visualization (video_input_path, vehicle_cropping_line) #to visualize the line

fig = plt.figure(figsize=(8, 6), dpi=80)

# Track_only = ["person","car","bus", "truck"]
# turn on option "show" to view results
AVATools.Object_tracking(video_input_path, video_output_path,fps=fps, show=True,  \
                        cropping_line=vehicle_cropping_line, violation_process=violation_process )
# +
# part 1 - Detection

import importlib
importlib.reload(AVATools)

# input and output path

video_input_path   = "/workspace/AVA_model/DMM_3.ts"
video_output_path = "/workspace/AVA_model/OutResults_DMM_3/DMM_3_out.mp4" 

# vehicle_cropping_line = np.array([(330,770), (200,260)]) #define from left to right [(x1,x2), (y1,y2)]
# vehicle_cropping_line = np.array([(300,770), (240,270)]) #test1
# vehicle_cropping_line = np.array([(100,770), (350,270)]) #test2
# vehicle_cropping_line = np.array([(400,940), (420,150)]) #test3

        # test with videos from Vitronics
vehicle_cropping_line = np.array([(430,1200), (450,200)]) # DMM_1-4

# vehicle_cropping_line = np.array([(1000,1400), (210,400)]) # DMM_5


# vehicle_cropping_line = np.array([(300,770), (240,270)]) # Dubai_1,2

# vehicle_cropping_line = np.array([(780,1600), (700,680)]) # PACIDAL_1
# vehicle_cropping_line = np.array([(900,1700), (400,340)]) # PACIDAL_2

# AVATools.focal_line_visualization (video_input_path, vehicle_cropping_line)
fig = plt.figure(figsize=(8, 6), dpi=80)

# Track_only = ["person","car","bus", "truck"]
AVATools.Object_tracking(\
            video_input_path, video_output_path, show=False, \
            cropping_line=vehicle_cropping_line, violation_process=violation_process )
# -



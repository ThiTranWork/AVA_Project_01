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

## Track_only = ["person","car","bus", "truck"]
## turn on option "show" to view results
AVATools.Object_tracking(video_input_path, video_output_path,fps=fps, show=True, cropping_line=vehicle_cropping_line)
# -


FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3 
COPY ./YOLOV3_M/ /workspace/YOLOV3_M
RUN wget -P /workspace/YOLOV3_M/model_data https://pjreddie.com/media/files/yolov3.weights
COPY ./requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt && pip install jupyter notebook \
    && apt update && apt-get install -y libsm6 libxext6 libxrender-dev
WORKDIR /workspace/
CMD jupyter notebook

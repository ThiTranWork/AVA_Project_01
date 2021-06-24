FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3 
COPY ./YOLOV3 /workspace/projectAVA/
RUN pip install -r /workspace/projectAVA/requirements.txt && pip install jupyter notebook \
    && apt update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN wget -P /workspace/projectAVA/model_data https://pjreddie.com/media/files/yolov3.weights
WORKDIR /workspace/projectAVA/
CMD jupyter notebook

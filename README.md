# AVA_Project
traffic violation detection

- based on opensource Yolov3  GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

- Prequisite:
Docker image nvcr.io/nvidia/tensorflow:20.12-tf2-py3 is used in the docker file and for more details, please refer to 
https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags


to build docker image using Dockerfile defined in this repo:
docker build . -t <image_name>:<image_tag>


to run the docket image
docker run --gpus all -it --rm -p 8888:8888 -v <path_to_repo>:/workspace/project <image_name>:<image_tag>


CMD:
docker build -t thitranwork/ava:02 .
 docker run --gpus all -it --rm -p 8888:8888 -v /home/thi/Vitronic_Proj/AVA_Project_02/AVA_model:/workspace/projectAVA/AVA_model thitranwork/ava:02


Github: https://github.com/ThiTranWork/AVA_Project_02
docker hub: docker pull thitranwork/ava:02


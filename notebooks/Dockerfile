FROM nvcr.io/nvidia/pytorch:24.07-py3
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /notebooks
COPY requirements.txt requirements.txt
RUN apt update -y && apt install -y libglu1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 build-essential aria2
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
RUN pip install -U opencv-python
RUN pip install -r requirements.txt
RUN pip install opencv-python==4.5.5.64 Pillow==10.4.0
CMD jupyter lab --allow-root --ip 0.0.0.0 --port 8888 --no-browser --NotebookApp.token='algofly'

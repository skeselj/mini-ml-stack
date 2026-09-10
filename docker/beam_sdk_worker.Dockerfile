# SDK harness for Beam workers that run Python.

FROM apache/beam_python3.11_sdk:2.76.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision

RUN pip install --no-cache-dir \
    ultralytics==8.4.144 \
    opencv-python-headless==5.0.0.93

COPY models/yolo26n.pt /opt/models/yolo26n.pt
ENV YOLO_WEIGHTS=/opt/models/yolo26n.pt

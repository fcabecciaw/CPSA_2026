# Vitis AI 3.5 CPU PyTorch image plus Ultralytics.
#
# The quantizer (`pytorch_nndct`) ships in the Vitis AI image and stays exactly
# as released: only the Ultralytics wheel is added, with `--no-deps`, so numpy
# 1.22, PyTorch 1.13.1 and torchvision 0.14.1 are not upgraded.
#
# Base image (pinned by tag in run_vitis.sh):
#   xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
#   digest sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298
FROM xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306

RUN /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/pip install --no-cache-dir \
      --no-deps ultralytics==8.4.203 \
 && /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -c \
      "import ultralytics; print('ultralytics', ultralytics.__version__)"

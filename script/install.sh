# Install SAM
cd sam; pip install -e .
cd -

# Install Grounding-Dino
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO

# Install other lib
pip install numpy opencv-python pycocotools matplotlib Pillow scikit-image
pip install gradio gdown ffmpeg
apt-get install zip

pip install spatial_correlation_sampler

# # Install Pytorch Correlation
# git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
# cd Pytorch-Correlation-extension
# python setup.py install
# cd -


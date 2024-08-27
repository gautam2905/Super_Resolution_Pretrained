# Super_Resolution

This web app implements a pre-trained [Super-Resolution paper](https://arxiv.org/pdf/2107.10833). The pre-trained model I used is provided on GitHub by [XINNTAO](https://github.com/xinntao/ESRGAN).

## Steps to implement
1. git clone https://github.com/gautam2905/Super_Resolution_Pretrained.
2. Create a virtual environment.
3. Install the dependencies.
    * pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    * pip install -r requirements.txt
4. Run the `webapp.py`
5. Upload the image.
6. The resulting image will also be saved in the result folder.

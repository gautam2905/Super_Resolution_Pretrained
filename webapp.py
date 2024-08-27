from taipy.gui import Gui
from PIL import Image
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# print('Model path {:s}. \nTesting...'.format(model_path))

# idx = 0
# for path in glob.glob(test_img_folder):
def predict_hr(path):
    base = osp.splitext(osp.basename(path))[0]
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    print("image_saved")
    output_path = "results/{:s}_rlt.png".format(base)
    
    return output_path

# def predict_image(model, path_to_img):
#     results = model(path_to_img)
#     name_list = model.names
#     prob = results[0].probs
#     top_prob = prob.top1
#     top_prob1 = prob.top1conf.item()
#     top_pred =  name_list[top_prob]

#     return top_prob1, top_pred
    
content = ""
img_path = r"placeholder_image.png"
img2_path = r"placeholder_image.png"
arrow = r"arrow1.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>
>
<|layout|columns=1 1 1

<|first column
<|{img_path}|image|>
|>

<|second column
<|{arrow}|image|>
|>

<|third column
<|{img2_path}|image|>
|>

|>

"""
# <|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob = predict_hr(var_val)
        # state.prob = round(top_prob * 100)
        # state.pred = "this is a " + top_pred
        state.img_path = var_val
        state.img2_path = top_prob
    #print(var_name, var_val)


app = Gui(page=index)

# my_theme = {
#   "palette": {
#     "background": {"green": "#808080"},
#     "primary": {"main": "#a25221"}
#   }
# }

if __name__ == "__main__":
    app.run(use_reloader=True) # , theme=my_theme
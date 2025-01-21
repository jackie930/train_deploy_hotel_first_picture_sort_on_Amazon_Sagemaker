import torch
from PIL import Image

def expand2square(pil_img, background_color):
    width, height = pil_img.size  # 获得图像宽高
    if width == height:  # 相等直接返回不用重搞
        return pil_img
    elif width > height:  # w大构建w尺寸图
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))  # w最大，以坐标x=0,y=(width - height) // 2位置粘贴原图
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
class Pairwise_ViT_Infer(torch.nn.Module):
    def __init__(self, vision_tower, num_labels=2):
        super(Pairwise_ViT_Infer, self).__init__()

        self.vit = vision_tower
        self.score_layer = torch.nn.Linear(1024, 1)

    def forward(self, x1):
        x1 = self.vit(x1, output_hidden_states=True)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = torch.sigmoid(self.score_layer(x1[:, 0, :]))

        return output

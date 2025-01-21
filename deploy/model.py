import os
import boto3
import torch
import logging

from PIL import Image
from djl_python import Input, Output
import requests
from io import BytesIO

from first_page_pic_infer import expand2square,Pairwise_ViT_Infer
from transformers import CLIPVisionModel, CLIPImageProcessor

model_dict = None


class ModelConfig:
    def __init__(self):
        image_aspect_ratio = "pad"


def load_model(properties):
    s3 = boto3.client('s3')
    # print('!!!!!',properties)
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    logging.info(f"Loading model contains: {os.path.listdir(model_location)}")

    ## load first-page-pic-scoring model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model_name=os.path.join(model_location,'model.pth')
    vit_image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像模型
    vit_model = Pairwise_ViT_Infer(vision_tower).to(device)
    vit_model.load_state_dict(torch.load(model_name, map_location=device), strict=True)
    
    model_dict = {
        "vit_model": vit_model,
        "vit_image_processor": vit_image_processor
    }

    return model_dict


def handle(inputs: Input):
    global model_dict

    if not model_dict:
        model_dict = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    image_file = data["input_image"]
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    ## first-page-pic-scoring inference
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    vit_image_processor = model_dict['vit_image_processor']
    vit_model = model_dict['vit_model']
    image = expand2square(image, tuple(int(x * 255) for x in vit_image_processor.image_mean))
    image = vit_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).to(device)    
    score = vit_model(image)

    return Output().add({"score":str(score.cpu().detach().numpy()[0][0])})

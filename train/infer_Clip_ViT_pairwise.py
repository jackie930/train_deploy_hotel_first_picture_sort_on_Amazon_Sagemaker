import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as Dataloader
import torch.nn as nn
import json
import pandas as pd
import argparse
import os
from tqdm import tqdm
from accelerate import Accelerator
from PIL import Image
import time

accelerator = Accelerator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class myDataset(Dataset):
    def __init__(self, file_path, image_dir, image_processor):
        with open(file_path, 'r') as f:
            self.input_data = json.load(f)
        self.image_dir = image_dir
        self.image_processor = image_processor

    def __len__(self):
        return len(self.input_data)

    def get_images(self, idx):
        image1 = Image.open(os.path.join(self.image_dir, self.input_data[idx]['image'])).convert('RGB')
        image1 = expand2square(image1, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image1 = self.image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        return image1

    def get_labels(self, idx):
        return self.input_data[idx]['label']

    def get_ids(self, idx):
        return self.input_data[idx]['image']

    def __getitem__(self, idx):
        # Get input data in a batch
        id = self.get_ids(idx)
        image1 = self.get_images(idx)
        label = self.get_labels(idx)

        return id, image1, label


class Pairwise_ViT_Infer(nn.Module):
    def __init__(self, vision_tower, num_labels=2):
        super(Pairwise_ViT_Infer, self).__init__()

        self.vit = vision_tower
        self.score_layer = torch.nn.Linear(1024, 1)

    def forward(self, x1):
        x1 = self.vit(x1, output_hidden_states=True)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = self.score_layer(x1[:, 0, :])

        return output


def infer(args):
    model_name = args.model_name
    test_data_file = args.test_data_file
    image_dir = args.image_dir
    num_workers = args.num_workers

    t0 = time.time()
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')  # 加载图像模型
    model = Pairwise_ViT_Infer(vision_tower).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device), strict=True)
    print('load model time:', time.time() - t0)

    t0 = time.time()
    dataset_test = myDataset(test_data_file, image_dir, image_processor)
    data_loader_test = Dataloader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    data_loader_test, model = accelerator.prepare(
        data_loader_test, model
    )

    print('infer data number:', len(data_loader_test))

    model.eval()
    ss = []
    for id, image, label in tqdm(data_loader_test):
        score = model(image)
        score = accelerator.gather_for_metrics(score.item())
        ss.append([id[0], score, label.cpu().numpy()[0]])
    ss = pd.DataFrame(ss, columns=['image', 'predict', 'label'])
    ss.to_csv(os.path.join(args.save_path, 'pred_single_sample.csv'), index=False)
    print('infer time:', time.time() - t0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default='vit_pairwise_clip_model/model.pth',
        help="The name of the backbone (default BERT)."
    )
    parser.add_argument("--test_data_file", type=str)
    parser.add_argument("--image_dir", type=str, default='imgs')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default='./')

    args = parser.parse_args()
    print(args)

    t0 = time.time()
    infer(args)
    print('all used time:', time.time() - t0)

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

accelerator = Accelerator()

# 检查是否有GPU可用
print(f"是否有GPU可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
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
        image1 = Image.open(os.path.join(self.image_dir, self.input_data[idx]['image1'])).convert('RGB')
        image2 = Image.open(os.path.join(self.image_dir, self.input_data[idx]['image2'])).convert('RGB')
        image1 = expand2square(image1, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image2 = expand2square(image2, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image1 = self.image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        image2 = self.image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]
        return image1, image2

    def get_labels(self, idx):
        return self.input_data[idx]['label']

    def get_ids(self, idx):
        return self.input_data[idx]['image1'] + '^^^' + self.input_data[idx]['image2']

    def __getitem__(self, idx):
        # Get input data in a batch
        id = self.get_ids(idx)
        image1, image2 = self.get_images(idx)
        label = self.get_labels(idx)

        return id, image1, image2, label


class PairwiseViT(nn.Module):
    def __init__(self, vision_tower, num_labels=2):
        super(PairwiseViT, self).__init__()

        self.vit = vision_tower
        self.score_layer = torch.nn.Linear(1024, 1)

    def forward(self, x1, x2):
        x1 = self.vit(x1, output_hidden_states=True)['last_hidden_state']
        # Use the embedding of [CLS] token
        output1 = self.score_layer(x1[:, 0, :])

        x2 = self.vit(x2, output_hidden_states=True)['last_hidden_state']
        # Use the embedding of [CLS] token
        output2 = self.score_layer(x2[:, 0, :])

        output = torch.sigmoid(output1 - output2)

        return output, output1, output2


def train(args):
    model_name = args.model_name
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    image_dir = args.image_dir
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    save_path = args.save_path

    image_processor = CLIPImageProcessor.from_pretrained(model_name)  # 加载图像预处理
    vision_tower = CLIPVisionModel.from_pretrained(model_name)  # 加载图像模型

    dataset = myDataset(train_data_file, image_dir, image_processor)
    data_loader = Dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

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

    model = PairwiseViT(vision_tower).to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    data_loader, data_loader_test, model, optimizer = accelerator.prepare(
        data_loader, data_loader_test, model, optimizer
    )

    print(len(data_loader), len(data_loader_test))

    model.train()
    for epoch in range(max_epoch):
        total_loss_train = 0.0
        for ids, train_image1, train_image2, train_label in tqdm(data_loader):
            output, score1, score2 = model(train_image1, train_image2)
            loss = criterion(output.to(device), train_label.reshape(output.shape).to(device).to(torch.float32))
            total_loss_train += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        print(
            f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(data_loader): .3f} ')

    print("Training is finished!")
    os.makedirs(save_path, exist_ok=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(args.save_path, exist_ok=True)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(args.save_path, "model.pth"))
        print("Model is saved: {}".format(os.path.join(args.save_path)))

    model.eval()
    ss = []
    for id, image1, image2, label in tqdm(data_loader_test):
        output, score1, score2 = model(image1, image2)
        prediction = accelerator.gather_for_metrics(output.item())
        score1 = accelerator.gather_for_metrics(score1.item())
        score2 = accelerator.gather_for_metrics(score2.item())
        ss.append([id[0], prediction, score1, score2, label.cpu().numpy()[0]])
    ss = pd.DataFrame(ss, columns=['image', 'predict', 'score1', 'score2', 'label'])
    ss.to_csv(os.path.join(args.save_path, 'pred.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default='Salesforce/SFR-Embedding-Mistral',
        help="The name of the backbone (default BERT)."
    )
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--test_data_file", type=str)
    parser.add_argument("--image_dir", type=str, default='imgs')
    parser.add_argument("--save_path", type=str, default='vit_pairwise_clip_model')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    print(args)

    train(args)

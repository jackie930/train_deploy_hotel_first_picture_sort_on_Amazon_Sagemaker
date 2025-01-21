import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel,ViTModel, ViTConfig
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as Dataloader
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import json
import pandas as pd
import argparse
import os
from tqdm import tqdm
from accelerate import Accelerator
import cv2

accelerator = Accelerator()

# 检查是否有GPU可用
print(f"是否有GPU可用: {torch.cuda.is_available()}")

# 检查GPU数量
print(f"GPU数量: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myDataset(Dataset):
    def __init__(self, file_path,image_dir):
        with open(file_path, 'r') as f:
            self.input_data = json.load(f)
        # Transform input data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.image_dir=image_dir

    def __len__(self):
        return len(self.input_data)

    def get_images(self, idx):
        # image=Image.open(os.path.join(self.image_dir,self.input_data[idx]['image']))
        image=cv2.imread(os.path.join(self.image_dir,self.input_data[idx]['image']))
        return self.transform(image)

    def get_labels(self, idx):
        return self.input_data[idx]['label']

    def get_ids(self, idx):
        return self.input_data[idx]['image']

    def __getitem__(self, idx):

        # Get input data in a batch
        train_ids = self.get_ids(idx)
        train_images = self.get_images(idx)
        train_labels = self.get_labels(idx)

        return train_ids, train_images, train_labels
    
#     def collate_fn(self, batch_data):
#         data = zip(*batch_data)
#         ids, img, label = data

#         label = torch.tensor(label)
#         # ids = torch.tensor(ids, dtype=torch.bfloat16)
#         img = torch.tensor(img)
#         return ids, img, label 

#
# class scoreModel(torch.nn.Module):
#     def __init__(self, backbone, in_channels=776, use_dropout=False, keep_prob=0.8):
#         super().__init__()
#         self.backbone = backbone
#         self.score_layer = torch.nn.Linear(in_channels, 1)
#         self.bce = torch.nn.BCELoss()
#         self.use_dropout = use_dropout
#         if use_dropout:
#             self.dropout = torch.nn.Dropout(p=1 - keep_prob)
#
#     def forward(self, a_text_ids, a_text_type_ids, a_text_masks,
#                 b_text_ids, b_text_type_ids, b_text_masks, feat_A, feat_B, label):
#         outputa = self.backbone(**a_text_ids)
#         a_text_feat= last_token_pool(outputa.last_hidden_state, a_text_ids['attention_mask'])
#         # normalize embeddings
#         a_text_feat = F.normalize( a_text_feat, p=2, dim=1).to(torch.bfloat16)
#         outputb = self.backbone(**b_text_ids)
#         b_text_feat = last_token_pool(outputb.last_hidden_state, b_text_ids['attention_mask'])
#         b_text_feat = F.normalize(b_text_feat, p=2, dim=1).to(torch.bfloat16)
#
#         a_text_feat = torch.cat([a_text_feat, feat_A], 1)
#         b_text_feat = torch.cat([b_text_feat, feat_B], 1)
#
#         if self.use_dropout:
#             a_text_score = self.score_layer(self.dropout(a_text_feat))
#             b_text_score = self.score_layer(self.dropout(b_text_feat))
#         else:
#             a_text_score = self.score_layer(a_text_feat)
#             b_text_score = self.score_layer(b_text_feat)
#
#         pred = torch.sigmoid(b_text_score - a_text_score)
#         loss = self.bce(pred, label.reshape(pred.shape).to(torch.bfloat16))
#         return pred, loss
#
#     @torch.no_grad()
#     def predict(self, text_ids, text_type_ids, text_masks, feat):
#         output = self.backbone(**text_ids)
#         text_feat = last_token_pool(output.last_hidden_state, text_ids['attention_mask'])
#         text_feat = F.normalize(text_feat, p=2, dim=1).to(torch.bfloat16)
#         text_feat = torch.cat([text_feat, feat], 1)
#         text_score = self.score_layer(text_feat)
#
#         return text_score
#
#     @torch.no_grad()
#     def inference(self, text_ids, text_type_ids, text_masks, feat):
#         score = self.predict(text_ids, text_type_ids, text_masks, feat)
#
#         return score


class ViT(nn.Module):
    def __init__(self, config=ViTConfig(), num_labels=2,
                 model_checkpoint='google/vit-base-patch16-224-in21k'):
        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels)
        )
    def forward(self, x):
        x = self.vit(x)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = self.classifier(x[:, 0, :])
        return output


def train(args):
    model_name = args.model_name
    in_channels = args.in_channels
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_cls = args.num_cls
    max_epoch = args.max_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    print_interval_steps = args.print_interval_steps
    use_score_model = args.use_score_model
    use_dropout = args.use_dropout
    keep_prob = args.keep_prob
    clip_value = args.clip_value
    save_path = args.save_path
    save_name = args.save_name

    infer_hyper_params = {
        "model_name": model_name,
        "in_channels": in_channels,
        "num_cls": num_cls,
        "use_score_model": use_score_model,
        "use_dropout": use_dropout,
        "keep_prob": keep_prob,
        "save_name": save_name
    }

    dataset = myDataset(train_data_file,image_dir)
    data_loader = Dataloader(
        dataset,
        batch_size=batch_size,
        # collate_fn=dataset.collate_fn,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dataset_test = myDataset(test_data_file,image_dir)
    data_loader_test = Dataloader(
        dataset_test,
        batch_size=1,
        # collate_fn=dataset_test.collate_fn,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    print('-----!!',len(data_loader),len(data_loader_test))


    print('!!!!!',model_name)

    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    data_loader, data_loader_test, model, optimizer = accelerator.prepare(
        data_loader,data_loader_test,  model, optimizer
    )
    
    print('!!!!!!',len(data_loader),len(data_loader_test))
    
            
    model.train()
    for epoch in range(max_epoch):
        total_acc_train = 0
        total_loss_train = 0.0

        for ids,train_image, train_label in tqdm(data_loader):
            # print(train_image.shape)
            output = model(train_image)
            loss = criterion(output, train_label.to(device))
            acc = (output.argmax(dim=1) == train_label.to(device)).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        print(
            f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(data_loader): .3f} | Accuracy: {total_acc_train / len(data_loader): .3f}')

    
    # torch.save(model, os.path.join(save_path, save_name))
    print("Training is finished!")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "infer_hyper_params.json"), "w") as f:
        json.dump(infer_hyper_params, f)
     
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        print(os.path.join(args.save_path, "model.pth"))
        os.makedirs(args.save_path, exist_ok=True)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(args.save_path, "model.pth"))
        print("Model is saved: {}".format(os.path.join(args.save_path, "model.pth")))

    model.eval()
    ss = []
    for id, image, label in tqdm(data_loader_test):
        output = model(image).argmax(dim=1).item()
        prediction = accelerator.gather_for_metrics(output)
        # prediction = prediction.cpu().numpy()


        ss.append([id,prediction,label.cpu().numpy()[0]])
    ss = pd.DataFrame(ss,columns=['image','predict','label'])
    ss.to_csv(os.path.join(args.save_path, 'pred.csv'), index=False)


if __name__ == "__main__":
    try:
        train_data_dir = os.environ['SM_CHANNEL_TRAIN']
        test_data_dir = os.environ['SM_CHANNEL_TEST']
        model_dir = os.environ['SM_MODEL_DIR']
    except:
        train_data_dir = ''
        test_data_dir = ''
        model_dir = 'vit_model'
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    

    parser.add_argument(
        "--model_name", type=str, default='Salesforce/SFR-Embedding-Mistral',
        help="The name of the backbone (default BERT)."
    )
    parser.add_argument("--in_channels", type=int, default=1024+8,
                        help="The channels of the backbone output (default 768).")
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--test_data_file", type=str)
    parser.add_argument("--image_dir", type=str, default='imgs')              
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_cls", type=int, default=3, help="It is not used in the score model.")
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval_steps", type=int, default=100)
    parser.add_argument("--use_score_model", type=int, default=1,
                        help="Whether to use score model (default 1). 1 denotes yes, 0 denotes no."
                        )
    parser.add_argument("--use_dropout", type=int, default=0,
                        help="Whether to use dropout to mitigate overfitting (default 0). 1 denotes yes, 0 denotes no."
                        )
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default=model_dir)
    parser.add_argument("--save_name", type=str)

    args = parser.parse_args()
    print(args)

    train(args)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from torchvision.transforms import functional as F
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt


class LocalMatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def random_crop(images, crop_size=224, scale=(0.5, 1.0), ratio=(0.75, 1.33)):
    return torch.stack([
        F.resized_crop(
            img,
            *transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio),
            (crop_size, crop_size)
        )
        for img in images
    ])


class LocalMatcher(nn.Module):
    def __init__(self, backbone='dinov2_vitb14'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False  # 冻结参数

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]

        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        query1 = feat1.unsqueeze(1)  # [B, 1, feat_dim]
        key2 = feat2.unsqueeze(1)
        value2 = feat2.unsqueeze(1)
        attn1, _ = self.cross_attn(query1, key2, value2)
        attn1 = attn1.squeeze(1)  # [B, feat_dim]

        query2 = feat2.unsqueeze(1)
        key1 = feat1.unsqueeze(1)
        value1 = feat1.unsqueeze(1)
        attn2, _ = self.cross_attn(query2, key1, value1)
        attn2 = attn2.squeeze(1)

        combined = torch.cat([attn1, attn2], dim=1)
        return self.mlp(combined).squeeze()


def train():
    batch_size = 8
    epochs = 20
    lr = 1e-4
    crop_size = 224
    K = 5  # 每张图像生成正样本数量

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LocalMatchDataset(root_dir='/root/autodl-tmp/pixiv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalMatcher().to(device)
    optimizer = optim.AdamW(
        list(model.cross_attn.parameters()) + list(model.mlp.parameters()),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for images in dataloader:
            images = images.to(device)
            B = images.size(0)

            images_pos = images.repeat_interleave(K, dim=0)  # [B*K, C, H, W]
            crops_pos = random_crop(images_pos, crop_size=crop_size)

            perm = torch.randperm(B)
            images_neg_source = images[perm]

            crops_neg = random_crop(
                images_neg_source.repeat(K, 1, 1, 1),
                crop_size=crop_size
            )

            images_neg = images.repeat(K, 1, 1, 1)

            input1 = torch.cat([images_pos, images_neg], dim=0)
            input2 = torch.cat([crops_pos, crops_neg], dim=0)
            # print(input1.shape, input2.shape, crops_neg.shape, crops_pos.shape)
            labels = torch.cat([
                torch.ones(B * K),
                torch.zeros(B * K)
            ]).to(device)

            outputs = model(input1, input2)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    torch.save(model.state_dict(), "local_matcher.pth")


def is_image_cropped(full_img_path, crop_img_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = transform(Image.open(full_img_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(crop_img_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1, img2)
        prob = torch.sigmoid(output).item()

    return {
        "confidence": prob,
        "is_cropped": prob > 0.5,
        "message": "该图像是原图裁剪" if prob > 0.5 else "该图非原图裁剪"
    }


def demo(image1, image2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("local_matcher.pth").to(device)
    model.eval()
    result = is_image_cropped("test_full.jpg", "test_crop.jpg")
    print(result)


def demo2(original_path, crop_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalMatcher().to(device)  # 需要定义LocalMatcher类
    model.load_state_dict(torch.load("local_matcher.pth", map_location=device))
    model.eval()

    original_img = Image.open(original_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # 先调整到稍大尺寸
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crop_img = transform(original_img).unsqueeze(0).to(device)  # 添加批次维度

    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    original_input = original_transform(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # print(f"original_input shape: {original_input.shape}, crop_img shape: {crop_img.shape}")
        output = model(original_input, crop_img)
        probability = torch.sigmoid(output).item()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(original_img)
    ax[0].set_title(f"Original Image\n{original_path}")
    ax[0].axis('off')

    crop_display = crop_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    crop_display = crop_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    crop_display = np.clip(crop_display, 0, 1)
    ax[1].imshow(crop_display)
    ax[1].set_title(f"Cropped Region\nProbability: {probability:.4f}")
    ax[1].axis('off')

    result = "Same Image" if probability > 0.5 else "Different Image"
    plt.suptitle(f"Prediction: {result} (Confidence: {probability:.4f})", fontsize=16)

    plt.tight_layout()
    plt.show()

    return probability


if __name__ == "__main__":
    train()

    original_path = "/root/autodl-tmp/pixiv/_D_94039006.jpg"
    probability = demo2(original_path)
    print(f"Probability that crop is from original: {probability:.4f}")

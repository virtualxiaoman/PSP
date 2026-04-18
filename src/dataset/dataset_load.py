import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocess.degradation import HighOrderDegradationPipeline
from src.utils.img_read import read_image


# 数据集定义
class BicycleDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = []
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        self.img_paths.sort()  # 保证顺序稳定
        self.transform = transform
        self.degradation_pipeline = HighOrderDegradationPipeline(num_orders=4)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path)
        img_pil = Image.fromarray(img.astype('uint8')).convert('RGB')
        degraded_np = self.degradation_pipeline.process(img)
        degraded_img_pil = Image.fromarray(degraded_np.astype('uint8')).convert('RGB')

        if self.transform:
            original_tensor = self.transform(img_pil)
            degraded_tensor = self.transform(degraded_img_pil)
        else:
            t = transforms.ToTensor()
            original_tensor = t(img_pil)
            degraded_tensor = t(degraded_img_pil)

        return original_tensor, degraded_tensor, idx


class SOPDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 传入根目录，例如 "G:/DataSets/CV/ImageRetrieval/Stanford_Online_Products"
        """
        self.img_paths = []
        # 使用 "**" 和 recursive=True 递归遍历目录下所有的子文件夹
        self.img_paths = (glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) +
                          glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True))
        self.img_paths.sort()  # 保证顺序稳定
        self.transform = transform
        # 保持与之前相同的 Degradation Pipeline
        self.degradation_pipeline = HighOrderDegradationPipeline(num_orders=4)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path)
        img_pil = Image.fromarray(img.astype('uint8')).convert('RGB')
        degraded_np = self.degradation_pipeline.process(img)
        degraded_img_pil = Image.fromarray(degraded_np.astype('uint8')).convert('RGB')

        if self.transform:
            original_tensor = self.transform(img_pil)
            degraded_tensor = self.transform(degraded_img_pil)
        else:
            t = transforms.ToTensor()
            original_tensor = t(img_pil)
            degraded_tensor = t(degraded_img_pil)

        return original_tensor, degraded_tensor, idx

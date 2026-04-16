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
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                         glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transform
        self.degradation_pipeline = HighOrderDegradationPipeline(num_orders=4)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path)
        img_pil = Image.fromarray(img.astype('uint8')).convert('RGB')
        # try:
        #     original_img_pil = Image.open(img_path).convert('RGB')
        # except Exception:
        #     original_img_pil = Image.new('RGB', (224, 224), (255, 255, 255))
        #
        # # 模拟退化过程 (如果 pipeline 不在，这里仅作示意)
        # img_np = np.array(original_img_pil)
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

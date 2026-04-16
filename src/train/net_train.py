import os
import time
import warnings
import torch

from torch.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from src.config.path import MODEL_DISTILL_DIR
from src.dataset.dataset_load import BicycleDataset
from src.net.stu_net import DistillationStudent


# 蒸馏训练器类
class ImageDistillationTrainer:
    def __init__(self, student, teacher, train_loader, eval_loader, config):
        """
        config: 包含 lr, epochs, device, temp 等参数的字典
        """
        self.student_net = student
        self.teacher_net = teacher
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # 参数配置
        self.device = config.get('device', torch.device('cuda'))
        self.lr = config.get('lr', 5e-4)
        self.epochs = config.get('epochs', 20)
        self.temperature = config.get('temp', 0.1)
        self.distill_weight = config.get('distill_weight', 2.0)
        self.save_name = config.get('save_name', "student_vit_256d.pth")

        # 优化器与调度器
        self.optimizer = torch.optim.AdamW(self.student_net.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.student_net.to(self.device)
        self.teacher_net.to(self.device)
        self.teacher_net.eval()

    def nt_xent_loss(self, embed1, embed2):
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)  # 归一化
        logits = torch.matmul(embed1, embed2.T) / self.temperature  # 相似度矩阵，shape为 (batch_size, batch_size)
        batch_size = embed1.size(0)
        labels = torch.arange(batch_size, device=self.device)  # 构造标签，正确匹配的索引为 (i, i)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)  # 前者保证embed1找embed2，后者保证2找1
        return loss / 2

    def distillation_loss(self, student_out, teacher_out):
        student_out = F.normalize(student_out, dim=-1)
        teacher_out = F.normalize(teacher_out, dim=-1)
        return 1.0 - (student_out * teacher_out).sum(dim=-1).mean()  # *是逐元素乘，shape: (batch_size, D)，求平均余弦相似度

    def evaluate(self, ks=(1, 3, 5)):
        self.student_net.eval()
        # print(f"\n--- 正在评估 (Recall@{ks}) ---")
        original_features = []
        degraded_features = []

        with torch.no_grad():
            for orig, deg, _ in self.eval_loader:
                orig, deg = orig.to(self.device), deg.to(self.device)
                feat_o = F.normalize(self.student_net(orig), dim=-1)
                feat_d = F.normalize(self.student_net(deg), dim=-1)
                original_features.append(feat_o.cpu())
                degraded_features.append(feat_d.cpu())

        original_features = torch.cat(original_features, dim=0)
        degraded_features = torch.cat(degraded_features, dim=0)
        sim_matrix = torch.matmul(degraded_features, original_features.T)

        num_queries = sim_matrix.size(0)
        topk_indices = torch.topk(sim_matrix, max(ks), dim=1).indices
        labels = torch.arange(num_queries).view(-1, 1)

        results = {}
        for k in ks:
            correct_k = (topk_indices[:, :k] == labels).any(dim=1).sum().item()
            recall_k = correct_k / num_queries
            results[k] = recall_k
            print(f"Recall@{k}: {recall_k:.4f}", end='\t')
        print()
        return results

    def train_net(self):
        for epoch in range(self.epochs):
            self.student_net.train()
            total_loss = 0

            for orig_img, deg_img, _ in self.train_loader:
                orig_img, deg_img = orig_img.to(self.device), deg_img.to(self.device)

                self.optimizer.zero_grad()
                with torch.no_grad():
                    t_features = self.teacher_net(orig_img)

                s_orig_256, s_orig_1024 = self.student_net(orig_img)
                s_deg_256, _ = self.student_net(deg_img)

                loss_nce = self.nt_xent_loss(s_orig_256, s_deg_256)
                loss_dist = self.distillation_loss(s_orig_1024, t_features)

                loss = loss_nce + (loss_dist * self.distill_weight)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{self.epochs} | Avg Loss: {total_loss / len(self.train_loader):.4f} | "
                  f"LR: {current_lr:.6e}")

            self.scheduler.step()
            self.evaluate()

        # 保存模型
        save_path = os.path.join(MODEL_DISTILL_DIR, self.save_name)
        torch.save(self.student_net.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")


class ImageDistillationTrainer2:
    """
    图像检索蒸馏训练器
    支持：
    1. AMP 自动混合精度
    2. 自动保存最佳模型
    3. Recall@K评估
    4. GPU显存日志
    5. Cosine LR调度
    """

    def __init__(
            self,
            student,
            teacher,
            train_loader,
            eval_loader,
            config,
            scheduler=None,
            use_amp=True,
            eval_interval=1,
            eval_during_training=True,
            **kwargs
    ):
        # =========================
        # 基础参数
        # =========================
        self.student_net = student
        self.teacher_net = teacher
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = config.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.lr = config.get("lr", 5e-4)
        self.epochs = config.get("epochs", 20)
        self.temperature = config.get("temp", 0.1)
        self.distill_weight = config.get("distill_weight", 2.0)
        self.save_name = config.get("save_name", "student_vit_256d_best.pth")

        self.eval_interval = eval_interval
        self.eval_during_training = eval_during_training
        self.free_memory = kwargs.get("free_memory", False)

        # =========================
        # AMP
        # =========================
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        # =========================
        # 网络初始化
        # =========================
        self.student_net.to(self.device)
        self.teacher_net.to(self.device)
        self.teacher_net.eval()

        for p in self.teacher_net.parameters():
            p.requires_grad = False

        # =========================
        # 优化器
        # =========================
        self.optimizer = torch.optim.AdamW(
            self.student_net.parameters(),
            lr=self.lr,
            weight_decay=0.05
        )

        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )

        # =========================
        # 日志记录
        # =========================
        self.loss_epoch = None
        self.best_recall1 = None
        self.best_epoch = None
        self.current_gpu_memory = None
        self.time_list = []

        self.auto_save_best_net = False

        self.log_model_info()

    # =====================================================
    # Info Logging
    # =====================================================
    def log_model_info(self):
        train_samples = len(self.train_loader.dataset)
        eval_samples = len(self.eval_loader.dataset)

        print(f"[log] Train Samples: {train_samples}, Eval Samples: {eval_samples}")

        x, _, _ = next(iter(self.train_loader))
        x = x.to(self.device)

        with torch.no_grad():
            out_256, out_1024 = self.student_net(x)

        print(
            f"[log] 输入shape: {x.shape}, "
            f"Student256: {out_256.shape}, "
            f"Student1024: {out_1024.shape}"
        )

        total_params_stu = sum(p.numel() for p in self.student_net.parameters())
        print(f"[log] Student总参数量: {total_params_stu}")
        total_params_tea = sum(p.numel() for p in self.teacher_net.parameters())
        print(f"[log] Teacher总参数量: {total_params_tea}")

    # =====================================================
    # Loss Functions
    # =====================================================
    def nt_xent_loss(self, embed1, embed2):
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        logits = torch.matmul(embed1, embed2.T) / self.temperature
        batch_size = embed1.size(0)
        labels = torch.arange(batch_size, device=self.device)

        loss = F.cross_entropy(logits, labels)
        loss += F.cross_entropy(logits.T, labels)

        return loss / 2

    def distillation_loss(self, student_out, teacher_out):
        student_out = F.normalize(student_out, dim=-1)
        teacher_out = F.normalize(teacher_out, dim=-1)

        return 1.0 - (student_out * teacher_out).sum(dim=-1).mean()

    # =====================================================
    # Train Main
    # =====================================================
    def train_net(self, net_save_path=None):
        print(f">>> 开始训练 | epochs={self.epochs}, device={self.device}")

        self.__check_best_net_save_path(net_save_path)

        for epoch in range(self.epochs):
            self.student_net.train()
            self.train_epoch()

            if epoch % self.eval_interval == 0:
                self.log_and_update_eval(epoch, net_save_path)

        print(">>> 训练结束")
        if self.best_recall1 is not None:
            print(f"[Best] Epoch={self.best_epoch + 1}, Recall@1={self.best_recall1:.4f}")

    # =====================================================
    # Train One Epoch
    # =====================================================
    def train_epoch(self):
        start_time = time.time()
        total_loss = 0.0

        for orig_img, deg_img, _ in self.train_loader:
            orig_img = orig_img.to(self.device)
            deg_img = deg_img.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_feat = self.teacher_net(orig_img)
                teacher_feat = F.normalize(teacher_feat, dim=-1)

            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                s_orig_256, s_orig_1024 = self.student_net(orig_img)
                s_deg_256, _ = self.student_net(deg_img)

                loss_nce = self.nt_xent_loss(s_orig_256, s_deg_256)
                loss_dist = self.distillation_loss(s_orig_1024, teacher_feat)

                loss = loss_nce + self.distill_weight * loss_dist

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            self.current_gpu_memory = self._log_gpu_memory()

            if self.free_memory:
                del orig_img, deg_img, loss

        self.scheduler.step()

        self.loss_epoch = total_loss / len(self.train_loader)
        self.time_list.append(time.time() - start_time)

    # =====================================================
    # Evaluate
    # =====================================================
    def evaluate(self, ks=(1, 3, 5, 10)):
        if not self.eval_during_training:
            return None

        self.student_net.eval()
        # print(f"\n--- 正在评估 (Recall@{ks}) ---")
        original_features = []
        degraded_features = []

        with torch.no_grad():
            for orig, deg, _ in self.eval_loader:
                orig, deg = orig.to(self.device), deg.to(self.device)
                feat_o = F.normalize(self.student_net(orig), dim=-1)  # 单位向量
                feat_d = F.normalize(self.student_net(deg), dim=-1)
                original_features.append(feat_o.cpu())
                degraded_features.append(feat_d.cpu())

        original_features = torch.cat(original_features, dim=0)
        degraded_features = torch.cat(degraded_features, dim=0)
        sim_matrix = torch.matmul(degraded_features, original_features.T)  # [n, D] x [D, n] -> [n, n]，单位向量点积=余弦相似度

        num_queries = sim_matrix.size(0)
        topk_indices = torch.topk(sim_matrix, max(ks), dim=1).indices
        labels = torch.arange(num_queries).view(-1, 1)

        results = {}
        for k in ks:
            correct_k = (topk_indices[:, :k] == labels).any(dim=1).sum().item()  # 预测[n, k]，真实[n, 1]
            recall_k = correct_k / num_queries
            results[k] = recall_k
        #     print(f"Recall@{k}: {recall_k:.4f}", end='\t')
        # print()
        return results

    # =====================================================
    # Logging + Save Best
    # =====================================================
    def log_and_update_eval(self, epoch, net_save_path):
        current_lr = self.optimizer.param_groups[0]['lr']
        results = self.evaluate()

        recall1 = results[1]

        print(
            f"Epoch {epoch + 1}/{self.epochs} | "
            f"Loss: {self.loss_epoch:.6f} | "
            f"R@1: {results[1]:.4f} | "
            f"R@3: {results[3]:.4f} | "
            f"R@5: {results[5]:.4f} | "
            f"R@10: {results[10]:.4f} | "
            f"Time: {self.time_list[-1]:.2f}s | "
            f"LR: {current_lr:.4e} | "
            f"GPU: {self.current_gpu_memory}"
        )

        if self.best_recall1 is None or recall1 > self.best_recall1:
            self.best_recall1 = recall1
            self.best_epoch = epoch

            if self.auto_save_best_net:
                self.__save_net(net_save_path)

    # =====================================================
    # GPU Log
    # =====================================================
    def _log_gpu_memory(self):
        if self.device.type != "cuda":
            return "CPU"

        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)

        used = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)

        return f"U{used:.2f}+R{reserved:.2f}/T{total:.2f}GB"

    # =====================================================
    # Save
    # =====================================================
    def __check_best_net_save_path(self, net_save_path):
        if isinstance(net_save_path, str):
            self.auto_save_best_net = True
            dir_path = os.path.dirname(net_save_path)
            os.makedirs(dir_path, exist_ok=True)
            print(f"[save] 最佳模型保存地址: {net_save_path}")
        else:
            self.auto_save_best_net = False
            print("[save] 未启用自动保存最佳模型")

    def __save_net(self, net_save_path):
        try:
            torch.save(self.student_net.state_dict(), net_save_path)
        except Exception as e:
            warnings.warn(f"保存模型失败: {e}")


def main():
    # 配置信息
    DATA_DIR = r"G:\DataSets\CV\ImageRetrieval\Stanford_Online_Products\bicycle_final"
    config = {
        'lr': 5e-4,
        'epochs': 20,
        'batch_size': 32,
        'temp': 0.1,
        'distill_weight': 2.0,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_name': "student_vit_256d.pth"
    }

    # 数据准备
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    full_dataset = BicycleDataset(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    # train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    # eval_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 模型准备
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    student = DistillationStudent(target_dim=256, teacher_dim=1024)

    # 启动训练器
    trainer = ImageDistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config
    )

    trainer.train_net()


def main2():
    # 配置信息
    DATA_DIR = r"G:\DataSets\CV\ImageRetrieval\Stanford_Online_Products\bicycle_final"
    config = {
        'lr': 5e-4,
        'epochs': 10,
        'batch_size': 32,
        'temp': 0.1,
        'distill_weight': 2.0,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_name': "student_vit_256d.pth"
    }

    # 数据准备
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    full_dataset = BicycleDataset(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    student = DistillationStudent(target_dim=256, teacher_dim=1024)

    # 启动训练器
    trainer = ImageDistillationTrainer2(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config)
    trainer.train_net(net_save_path=os.path.join(MODEL_DISTILL_DIR, config['save_name']))


if __name__ == "__main__":
    main2()
    # Epoch 1/10 | Loss: 1.474065 | R@1: 0.8539 | R@3: 0.9314 | R@5: 0.9579 | R@10: 0.9771 | Time: 155.52s | LR: 4.877641e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 2/10 | Loss: 0.774329 | R@1: 0.8912 | R@3: 0.9639 | R@5: 0.9735 | R@10: 0.9862 | Time: 151.32s | LR: 4.522542e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 3/10 | Loss: 0.652702 | R@1: 0.9164 | R@3: 0.9771 | R@5: 0.9862 | R@10: 0.9946 | Time: 157.17s | LR: 3.969463e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 4/10 | Loss: 0.573386 | R@1: 0.9290 | R@3: 0.9814 | R@5: 0.9880 | R@10: 0.9922 | Time: 171.04s | LR: 3.272542e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 5/10 | Loss: 0.522455 | R@1: 0.9429 | R@3: 0.9910 | R@5: 0.9964 | R@10: 0.9982 | Time: 170.62s | LR: 2.500000e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 6/10 | Loss: 0.484162 | R@1: 0.9465 | R@3: 0.9928 | R@5: 0.9970 | R@10: 0.9994 | Time: 163.46s | LR: 1.727458e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 7/10 | Loss: 0.444535 | R@1: 0.9537 | R@3: 0.9886 | R@5: 0.9922 | R@10: 0.9964 | Time: 177.02s | LR: 1.030537e-04 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 8/10 | Loss: 0.424371 | R@1: 0.9519 | R@3: 0.9928 | R@5: 0.9964 | R@10: 0.9988 | Time: 173.91s | LR: 4.774575e-05 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 9/10 | Loss: 0.412826 | R@1: 0.9621 | R@3: 0.9922 | R@5: 0.9946 | R@10: 0.9982 | Time: 165.79s | LR: 1.223587e-05 | GPU: U1.64+R10.42/T15.92GB
    # Epoch 10/10 | Loss: 0.405728 | R@1: 0.9567 | R@3: 0.9946 | R@5: 0.9970 | R@10: 0.9976 | Time: 167.11s | LR: 0.000000e+00 | GPU: U1.64+R10.42/T15.92GB
    # >>> 训练结束
    # [Best] Epoch=9, Recall@1=0.9621

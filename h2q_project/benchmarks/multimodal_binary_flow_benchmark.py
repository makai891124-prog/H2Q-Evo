#!/usr/bin/env python3
"""
Multimodal Binary Flow Encoder Benchmark
- 图像：CIFAR-10 子集分类
- 视频：合成移动方块序列分类
"""

import argparse
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets, transforms
except Exception:  # noqa: BLE001
    datasets = None
    transforms = None

from transformers import AutoTokenizer, AutoModel
from h2q_project.h2q.core.multimodal_binary_flow import MultimodalBinaryFlowEncoder


class ImageBinaryFlowClassifier(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.encoder = MultimodalBinaryFlowEncoder(dim=dim, flow_dim=dim)
        # 扩展头：MLP 以提升能力上限
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim // 4, 10)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        img_sig, _ = self.encoder(images=images, videos=None)
        fused = self.encoder.fuse_signature(img_sig, None)
        return self.classifier(fused)


class MultimodalJointClassifier(nn.Module):
    def __init__(self, dim: int = 256, num_classes: int = 101, text_model: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = MultimodalBinaryFlowEncoder(dim=dim, flow_dim=dim)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.text_proj = nn.Linear(768, dim)  # BERT hidden size
        self.fusion_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, videos: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        _, vid_sig = self.encoder(images=None, videos=videos)
        fused_v = self.encoder.fuse_signature(None, vid_sig)
        text_outputs = self.text_model(**texts)
        text_emb = text_outputs.last_hidden_state[:, 0]  # [CLS]
        text_proj = self.text_proj(text_emb)
        # 联合注意力
        joint = torch.stack([fused_v, text_proj], dim=1)  # [B, 2, dim]
        attn_out, _ = self.fusion_attn(joint, joint, joint)
        pooled = attn_out.mean(dim=1)
        return self.classifier(pooled)


class SpatioTemporalBinaryFlowClassifier(nn.Module):
    def __init__(self, dim: int = 256, num_classes: int = 101, seq_len: int = 8):
        super().__init__()
        self.encoder = MultimodalBinaryFlowEncoder(dim=dim, flow_dim=dim)
        # 时空注意力头
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = videos.shape
        videos_flat = videos.view(b*t, c, h, w)
        img_sigs = []
        for i in range(t):
            img_sig, _ = self.encoder(images=videos_flat[i*b:(i+1)*b], videos=None)
            img_sigs.append(img_sig)
        seq_sig = torch.stack(img_sigs, dim=1)  # [B, T, dim]
        attn_out, _ = self.temporal_attn(seq_sig, seq_sig, seq_sig)
        pooled = attn_out.mean(dim=1)  # [B, dim]
        return self.classifier(pooled)


class MovingSquareVideoDataset(Dataset):
    def __init__(self, num_samples: int = 2000, seq_len: int = 8, size: int = 32):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.size = size
        self.square = 8

    def __len__(self) -> int:
        return self.num_samples

    def _render(self, direction: int) -> torch.Tensor:
        # direction: 0 = left->right, 1 = right->left
        t = self.seq_len
        s = self.size
        img = torch.zeros(t, 3, s, s)
        y = random.randint(4, s - self.square - 4)
        start = 2 if direction == 0 else s - self.square - 2
        step = 2 if direction == 0 else -2
        for i in range(t):
            x = start + i * step
            x = max(0, min(s - self.square, x))
            img[i, :, y:y + self.square, x:x + self.square] = 1.0
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = idx % 2
        video = self._render(label)
        return video, label


class UCF101VideoWrapper(Dataset):
    def __init__(self, dataset: Dataset, max_samples: int = 0, video_size: int = 32):
        self.dataset = dataset
        self.max_samples = max_samples
        self.video_size = video_size

    def __len__(self) -> int:
        if self.max_samples > 0:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video, _, label = self.dataset[idx]
        # video: [T, H, W, C] numpy or tensor
        if isinstance(video, torch.Tensor):
            video = video.float() / 255.0
        else:
            video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        if video.shape[-2:] != (self.video_size, self.video_size):
            video = F.interpolate(video, size=(self.video_size, self.video_size), mode='bilinear', align_corners=False)
        return video.float(), int(label)


def get_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch
        else:
            inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def run_image_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    if datasets is None:
        raise RuntimeError("torchvision 未安装，无法运行 CIFAR-10 基准")

    transform = transforms.Compose([transforms.ToTensor()])
    if args.image_dataset.lower() == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
    else:
        raise RuntimeError(f"不支持的图像数据集: {args.image_dataset}")

    train_subset = Subset(train_set, list(range(min(args.image_train_samples, len(train_set)))))
    test_subset = Subset(test_set, list(range(min(args.image_test_samples, len(test_set)))))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ImageBinaryFlowClassifier(dim=args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"[Image] Epoch {epoch}/{args.epochs} | loss={loss:.4f} | acc={acc:.4f}")


def _get_ucf101_dataset(args: argparse.Namespace, split: str):
    if datasets is None:
        raise RuntimeError("torchvision 未安装，无法运行 UCF101 基准")

    if not hasattr(datasets, "UCF101"):
        raise RuntimeError("当前 torchvision 版本不包含 UCF101")

    # UCF101 需要视频文件目录，默认 data_root/ucf101
    root = args.video_root
    annotation_path = args.ucf101_annotation_path
    frames_per_clip = args.seq_len
    step_between_clips = max(1, args.seq_len // 2)
    dataset = datasets.UCF101(
        root=root,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        train=(split == "train"),
        transform=None,  # Remove transform to avoid type issues
    )
    max_samples = args.video_train_samples if split == "train" else args.video_test_samples
    return UCF101VideoWrapper(dataset, max_samples=max_samples, video_size=args.video_size)


def run_video_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    if args.video_dataset.lower() == "synthetic":
        train_set = MovingSquareVideoDataset(num_samples=args.video_train_samples, seq_len=args.seq_len, size=args.video_size)
        test_set = MovingSquareVideoDataset(num_samples=args.video_test_samples, seq_len=args.seq_len, size=args.video_size)
    elif args.video_dataset.lower() == "ucf101":
        if datasets is None or transforms is None:
            raise RuntimeError("torchvision 未安装，无法运行 UCF101 基准")
        train_set = _get_ucf101_dataset(args, "train")
        test_set = _get_ucf101_dataset(args, "test")
    else:
        raise RuntimeError(f"不支持的视频数据集: {args.video_dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = 2 if args.video_dataset.lower() == "synthetic" else args.video_num_classes
    if args.use_spatiotemporal:
        model = SpatioTemporalBinaryFlowClassifier(dim=args.dim, num_classes=num_classes, seq_len=args.seq_len).to(device)
    else:
        model = VideoBinaryFlowClassifier(dim=args.dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"[Video] Epoch {epoch}/{args.epochs} | loss={loss:.4f} | acc={acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Binary Flow Encoder Benchmark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps")

    parser.add_argument("--image-dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--image-train-samples", type=int, default=5000)
    parser.add_argument("--image-test-samples", type=int, default=1000)

    parser.add_argument("--video-dataset", type=str, default="synthetic")
    parser.add_argument("--video-root", type=str, default="./data/ucf101")
    parser.add_argument("--ucf101-annotation-path", type=str, default="./data/ucf101/ucfTrainTestlist")
    parser.add_argument("--video-num-classes", type=int, default=101)
    parser.add_argument("--video-train-samples", type=int, default=2000)
    parser.add_argument("--video-test-samples", type=int, default=500)
    parser.add_argument("--use-spatiotemporal", action="store_true", help="Use spatiotemporal attention head")
    parser.add_argument("--use-joint-text", action="store_true", help="Use joint text-video training")
    parser.add_argument("--text-dataset", type=str, default="imdb", help="Text dataset for joint training")
    parser.add_argument("--text-model", type=str, default="bert-base-uncased")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--video-size", type=int, default=32)

    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-video", action="store_true")

    args = parser.parse_args()
    device = get_device(args.device)

    print(f"Using device: {device}")

    if not args.skip_image:
        run_image_benchmark(args, device)

    if not args.skip_video:
        run_video_benchmark(args, device)


if __name__ == "__main__":
    main()

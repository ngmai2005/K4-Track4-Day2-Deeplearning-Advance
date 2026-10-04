"""DeepWeeds fold loading, safeguards, transforms and loaders."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN, IMAGENET_STD = (0.485, .456, .406), (.229, .224, .225)

def load_split(labels_dir: str | Path, fold: int = 0):
    root = Path(labels_dir)
    files = [root / f"{s}_subset{fold}.csv" for s in ("train", "val", "test")]
    if not all(p.exists() for p in files):
        raise FileNotFoundError(f"Missing fold {fold} CSV in {root}")
    return tuple(pd.read_csv(p) for p in files)

def check_split(train_df, val_df, test_df, images_dir: str | Path):
    required = {"Filename", "Label", "Species"}
    groups = {"train": train_df, "val": val_df, "test": test_df}
    for name, df in groups.items():
        if not required.issubset(df.columns): raise ValueError(f"{name} lacks required columns")
        if df.Filename.duplicated().any(): raise ValueError(f"Duplicate Filename in {name}")
        if not df.Label.between(0, NUM_CLASSES - 1).all(): raise ValueError(f"Invalid labels in {name}")
    names = {k: set(v.Filename) for k, v in groups.items()}
    overlaps = {"train_val": len(names["train"] & names["val"]), "train_test": len(names["train"] & names["test"]), "val_test": len(names["val"] & names["test"])}
    if any(overlaps.values()): raise ValueError(f"Split overlap: {overlaps}")
    union = set.union(*names.values())
    if len(union) != 17509: raise ValueError(f"Expected 17509 files, got {len(union)}")
    image_root = Path(images_dir)
    missing = [n for n in union if not (image_root / n).is_file()]
    if missing: raise FileNotFoundError(f"{len(missing)} image files missing, e.g. {missing[:3]}")
    return {"n": {k: len(v) for k, v in groups.items()}, "per_class": {k: v.Label.value_counts().sort_index().to_dict() for k,v in groups.items()}, "overlap": overlaps}

def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    # DeepWeeds originals are 256x256; a centre crop keeps evaluation deterministic.
    if not train: return T.Compose([T.CenterCrop(img_size), *norm])
    ops = [T.RandomResizedCrop(img_size, scale=(.7, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color": ops += [T.ColorJitter(.2, .2, .15, .05)]
    elif aug == "trivial": ops += [T.TrivialAugmentWide()]
    elif aug == "randaug": ops += [T.RandAugment(num_ops=2, magnitude=7)]
    elif aug != "basic": raise ValueError(f"Unknown aug: {aug}")
    return T.Compose([*ops, *norm])

class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None): self.df, self.images_dir, self.transform = df.reset_index(drop=True), Path(images_dir), transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]; name = str(row.Filename)
        with Image.open(self.images_dir / name) as im: image = im.convert("RGB")
        return (self.transform(image) if self.transform else image), int(row.Label), name

def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32; random.seed(seed); np.random.seed(seed)

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    ds = DeepWeedsDataset(df, images_dir, transform)
    weighted = None
    if sampler == "balanced":
        counts = df.Label.value_counts(); weighted = WeightedRandomSampler(df.Label.map(lambda x: 1 / counts[x]).to_numpy(), len(df), replacement=True)
    elif sampler is not None: raise ValueError("sampler must be None or 'balanced'")
    return DataLoader(ds, batch_size=batch_size, shuffle=train and weighted is None, sampler=weighted, drop_last=train, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, worker_init_fn=_seed_worker)

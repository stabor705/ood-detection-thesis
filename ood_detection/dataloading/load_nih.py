from dataclasses import dataclass

import kagglehub
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch


class NIHChestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(row["target"], dtype=torch.float32)
        return img, target


def _download_and_find_csv():
    output_dir = Path("data/nih-chest-xrays")
    try:
        root = Path(
            kagglehub.dataset_download("nih-chest-xrays/data", output_dir=output_dir)
        )
    except Exception:
        root = Path(output_dir)

    candidate_csv = [
        root / "Data_Entry_2017.csv",
        root / "data" / "Data_Entry_2017.csv",
    ]
    csv_path = next((p for p in candidate_csv if p.exists()), None)
    if csv_path is None:
        csv_matches = list(root.rglob("Data_Entry_2017*.csv"))
        if not csv_matches:
            raise FileNotFoundError(
                "Could not find Data_Entry_2017 CSV in downloaded dataset."
            )
        csv_path = csv_matches[0]
    return root, csv_path


def _clean_dataframe_columns(df):
    if "Image Index" not in df.columns:
        alt_cols = [
            c for c in df.columns if c.lower().replace("_", " ") == "image index"
        ]
        if alt_cols:
            df = df.rename(columns={alt_cols[0]: "Image Index"})
    if "Finding Labels" not in df.columns:
        alt_cols = [
            c for c in df.columns if c.lower().replace("_", " ") == "finding labels"
        ]
        if alt_cols:
            df = df.rename(columns={alt_cols[0]: "Finding Labels"})

    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError("CSV must contain 'Image Index' and 'Finding Labels' columns.")
    return df


def _match_images_to_df(df, root):
    img_exts = {".png", ".jpg", ".jpeg"}
    image_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in img_exts])
    if not image_files:
        raise FileNotFoundError(
            "No image files found. Make sure the dataset extracted correctly."
        )

    name_to_path = {p.name: p for p in image_files}

    df["image_path"] = df["Image Index"].map(
        lambda x: str(name_to_path.get(str(x), ""))
    )
    df = df[df["image_path"] != ""].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No metadata rows matched image files.")
    return df


def _prepare_labels(df):
    id_df = df[df["Finding Labels"] != "No Finding"].copy()
    ood_df = df[df["Finding Labels"] == "No Finding"].copy()

    # Extract all unique diseases split by '|'
    id_df["finding_list"] = id_df["Finding Labels"].apply(lambda x: str(x).split("|"))
    id_df = id_df[id_df["finding_list"].apply(len) == 1]
    id_df["finding_list"] = id_df["finding_list"].apply(lambda x: x[0])

    lb = LabelBinarizer()
    targets = lb.fit_transform(id_df["finding_list"])

    # Add target column as float32 arrays suitable for PyTorch BCEWithLogitsLoss
    id_df["target"] = list(targets.astype(np.float32))

    NUM_CLASSES = len(lb.classes_)
    LABEL_NAMES = list(lb.classes_)

    # OOD samples (No Finding) have 0 for all disease classes
    ood_df["target"] = [np.zeros(NUM_CLASSES, dtype=np.float32)] * len(ood_df)

    return id_df, ood_df, NUM_CLASSES, LABEL_NAMES


def normalize_x_ray(x):
    return x * 2048 - 1024


def get_transforms(image_size=224):
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Lambda(normalize_x_ray),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(normalize_x_ray),
        ]
    )
    return train_tfms, val_tfms


def load_nih(image_size=224, batch_size=64):
    root, csv_path = _download_and_find_csv()
    print("Using metadata:", csv_path)

    df = pd.read_csv(csv_path)
    df = _clean_dataframe_columns(df)
    df = _match_images_to_df(df, root)

    print(f"Matched rows with images: {len(df):,}")

    id_df, ood_df, num_classes, label_names = _prepare_labels(df)
    return NIHData(
        id_df=id_df, ood_df=ood_df, num_classes=num_classes, label_names=label_names
    )


@dataclass
class NIHData:
    id_df: pd.DataFrame
    ood_df: pd.DataFrame
    num_classes: int
    label_names: list

import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional
from glob import glob

from monai.data import Dataset, DataLoader
from monai.data.image_reader import ITKReader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    ToTensord,
    NormalizeIntensityd,
)
from sklearn.model_selection import train_test_split


def load_brats(path: str = "../data/brats"):
    dicom_reader = ITKReader()
    brats_base_dir = Path(path)

    train_data_dirs = [{"image": d} for d in glob(str(brats_base_dir / "train" / "*" / "T1w"))]
    test_data_dirs = [{"image": d} for d in glob(str(brats_base_dir / "test" / "*" / "T1w"))]

    print(f"Found {len(train_data_dirs)} Train BRATS volumes")
    print(f"Found {len(test_data_dirs)} Test BRATS volumes")

    data = train_data_dirs + test_data_dirs

    transform = Compose([
        LoadImaged(keys=["image"], reader=dicom_reader, image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        # CT data typically has HU values (-1000 to 1000)
        # MR data has different intensity ranges, but we'll normalize similarly
        # For OOD detection, we want to normalize to [0, 1] range
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])

    dataset = Dataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return dataloader



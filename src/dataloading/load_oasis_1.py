import glob
from pathlib import Path
from monai.data import Dataset, DataLoader
from monai.data.image_reader import NibabelReader
from sklearn.model_selection import train_test_split
import nibabel  # ensure dependency available
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Rotate90d,
    ToTensord,
)


def load_oasis_1(path: str = "../data/oasis-1"):
    """
    Load OASIS-1 dataset and return train, validation, and test data loaders.

    OASIS-1 contains cross-sectional MRI data of subjects.
    Data is stored in Analyze format (.hdr/.img pairs).
    Uses the processed averaged images from PROCESSED/MPRAGE/SUBJ_111.
    """
    base_path = Path(path)

    # Find all image files from all discs
    all_data = find_image_files(base_path)

    # Split into train, validation, and test (60/20/20)
    train_data, temp_data = train_test_split(all_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train data: {len(train_data)} files")
    print(f"Validation data: {len(val_data)} files")
    print(f"Test data: {len(test_data)} files")

    train_loader = create_loader(train_data)
    val_loader = create_loader(val_data)
    test_loader = create_loader(test_data)

    return train_loader, val_loader, test_loader


def find_image_files(base_path: Path):
    """
    Find all processed MRI image files in the OASIS-1 dataset.

    Structure: oasis-1/oasis_cross-sectional_disc{1-12}/disc{1-12}/OAS1_XXXX_MR1/
               - RAW/: raw scans (mpr-{1,2,3,4}_anon.hdr/.img)
               - PROCESSED/MPRAGE/SUBJ_111/: processed averaged images
               - FSL_SEG/: FSL segmentation masks

    We use the processed SUBJ_111 images as they are averaged and registered.
    """
    image_files = []

    # Search in all disc directories (disc1 to disc12)
    for disc_num in range(1, 13):
        disc_dir = (
            base_path / f"oasis_cross-sectional_disc{disc_num}" / f"disc{disc_num}"
        )
        if not disc_dir.exists():
            continue

        # Find all processed images in SUBJ_111 folders
        hdr_files = sorted(
            glob.glob(
                str(disc_dir / "**/PROCESSED/MPRAGE/SUBJ_111/*.hdr"), recursive=True
            )
        )
        image_files.extend(hdr_files)

    print(f"Found {len(image_files)} image files")

    # Create data dictionaries (OASIS-1 doesn't have lesion segmentation labels)
    data = [{"image": img} for img in image_files]
    return data


def find_image_files_with_segmentation(base_path: Path):
    """
    Find all processed MRI image files with their FSL segmentation masks.

    Returns data dictionaries with both image and label keys.
    """
    data = []

    # Search in all disc directories (disc1 to disc12)
    for disc_num in range(1, 13):
        disc_dir = (
            base_path / f"oasis_cross-sectional_disc{disc_num}" / f"disc{disc_num}"
        )
        if not disc_dir.exists():
            continue

        # Find all subject directories
        subject_dirs = sorted(glob.glob(str(disc_dir / "OAS1_*_MR*")))

        for subj_dir in subject_dirs:
            subj_path = Path(subj_dir)

            # Find processed image
            img_pattern = str(subj_path / "PROCESSED/MPRAGE/SUBJ_111/*.hdr")
            img_files = glob.glob(img_pattern)

            # Find FSL segmentation
            seg_pattern = str(subj_path / "FSL_SEG/*_fseg.hdr")
            seg_files = glob.glob(seg_pattern)

            if img_files and seg_files:
                data.append({"image": img_files[0], "label": seg_files[0]})

    print(f"Found {len(data)} image-segmentation pairs")
    return data


def create_loader(data, batch_size: int = 1, num_workers: int = 4):
    """
    Create a DataLoader with appropriate transforms for OASIS-1 MRI data.
    """
    nibabel_reader = NibabelReader()

    # Check if data contains labels
    has_labels = "label" in data[0] if data else False
    keys = ["image", "label"] if has_labels else ["image"]

    transforms_list = [
        LoadImaged(keys=keys, reader=nibabel_reader, image_only=False),
        EnsureChannelFirstd(keys=keys),
    ]

    if has_labels:
        transforms_list.extend(
            [
                Spacingd(
                    keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")
                ),
                Orientationd(keys=keys, axcodes="RAS"),
                # Rotate to match ATLAS orientation (frontal lobe on right, view from top)
                Rotate90d(keys=keys, k=1, spatial_axes=(0, 2)),
                Rotate90d(keys=keys, k=1, spatial_axes=(1, 2)),
            ]
        )
    else:
        transforms_list.extend(
            [
                Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Orientationd(keys=keys, axcodes="RAS"),
                # Rotate to match ATLAS orientation (frontal lobe on right, view from top)
                Rotate90d(keys=keys, k=1, spatial_axes=(0, 2)),
                Rotate90d(keys=keys, k=1, spatial_axes=(1, 2)),
            ]
        )

    transforms_list.extend(
        [
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=keys, source_key="image"),
            ToTensord(keys=keys),
        ]
    )

    transform = Compose(transforms_list)
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    return dataloader


def load_oasis_1_with_segmentation(path: str = "../data/oasis-1"):
    """
    Load OASIS-1 dataset with FSL segmentation masks.

    Returns train, validation, and test data loaders with both images and labels.
    """
    base_path = Path(path)

    # Find all image files with segmentation from all discs
    all_data = find_image_files_with_segmentation(base_path)

    # Split into train, validation, and test (60/20/20)
    train_data, temp_data = train_test_split(all_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train data: {len(train_data)} files")
    print(f"Validation data: {len(val_data)} files")
    print(f"Test data: {len(test_data)} files")

    train_loader = create_loader(train_data)
    val_loader = create_loader(val_data)
    test_loader = create_loader(test_data)

    return train_loader, val_loader, test_loader

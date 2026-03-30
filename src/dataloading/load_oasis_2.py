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
    ToTensord,
)


def load_oasis_2(path: str = "../data/oasis-2"):
    """
    Load OASIS-2 dataset and return train, validation, and test data loaders.

    OASIS-2 contains longitudinal MRI data of subjects.
    Data is stored in NIfTI format (.nifti.hdr/.nifti.img pairs).
    """
    base_path = Path(path)

    # Find all image files from both parts
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
    Find all NIfTI image files in the OASIS-2 dataset.

    Structure: oasis-2/OAS2_RAW_PART{1,2}/OAS2_XXXX_MRX/RAW/mpr-{1,2,3}.nifti.{hdr,img}
    """
    image_files = []

    # Search in both PART1 and PART2 directories
    for part_dir in ["OAS2_RAW_PART1", "OAS2_RAW_PART2"]:
        part_path = base_path / part_dir
        if not part_path.exists():
            continue

        # Find all .nifti.hdr files (we use .hdr as the reference, nibabel will load both .hdr and .img)
        hdr_files = sorted(
            glob.glob(str(part_path / "**/RAW/*.nifti.hdr"), recursive=True)
        )
        image_files.extend(hdr_files)

    print(f"Found {len(image_files)} image files")

    # Create data dictionaries (OASIS-2 doesn't have segmentation labels)
    data = [{"image": img} for img in image_files]
    return data


def create_loader(data, batch_size: int = 1, num_workers: int = 4):
    """
    Create a DataLoader with appropriate transforms for OASIS-2 MRI data.
    """
    nibabel_reader = NibabelReader()
    transform = Compose(
        [
            LoadImaged(keys=["image"], reader=nibabel_reader, image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    return dataloader

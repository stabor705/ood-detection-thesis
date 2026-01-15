from monai.data.image_reader import ITKReader
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
)
from monai.data import Dataset, DataLoader

def load_chaos(path: str = "../data/chaos"):
    # Initialize DICOM reader (ITKReader can handle DICOM series)
    dicom_reader = ITKReader()

    # Base CHAOS data directory
    chaos_base_dir = Path(path)

    # Collect all CHAOS data
    chaos_data = []

    # Train CT
    train_ct_dir = chaos_base_dir / "CHAOS_Train_Sets" / "Train_Sets" / "CT"
    train_ct_data = collect_ct_data(train_ct_dir)
    chaos_data.extend(train_ct_data)
    print(f"Found {len(train_ct_data)} Train CT volumes")

    # Test CT
    test_ct_dir = chaos_base_dir / "CHAOS_Test_Sets" / "Test_Sets" / "CT"
    test_ct_data = collect_ct_data(test_ct_dir)
    chaos_data.extend(test_ct_data)
    print(f"Found {len(test_ct_data)} Test CT volumes")

    # Train MR
    train_mr_dir = chaos_base_dir / "CHAOS_Train_Sets" / "Train_Sets" / "MR"
    train_mr_data = collect_mr_data(train_mr_dir)
    chaos_data.extend(train_mr_data)
    print(f"Found {len(train_mr_data)} Train MR volumes")

    # Test MR
    test_mr_dir = chaos_base_dir / "CHAOS_Test_Sets" / "Test_Sets" / "MR"
    test_mr_data = collect_mr_data(test_mr_dir)
    chaos_data.extend(test_mr_data)
    print(f"Found {len(test_mr_data)} Test MR volumes")

    print(f"Total CHAOS volumes: {len(chaos_data)}")

    # CHAOS transforms - similar to ATLAS but adapted for different modalities
    # Note: CHAOS data doesn't have labels for OOD detection, so we only load images
    chaos_transform = Compose([
        LoadImaged(keys=["image"], reader=dicom_reader, image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        # CT data typically has HU values (-1000 to 1000)
        # MR data has different intensity ranges, but we'll normalize similarly
        # For OOD detection, we want to normalize to [0, 1] range
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])

    chaos_ds = Dataset(data=chaos_data, transform=chaos_transform)
    chaos_loader = DataLoader(chaos_ds, batch_size=1, num_workers=4, pin_memory=True)

    print(f"CHAOS dataloader created with {len(chaos_ds)} samples")

    return chaos_loader

def collect_ct_data(ct_dir):
    """Collect CT DICOM series from a CT directory."""
    ct_data = []
    if not ct_dir.exists():
        return ct_data
    
    patient_dirs = sorted([d for d in ct_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    for patient_dir in patient_dirs:
        dicom_dir = patient_dir / "DICOM_anon"
        if dicom_dir.exists():
            dicom_files = sorted(list(dicom_dir.glob("*.dcm")))
            if len(dicom_files) > 0:
                ct_data.append({"image": str(dicom_dir), "modality": "CT"})
    return ct_data

# Function to collect MR data from a directory
def collect_mr_data(mr_dir):
    """Collect MR DICOM series from a MR directory (T1DUAL and T2SPIR)."""
    mr_data = []
    if not mr_dir.exists():
        return mr_data
    
    patient_dirs = sorted([d for d in mr_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    for patient_dir in patient_dirs:
        # CHAOS MR has T1DUAL and T2SPIR sequences
        for sequence in ["T1DUAL", "T2SPIR"]:
            seq_dir = patient_dir / sequence
            if seq_dir.exists():
                dicom_files = sorted(list(seq_dir.glob("*.dcm")))
                if len(dicom_files) > 0:
                    mr_data.append({
                        "image": str(seq_dir), 
                        "modality": "MR",
                        "sequence": sequence
                    })
    return mr_data
"""Industrial dataset adapters, registry, and preprocessing utilities."""

from forgegan.data.audit import audit_dataset
from forgegan.data.loader import collate_anomaly_samples, create_dataloader
from forgegan.data.mvtec import MVTecDataset, MVTecRecord, load_mvtec_records, validate_mvtec_split
from forgegan.data.registry import (
    DATASET_NAMES,
    DatasetConfig,
    NormalizedDataset,
    create_dataset,
    validate_source_splits,
)
from forgegan.data.transforms import SpatialTransform
from forgegan.data.visa import VisaDataset, VisaRecord, load_visa_records, validate_visa_split

__all__ = [
    "DATASET_NAMES",
    "DatasetConfig",
    "MVTecDataset",
    "MVTecRecord",
    "NormalizedDataset",
    "SpatialTransform",
    "VisaDataset",
    "VisaRecord",
    "audit_dataset",
    "collate_anomaly_samples",
    "create_dataloader",
    "create_dataset",
    "load_mvtec_records",
    "load_visa_records",
    "validate_mvtec_split",
    "validate_source_splits",
    "validate_visa_split",
]

from .kitti import KittiDataset
TartanKittiDataset = KittiDataset  # backward-compat alias
from .tartan_pt import TartanPT
from .torch_wrappers import TorchKittiDataset, TorchKittiIterDataset, TorchTartanPTDataset
# backward-compat aliases
TorchTKDataset = TorchKittiDataset
TorchTKIterDataset = TorchKittiIterDataset
TorchTPTDataset = TorchTartanPTDataset

__all__ = [
    "TartanKittiDataset",
    "TorchKittiDataset",
    "TorchKittiIterDataset",
    "TorchTartanPTDataset",
    "TorchTKDataset",
    "TorchTKIterDataset",
    "TorchTPTDataset",
    "TartanPT",
]

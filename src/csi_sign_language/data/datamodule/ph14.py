from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from ..dataset.phoenix14 import MyPhoenix14Dataset, MyPhoenix14DatasetV2
from ...data_utils.ph14.post_process import PostProcess
from ...data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from ...data_utils.base import IPostProcess, IEvaluator
from ..dataset.phoenix14 import CollateFn


class Ph14DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        feature_dir,
        batch_size,
        num_workers,
        train_shuffle,
        thread_pool=None,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        excluded_ids=[],
    ) -> None:
        super().__init__()
        self.data_root = data_dir
        self.feature_dir = feature_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.v_transform = val_transform
        self.test_transform = test_transform

        self.train_shuffle = train_shuffle

        self.train_set = MyPhoenix14DatasetV2(
            self.data_root,
            self.feature_dir,
            "train",
            thread_pool=thread_pool,
            transform=self.train_transform,
            excluded_ids=excluded_ids,
        )
        self.val_set = MyPhoenix14DatasetV2(
            self.data_root,
            self.feature_dir,
            "dev",
            thread_pool=thread_pool,
            transform=self.v_transform,
            excluded_ids=excluded_ids,
        )
        self.test_set = MyPhoenix14DatasetV2(
            self.data_root,
            self.feature_dir,
            "test",
            thread_pool=thread_pool,
            transform=self.test_transform,
            excluded_ids=excluded_ids,
        )
        self.collate_fn = CollateFn()

    def get_vocab(self):
        return self.train_set.get_vocab()

    def create_evaluator(self, origin_data_root: str, mode: str):
        return Pheonix14Evaluator(
            data_root=origin_data_root,
            subset="multisigner",
            mode=mode,
        )

    @staticmethod
    def get_post_process():
        return PostProcess()

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=CollateFn(),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=CollateFn(),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=CollateFn(),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        return self.test_dataloader()

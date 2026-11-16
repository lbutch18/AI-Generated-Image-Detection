import pandas as pd 
import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image 
import torch
class AIGenDetDataset(Dataset):
    def __init__(self, shard_dir=None, shard_nums=None, transform=None,
                 *, image_dir=None, labels_csv=None):
        """ 
        Example dataset class for NTIRE 2026 Robust AI-generated Image Detection In The Wild challenge.

        Arguments:
            - shard_dir (str): base directory where shards are stored 
            - image_dir, labels_csv: alternative image folder and labeled CSV
                for official validation/test splits; do not combine with shards.
            - shard_nums (None or list[int]): a list of specific shard numbers (0 to 6) to use. 
                Uses all shards found in shard_dir by default.
        """
        self.transform = transform
        self.image_dir = Path(image_dir) if image_dir is not None else None
        if image_dir is not None or labels_csv is not None:
            if image_dir is None or labels_csv is None:
                raise ValueError('Provide both image_dir and labels_csv.')
            if shard_dir is not None or shard_nums is not None:
                raise ValueError('Use either shards or image_dir/labels_csv.')
            self.label_df = pd.read_csv(labels_csv)
            if not {'image_name', 'label'}.issubset(self.label_df.columns):
                raise ValueError('Labels CSV must contain image_name and label.')
            print(f'Found {len(self.label_df)} images in {image_dir}.')
            return
        if shard_dir is None:
            raise ValueError('Provide shard_dir or image_dir/labels_csv.')
        self.shard_root_dir = shard_dir
        if shard_nums is None:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in range(0,6)]
        else:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in shard_nums]
        self.shard_dirs = [x for x in self.shard_dirs if os.path.isdir(x)]
        label_dfs = [pd.read_csv(os.path.join(x, 'labels.csv'), index_col=0) for x in self.shard_dirs]
        self.label_df = pd.DataFrame(columns=['image_name', 'label', 'shard_name'])
        for idx, ldf in enumerate(label_dfs):
            ldf['shard_name'] = Path(self.shard_dirs[idx]).name
            self.label_df = pd.concat([self.label_df, ldf], ignore_index=True)
        print(f'Found {len(self.shard_dirs)} shards, {len(self.label_df)} images in total.')

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.image_dir is not None:
            img_path = self.image_dir / self.label_df.loc[idx, 'image_name']
        else:
            img_path = os.path.join(self.shard_root_dir,
                                    self.label_df.loc[idx, 'shard_name'], 'images', self.label_df.loc[idx, 'image_name'])
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_df.loc[idx, 'label']
        sample = {'image': image, 'label': label}
        if 'is_distorted' in self.label_df.columns:
            sample['is_distorted'] = int(self.label_df.loc[idx, 'is_distorted'])
        return sample

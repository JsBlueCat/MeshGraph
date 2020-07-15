import os
import os.path as osp
import glob


import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from data.load_obj import read_obj


class HumanSeg(InMemoryDataset):
    def __init__(self,
                 root,
                 name='seg',
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert name in ['seg']
        self.name = name
        self.train = train
        super(HumanSeg, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'seg', 'sseg', 'test', 'train'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(
            'https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz', self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'human_seg')
        os.rename(folder, self.raw_dir)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])
        data_list = []
        seg_folder = osp.join(self.raw_dir, 'seg')
        sseg_folder = osp.join(self.raw_dir, 'sseg')
        data_folder = osp.join(self.raw_dir, dataset)
        paths = glob.glob('{}/*.obj'.format(data_folder))
        for path in paths:
            data = read_obj(path)
            print(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

import unittest
import torch
from tensorboardX import SummaryWriter


class EmbeddingTest(unittest.TestCase):
    def test_embedding(self):
        w = SummaryWriter()
        all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
        all_labels = torch.Tensor([33, 44, 55])
        all_images = torch.zeros(3, 3, 5, 5)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)
        # assert...

    def test_embedding_64(self):
        w = SummaryWriter()
        all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
        all_labels = torch.Tensor([33, 44, 55])
        all_images = torch.zeros((3, 3, 5, 5), dtype=torch.float64)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)

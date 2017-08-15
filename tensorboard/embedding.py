import os


def make_tsv(metadata, save_path):
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    # we have to make labels_img squared!
    import math
    import torch
    import torchvision
    # this ensure we have enought space for the images
    base_size = int(math.ceil((label_img.size(0)) ** 0.5))
    # cat the images to reach the square of base_size
    label_img = torch.cat((label_img, torch.zeros(base_size ** 2 - label_img.size(0), *label_img.size()[1:])), 0)
    # this call make_grid insied, but now we can ensure a square grid
    torchvision.utils.save_image(label_img, os.path.join(save_path, 'sprite.png'), nrow=base_size, padding=0)


def make_pbtxt(save_path, metadata, label_img):
    with open(os.path.join(save_path, 'projector_config.pbtxt'), 'w') as f:
        f.write('embeddings {\n')
        f.write('tensor_name: "embedding:0"\n')
        f.write('tensor_path: "tensors.tsv"\n')
        if metadata is not None:
            f.write('metadata_path: "metadata.tsv"\n')
        if label_img is not None:
            f.write('sprite {\n')
            f.write('image_path: "sprite.png"\n')
            f.write('single_image_dim: {}\n'.format(label_img.size(3)))
            f.write('single_image_dim: {}\n'.format(label_img.size(2)))
            f.write('}\n')
        f.write('}\n')


def make_mat(matlist, save_path):
    with open(os.path.join(save_path, 'tensors.tsv'), 'w') as f:
        for x in matlist:
            x = [str(i) for i in x]
            f.write('\t'.join(x) + '\n')


def add_embedding(mat, save_path, metadata=None, label_img=None):
    """add embedding

    Args:
        mat (torch.Tensor): A matrix which each row is the feature vector of the data point
        save_path (string): Save path (use ``writer.file_writer.get_logdir()`` to show embedding along with other summaries)
        metadata (list): A list of labels, each element will be convert to string
        label_img (torch.Tensor): Images correspond to each data point
    Shape:
        mat: :math:`(N, D)`, where N is number of data and D is feature dimension

        label_img: :math:`(N, C, H, W)`

    .. note::
        ~~This function needs tensorflow installed. It invokes tensorflow to dump data. ~~
        Therefore I separate it from the SummaryWriter class. Please pass ``writer.file_writer.get_logdir()`` to ``save_path`` to prevent glitches.

        If ``save_path`` is different than SummaryWritter's save path, you need to pass the leave directory to tensorboard's logdir argument,
        otherwise it cannot display anything. e.g. if ``save_path`` equals 'path/to/embedding',
        you need to call 'tensorboard --logdir=path/to/embedding', instead of 'tensorboard --logdir=path'.


    Examples::

        from tensorboard.embedding import add_embedding
        import keyword
        import torch
        meta = []
        while len(meta)<100:
            meta = meta+keyword.kwlist # get some strings
        meta = meta[:100]

        for i, v in enumerate(meta):
            meta[i] = v+str(i)

        label_img = torch.rand(100, 3, 10, 32)
        for i in range(100):
            label_img[i]*=i/100.0

        add_embedding(torch.randn(100, 5), 'embedding1', metadata=meta, label_img=label_img)
        add_embedding(torch.randn(100, 5), 'embedding2', label_img=label_img)
        add_embedding(torch.randn(100, 5), 'embedding3', metadata=meta)
    """
    try:
        os.makedirs(save_path)
    except OSError:
        print('warning: dir exists')
    if metadata is not None:
        assert mat.size(0) == len(metadata), '#labels should equal with #data points'
        make_tsv(metadata, save_path)
    if label_img is not None:
        assert mat.size(0) == label_img.size(0), '#images should equal with #data points'
        make_sprite(label_img, save_path)
    assert mat.dim() == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
    make_mat(mat.tolist(), save_path)
    make_pbtxt(save_path, metadata, label_img)


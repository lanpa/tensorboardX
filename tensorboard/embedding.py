import os

def make_tsv(metadata, save_path):
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    import math
    import torch
    import torchvision
    # this ensures the sprite image has correct dimension as described in 
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = int(math.ceil((label_img.size(0)) ** 0.5))
    
    # augment images so that #images equals nrow*nrow
    label_img = torch.cat((label_img, torch.randn(nrow ** 2 - label_img.size(0), *label_img.size()[1:]) * 255), 0)
    
    # Dirty fix: no pixel are appended by make_grid call in save_image (https://github.com/pytorch/vision/issues/206)
    xx = torchvision.utils.make_grid(torch.Tensor(1, 3, 32, 32), padding=0)
    if xx.size(2) == 33:
        sprite = torchvision.utils.make_grid(label_img, nrow=nrow, padding=0)
        sprite = sprite[:, 1:, 1:]
        torchvision.utils.save_image(sprite, os.path.join(save_path, 'sprite.png'))
    else:
        torchvision.utils.save_image(label_img, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)

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
        assert mat.size(0)==len(metadata), '#labels should equal with #data points'
        make_tsv(metadata, save_path)
    if label_img is not None:
        assert mat.size(0)==label_img.size(0), '#images should equal with #data points'
        make_sprite(label_img, save_path)
    assert mat.dim()==2, 'mat should be 2D, where mat.size(0) is the number of data points'
    make_mat(mat.tolist(), save_path)
    make_pbtxt(save_path, metadata, label_img)

def append_pbtxt(f, metadata, label_img,path):

    f.write('embeddings {\n')
    f.write('tensor_name: "{}"\n'.format(os.path.join(path,"embedding")))
    f.write('tensor_path: "{}"\n'.format(os.path.join(path,"tensors.tsv")))
    if metadata is not None:
        f.write('metadata_path: "{}"\n'.format(os.path.join(path,"metadata.tsv")))
    if label_img is not None:
        f.write('sprite {\n')
        f.write('image_path: "{}"\n'.format(os.path.join(path,"sprite.png")))
        f.write('single_image_dim: {}\n'.format(label_img.size(3)))
        f.write('single_image_dim: {}\n'.format(label_img.size(2)))
        f.write('}\n')
    f.write('}\n')


class EmbeddingWriter(object):
    """
    Class to allow writing embeddings ad defined timestep

    """
    def __init__(self,save_path):
        """

        :param save_path: should be the same path of you SummaryWriter
        """
        self.save_path = save_path
        #make dir if needed, it should not
        try:
            os.makedirs(save_path)
        except OSError:
            print('warning: dir exists')
        #create config file to store all embeddings conf
        self.f = open(os.path.join(save_path, 'projector_config.pbtxt'), 'w')

    def add_embedding(self,mat, metadata=None, label_img=None,timestep=0):
        """
        add an embedding at the defined timestep

        :param mat:
        :param metadata:
        :param label_img:
        :param timestep:
        :return:
        """
        # TODO make doc
        #path to the new subdir
        timestep_path = "{}".format(timestep)
        # TODO should this be handled?
        os.makedirs(os.path.join(self.save_path,timestep_path))
        #check other info
        #save all this metadata in the new subfolder
        if metadata is not None:
            assert mat.size(0) == len(metadata), '#labels should equal with #data points'
            make_tsv(metadata, os.path.join(self.save_path,timestep_path))
        if label_img is not None:
            assert mat.size(0) == label_img.size(0), '#images should equal with #data points'
            make_sprite(label_img, os.path.join(self.save_path,timestep_path))
        assert mat.dim() == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat.tolist(), os.path.join(self.save_path,timestep_path))
        #new funcion to append to the config file a new embedding
        append_pbtxt(self.f, metadata, label_img,timestep_path)


    def __del__(self):
        #close the file at the end of the script
        self.f.close()
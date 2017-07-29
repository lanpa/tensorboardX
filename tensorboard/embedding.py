import torch
import torchvision
import os

def make_tsv(metadata, save_path):
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x+'\n')


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    import math
    nrow = int(math.floor(math.sqrt(label_img.size(0))))
    xx = torchvision.utils.make_grid(torch.Tensor(1,3,32,32), padding=0)
    if xx.size(2)==33: # https://github.com/pytorch/vision/issues/206
        sprite = torchvision.utils.make_grid(label_img, nrow=nrow, padding=0)
        sprite = sprite[:,1:,1:]
        torchvision.utils.save_image(sprite, os.path.join(save_path, 'sprite.png'))
    else:
        torchvision.utils.save_image(label_img, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)

def make_pbtxt(save_path, metadata, label_img):
    with open(os.path.join(save_path, 'projector_config.pbtxt'), 'w') as f:
        f.write('embeddings {\n')
        f.write('tensor_name: "embedding:0"\n')
        if metadata is not None:
            f.write('metadata_path: "metadata.tsv"\n')
        if label_img is not None:
            f.write('sprite {\n')
            f.write('image_path: "sprite.png"\n')
            f.write('single_image_dim: {}\n'.format(label_img.size(3)))
            f.write('single_image_dim: {}\n'.format(label_img.size(2)))
            f.write('}\n')
        f.write('}\n')


'''
mat: torch tensor. mat.size(0) is the number of data. mat.size(1) is the cardinality of feature dimensions
save_path: self-explained.
metadata: a list of {int, string} of length equals mat.size(0)
label_img: 4D torch tensor. label_img.size(0) equals mat.size(0). 

'''

def add_embedding(mat, save_path, metadata=None, label_img=None):
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
    import tensorflow as tf
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        emb = tf.Variable(mat.tolist(), name="embedding")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(emb.initializer)
        saver = tf.train.Saver()
        saver.save(sess, save_path=os.path.join(save_path, 'model.ckpt'), global_step=None, write_meta_graph=False)
    make_pbtxt(save_path, metadata, label_img)



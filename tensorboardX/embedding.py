import os
import sys

# Maximum sprite size allowed by TB frontend,
# see https://github.com/lanpa/tensorboardX/issues/516
TB_MAX_SPRITE_SIZE = 8192


def maybe_upload_file(local_path):
    '''Upload a file to remote cloud storage
    if the path starts with gs:// or s3://
    '''
    if local_path.startswith(('s3://', 'gs://')):
        prefix = local_path.split(':')[0]
        remote_bucket_path = local_path[len("s3://"):]  # same length
        bp = remote_bucket_path.split("/")
        bucket = bp[0]
        path = remote_bucket_path[1 + len(bucket):]

        # s3://example/file becomes s3:/example/file in Linux
        local_path = prefix + ':/' + remote_bucket_path
        if prefix == 's3':
            import boto3
            s3 = boto3.client('s3', endpoint_url=os.environ.get('S3_ENDPOINT'))
            s3.upload_file(local_path, bucket, path)

        elif prefix == 'gs':
            from google.cloud import storage
            client = storage.Client()

            Hbucket = storage.Bucket(client, bucket)
            blob = storage.Blob(path, Hbucket)
            blob.upload_from_filename(local_path)


def make_tsv(metadata, save_path, metadata_header=None):
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        assert len(metadata_header) == len(metadata[0]), \
            'len of header must be equal to the number of columns in metadata'
        metadata = ['\t'.join(str(e) for e in l)
                    for l in [metadata_header] + metadata]

    named_path = os.path.join(save_path, 'metadata.tsv')

    with open(named_path, 'w', encoding='utf8') as f:
        for x in metadata:
            f.write(x + '\n')
    maybe_upload_file(named_path)


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    import math

    import numpy as np
    from PIL import Image

    from .utils import make_grid
    from .x2num import make_np
    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    # There are some constraints for the sprite image:
    # 1. The sprite image should be square.
    # 2. Each image patch in the sprite image should be square.
    # 2. The content is row major order, so we can padding the image on the
    #    bottom, but not on the right, otherwise, TB will treat some padded location
    #    as images to be shown.
    # args: label_img: tensor in NCHW

    assert label_img.shape[2] == label_img.shape[3], 'Image should be square, see tensorflow/tensorboard#670'
    total_pixels = label_img.shape[0] * label_img.shape[2] * label_img.shape[3]
    pixels_one_side = total_pixels ** 0.5
    number_of_images_per_row = int(math.ceil(pixels_one_side / label_img.shape[3]))
    arranged_img_CHW = make_grid(make_np(label_img), ncols=number_of_images_per_row)
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc

    sprite_size = arranged_img_CHW.shape[2]
    assert sprite_size <= TB_MAX_SPRITE_SIZE, 'Sprite too large, see label_img shape limits'
    arranged_augment_square_HWC = np.ndarray((sprite_size, sprite_size, 3))
    arranged_augment_square_HWC[:arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))
    named_path = os.path.join(save_path, 'sprite.png')
    im.save(named_path)
    maybe_upload_file(named_path)


def append_pbtxt(metadata, label_img, save_path, subdir, global_step, tag):
    from posixpath import join
    named_path = os.path.join(save_path, 'projector_config.pbtxt')
    with open(named_path, 'a') as f:
        # step = os.path.split(save_path)[-1]
        f.write('embeddings {\n')
        f.write(f'tensor_name: "{tag}:{str(global_step).zfill(5)}"\n')
        f.write('tensor_path: "{}"\n'.format(join(subdir, 'tensors.tsv')))
        if metadata is not None:
            f.write('metadata_path: "{}"\n'.format(
                join(subdir, 'metadata.tsv')))
        if label_img is not None:
            f.write('sprite {\n')
            f.write('image_path: "{}"\n'.format(join(subdir, 'sprite.png')))
            f.write(f'single_image_dim: {label_img.shape[3]}\n')
            f.write(f'single_image_dim: {label_img.shape[2]}\n')
            f.write('}\n')
        f.write('}\n')
    maybe_upload_file(named_path)


def make_mat(matlist, save_path):
    named_path = os.path.join(save_path, 'tensors.tsv')
    with open(named_path, 'w') as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write('\t'.join(x) + '\n')
    maybe_upload_file(named_path)

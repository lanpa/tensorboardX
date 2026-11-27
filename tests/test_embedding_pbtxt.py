import os
import tempfile
import unittest

import numpy as np
from google.protobuf import (
    descriptor_pb2,
    descriptor_pool,
    message_factory,
    text_format,
)

from tensorboardX import SummaryWriter
from tensorboardX.embedding import append_pbtxt


def _make_projector_config_cls():
    fdp = descriptor_pb2.FileDescriptorProto(name="projector_config.proto", package="tb")
    sprite = fdp.message_type.add(name="SpriteMetadata")
    sprite.field.add(name="image_path", number=1, label=1, type=9)
    sprite.field.add(name="single_image_dim", number=2, label=3, type=13)

    emb = fdp.message_type.add(name="EmbeddingInfo")
    emb.field.add(name="tensor_name", number=1, label=1, type=9)
    emb.field.add(name="tensor_path", number=2, label=1, type=9)
    emb.field.add(name="metadata_path", number=3, label=1, type=9)
    emb.field.add(name="sprite", number=4, label=1, type=11, type_name=".tb.SpriteMetadata")

    cfg = fdp.message_type.add(name="ProjectorConfig")
    cfg.field.add(name="embeddings", number=1, label=3, type=11, type_name=".tb.EmbeddingInfo")

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tb.ProjectorConfig"))


class EmbeddingPbtxtTest(unittest.TestCase):
    def test_append_pbtxt_escapes_tag_and_paths(self):
        projector_config_cls = _make_projector_config_cls()
        tag = 'tag"with\\backslashes\nand\r\ttabs'
        subdir = '00003/sub"dir\\path\nline'
        label_img = np.zeros((2, 3, 4, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            append_pbtxt(
                metadata=["a", "b"],
                label_img=label_img,
                save_path=tmpdir,
                subdir=subdir,
                global_step=3,
                tag=tag,
            )
            pbtxt_path = os.path.join(tmpdir, "projector_config.pbtxt")
            with open(pbtxt_path, encoding="utf-8") as f:
                content = f.read()

        parsed = text_format.Parse(content, projector_config_cls())
        self.assertEqual(len(parsed.embeddings), 1)
        entry = parsed.embeddings[0]
        self.assertEqual(entry.tensor_name, f"{tag}:00003")
        self.assertEqual(entry.tensor_path, f"{subdir}/tensors.tsv")
        self.assertEqual(entry.metadata_path, f"{subdir}/metadata.tsv")
        self.assertEqual(entry.sprite.image_path, f"{subdir}/sprite.png")
        self.assertEqual(list(entry.sprite.single_image_dim), [4, 4])

    def test_add_embedding_backslash_tag_produces_valid_pbtxt(self):
        projector_config_cls = _make_projector_config_cls()
        mat = np.zeros((2, 3), dtype=np.float32)
        tag = "layer\\name\\"

        with tempfile.TemporaryDirectory() as tmpdir:
            with SummaryWriter(logdir=tmpdir) as writer:
                writer.add_embedding(mat, global_step=1, tag=tag)
            pbtxt_path = os.path.join(tmpdir, "projector_config.pbtxt")
            with open(pbtxt_path, encoding="utf-8") as f:
                content = f.read()

        parsed = text_format.Parse(content, projector_config_cls())
        self.assertEqual(parsed.embeddings[0].tensor_name, f"{tag}:00001")

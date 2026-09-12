import numpy as np
import pytest
from google.protobuf import text_format
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig

from tensorboardX.embedding import append_pbtxt


@pytest.mark.parametrize("tag", ['layer "quoted"', 'layer\\weights', 'line\nnext', '模型'])
def test_projector_preserves_string_values(tmp_path, tag):
    subdir = 'vectors/"quoted"'
    append_pbtxt(["row"], np.ones((1, 3, 2, 2)), str(tmp_path), subdir, 1, tag)
    config = ProjectorConfig()
    text_format.Parse((tmp_path / "projector_config.pbtxt").read_text(), config)
    embedding = config.embeddings[0]
    assert embedding.tensor_name == f"{tag}:00001"
    assert embedding.tensor_path == subdir + "/tensors.tsv"
    assert embedding.metadata_path == subdir + "/metadata.tsv"
    assert embedding.sprite.image_path == subdir + "/sprite.png"

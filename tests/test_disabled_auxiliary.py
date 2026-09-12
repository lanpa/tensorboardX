import numpy as np
import pytest

from tensorboardX import SummaryWriter


@pytest.mark.parametrize("operation", ["embedding", "hparams"])
def test_auxiliary_logging_respects_write_to_disk(tmp_path, monkeypatch, operation):
    monkeypatch.chdir(tmp_path)
    with SummaryWriter(logdir=str(tmp_path / "logs"), write_to_disk=False,
                       comet_config={"disabled": True}) as writer:
        if operation == "embedding":
            writer.add_embedding(np.ones((2, 3)), metadata=["first", "second"])
        else:
            writer.add_hparams({"lr": 0.1}, {"loss": 0.2}, name="trial")
    assert list(tmp_path.iterdir()) == []

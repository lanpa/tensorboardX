from tensorboardX import SummaryWriter


def test_summary_writer_ctx():
    # after using a SummaryWriter as a ctx it should be closed
    with SummaryWriter() as writer:
        writer.add_scalar('test', 1)
    assert writer.file_writer is None

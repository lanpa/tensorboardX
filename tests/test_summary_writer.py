from tensorboardX import SummaryWriter


def test_summary_writer_ctx():
    # after using a SummaryWriter as a ctx it should be closed
    with SummaryWriter(filename_suffix='.test') as writer:
        writer.add_scalar('test', 1)
    assert writer.file_writer is None


def test_summary_writer_close():
    # Opening and closing SummaryWriter a lot should not run into
    # OSError: [Errno 24] Too many open files
    passed = True
    try:
        for _ in range(2048):
            writer = SummaryWriter()
            writer.close()
    except OSError:
        passed = False

    assert passed

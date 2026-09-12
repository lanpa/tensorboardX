from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

from tensorboardX.event_file_writer import EventsWriter
from tensorboardX.proto.event_pb2 import Event


def test_writers_created_in_same_second_keep_both_event_files(tmp_path, monkeypatch):
    monkeypatch.setattr('tensorboardX.event_file_writer.time.time', lambda: 1234567890.0)
    for step in [1, 2]:
        writer = EventsWriter(str(tmp_path / 'events'))
        writer.write_event(Event(step=step))
        writer.close()
    files = list(tmp_path.glob('events.out.tfevents.*'))
    assert len(files) == 2
    steps = {event.step for file in files for event in EventFileLoader(str(file)).Load() if event.step}
    assert steps == {1, 2}

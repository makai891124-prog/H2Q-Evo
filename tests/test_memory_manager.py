from pathlib import Path

from tools.memory_manager import MetricsRotator, SlidingWindowBuffer, StreamingJSONWriter


def test_sliding_window_buffer_keeps_recent_entries():
    buffer = SlidingWindowBuffer(max_size=3)
    for idx in range(5):
        buffer.append({"idx": idx})
    assert [item["idx"] for item in buffer.get_all()] == [2, 3, 4]


def test_metrics_rotator_trims_records_and_writes_checkpoint(tmp_path: Path):
    rotator = MetricsRotator(max_records=3, checkpoint_size=2, checkpoint_dir=tmp_path)
    for idx in range(5):
        rotator.append({"idx": idx})

    records = rotator.get_records()
    assert [item["idx"] for item in records] == [2, 3, 4]
    checkpoint_files = sorted(tmp_path.glob("metrics_checkpoint_*.json"))
    assert checkpoint_files


def test_streaming_json_writer_writes_iterable_array(tmp_path: Path):
    out = tmp_path / "daily.json"
    with StreamingJSONWriter(out) as writer:
        writer.write_key_value("meta", {"ok": True})
        writer.write_iterable_array("rounds", ({"id": i} for i in range(3)))

    content = out.read_text(encoding="utf-8")
    assert '"rounds": [{"id":0},{"id":1},{"id":2}]' in content

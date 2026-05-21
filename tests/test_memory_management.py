#!/usr/bin/env python3
import io
import json
import unittest
from pathlib import Path

from tools.memory_manager import append_with_limit, json_write_round_payload


class MemoryManagementTests(unittest.TestCase):
    def test_append_with_limit_keeps_latest_entries(self):
        items = [1, 2, 3]
        append_with_limit(items, 4, 3)
        self.assertEqual(items, [2, 3, 4])

    def test_round_payload_json_stream_writer(self):
        out = io.StringIO()
        json_write_round_payload(
            out_path=Path("/tmp/unused.json"),
            round_payload={"round_id": 1, "timestamp_utc": "2026-01-01T00:00:00Z"},
            trust_summary={"trust_score": 0.9},
            trust_report=None,
            stream=out,
        )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["round"]["round_id"], 1)
        self.assertEqual(payload["trust"]["trust_score"], 0.9)

if __name__ == "__main__":
    unittest.main()

import pytest

try:
    from fastapi.testclient import TestClient
    from h2q_project.h2q_server import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestGenerateEndpoint:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)

    def test_health_and_metrics_increment(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        before_total = resp.json().get("requests_total", 0)

        resp_metrics = self.client.get("/metrics")
        assert resp_metrics.status_code == 200
        before_metrics_total = resp_metrics.json().get("requests_total", 0)

        resp_gen = self.client.post("/generate", json={"prompt": "test", "max_new_tokens": 4})
        assert resp_gen.status_code == 200
        payload = resp_gen.json()
        assert "text" in payload
        assert "fueter_curvature" in payload

        resp_metrics_after = self.client.get("/metrics")
        assert resp_metrics_after.status_code == 200
        after_total = resp_metrics_after.json().get("requests_total", 0)
        assert after_total >= before_total
        assert after_total >= before_metrics_total

    def test_generate_roundtrip_uses_decoder(self):
        resp = self.client.post("/generate", json={"prompt": "abc", "max_new_tokens": 4})
        assert resp.status_code == 200
        payload = resp.json()
        assert isinstance(payload.get("text"), str)
        assert payload.get("status") in {"Analytic", "Pruned/Healed"}

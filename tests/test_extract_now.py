import asyncio
import time
from pathlib import Path

import httpx
import pytest

from app import worker
from app.domain.features import AudioFeatures
from app.services import jobs, store
from app.services.download import DownloadedAudio

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.fixture
def client(logged_client: httpx.AsyncClient) -> httpx.AsyncClient:
    return logged_client


@pytest.fixture
def pipeline(monkeypatch):
    """Duplo do download e da análise; o download dorme `delay["first"]` s só na 1ª chamada."""
    monkeypatch.delenv("PROXY_USERNAME", raising=False)
    monkeypatch.delenv("PROXY_PASSWORD", raising=False)
    calls: list[str] = []
    delay = {"first": 0.0}

    def download(video_id: str, dest_dir: Path, proxy: str | None = None) -> DownloadedAudio:
        calls.append(video_id)
        if len(calls) == 1:
            time.sleep(delay["first"])
        return DownloadedAudio(dest_dir / "audio.wav", "Música", 100)

    features = AudioFeatures(
        top_midi=71, ranking=[(71, 150)], top_times=[1.0], chroma_windows=[[0.1] * 12]
    )
    monkeypatch.setattr(worker, "download_audio", download)
    monkeypatch.setattr(worker, "analyze_audio", lambda path: features)
    return calls, delay


async def _progress():
    async with store.session() as session:
        return await jobs.get(session, VIDEO_ID)


async def _saved() -> bool:
    async with store.session() as session:
        return await store.find(session, VIDEO_ID) is not None


async def test_default_only_enqueues(client, pipeline):
    calls, _ = pipeline
    response = await client.post("/api/analyses", json={"url": URL})
    assert response.status_code == 202
    assert calls == []
    assert await _progress() == jobs.Progress("pending")


async def test_extract_now_finishes_without_worker(client, pipeline):
    calls, _ = pipeline
    response = await client.post("/api/analyses", json={"url": URL, "extract_now": True})
    assert response.status_code == 202
    assert calls == [VIDEO_ID]
    assert await _saved()
    assert await _progress() is None


async def test_slow_download_returns_to_pending_and_worker_finishes(client, pipeline, monkeypatch):
    calls, delay = pipeline
    monkeypatch.setattr(worker, "EXTRACT_TIMEOUT_SECONDS", 0.2)
    delay["first"] = 0.6
    await client.post("/api/analyses", json={"url": URL, "extract_now": True})
    assert await _progress() == jobs.Progress("pending")
    await asyncio.sleep(0.8)  # a thread tardia termina, mas o resultado dela é descartado
    assert not await _saved()
    assert await _progress() == jobs.Progress("pending")
    assert await worker.process_next() is True
    assert await _saved()
    assert calls == [VIDEO_ID, VIDEO_ID]


OTHER_ID = "9bZkp7q19f0"
OTHER_URL = f"https://www.youtube.com/watch?v={OTHER_ID}"


async def _post_during_extraction(client, delay):
    """Segundo pedido enquanto o primeiro extrai; devolve (tarefa do primeiro, tempo do segundo)."""
    delay["first"] = 0.5
    first = asyncio.create_task(
        client.post("/api/analyses", json={"url": URL, "extract_now": True})
    )
    await asyncio.sleep(0.2)
    started = time.monotonic()
    response = await client.post("/api/analyses", json={"url": OTHER_URL, "extract_now": True})
    assert response.status_code == 202
    return first, time.monotonic() - started


async def test_request_during_extraction_is_queued_and_extracted_next(client, pipeline):
    calls, delay = pipeline
    first, waited = await _post_during_extraction(client, delay)
    assert waited < 0.3  # entrou na fila e voltou, sem esperar a extração em curso
    assert calls == [VIDEO_ID]
    await first
    assert calls == [VIDEO_ID, OTHER_ID]
    async with store.session() as session:
        assert await store.find(session, OTHER_ID) is not None
        assert await jobs.get(session, OTHER_ID) is None


async def test_queue_beyond_the_budget_is_left_pending_for_the_worker(
    client, pipeline, monkeypatch
):
    calls, delay = pipeline
    monkeypatch.setattr(worker, "DRAIN_BUDGET_SECONDS", 0)
    first, _ = await _post_during_extraction(client, delay)
    await first
    assert calls == [VIDEO_ID]
    async with store.session() as session:
        assert await jobs.get(session, OTHER_ID) == jobs.Progress("pending")
    assert not worker._api_queue and not worker._api_busy

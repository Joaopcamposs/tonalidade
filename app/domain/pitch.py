"""Extração da melodia e do chroma de um áudio: nota mais aguda e onde ela toca.

Usa librosa e scipy, então só o worker importa este módulo.
"""

from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import soxr
from scipy.signal import medfilt

from app.domain.chords import AFTER_SECONDS, BEFORE_SECONDS, WINDOW_SECONDS, moment_chord
from app.domain.errors import NoMelodyError
from app.domain.features import CHROMA_WINDOW_SECONDS, AudioFeatures
from app.domain.key import KeySection, key_sections

BINS_PER_OCTAVE = 36  # 3 divisões por semitom na CQT
HOP = 512
SAMPLE_RATE = 22050
MIN_FRAMES = 100  # a nota mais aguda precisa aparecer em pelo menos ~2,3 s no total
RANKING_SIZE = 8
MAX_TIMES = 5
CHUNK = 5 * SAMPLE_RATE // HOP * HOP  # bloco da análise (~5 s), múltiplo do HOP
OVERLAP = 2 * SAMPLE_RATE // HOP * HOP  # sobra de cada lado, descartada, para as bordas
MIN_RUN_FRAMES = 10  # trechos menores que isso (~0,23 s) não contam como ocorrência
RUN_GAP_SECONDS = 1.0  # quadros separados por menos que isso formam o mesmo trecho


def _load(path: Path) -> np.ndarray:
    """Lê o áudio como mono em SAMPLE_RATE, em pedaços de 10 s, sem carregar o arquivo inteiro.

    Ler e reamostrar o arquivo todo de uma vez (`librosa.load`) tem um pico de ~350 MB.
    """
    with sf.SoundFile(path) as file:
        resampler = soxr.ResampleStream(file.samplerate, SAMPLE_RATE, 1, dtype="float32")
        blocks = file.blocks(file.samplerate * 10, dtype="float32", always_2d=True)
        parts = [resampler.resample_chunk(block.mean(axis=1)) for block in blocks]
        parts.append(resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
    return np.concatenate(parts)


def _harmonic(y: np.ndarray) -> np.ndarray:
    """Parte harmônica do áudio (tira a percussão), calculada em blocos para limitar a memória.

    O HPSS da música inteira passa de 1 GB de pico; os filtros de mediana são locais, então cada
    bloco leva uma sobra de cada lado que é descartada e o resultado praticamente não muda.
    """
    out = np.empty_like(y)
    for start in range(0, len(y), CHUNK):
        end = min(start + CHUNK, len(y))
        left = max(0, start - OVERLAP)
        harmonic = librosa.effects.harmonic(y[left : end + OVERLAP], margin=3.0)
        out[start:end] = harmonic[start - left : end - left]
    return out


def _by_frames(y: np.ndarray, compute: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Aplica `compute` (quadros no último eixo) em blocos de `y` e costura os quadros.

    Mesma ideia do `_harmonic`: cada bloco leva uma sobra que é descartada. Os blocos começam em
    múltiplos de HOP, então os quadros de cada um caem nos mesmos instantes de uma passada única.
    """
    parts: list[np.ndarray] = []
    for start in range(0, len(y), CHUNK):
        end = min(start + CHUNK, len(y))
        left = max(0, start - OVERLAP)
        frames = compute(y[left : end + OVERLAP])
        stop = frames.shape[-1] if end == len(y) else (end - left) // HOP
        parts.append(frames[..., (start - left) // HOP : stop])
    return np.concatenate(parts, axis=-1)


def _tuning(y: np.ndarray) -> float:
    """Afinação da música inteira, como `librosa.estimate_tuning(y=y)`, mas calculada em blocos.

    Sem isso, cada bloco estimaria a sua e a CQT mudaria de um bloco para o outro.
    """
    pitches: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    for start in range(0, len(y), CHUNK):
        end = min(start + CHUNK, len(y))
        left = max(0, start - OVERLAP)
        pitch, magnitude = librosa.piptrack(y=y[left : end + OVERLAP], sr=SAMPLE_RATE)
        stop = pitch.shape[1] if end == len(y) else (end - left) // HOP
        pitch, magnitude = (
            pitch[:, (start - left) // HOP : stop],
            magnitude[:, (start - left) // HOP : stop],
        )
        voiced = pitch > 0
        pitches.append(pitch[voiced])
        magnitudes.append(magnitude[voiced])
    pitch, magnitude = np.concatenate(pitches), np.concatenate(magnitudes)
    threshold = np.median(magnitude) if pitch.size else 0.0
    return librosa.pitch_tuning(pitch[magnitude >= threshold], bins_per_octave=BINS_PER_OCTAVE)


def _melody(y_harmonic: np.ndarray, tuning: float) -> np.ndarray:
    """Nota MIDI e confiança por quadro (linhas 0 e 1), por soma harmônica sobre a CQT."""
    fmin = librosa.note_to_hz("C2")
    cqt = np.abs(
        librosa.cqt(
            y_harmonic,
            sr=SAMPLE_RATE,
            hop_length=HOP,
            fmin=fmin,
            n_bins=BINS_PER_OCTAVE * 6,
            bins_per_octave=BINS_PER_OCTAVE,
            tuning=tuning,
        )
    )
    cqt = librosa.util.normalize(cqt, axis=0, norm=np.inf)

    low = round(BINS_PER_OCTAVE * np.log2(librosa.note_to_hz("E3") / fmin))
    high = round(BINS_PER_OCTAVE * np.log2(librosa.note_to_hz("G5") / fmin))
    salience = np.zeros((high - low + 1, cqt.shape[1]))
    for k, b in enumerate(range(low, high + 1)):  # f0 mais os harmônicos 2 a 5
        for h, w in zip(range(1, 6), (1.0, 0.8, 0.6, 0.45, 0.3), strict=True):
            bb = b + round(BINS_PER_OCTAVE * np.log2(h))
            if bb < cqt.shape[0]:
                salience[k] += w * cqt[bb]

    midi = librosa.hz_to_midi(fmin * 2 ** ((salience.argmax(axis=0) + low) / BINS_PER_OCTAVE))
    return np.stack([medfilt(midi, 9), salience.max(axis=0)])


def _runs(frames: np.ndarray) -> list[float]:
    """Início (em segundos) de cada trecho contínuo de quadros, ignorando blips."""
    if frames.size == 0:
        return []
    gap = RUN_GAP_SECONDS * SAMPLE_RATE / HOP
    starts: list[float] = []
    begin = previous = int(frames[0])
    count = 1
    for frame in (int(f) for f in frames[1:]):
        if frame - previous > gap:
            if count >= MIN_RUN_FRAMES:
                starts.append(begin * HOP / SAMPLE_RATE)
            begin, count = frame, 0
        previous = frame
        count += 1
    if count >= MIN_RUN_FRAMES:
        starts.append(begin * HOP / SAMPLE_RATE)
    return starts


def rank_notes(notes: list[int], counts: list[int]) -> tuple[int, list[tuple[int, int]]]:
    """Nota mais aguda sustentada (>= MIN_FRAMES) e o ranking das sustentadas, do agudo ao grave."""
    pairs = zip(notes, counts, strict=True)
    sustained = sorted([(n, c) for n, c in pairs if c >= MIN_FRAMES], reverse=True)
    if not sustained:
        raise NoMelodyError("Não encontrei uma melodia detectável neste áudio.")
    return sustained[0][0], sustained[:RANKING_SIZE]


def _chroma_windows(chroma: np.ndarray) -> list[list[float]]:
    """Média do chroma em janelas de CHROMA_WINDOW_SECONDS; a sobra final só conta se couber meia.

    Sempre há ao menos uma janela: sem ~5 s de áudio a melodia nem seria detectada.
    """
    size = round(CHROMA_WINDOW_SECONDS * SAMPLE_RATE / HOP)
    starts = [s for s in range(0, chroma.shape[1], size) if chroma.shape[1] - s >= size / 2]
    return [chroma[:, s : s + size].mean(axis=1).tolist() for s in starts]


def _chords_at(
    chroma: np.ndarray, times: list[float], sections: list[KeySection]
) -> list[str | None]:
    """Acorde provável de cada instante, pelo chroma de `t - BEFORE` a `t + AFTER` em janelas.

    O tom da seção em que o instante cai só desempata acordes parecidos.
    """
    frames_per_second = SAMPLE_RATE / HOP
    size = round(WINDOW_SECONDS * frames_per_second)
    chords: list[str | None] = []
    for t in times:
        start = max(0, round((t - BEFORE_SECONDS) * frames_per_second))
        end = min(chroma.shape[1], round((t + AFTER_SECONDS) * frames_per_second))
        windows = [
            chroma[:, s : min(s + size, end)].mean(axis=1).tolist() for s in range(start, end, size)
        ]
        section = [k for k in sections if k.start <= t][-1]
        chords.append(moment_chord(windows, (section.tonic_pc, section.mode)))
    return chords


def analyze_audio(path: Path) -> AudioFeatures:
    """Analisa o arquivo e devolve a nota mais aguda sustentada, o ranking e o chroma por janela."""
    y = _load(path)
    y_harmonic = _harmonic(y)
    del y
    tuning = _tuning(y_harmonic)
    midi, confidence = _by_frames(y_harmonic, lambda chunk: _melody(chunk, tuning))
    chroma = _by_frames(
        y_harmonic,
        lambda chunk: librosa.feature.chroma_cqt(
            y=chunk, sr=SAMPLE_RATE, hop_length=HOP, tuning=tuning
        ),
    )
    del y_harmonic

    strong = confidence > np.percentile(confidence, 60)
    rounded = np.round(midi).astype(int)
    notes, counts = np.unique(rounded[strong], return_counts=True)
    top_midi, ranking = rank_notes(notes.tolist(), counts.tolist())

    top_frames = np.flatnonzero(strong & (rounded == top_midi))
    times = [round(t, 1) for t in _runs(top_frames)[:MAX_TIMES]]
    windows = _chroma_windows(chroma)
    chords = _chords_at(chroma, times, key_sections(windows))
    return AudioFeatures(top_midi, ranking, times, windows, chords)

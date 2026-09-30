"""
스트리밍 추론의 테스트. 모델 파일·GPU 없이 CPU 에서 실행된다.

실행 방법: pytest tests/test_streaming.py
"""

import io
import wave

import numpy as np
import pytest

from style_bert_vits2.models.hyper_parameters import HyperParametersModel
from style_bert_vits2.models.streaming import (
    ChunkPlan,
    DecoderGeometry,
    decoder_geometry,
    float_to_pcm16_fixed,
    plan_chunks,
    wav_header,
)


def _covered_frames(plans: list[ChunkPlan]) -> list[int]:
    """각 청크에서 잘라낸 뒤 실제로 출력되는 프레임 번호를 순서대로 모은다"""
    frames: list[int] = []
    for p in plans:
        frames.extend(range(p.start + p.trim_left, p.end - p.trim_right))
    return frames


class TestPlanChunks:
    @pytest.mark.parametrize("chunk_size,overlap_size", [(100, 32), (100, 28), (60, 16), (40, 2), (30, 28)])
    def test_covers_every_frame_exactly_once(self, chunk_size, overlap_size):
        for total in range(1, 601):
            plans = plan_chunks(total, chunk_size, overlap_size)
            assert _covered_frames(plans) == list(range(total)), (total, chunk_size, overlap_size)
            assert plans[-1].end == total
            assert all(p.end - p.start <= chunk_size for p in plans)

    def test_single_chunk_when_total_fits(self):
        assert plan_chunks(1, 100, 32) == [ChunkPlan(0, 1, 0, 0)]
        assert plan_chunks(10, 100, 32) == [ChunkPlan(0, 10, 0, 0)]
        assert plan_chunks(100, 100, 32) == [ChunkPlan(0, 100, 0, 0)]

    def test_chunk_size_plus_one(self):
        assert plan_chunks(101, 100, 32) == [ChunkPlan(0, 100, 0, 16), ChunkPlan(68, 101, 16, 0)]

    def test_no_tail_duplication_regression(self):
        # upstream d88d4aa 은 779 프레임·100/32 에서 끝에 닿은 뒤에도 루프를 돌아 꼬리 15 프레임을 두 번 출력했다
        plans = plan_chunks(779, 100, 32)
        assert plans[-1] == ChunkPlan(680, 779, 16, 0)
        assert len(_covered_frames(plans)) == 779

    @pytest.mark.parametrize("total,chunk_size,overlap_size", [(0, 100, 32), (-1, 100, 32), (10, 100, 31), (10, 32, 32), (10, 30, 32), (10, 100, 0)])
    def test_invalid_arguments(self, total, chunk_size, overlap_size):
        with pytest.raises(ValueError):
            plan_chunks(total, chunk_size, overlap_size)


class TestDecoderGeometry:
    def test_ko_default_config(self):
        # KO 모델(kss_ko_warm)의 디코더 설정은 HyperParametersModel 기본값과 같다 (R = 13.35 → 최소 겹침 28)
        assert decoder_geometry(HyperParametersModel()) == DecoderGeometry(upsample_factor=512, min_overlap=28)

    def test_resblock2_uses_only_two_dilations(self):
        # ResBlock2 는 dilation 앞 2개만 쓴다: 반경 = (3-1)/2 * (1+3) = 4 → R = 4.71 → 최소 겹침 10 (3개를 다 쓰면 12)
        hps = HyperParametersModel(resblock="2", resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]])
        assert decoder_geometry(hps) == DecoderGeometry(upsample_factor=512, min_overlap=10)

    def test_odd_kernel_minus_rate_is_unsupported(self):
        # 16 - 7 = 9 (홀수) 이면 ConvTranspose 출력 길이가 L*u 가 아니게 된다
        with pytest.raises(ValueError):
            decoder_geometry(HyperParametersModel(upsample_rates=[7, 8, 2, 2, 2]))


class TestFloatToPcm16Fixed:
    def test_full_scale_and_zero(self):
        out = float_to_pcm16_fixed(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        assert out.dtype == np.int16
        assert out.tolist() == [-32767, 0, 32767]

    def test_rounds_to_nearest(self):
        # 기존 peak 정규화 변환은 0 방향 절삭이지만, 고정 스케일 변환은 가장 가까운 정수로 반올림한다
        assert float_to_pcm16_fixed(np.array([0.9, -0.9, 0.4, -0.4]) / 32767.0).tolist() == [1, -1, 0, 0]

    def test_clips_out_of_range(self):
        assert float_to_pcm16_fixed(np.array([1.7, -3.0])).tolist() == [32767, -32767]

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError):
            float_to_pcm16_fixed(np.array([0.0, bad]))


class TestWavHeader:
    def test_exact_length_is_readable_by_wave(self):
        samples = np.arange(-50, 50, dtype=np.int16)
        data = wav_header(44100, len(samples)) + samples.tobytes()
        with wave.open(io.BytesIO(data)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 44100, 100)
            assert np.frombuffer(w.readframes(100), dtype=np.int16).tolist() == samples.tolist()

    def test_unknown_length_uses_max_size(self):
        header = wav_header(44100, None)
        assert len(header) == 44
        assert header[4:8] == b"\xff\xff\xff\xff" and header[40:44] == b"\xff\xff\xff\xff"
        assert header[24:28] == (44100).to_bytes(4, "little")

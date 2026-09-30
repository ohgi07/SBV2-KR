"""
스트리밍 추론의 테스트. 모델 파일·GPU 없이 CPU 에서 실행된다.

실행 방법: pytest tests/test_streaming.py
"""

import io
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParametersModel
from style_bert_vits2.models.streaming import (
    ChunkPlan,
    DecoderGeometry,
    decoder_geometry,
    float_to_pcm16_fixed,
    plan_chunks,
    wav_header,
)
from style_bert_vits2.nlp.symbols import SYMBOLS


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

    @pytest.mark.parametrize(
        "total,chunk_size,overlap_size",
        [(0, 100, 32), (-1, 100, 32), (10, 100, 31), (10, 32, 32), (10, 30, 32), (10, 100, 0)],
    )
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

    @pytest.mark.parametrize(
        "hps_kwargs",
        [
            # 16 - 7 = 9 (홀수) 이면 ConvTranspose 출력 길이가 L*u 가 아니게 된다
            dict(upsample_rates=[7, 8, 2, 2, 2]),
            # kernel < upsample_rate
            dict(upsample_kernel_sizes=[4, 16, 8, 2, 2]),
            # 짝수 ResBlock kernel 은 출력 길이가 1 줄어든다
            dict(resblock_kernel_sizes=[4, 7, 11]),
            dict(upsample_kernel_sizes=[16, 16, 8, 2]),
            dict(resblock_dilation_sizes=[[1, 3, 5]] * 2),
        ],
    )
    def test_unsupported_config(self, hps_kwargs):
        with pytest.raises(ValueError):
            decoder_geometry(HyperParametersModel(**hps_kwargs))


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


# 디코더 구조는 KO 모델과 같고 나머지는 작게 줄인 무작위 가중치 JP-Extra 모델 설정 (CPU 에서 0.1초 내외로 생성된다)
TINY_MODEL_KWARGS = dict(
    n_vocab=len(SYMBOLS), spec_channels=513, segment_size=32, inter_channels=16, hidden_channels=16,
    filter_channels=32, n_heads=2, n_layers=3, kernel_size=3, p_dropout=0.1, n_speakers=1, gin_channels=16,
    n_layers_trans_flow=3, resblock="1", resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3,
    upsample_rates=[8, 8, 2, 2, 2], upsample_initial_channel=32, upsample_kernel_sizes=[16, 16, 8, 2, 2],
)


def _tiny_net_g():
    from style_bert_vits2.models.models_jp_extra import SynthesizerTrn

    torch.manual_seed(0)
    return SynthesizerTrn(**TINY_MODEL_KWARGS).eval()


def _tiny_inputs(n_phones: int = 12) -> tuple[torch.Tensor, ...]:
    """infer() 의 위치 인자 (x, x_lengths, sid, tone, language, bert, style_vec)"""
    gen = torch.Generator().manual_seed(3)
    x = torch.randint(1, len(SYMBOLS), (1, n_phones), generator=gen)
    zeros = torch.zeros(1, n_phones, dtype=torch.long)
    bert, style_vec = torch.randn(1, 1024, n_phones, generator=gen), torch.randn(1, 256, generator=gen)
    return x, torch.LongTensor([n_phones]), torch.LongTensor([0]), zeros, zeros, bert, style_vec


class TestInferInputFeature:
    def test_infer_equals_input_feature_then_decoder(self):
        net_g, args = _tiny_net_g(), _tiny_inputs()
        kwargs = dict(noise_scale=0.6, noise_scale_w=0.8, sdp_ratio=0.5)
        with torch.no_grad():
            torch.manual_seed(1)
            o, attn, y_mask, (z, z_p, m_p, logs_p) = net_g.infer(*args, **kwargs)
            torch.manual_seed(1)
            z2, y_mask2, g2, attn2, z_p2, m_p2, logs_p2 = net_g.infer_input_feature(*args, **kwargs)
            o2 = net_g.dec(z2 * y_mask2, g=g2)
        for a, b in [(o, o2), (attn, attn2), (y_mask, y_mask2), (z, z2), (z_p, z_p2), (m_p, m_p2), (logs_p, logs_p2)]:
            assert torch.equal(a, b)

    def test_max_len_only_truncates_decoder_input(self):
        net_g, args = _tiny_net_g(), _tiny_inputs()
        with torch.no_grad():
            torch.manual_seed(1)
            o, _, y_mask, (z, *_) = net_g.infer(*args, max_len=5)
        # 음소마다 최소 1 프레임이므로 전체 프레임은 12 이상이고, 디코더 입력만 5 프레임으로 잘린다
        assert o.shape[2] == 5 * 512
        assert z.shape[2] == y_mask.shape[2] >= 12


class TestDecodeChunks:
    def test_concatenated_chunks_match_full_decode(self):
        from style_bert_vits2.models.infer import decode_chunks

        net_g, geometry = _tiny_net_g(), decoder_geometry(HyperParametersModel())
        gen = torch.Generator().manual_seed(2)
        z, g = torch.randn(1, 16, 250, generator=gen), torch.randn(1, 16, 1, generator=gen)
        with torch.no_grad():
            full = net_g.dec(z, g=g)[0, 0].numpy()
        # 바깥에 no_grad 가 없어도 동작해야 한다 (서버에서는 반복이 작업 스레드에서 일어난다)
        plans = plan_chunks(250, 100, geometry.min_overlap)
        chunks = list(decode_chunks(net_g, z, g, plans, geometry.upsample_factor))
        assert len(chunks) == len(plans) > 1
        assert all(c.dtype == np.float32 for c in chunks)
        out = np.concatenate(chunks)
        assert out.shape == full.shape
        np.testing.assert_allclose(out, full, atol=1e-5)


def _dummy_tts_model(model_file: str = "dummy.safetensors", **hps_kwargs):
    """모델 파일 없이 검증 로직만 확인하기 위한 TTSModel (실제로 로드하면 파일이 없어 실패한다)"""
    from style_bert_vits2.models.hyper_parameters import HyperParameters
    from style_bert_vits2.tts_model import TTSModel

    return TTSModel(Path(model_file), HyperParameters(**hps_kwargs), np.zeros((1, 256), dtype=np.float32))


class TestInferStreamValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(overlap_size=16), "overlap_size must be at least 28"),
            (dict(chunk_size=32, overlap_size=32), "chunk_size"),
            (dict(overlap_size=31), "chunk_size"),
            (dict(split_interval=-0.1), "split_interval"),
            (dict(split_interval=float("nan")), "split_interval"),
            (dict(language=Languages.EN), "not JP or KO"),
            (dict(text=""), "empty"),
            (dict(text="\n\n", line_split=True), "empty"),
        ],
    )
    def test_rejects_invalid_input_before_loading(self, kwargs, match):
        model = _dummy_tts_model()
        with pytest.raises(ValueError, match=match):
            model.infer_stream(**{"text": "안녕하세요", "language": Languages.KO, **kwargs})
        assert model.net_g is None

    def test_rejects_onnx_model(self):
        with pytest.raises(ValueError, match="ONNX"):
            _dummy_tts_model("dummy.onnx").infer_stream("안녕하세요", language=Languages.KO)

    def test_rejects_non_jp_extra_model(self):
        with pytest.raises(ValueError, match="JP-Extra"):
            _dummy_tts_model(version="2.7.0").infer_stream("안녕하세요", language=Languages.KO)

    def test_infer_rejects_unknown_pcm_scale_before_loading(self):
        model = _dummy_tts_model()
        with pytest.raises(ValueError, match="pcm_scale"):
            model.infer("안녕하세요", language=Languages.KO, pcm_scale="peak")
        assert model.net_g is None


class TestAudioStream:
    def test_close_runs_generator_cleanup_once(self):
        from style_bert_vits2.tts_model import AudioStream

        closed: list[bool] = []

        def chunks():
            try:
                yield np.zeros(3, dtype=np.int16)
                yield np.ones(3, dtype=np.int16)
            finally:
                closed.append(True)

        stream = AudioStream(sample_rate=44100, num_samples=6, chunks=chunks())
        assert next(iter(stream)).tolist() == [0, 0, 0]
        stream.close()
        stream.close()
        assert closed == [True]
        assert list(stream) == []


class TestConvertTo16BitWav:
    def test_silence_returns_zeros_without_warning(self):
        import warnings

        from style_bert_vits2.tts_model import TTSModel

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = TTSModel.convert_to_16_bit_wav(np.zeros(10, dtype=np.float32))
        assert out.dtype == np.int16 and not out.any()

    def test_peak_normalization_unchanged(self):
        from style_bert_vits2.tts_model import TTSModel

        # 기존 동작: 최대 진폭으로 나눈 뒤 ×32767, 0 방향 절삭 (-16383.5 → -16383)
        assert TTSModel.convert_to_16_bit_wav(np.array([0.5, -0.25], dtype=np.float32)).tolist() == [32767, -16383]

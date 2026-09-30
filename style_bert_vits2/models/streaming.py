"""
스트리밍 추론용 순수 함수 모음.

디코더를 프레임 청크 단위로 나눠 실행할 때 필요한 청크 계획, 디코더 기하(1 프레임당 출력 샘플 수·최소 겹침),
고정 스케일 16bit PCM 변환, 스트리밍용 WAV 헤더를 다룬다. 모델·GPU 없이 테스트할 수 있다.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from style_bert_vits2.models.hyper_parameters import HyperParametersModel


# 길이를 미리 알 수 없는 WAV 스트림에서 RIFF·data 크기 필드에 넣는 값 (많은 플레이어가 스트림 끝까지 읽는다)
UNKNOWN_WAV_SIZE = 0xFFFFFFFF


@dataclass(frozen=True)
class ChunkPlan:
    """디코더에 넣을 프레임 구간 [start, end) 와, 그 출력에서 앞뒤로 잘라낼 프레임 수"""

    start: int
    end: int
    trim_left: int
    trim_right: int


@dataclass(frozen=True)
class DecoderGeometry:
    """디코더 1 프레임당 출력 샘플 수와, 청크 경계 오차가 생기지 않는 겹침 프레임 수의 하한 (한쪽 1 프레임 여유를 둔 보수적인 값)"""

    upsample_factor: int
    min_overlap: int


def decoder_geometry(model_hps: HyperParametersModel) -> DecoderGeometry:
    """
    Generator 설정에서 1 프레임당 출력 샘플 수와 최소 겹침을 계산한다.
    출력 길이가 정확히 `프레임 수 × 업샘플 배율` 이 되는 설정만 지원하며, 그 밖에는 ValueError 를 낸다.
    """

    rates, kernels = model_hps.upsample_rates, model_hps.upsample_kernel_sizes
    rb_kernels, rb_dilations = model_hps.resblock_kernel_sizes, model_hps.resblock_dilation_sizes
    if len(rates) == 0 or len(rates) != len(kernels):
        raise ValueError(f"upsample_rates 와 upsample_kernel_sizes 의 길이가 맞지 않습니다: {rates}, {kernels}")
    if len(rb_kernels) == 0 or len(rb_kernels) != len(rb_dilations):
        raise ValueError(f"resblock_kernel_sizes 와 resblock_dilation_sizes 의 길이가 맞지 않습니다: {rb_kernels}, {rb_dilations}")
    # ConvTranspose1d(padding=(k-u)//2) 의 출력 길이는 L*u + (k-u)%2 이므로, (k-u) 가 짝수일 때만 정확히 L*u 가 된다
    for u, k in zip(rates, kernels):
        if u <= 0 or k < u or (k - u) % 2 != 0:
            raise ValueError(f"스트리밍을 지원하지 않는 디코더 설정입니다 (upsample_rate={u}, kernel={k})")

    # ResBlock1 은 dilation 앞 3개 (각 dilated conv 뒤에 dilation 1 conv), ResBlock2 는 앞 2개 (dilated conv 만) 를 쓴다
    n_dilations = 3 if model_hps.resblock == "1" else 2
    resblock_radius = 0.0
    for k, dilations in zip(rb_kernels, rb_dilations):
        used = dilations[:n_dilations]
        if len(used) != n_dilations or k <= 0 or k % 2 == 0 or any(d <= 0 for d in used):
            raise ValueError(f"스트리밍을 지원하지 않는 ResBlock 설정입니다 (kernel={k}, dilation={dilations})")
        span = sum(d + 1 for d in used) if model_hps.resblock == "1" else sum(used)
        resblock_radius = max(resblock_radius, (k - 1) / 2 * span)

    # 한쪽 수용 영역을 입력 프레임 단위로 환산: conv_pre(kernel 7) + 단계별 (ConvTranspose 반경 + ResBlock 반경) + conv_post(kernel 7)
    radius, scale = 3.0, 1
    for u, k in zip(rates, kernels):
        scale *= u
        radius += ((k - 1) / 2 + resblock_radius) / scale
    radius += 3 / scale
    return DecoderGeometry(upsample_factor=scale, min_overlap=2 * math.ceil(radius))


def plan_chunks(total_frames: int, chunk_size: int, overlap_size: int) -> list[ChunkPlan]:
    """
    total_frames 프레임을 chunk_size 씩, 이웃 청크와 overlap_size 만큼 겹치도록 나눈다.
    각 청크의 출력에서 trim_left·trim_right 만큼 잘라내면 [0, total_frames) 를 빈틈·중복 없이 정확히 한 번 덮는다.
    """

    if total_frames < 1:
        raise ValueError(f"total_frames 는 1 이상이어야 합니다: {total_frames}")
    if not (chunk_size > overlap_size > 0) or overlap_size % 2 != 0:
        raise ValueError(f"chunk_size > overlap_size > 0 이고 overlap_size 는 짝수여야 합니다: chunk_size={chunk_size}, overlap_size={overlap_size}")
    step, margin = chunk_size - overlap_size, overlap_size // 2
    plans: list[ChunkPlan] = []
    start = 0
    while True:
        end = min(start + chunk_size, total_frames)
        plans.append(ChunkPlan(start, end, margin if start > 0 else 0, margin if end < total_frames else 0))
        # 끝에 닿은 청크 뒤에서 멈춘다 (upstream 은 여기서 멈추지 않아 꼬리가 중복 출력됐다)
        if end == total_frames:
            return plans
        start += step


def float_to_pcm16_fixed(audio: NDArray[Any]) -> NDArray[np.int16]:
    """
    float 파형을 정규화 없이 고정 스케일로 16bit PCM 으로 바꾼다: clip(x, -1, 1) → rint(x × 32767).
    청크마다 따로 바꿔도 전체를 한 번에 바꾼 것과 같은 결과가 된다 (샘플별 함수이므로).
    """

    x = np.asarray(audio, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("음성 데이터에 NaN 또는 inf 가 포함되어 있습니다")
    return np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def wav_header(sample_rate: int, num_samples: Optional[int]) -> bytes:
    """16bit PCM 모노 WAV 헤더 (44 바이트). num_samples 가 None 이면 RIFF·data 크기를 0xFFFFFFFF 로 둔다."""

    if num_samples is None:
        riff_size = data_size = UNKNOWN_WAV_SIZE
    else:
        data_size = num_samples * 2
        riff_size = 36 + data_size
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", data_size)

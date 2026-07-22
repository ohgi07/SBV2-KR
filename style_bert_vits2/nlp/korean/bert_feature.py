"""
한국어 텍스트로부터의 BERT 특징량 추출.

BERT 모델은 기본적으로 klue/roberta-large (hidden_size=1024)를 사용한다.
(beomi/kcbert-large 등 hidden_size=1024인 다른 한국어 모델로 교체도 가능)
중국어 (문자 단위 토크나이즈)와 달리 한국어 모델은 WordPiece 서브워드
토크나이즈를 하므로 토큰 수와 문자 수가 일치하지 않는다.
그래서 offset mapping을 이용해 토큰열을 문자 단위 특징량으로 전개한 뒤,
word2ph에 따라 음소 단위 특징량으로 전개한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.utils import get_onnx_device_options


if TYPE_CHECKING:
    import torch


# 모델의 입력 상한을 넘는 텍스트는 잘라낸다 (초과분의 문자는 근처 토큰의 특징량으로 대체됨)
# 상한은 토크나이저의 model_max_length에서 가져온다
# (kcbert-large: 300, klue/roberta-large: 512 등 모델마다 다름)
__MAX_LENGTH_FALLBACK = 512


def __get_max_length(tokenizer: Any) -> int:
    """토크나이저에서 모델에 입력 가능한 최대 토큰 수를 가져온다"""
    max_length = getattr(tokenizer, "model_max_length", None)
    # model_max_length 미설정 토크나이저는 거대한 플레이스홀더 값을 반환할 수 있다
    if not isinstance(max_length, int) or not (0 < max_length <= 100_000):
        return __MAX_LENGTH_FALLBACK
    return max_length


def __build_char_to_token_map(offsets: list[tuple[int, int]], num_chars: int) -> list[int]:  # fmt: skip
    """
    토큰별 (시작, 끝) 문자 오프셋에서, 문자 인덱스 → 토큰 인덱스 대응표를 만든다.
    어느 토큰에도 속하지 않는 문자 (스페이스나 잘려나간 문자)에는 직전의 유효한
    토큰 (없으면 직후)을 할당한다.
    """
    char_to_token = [-1] * num_chars
    for token_index, (start, end) in enumerate(offsets):
        # 특수 토큰 ([CLS], [SEP] 등)은 (0, 0)이 된다
        if start == end:
            continue
        for char_index in range(start, min(end, num_chars)):
            char_to_token[char_index] = token_index

    # 미할당 문자를 앞쪽 (없으면 뒤쪽)의 유효한 토큰으로 채운다
    last_valid = -1
    for i in range(num_chars):
        if char_to_token[i] != -1:
            last_valid = char_to_token[i]
        elif last_valid != -1:
            char_to_token[i] = last_valid
    next_valid = -1
    for i in range(num_chars - 1, -1, -1):
        if char_to_token[i] != -1:
            next_valid = char_to_token[i]
        else:
            char_to_token[i] = next_valid if next_valid != -1 else 0
    return char_to_token


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    한국어 텍스트에서 BERT 특징량을 추출한다 (PyTorch 추론)

    Args:
        text (str): 한국어 텍스트 (정규화 완료)
        word2ph (list[int]): 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지 나타내는 리스트
        device (str): 추론에 사용할 디바이스
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        torch.Tensor: BERT 특징량
    """

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.KO, device_map=device)
    bert_models.transfer_model(Languages.KO, device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.KO)
        max_length = __get_max_length(tokenizer)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(
                assist_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2
    char_to_token = __build_char_to_token_map(offsets, len(text))

    # 문자 단위 특징량: [CLS] + 각 문자 + [SEP]
    char_level_feature = [res[0]]
    for char_index in range(len(text)):
        char_level_feature.append(res[char_to_token[char_index]])
    char_level_feature.append(res[-1])

    phone_level_feature = []
    for i in range(len(word2ph)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                char_level_feature[i].repeat(word2ph[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2ph[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = char_level_feature[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    한국어 텍스트에서 BERT 특징량을 추출한다 (ONNX 추론)

    Args:
        text (str): 한국어 텍스트 (정규화 완료)
        word2ph (list[int]): 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지 나타내는 리스트
        onnx_providers (list[str]): ONNX 추론에 사용할 ExecutionProvider
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        NDArray[Any]: BERT 특징량
    """

    # 토크나이저와 모델 로드
    tokenizer = onnx_bert_models.load_tokenizer(Languages.KO)
    session = onnx_bert_models.load_model(
        language=Languages.KO,
        onnx_providers=onnx_providers,
    )
    input_names = [input.name for input in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # 입력 텐서 전송에 사용할 디바이스 종류, 디바이스 ID, 실행 옵션을 가져온다
    device_type, device_id, run_options = get_onnx_device_options(session, onnx_providers)  # fmt: skip

    # 입력을 텐서로 변환
    max_length = __get_max_length(tokenizer)
    inputs = tokenizer(
        text,
        return_tensors="np",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = inputs["offset_mapping"][0].tolist()
    input_tensor = [
        inputs["input_ids"].astype(np.int64),  # type: ignore
        inputs["token_type_ids"].astype(np.int64),  # type: ignore
        inputs["attention_mask"].astype(np.int64),  # type: ignore
    ]
    # 추론 디바이스에 입력 텐서를 할당
    io_binding = session.io_binding()
    for name, value in zip(input_names, input_tensor):
        gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
            value, device_type, device_id
        )
        io_binding.bind_ortvalue_input(name, gpu_tensor)
    # text에서 BERT 특징량을 추출
    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)
    res = io_binding.get_outputs()[0].numpy()

    style_res_mean = None
    if assist_text:
        style_inputs = tokenizer(
            assist_text,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
        )
        style_input_tensor = [
            style_inputs["input_ids"].astype(np.int64),  # type: ignore
            style_inputs["token_type_ids"].astype(np.int64),  # type: ignore
            style_inputs["attention_mask"].astype(np.int64),  # type: ignore
        ]
        io_binding = session.io_binding()  # IOBinding은 새로 만들어야 한다
        for name, value in zip(input_names, style_input_tensor):
            gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
                value, device_type, device_id
            )
            io_binding.bind_ortvalue_input(name, gpu_tensor)
        io_binding.bind_output(output_name, device_type)
        session.run_with_iobinding(io_binding, run_options=run_options)
        style_res = io_binding.get_outputs()[0].numpy()
        style_res_mean = np.mean(style_res, axis=0)

    assert len(word2ph) == len(text) + 2
    char_to_token = __build_char_to_token_map(offsets, len(text))

    # 문자 단위 특징량: [CLS] + 각 문자 + [SEP]
    char_level_feature = [res[0]]
    for char_index in range(len(text)):
        char_level_feature.append(res[char_to_token[char_index]])
    char_level_feature.append(res[-1])

    phone_level_feature = []
    for i in range(len(word2ph)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(char_level_feature[i], (word2ph[i], 1))
                * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2ph[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(char_level_feature[i], (word2ph[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T

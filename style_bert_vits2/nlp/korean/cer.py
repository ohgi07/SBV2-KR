"""
한국어 CER (Character Error Rate) 계산.

TTS 명료도 평가 (합성 음성 → ASR → 참조 텍스트와 비교)를 위한 모듈.
정서법끼리 직접 비교하면 발음이 같아도 표기만 달라서 오류로 계산되어 버리므로
(예: 참조 "맛있다" vs ASR 출력 "마싯따"), 양쪽 텍스트를 발음형으로 변환한 뒤 비교한다.

- 숫자·기호는 정규화로 읽기 한글로 변환된다 (3개 vs 세 개 → 일치)
- 공백·문장 기호는 비교에서 제외된다 (ASR 띄어쓰기의 흔들림을 무시)
- 단위는 자모 (jamo) 또는 음절 (syllable)을 선택할 수 있다. 자모 단위 쪽이
  부분적인 발음 오류에 대해 완만한 점수가 된다 (음절 단위에서는 자모 하나의 오류도
  음절 전체의 오류로 계산됨)
"""

from collections.abc import Iterable, Sequence
from typing import TypeVar

from style_bert_vits2.nlp.korean.morph import apply_morph_rules
from style_bert_vits2.nlp.korean.normalizer import normalize_text
from style_bert_vits2.nlp.korean.pronounce import (
    decompose,
    is_hangul_syllable,
    pronounce,
)


_Segment = TypeVar("_Segment")


def drop_hallucinated_segments(
    segments: Iterable[_Segment], max_chars_per_sec: float = 2.0, min_duration_sec: float = 5.0
) -> list[_Segment]:
    """
    ASR 세그먼트 중 환각을 제거한다 (start·end·text 속성만 사용).

    Whisper는 짧은 음성 뒤에 무음이 이어지면 자막 상투구를 30초 윈도우 끝까지 늘여
    출력한다. vad_filter로도 남기 때문에 사후에 걸러내야 하고, 하나만 섞여도 평균 CER이
    몇 배로 뛴다. 문구 목록 대신 발화 속도로 판정하므로 처음 보는 상투구에도 통한다.
    기본값은 실측 분포 (정상 4.24~8.33자/초, 환각 0.37~1.00자/초) 사이에서 잡았다.
    짧은 구간은 속도가 낮아도 정상일 수 있어 길이 조건으로 보호한다.
    """
    kept = []
    for segment in segments:
        duration = segment.end - segment.start
        if duration > min_duration_sec and len(segment.text.strip()) < max_chars_per_sec * duration:
            continue
        kept.append(segment)
    return kept


def levenshtein(a: Sequence, b: Sequence) -> int:
    """두 시퀀스 간의 편집 거리 (삽입·삭제·치환 각 비용 1)"""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a):
        current = [i + 1]
        for j, item_b in enumerate(b):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + cost))
        previous = current
    return previous[-1]


def text_to_pronounced_units(text: str, unit: str = "jamo") -> list[str]:
    """
    텍스트를 발음형의 비교 단위열로 변환한다.

    Args:
        text (str): 임의의 한국어 텍스트 (정규화 전이어도 됨)
        unit (str): "jamo" (자모 단위) 또는 "syllable" (음절 단위)

    Returns:
        list[str]: 비교 단위 리스트 (공백·문장 기호는 포함되지 않음)
    """
    assert unit in ("jamo", "syllable")
    norm = normalize_text(text)
    pronounced = pronounce(apply_morph_rules(norm))
    syllables = [c for c in pronounced if is_hangul_syllable(c)]
    if unit == "syllable":
        return syllables
    units: list[str] = []
    for syllable in syllables:
        cho, jung, coda = decompose(syllable)
        if cho != "ㅇ":
            units.append(cho)
        units.append(jung)
        units.extend(coda)
    return units


def korean_cer(reference: str, hypothesis: str, unit: str = "jamo") -> float:
    """
    발음형에 기반한 한국어 문자 오류율을 계산한다.

    Args:
        reference (str): 참조 텍스트 (TTS 입력 문장)
        hypothesis (str): 가설 텍스트 (ASR 출력 문장)
        unit (str): "jamo" (기본값) 또는 "syllable"

    Returns:
        float: 오류율 (0.0 = 완전 일치). 참조가 비어 있으면 가설도 비어 있을 때 0.0, 아니면 1.0
    """
    ref_units = text_to_pronounced_units(reference, unit)
    hyp_units = text_to_pronounced_units(hypothesis, unit)
    if len(ref_units) == 0:
        return 0.0 if len(hyp_units) == 0 else 1.0
    return levenshtein(ref_units, hyp_units) / len(ref_units)

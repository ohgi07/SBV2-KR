"""
한국어 G2P (Grapheme-to-Phoneme) 변환.

정규화된 텍스트를 다음 순서로 음소열로 변환한다:
1. 발음 변환 (표준 발음법): morph.py의 형태소 기반 보정 (예외 사전·ㄴ 첨가·의→에 등)을
   적용한 뒤, pronounce.py의 내장 규칙 엔진으로 발음형으로 변환한다
2. 발음형 한글의 각 음절을 자모 (초성·중성·종성)로 분해해 음소열을 얻는다

예전에는 g2pkk를 우선 백엔드로 사용했지만, 어절 경계 규칙·형태소 보정·
예외 사전을 갖춘 내장 엔진이 전수 비교 (BERT vocab 24,077 단어 + KSS 1000 문장)에서
동등 이상이 되어 의존성을 제거했다 (남은 차이는 어중 ㅢ의 허용 발음뿐).

한국어는 고저 악센트로 의미가 갈리는 언어가 아니므로 톤은 전부 0으로 한다.
word2ph는 정규화 텍스트의 각 문자에 대응하는 음소 수 리스트 (앞뒤 패딩 포함).
"""

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.pronounce import (
    CHOSEONG,
    JONGSEONG,
    JUNGSEONG,
    decompose,
    is_hangul_syllable,
    pronounce,
)
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


# 호환 자모 (Compatibility Jamo) → 심볼로 사용하는 Hangul Jamo 코드포인트 변환
__CHOSEONG_TO_SYMBOL = {c: chr(0x1100 + i) for i, c in enumerate(CHOSEONG)}
__JUNGSEONG_TO_SYMBOL = {c: chr(0x1161 + i) for i, c in enumerate(JUNGSEONG)}
__JONGSEONG_TO_SYMBOL = {c: chr(0x11A7 + i) for i, c in enumerate(JONGSEONG) if i > 0}


def __to_pronunciation(norm_text: str) -> str:
    """
    정규화된 텍스트를 발음형으로 변환한다.
    출력은 입력과 같은 문자 수이며, 한글 이외 문자의 위치는 변하지 않음이 보장된다.
    """
    # 형태소 정보에 기반한 보정 (예외 사전·ㄴ첨가·의→에·형태소 경계의 경음화)
    from style_bert_vits2.nlp.korean.morph import apply_morph_rules

    return pronounce(apply_morph_rules(norm_text))


def __syllable_to_phones(syllable: str) -> list[str]:
    """발음형 한글 음절 1글자를 자모 음소 리스트로 변환한다"""
    cho, jung, coda = decompose(syllable)
    phones: list[str] = []
    # 초성 ㅇ은 무음이므로 음소를 출력하지 않는다
    if cho != "ㅇ":
        phones.append(__CHOSEONG_TO_SYMBOL[cho])
    phones.append(__JUNGSEONG_TO_SYMBOL[jung])
    if coda:
        # 표준 발음법 적용 후의 종성은 7종성뿐이어야 한다
        phones.append(__JONGSEONG_TO_SYMBOL[coda[0]])
    return phones


def g2p(norm_text: str) -> tuple[list[str], list[int], list[int]]:
    """
    정규화된 한국어 텍스트를 음소열로 변환한다.

    Args:
        norm_text (str): normalize_text()로 정규화된 텍스트

    Returns:
        tuple[list[str], list[int], list[int]]: 음소·톤·word2ph 리스트
    """
    pronounced = __to_pronunciation(norm_text)

    phones: list[str] = []
    word2ph: list[int] = []
    for char in pronounced:
        if is_hangul_syllable(char):
            syllable_phones = __syllable_to_phones(char)
            phones.extend(syllable_phones)
            word2ph.append(len(syllable_phones))
        elif char == " ":
            phones.append("SP")
            word2ph.append(1)
        elif char in PUNCTUATIONS:
            phones.append(char)
            word2ph.append(1)
        else:
            # normalize_text()를 거쳤다면 올 수 없지만, 만약을 위해 미지의 문자는 UNK로 처리
            logger.warning(f"Unexpected character in Korean g2p: {char!r}")
            phones.append("UNK")
            word2ph.append(1)

    # 앞뒤에 패딩 추가 (다른 언어 구현과 같은 형식)
    phones = ["_"] + phones + ["_"]
    tones = [0] * len(phones)
    word2ph = [1] + word2ph + [1]

    assert len(word2ph) == len(norm_text) + 2, (
        f"word2ph length mismatch: {len(word2ph)} != {len(norm_text) + 2}"
    )
    assert sum(word2ph) == len(phones)

    return phones, tones, word2ph

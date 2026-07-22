"""
KO 심볼 임베딩 warm-start 매핑.

JP-Extra 베이스가 학습한 JP 음소 임베딩에서 음성학적으로 유사한 행을 가중 결합
(convex combination, WECHSEL식)해 KO 심볼(자모 46개)·톤·언어 임베딩의 초기값을
만든다. 행별 매핑 근거는 아래 KO_JP_INIT_MAP의 주석 참조.
"""

import torch

from style_bert_vits2.nlp.symbols import KO_SYMBOLS, SYMBOLS


# KO 심볼 -> [(JP 심볼, 가중치), ...] (가중치 합 = 1.0)
## 초성: 평음=유성, 경음·격음=무성 근사. ㅅ/ㅆ는 ㅣ·y계 앞 [ɕ] 고빈도라 sh 소량 혼합
## 중성: 활음은 0.3*활음 + 0.7*핵모음. ㅓ→o, ㅡ→u는 JP 베이스의 학습 분포([o̞], [ɯᵝ]) 기준
## 종성: 불파음 ᆨᆮᆸ→q(촉음), 비음 ᆫᆷᆼ→N(발음), ᆯ→r
KO_JP_INIT_MAP: dict[str, list[tuple[str, float]]] = {
    # 초성 18
    "ᄀ": [("g", 1.0)],
    "ᄁ": [("k", 1.0)],
    "ᄂ": [("n", 1.0)],
    "ᄃ": [("d", 1.0)],
    "ᄄ": [("t", 1.0)],
    "ᄅ": [("r", 1.0)],
    "ᄆ": [("m", 1.0)],
    "ᄇ": [("b", 1.0)],
    "ᄈ": [("p", 1.0)],
    "ᄉ": [("s", 0.8), ("sh", 0.2)],
    "ᄊ": [("s", 0.8), ("sh", 0.2)],
    "ᄌ": [("j", 1.0)],
    "ᄍ": [("ch", 1.0)],
    "ᄎ": [("ch", 1.0)],
    "ᄏ": [("k", 1.0)],
    "ᄐ": [("t", 1.0)],
    "ᄑ": [("p", 1.0)],
    "ᄒ": [("h", 1.0)],
    # 중성 21
    "ᅡ": [("a", 1.0)],
    "ᅢ": [("e", 1.0)],
    "ᅣ": [("y", 0.3), ("a", 0.7)],
    "ᅤ": [("y", 0.3), ("e", 0.7)],
    "ᅥ": [("o", 1.0)],
    "ᅦ": [("e", 1.0)],
    "ᅧ": [("y", 0.3), ("o", 0.7)],
    "ᅨ": [("y", 0.3), ("e", 0.7)],
    "ᅩ": [("o", 1.0)],
    "ᅪ": [("w", 0.3), ("a", 0.7)],
    "ᅫ": [("w", 0.3), ("e", 0.7)],
    "ᅬ": [("w", 0.3), ("e", 0.7)],
    "ᅭ": [("y", 0.3), ("o", 0.7)],
    "ᅮ": [("u", 1.0)],
    "ᅯ": [("w", 0.3), ("o", 0.7)],
    "ᅰ": [("w", 0.3), ("e", 0.7)],
    "ᅱ": [("w", 0.3), ("i", 0.7)],
    "ᅲ": [("y", 0.3), ("u", 0.7)],
    "ᅳ": [("u", 1.0)],
    "ᅴ": [("i", 1.0)],
    "ᅵ": [("i", 1.0)],
    # 종성 7
    "ᆨ": [("q", 1.0)],
    "ᆫ": [("N", 1.0)],
    "ᆮ": [("q", 1.0)],
    "ᆯ": [("r", 1.0)],
    "ᆷ": [("N", 1.0)],
    "ᆸ": [("q", 1.0)],
    "ᆼ": [("N", 1.0)],
}

# SYMBOLS 인덱스 캐시 (변환 스크립트·테스트 공유, 인덱스 하드코딩 방지)
SYMBOL_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}

# KO 추가 이전의 베이스 심볼 수 (= 112)
NUM_BASE_SYMBOLS = len(SYMBOLS) - len(KO_SYMBOLS)


def __validate() -> None:
    """임포트 시점에 데이터 정합성을 검증한다 (PRONUNCIATION_EXCEPTIONS와 같은 임포트 시점 검증)."""
    if set(KO_JP_INIT_MAP.keys()) != set(KO_SYMBOLS):
        missing = set(KO_SYMBOLS) - set(KO_JP_INIT_MAP.keys())
        extra = set(KO_JP_INIT_MAP.keys()) - set(KO_SYMBOLS)
        raise ValueError(f"KO_JP_INIT_MAP 키 불일치 (누락: {sorted(missing)}, 초과: {sorted(extra)})")  # fmt: skip
    for ko, sources in KO_JP_INIT_MAP.items():
        if abs(sum(w for _, w in sources) - 1.0) > 1e-6:
            raise ValueError(f"{ko}: 가중치 합이 1.0이 아닙니다 ({sources})")
        for jp, _ in sources:
            if SYMBOL_TO_IDX.get(jp, NUM_BASE_SYMBOLS) >= NUM_BASE_SYMBOLS:
                raise ValueError(f"{ko}: 소스 심볼 '{jp}'가 베이스 심볼 구간에 없습니다")


__validate()


def build_embedding(
    base_weight: torch.Tensor,
    target_rows: int,
    init_map: dict[int, list[tuple[int, float]]],
) -> torch.Tensor:
    """
    기존 행은 그대로 유지하고, 신규 행을 (소스 행, 가중치) 가중 결합으로 채운
    확장 임베딩 텐서를 반환한다. init_map은 신규 행 전체를 빠짐없이 채워야 하고,
    소스는 전부 기존 행이어야 한다. 단일 매핑(w=1.0)은 소스 행을 그대로 복사한다.
    """
    num_base = base_weight.shape[0]
    if target_rows < num_base:
        raise ValueError(f"target_rows({target_rows})가 base_weight 행수({num_base})보다 작습니다")  # fmt: skip
    new_rows = set(range(num_base, target_rows))
    if set(init_map.keys()) != new_rows:
        raise ValueError(f"init_map 키 불일치: {sorted(init_map.keys())} != {sorted(new_rows)}")  # fmt: skip
    expanded = base_weight.new_zeros((target_rows, *base_weight.shape[1:]))
    expanded[:num_base] = base_weight
    for row, sources in init_map.items():
        for src, _ in sources:
            if not (0 <= src < num_base):
                raise ValueError(f"행 {row}: 소스 행 {src}가 기존 행 범위(0~{num_base - 1}) 밖입니다")  # fmt: skip
        if len(sources) == 1 and sources[0][1] == 1.0:
            expanded[row] = base_weight[sources[0][0]]  # 단일 매핑은 직접 복사 (비트 단위 보존)
        else:
            for src, weight in sources:
                expanded[row] += weight * base_weight[src]
    return expanded


# KO_JP_INIT_MAP을 행 인덱스로 변환한 상수 (결정적이므로 함수 대신 임포트 시 1회 구축)
KO_PHONEME_INIT_MAP: dict[int, list[tuple[int, float]]] = {
    SYMBOL_TO_IDX[ko]: [(SYMBOL_TO_IDX[jp], w) for jp, w in sources]
    for ko, sources in KO_JP_INIT_MAP.items()
}

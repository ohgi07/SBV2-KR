"""
KO 추가 전 체크포인트(JP-Extra 사전학습 G_0 등)를 로드할 때 KO 임베딩 행을 warm-start 초기화한다.

KO 심볼(자모 46개)·톤·언어 임베딩의 신규 행을 음성학적으로 유사한 JP 행의 가중 결합
(convex combination, WECHSEL식)으로 채운다. 랜덤 초기화로 학습한 런보다 test CER이 약 10배 낮다.
checkpoints.expand_embedding_if_needed가 임베딩 확장 시 자동 적용한다. 행별 매핑 근거는 아래 KO_JP_INIT_MAP의 주석 참조.
"""

import torch

from style_bert_vits2.nlp.symbols import KO_SYMBOLS, LANGUAGE_ID_MAP, LANGUAGE_TONE_START_MAP, SYMBOLS


# KO 심볼 -> [(JP 심볼, 가중치), ...] (가중치 합 = 1.0)
## 가중치는 v1(음성학 프라이어) 초기화로 s7000~s23000까지 학습한 임베딩의 수렴 방향을
## 축 스캔(수렴 행과 α·A+(1-α)·B의 cos 최대화)으로 실측해 정한 값:
## - 평음(ㄱㄷㅂㅈ)=유성 단독(전 구간 드리프트≈0, 무성 혼합 기각), 격음=무성 단독
## - 경음은 무성 0.9~0.95 + 유성 소량(수렴 방향 실측), ㅅ/ㅆ=0.75/0.85 s + sh
## - 활음 모음은 0.35*활음 + 0.65*핵모음 (예외: ᅯ는 [wʌ]의 약한 활음 반영 0.2)
## - ㅓ→o, ㅡ→u는 JP 베이스의 학습 분포([o̞], [ɯᵝ]) 기준. ᅩ·ᅮ는 원순·고모음 보정 혼합
## - 종성: 불파음→q(촉음) 기반+조음위치 온셋 소량(ᆮ은 q 단독), 비음 ᆫᆷ은 N+n/m 혼합,
##   ᆼ=N 단독(대안 축 기각), ᆯ→r
KO_JP_INIT_MAP: dict[str, list[tuple[str, float]]] = {
    # 초성 18
    "ᄀ": [("g", 1.0)],
    "ᄁ": [("k", 0.95), ("g", 0.05)],
    "ᄂ": [("n", 1.0)],
    "ᄃ": [("d", 1.0)],
    "ᄄ": [("t", 0.95), ("d", 0.05)],
    "ᄅ": [("r", 1.0)],
    "ᄆ": [("m", 1.0)],
    "ᄇ": [("b", 1.0)],
    "ᄈ": [("p", 0.95), ("b", 0.05)],
    "ᄉ": [("s", 0.75), ("sh", 0.25)],
    "ᄊ": [("s", 0.85), ("sh", 0.15)],
    "ᄌ": [("j", 1.0)],
    "ᄍ": [("ch", 0.9), ("j", 0.1)],
    "ᄎ": [("ch", 1.0)],
    "ᄏ": [("k", 1.0)],
    "ᄐ": [("t", 1.0)],
    "ᄑ": [("p", 1.0)],
    "ᄒ": [("h", 1.0)],
    # 중성 21
    "ᅡ": [("a", 1.0)],
    "ᅢ": [("e", 1.0)],
    "ᅣ": [("y", 0.35), ("a", 0.65)],
    "ᅤ": [("y", 0.35), ("e", 0.65)],
    "ᅥ": [("o", 1.0)],
    "ᅦ": [("e", 1.0)],
    "ᅧ": [("y", 0.35), ("o", 0.65)],
    "ᅨ": [("y", 0.35), ("e", 0.65)],
    "ᅩ": [("o", 0.85), ("u", 0.15)],
    "ᅪ": [("w", 0.35), ("a", 0.65)],
    "ᅫ": [("w", 0.35), ("e", 0.65)],
    "ᅬ": [("w", 0.35), ("e", 0.65)],
    "ᅭ": [("y", 0.35), ("o", 0.65)],
    "ᅮ": [("u", 0.9), ("o", 0.1)],
    "ᅯ": [("w", 0.2), ("o", 0.8)],
    "ᅰ": [("w", 0.35), ("e", 0.65)],
    "ᅱ": [("w", 0.35), ("i", 0.65)],
    "ᅲ": [("y", 0.35), ("u", 0.65)],
    "ᅳ": [("u", 1.0)],
    "ᅴ": [("i", 1.0)],
    "ᅵ": [("i", 1.0)],
    # 종성 7
    "ᆨ": [("q", 0.9), ("k", 0.1)],
    "ᆫ": [("N", 0.85), ("n", 0.15)],
    "ᆮ": [("q", 1.0)],
    "ᆯ": [("r", 1.0)],
    "ᆷ": [("N", 0.7), ("m", 0.3)],
    "ᆸ": [("q", 0.9), ("p", 0.1)],
    "ᆼ": [("N", 1.0)],
}

SYMBOL_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}

NUM_BASE_SYMBOLS = len(SYMBOLS) - len(KO_SYMBOLS)


def __validate() -> None:
    """임포트 시점에 매핑 데이터 정합성을 검증한다."""
    if set(KO_JP_INIT_MAP.keys()) != set(KO_SYMBOLS):
        missing = set(KO_SYMBOLS) - set(KO_JP_INIT_MAP.keys())
        extra = set(KO_JP_INIT_MAP.keys()) - set(KO_SYMBOLS)
        raise ValueError(f"KO_JP_INIT_MAP 키 불일치 (누락: {sorted(missing)}, 초과: {sorted(extra)})")
    for ko, sources in KO_JP_INIT_MAP.items():
        if abs(sum(w for _, w in sources) - 1.0) > 1e-6:
            raise ValueError(f"{ko}: 가중치 합이 1.0이 아닙니다 ({sources})")
        for jp, _ in sources:
            if SYMBOL_TO_IDX.get(jp, NUM_BASE_SYMBOLS) >= NUM_BASE_SYMBOLS:
                raise ValueError(f"{ko}: 소스 심볼 '{jp}'가 베이스 심볼 구간에 없습니다")


__validate()


__JP_TONE = LANGUAGE_TONE_START_MAP["JP"]

# 임베딩 키 -> {KO 행: [(JP 소스 행, 가중치), ...]}. 톤은 JP 저·고 평균, 언어는 JP 행 복사
KO_WARM_START_MAPS: dict[str, dict[int, list[tuple[int, float]]]] = {
    "enc_p.emb.weight": {
        SYMBOL_TO_IDX[ko]: [(SYMBOL_TO_IDX[jp], w) for jp, w in sources]
        for ko, sources in KO_JP_INIT_MAP.items()
    },
    "enc_p.tone_emb.weight": {LANGUAGE_TONE_START_MAP["KO"]: [(__JP_TONE, 0.5), (__JP_TONE + 1, 0.5)]},
    "enc_p.language_emb.weight": {LANGUAGE_ID_MAP["KO"]: [(LANGUAGE_ID_MAP["JP"], 1.0)]},
}


def warm_start_new_rows(key: str, expanded: torch.Tensor, num_saved: int) -> bool:
    """
    저장본 행(0~num_saved-1)이 앞쪽에 복사된 확장 임베딩에서, 저장본에 없던 KO 행을 JP 행의 가중 결합으로 채운다.
    매핑이 있는 키이고 소스 JP 행이 전부 저장본 범위 안에 있을 때만 적용하며, 적용 여부를 반환한다.
    """
    init_map = KO_WARM_START_MAPS.get(key)
    if init_map is None or any(src >= num_saved for sources in init_map.values() for src, _ in sources):
        return False
    for row, sources in init_map.items():
        if row >= num_saved:
            expanded[row] = sum(w * expanded[src] for src, w in sources)
    return True

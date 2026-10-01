"""
JP-Extra 베이스 G_0.safetensors를 KO warm-start 초기화로 확장하는 변환 스크립트.

KO 심볼(자모 46개)·톤·언어 임베딩 행을 음성학적으로 유사한 JP 행의 가중 결합
(convex combination, WECHSEL식)으로 채운 158행 G_0.safetensors를 생성한다.
D_0/WD_0는 텍스트 임베딩이 없으므로 변환 불필요. 행별 매핑 근거는 아래 KO_JP_INIT_MAP의 주석 참조.

사용법:
    python warm_start_ko.py --input path/to/G_0.safetensors --output path/to/G_0_ko.safetensors

출력 파일을 G_0.safetensors로 개명해 Data/{model}/models/ 에 두고 학습을 시작하면 된다.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.symbols import (
    KO_SYMBOLS,
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    NUM_LANGUAGES,
    NUM_TONES,
    SYMBOLS,
)


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

# SYMBOLS 인덱스 캐시
SYMBOL_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}

NUM_BASE_SYMBOLS = len(SYMBOLS) - len(KO_SYMBOLS)


def __validate() -> None:
    """임포트 시점에 매핑 데이터 정합성을 검증한다."""
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
            expanded[row] = base_weight[sources[0][0]]
        else:
            for src, weight in sources:
                expanded[row] += weight * base_weight[src]
    return expanded


# KO_JP_INIT_MAP을 행 인덱스로 변환한 상수
KO_PHONEME_INIT_MAP: dict[int, list[tuple[int, float]]] = {
    SYMBOL_TO_IDX[ko]: [(SYMBOL_TO_IDX[jp], w) for jp, w in sources]
    for ko, sources in KO_JP_INIT_MAP.items()
}


EMB_KEY = "enc_p.emb.weight"
TONE_KEY = "enc_p.tone_emb.weight"
LANG_KEY = "enc_p.language_emb.weight"


def convert(input_path: Path, output_path: Path) -> None:
    """베이스 G_0를 읽어 KO 임베딩을 warm-start 초기화한 safetensors를 출력 경로에 쓴다."""
    tensors = {}
    with safe_open(str(input_path), framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    if EMB_KEY not in tensors:
        raise ValueError(f"{EMB_KEY}가 없습니다 — G_0(생성자) 체크포인트가 맞는지 확인하세요 (D_0/WD_0는 변환 대상이 아님)")  # fmt: skip

    num_rows = tensors[EMB_KEY].shape[0]
    if num_rows == len(SYMBOLS):
        raise ValueError(f"{EMB_KEY}가 이미 {num_rows}행입니다 (이중 적용 방지)")
    if num_rows != NUM_BASE_SYMBOLS:
        raise ValueError(f"{EMB_KEY} 행수가 베이스({NUM_BASE_SYMBOLS})와 다릅니다: {num_rows}")

    tensors[EMB_KEY] = build_embedding(tensors[EMB_KEY], len(SYMBOLS), KO_PHONEME_INIT_MAP)  # fmt: skip
    jp_tone = LANGUAGE_TONE_START_MAP["JP"]
    tone_map = {LANGUAGE_TONE_START_MAP["KO"]: [(jp_tone, 0.5), (jp_tone + 1, 0.5)]}
    tensors[TONE_KEY] = build_embedding(tensors[TONE_KEY], NUM_TONES, tone_map)
    lang_map = {LANGUAGE_ID_MAP["KO"]: [(LANGUAGE_ID_MAP["JP"], 1.0)]}
    tensors[LANG_KEY] = build_embedding(tensors[LANG_KEY], NUM_LANGUAGES, lang_map)

    for ko, sources in KO_JP_INIT_MAP.items():
        src = " + ".join(f"{w}*{jp}" for jp, w in sources)
        logger.info(f"init {ko} <- {src}")
    logger.info(f"tone: KO <- mean(JP low, high) / language: KO <- JP")

    save_file(tensors, str(output_path), metadata=metadata)
    logger.success(f"Saved warm-start G_0 to {output_path}")


if __name__ == "__main__":
    # Windows cp949 콘솔에서 자모(U+1100대) 로그가 UnicodeEncodeError로 깨지는 것 방지
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="JP-Extra 베이스 G_0.safetensors 경로")  # fmt: skip
    parser.add_argument("--output", "-o", type=Path, required=True, help="warm-start 초기화된 G_0 출력 경로")  # fmt: skip
    args = parser.parse_args()
    convert(args.input, args.output)

"""
JP-Extra 베이스 G_0.safetensors를 KO warm-start 초기화로 확장하는 변환 스크립트.

KO 심볼(자모 46개)·톤·언어 임베딩 행을 음성학적 유사 JP 행의 가중 결합으로 채운
158행 G_0.safetensors를 생성한다. D_0/WD_0는 텍스트 임베딩이 없으므로 변환 불필요.
매핑 테이블은 style_bert_vits2/nlp/korean/warm_start.py 참조.

사용법:
    python warm_start_ko.py --input path/to/G_0.safetensors --output path/to/G_0_ko.safetensors

출력 파일을 G_0.safetensors로 개명해 Data/{model}/models/ 에 두고 학습을 시작하면 된다.
"""

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP, KO_PHONEME_INIT_MAP, NUM_BASE_SYMBOLS, build_embedding  # fmt: skip
from style_bert_vits2.nlp.symbols import LANGUAGE_ID_MAP, LANGUAGE_TONE_START_MAP, NUM_LANGUAGES, NUM_TONES, SYMBOLS  # fmt: skip


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

    # 음소: KO 46행을 매핑 가중 결합으로
    tensors[EMB_KEY] = build_embedding(tensors[EMB_KEY], len(SYMBOLS), KO_PHONEME_INIT_MAP)  # fmt: skip
    # 톤: KO 행 = JP low·high 평균
    jp_tone = LANGUAGE_TONE_START_MAP["JP"]
    tone_map = {LANGUAGE_TONE_START_MAP["KO"]: [(jp_tone, 0.5), (jp_tone + 1, 0.5)]}
    tensors[TONE_KEY] = build_embedding(tensors[TONE_KEY], NUM_TONES, tone_map)
    # 언어: KO 행 = JP 행 복사
    lang_map = {LANGUAGE_ID_MAP["KO"]: [(LANGUAGE_ID_MAP["JP"], 1.0)]}
    tensors[LANG_KEY] = build_embedding(tensors[LANG_KEY], NUM_LANGUAGES, lang_map)

    for ko, sources in KO_JP_INIT_MAP.items():
        src = " + ".join(f"{w}*{jp}" for jp, w in sources)
        logger.info(f"init {ko} <- {src}")
    logger.info(f"tone: KO <- mean(JP low, high) / language: KO <- JP")

    save_file(tensors, str(output_path), metadata=metadata)
    logger.success(f"Saved warm-start G_0 to {output_path}")


if __name__ == "__main__":
    import sys

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

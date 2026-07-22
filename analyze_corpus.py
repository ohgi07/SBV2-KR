"""
韓国語データセット (esd.list) のコーパスカバレッジ分析ツール。

NDC 発表の教訓「基盤モデルのコーパスに登場しない記号・単語は合成品質が落ちる」
(特に歯擦音 ㅅ/ㅆ/ㅈ/ㅊ を含む単語や 말줄임표/말늘임표) に基づき、学習前に
データセットの音素・文章記号カバレッジと音声長の統計を検査する。

使い方:
    python analyze_corpus.py --model_name YourModel
    python analyze_corpus.py --esd_path path/to/esd.list [--json report.json]
"""

import argparse
import json
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.g2p import g2p
from style_bert_vits2.nlp.korean.normalizer import normalize_text
from style_bert_vits2.nlp.symbols import KO_SYMBOLS, PUNCTUATIONS


# 歯擦音 (치찰음) の初声シンボル (Hangul Jamo)
SIBILANT_INITIALS = {"ᄉ": "ㅅ", "ᄊ": "ㅆ", "ᄌ": "ㅈ", "ᄍ": "ㅉ", "ᄎ": "ㅊ"}

# 中声シンボルの範囲 (母音コンテキストの判定に使用)
VOWEL_SYMBOLS = {chr(code) for code in range(0x1161, 0x1176)}

# カバレッジ警告のしきい値
LOW_COVERAGE_THRESHOLD = 10
# 平均音声長の警告しきい値 (秒)。発表では平均発話長が長すぎるデータセットは収束しなかった
MEAN_DURATION_WARNING_SEC = 10.0


def get_wav_duration(path: Path) -> float | None:
    """WAV ファイルの長さ (秒) を返す。読めない場合は None"""
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return None


def analyze(esd_path: Path, check_audio: bool = True) -> dict:
    phone_counts: Counter[str] = Counter()
    punct_counts: Counter[str] = Counter()
    ellipsis_count = 0  # "..." (말줄임표)
    sibilant_vowel_contexts: dict[str, Counter[str]] = defaultdict(Counter)
    text_lengths: list[int] = []
    durations: list[float] = []
    long_texts: list[tuple[int, str]] = []
    speakers: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    error_lines: list[str] = []
    total_lines = 0

    with esd_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split("|")
            if len(parts) != 4:
                error_lines.append(line)
                continue
            wav_path, spk, lang, text = parts
            speakers[spk] += 1
            languages[lang] += 1
            if lang != "KO":
                continue

            try:
                norm = normalize_text(text)
                phones, _, _ = g2p(norm)
            except Exception as e:
                error_lines.append(f"{line} ({e})")
                continue

            text_lengths.append(len(norm))
            long_texts.append((len(norm), text))
            ellipsis_count += norm.count("...")

            for i, phone in enumerate(phones):
                phone_counts[phone] += 1
                if phone in PUNCTUATIONS:
                    punct_counts[phone] += 1
                # 歯擦音の直後の母音コンテキスト
                if phone in SIBILANT_INITIALS and i + 1 < len(phones):
                    nxt = phones[i + 1]
                    if nxt in VOWEL_SYMBOLS:
                        sibilant_vowel_contexts[SIBILANT_INITIALS[phone]][nxt] += 1

            if check_audio:
                duration = get_wav_duration(Path(wav_path))
                if duration is not None:
                    durations.append(duration)

    long_texts.sort(reverse=True)

    report: dict = {
        "esd_path": str(esd_path),
        "total_lines": total_lines,
        "speakers": dict(speakers),
        "languages": dict(languages),
        "error_lines": error_lines,
        "phoneme_coverage": {
            "missing": [s for s in KO_SYMBOLS if phone_counts[s] == 0],
            "low": {
                s: phone_counts[s]
                for s in KO_SYMBOLS
                if 0 < phone_counts[s] < LOW_COVERAGE_THRESHOLD
            },
            "counts": {s: phone_counts[s] for s in KO_SYMBOLS},
        },
        "punctuation": {
            "counts": {p: punct_counts[p] for p in PUNCTUATIONS},
            "ellipsis (...)": ellipsis_count,
            "sp (space)": phone_counts["SP"],
        },
        "sibilant_contexts": {
            sib: {
                "total": sum(ctx.values()),
                "distinct_vowels": len(ctx),
                "max_distinct_vowels": len(VOWEL_SYMBOLS),
            }
            for sib, ctx in sorted(sibilant_vowel_contexts.items())
        },
        "text_length": (
            {
                "mean": round(statistics.mean(text_lengths), 1),
                "median": statistics.median(text_lengths),
                "max": max(text_lengths),
                "longest_samples": [t for _, t in long_texts[:5]],
            }
            if text_lengths
            else {}
        ),
        "audio_duration_sec": (
            {
                "files_measured": len(durations),
                "mean": round(statistics.mean(durations), 2),
                "median": round(statistics.median(durations), 2),
                "max": round(max(durations), 2),
                "total_hours": round(sum(durations) / 3600, 2),
            }
            if durations
            else {}
        ),
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print(f"코퍼스 분석 리포트: {report['esd_path']}")
    print("=" * 60)
    print(f"총 라인 수: {report['total_lines']}")
    print(f"화자: {report['speakers']}")
    print(f"언어: {report['languages']}")
    if report["error_lines"]:
        print(f"\n[경고] 처리 실패 라인 {len(report['error_lines'])}개:")
        for line in report["error_lines"][:5]:
            print(f"  {line}")

    cov = report["phoneme_coverage"]
    print(f"\n--- 음소 커버리지 ({len(KO_SYMBOLS)}개 중) ---")
    covered = len(KO_SYMBOLS) - len(cov["missing"])
    print(f"등장 음소: {covered}/{len(KO_SYMBOLS)}")
    if cov["missing"]:
        print(f"[경고] 미등장 음소 ({len(cov['missing'])}개): {' '.join(cov['missing'])}")
    if cov["low"]:
        print(f"[경고] 저빈도 음소 (<{LOW_COVERAGE_THRESHOLD}회):")
        for s, c in cov["low"].items():
            print(f"  {s}: {c}회")

    print("\n--- 문장 기호 ---")
    for p, c in report["punctuation"]["counts"].items():
        # "…" 는 정규화 단계에서 항상 "..." 로 변환되므로 별도 항목으로 표시
        if p == "…":
            continue
        flag = "  [경고] 미등장 — 이 기호의 발성 패턴을 학습할 수 없습니다" if c == 0 else ""
        print(f"  {p!r}: {c}회{flag}")
    ellipsis = report["punctuation"]["ellipsis (...)"]
    flag = "  [경고] 미등장 — 말줄임표 표현을 학습할 수 없습니다" if ellipsis == 0 else ""
    print(f"  말줄임표(...): {ellipsis}회{flag}")

    print("\n--- 치찰음 문맥 다양성 (발음 품질에 중요) ---")
    sib_report = report["sibilant_contexts"]
    for sib in ["ㅅ", "ㅆ", "ㅈ", "ㅉ", "ㅊ"]:
        if sib in sib_report:
            info = sib_report[sib]
            ratio = info["distinct_vowels"] / info["max_distinct_vowels"]
            flag = "  [경고] 문맥 다양성 부족" if ratio < 0.5 else ""
            print(
                f"  {sib}: {info['total']}회, 모음 문맥 {info['distinct_vowels']}/{info['max_distinct_vowels']}종{flag}"
            )
        else:
            print(f"  {sib}: 0회  [경고] 미등장")

    if report["text_length"]:
        tl = report["text_length"]
        print(f"\n--- 텍스트 길이 (정규화 후 문자 수) ---")
        print(f"  평균 {tl['mean']} / 중앙값 {tl['median']} / 최대 {tl['max']}")

    if report["audio_duration_sec"]:
        ad = report["audio_duration_sec"]
        print(f"\n--- 음성 길이 ({ad['files_measured']}개 파일) ---")
        print(f"  평균 {ad['mean']}초 / 중앙값 {ad['median']}초 / 최대 {ad['max']}초 / 총 {ad['total_hours']}시간")  # fmt: skip
        if ad["mean"] > MEAN_DURATION_WARNING_SEC:
            print(
                f"  [경고] 평균 음성 길이가 {MEAN_DURATION_WARNING_SEC}초를 초과합니다. "
                "평균 발화 길이가 긴 데이터셋은 학습이 수렴하지 않을 수 있습니다."
            )
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--esd_path", type=str, default=None)
    parser.add_argument("--json", type=str, default=None, help="리포트를 JSON으로 저장할 경로")
    parser.add_argument("--no_audio", action="store_true", help="wav 길이 측정 생략")
    args = parser.parse_args()

    if args.esd_path:
        esd_path = Path(args.esd_path)
    elif args.model_name:
        from config import get_path_config

        esd_path = get_path_config().dataset_root / args.model_name / "esd.list"
    else:
        parser.error("--model_name 또는 --esd_path 중 하나를 지정하세요.")

    if not esd_path.exists():
        logger.error(f"esd.list not found: {esd_path}")
        sys.exit(1)

    report = analyze(esd_path, check_audio=not args.no_audio)
    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON report saved to {args.json}")

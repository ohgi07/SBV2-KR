"""
韓国語 TTS モデルの Whisper 往復 CER 自動評価。

学習された各チェックポイントでテスト文を合成し、faster-whisper で音声認識した
結果と元テキストの CER (発音形の字母基準) を計算する。グラフ上の loss が
下がっていても品質が悪い場合 (NDC 発表で指摘された問題) を、人が全部聴かずに
検出するための定量評価。CER は低いほど良い。

テスト文は 2 グループで集計する:
- test: 汎化性能用の固定文 (下記 test_texts)
- train: モデルの train.list から自動サンプリングした学習文 (暗記性能の追跡用。
  学習が正常なら test より先に CER が下がる)

使い方:
    python speech_cer.py --model_name YourModel [--device cuda] [--whisper_model large-v3]

結果は cer_results/cer_{model_name}.csv と .png に保存される。
"""

import argparse
import csv
import random
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.cer import drop_hallucinated_segments, korean_cer
from style_bert_vits2.tts_model import TTSModel


warnings.filterwarnings("ignore")

cer_result_dir = Path("cer_results")
cer_result_dir.mkdir(exist_ok=True)

# 評価用テスト文
## 文章記号 (!?…-)・数字・歯擦音・激音/濃音など、発表で問題になりやすいと
## 指摘された要素を意図的に含めている
test_texts = [
    "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    "3일 전, 사과 5개를 샀는데 벌써 다 먹었어요.",
    "정말요? 그게 사실이에요? 믿을 수가 없어요...",
    "쉿, 조용히 하세요. 지금 시험 중이잖아요.",
    "숲속의 새싹이 쑥쑥 자라서 참 신기했습니다.",
    "값비싼 보석을 잃어버려서 정말 속상해요!",
    "출발 시간은 1시 30분이니까 늦지 마세요.",
    "책상 위에 있던 꽃잎이 바람에 날아갔어요.",
    ## 아래는 음운 현상별 묶음. CSV가 문장마다 열을 하나씩 쓰므로 이전 결과와 열을
    ## 맞추려면 순서를 건드리지 말고 새 문장은 끝에만 추가한다.
    # 연음 (옷을→오슬, 앞으로→아프로)
    "옷을 입고 밖에 나갔어요.",
    "꽃이 피었으니 앞으로 나와 보세요.",
    "이 책을 읽어 볼 만한 가치가 있어요.",
    # 경음화 (국밥집→국빱찝, 옷장→옫짱)
    "학교 앞 국밥집에서 먹고 갈까요?",
    "십 분 뒤에 출발할게요.",
    "늦게까지 남아서 옷장을 정리했다.",
    # 격음화 (축하→추카, 국화→구콰)
    "축하합니다, 정말 자랑스럽네요!",
    "그렇게 많이 좋아하는 줄은 몰랐어요.",
    "국화꽃 한 송이를 놓고 왔다.",
    # 비음화 (국립→궁닙, 밥물→밤물)
    "작년에는 국립 박물관에 갔었다.",
    "정답을 맞히는 것이 목표입니다.",
    "밥물을 잘 맞춰야 밥맛이 좋다.",
    # 유음화 (신라→실라, 권리→궐리)
    "신라의 천년 고도 경주에 다녀왔다.",
    "권리를 주장할 줄도 알아야 한다.",
    "설날에는 온 가족이 한자리에 모입니다.",
    # ㄴ 첨가 (색연필→생년필, 십육→심뉵)
    "색연필로 그림을 그렸어요.",
    "십육 년 만에 다시 만났다.",
    "담요를 덮고 한여름 밤을 보냈다.",
    # 구개음화 (굳이→구지, 해돋이→해도지)
    "굳이 그렇게까지 할 필요가 있을까?",
    "해돋이를 보러 새벽같이 나섰다.",
    "미닫이문이 잘 열리지 않는다.",
    # 겹받침 (밟고→밥꼬, 값을→갑쓸, 여덟→여덜)
    "닭이 앉아 있던 흙을 밟고 지나갔다.",
    "값을 깎아 주시면 여덟 개 살게요.",
    "삶이 없다면 앎도 없다.",
    # ㅎ 탈락 (낳은→나은, 싫어→시러)
    "낳은 정보다 기른 정이 더 크다.",
    "많은 사람이 싫어하는 일이다.",
    # 수사 (스물세·열두는 고유어계, 2026년·15분은 한자어계)
    "스물세 살에 처음 만나 열두 해를 함께했다.",
    "2026년 7월 31일 오후 3시 15분에 도착합니다.",
    "총액은 1,250,000원이고 할인율은 12.5퍼센트입니다.",
    # 외래어
    "카페에서 아메리카노 한 잔 주문했어요.",
    "이메일 주소는 알파벳 소문자로 입력하세요.",
    # 장문 — 끝까지 쉼과 억양이 유지되는지 본다
    "어제 저녁에 친구를 만나서 오랜만에 이런저런 이야기를 나누다 보니, 어느새 자정이 훌쩍 넘어 버렸습니다.",
    "이번 일에서 가장 중요한 것은 일정을 지키는 일이지만, 그보다 더 중요한 것은 결과물의 품질이라고 생각합니다.",
    # 단문 — 짧으면 자모 하나 차이로도 CER이 크게 튄다
    "네, 알겠습니다.",
]

def load_train_texts(train_list_path: Path, num: int, seed: int) -> list[str]:
    """전처리된 train.list에서 KO 발화의 정규화 텍스트를 결정론적으로 샘플링한다."""
    texts: list[str] = []
    with train_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            # wav_path|spk|lang|norm_text|phones|tones|word2ph
            if len(parts) >= 4 and parts[2] == "KO" and parts[3]:
                texts.append(parts[3])
    if len(texts) <= num:
        return texts
    return random.Random(seed).sample(texts, num)

if __name__ == "__main__":
    path_config = get_path_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument("--whisper_device", type=str, default=None)  # VRAM 不足時に whisper だけ CPU に逃がす
    parser.add_argument("--compute_type", type=str, default="bfloat16")
    parser.add_argument("--unit", type=str, default="jamo", choices=["jamo", "syllable"])  # fmt: skip
    parser.add_argument("--train_list", type=Path, default=None, help="전처리된 train.list 경로 (기본: {dataset_root}/{model_name}/train.list)")  # fmt: skip
    parser.add_argument("--num_train", type=int, default=4, help="train.list에서 샘플링할 학습 문장 수 (0으로 비활성)")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_name: str = args.model_name
    device: str = args.device
    unit: str = args.unit

    # 학습 문장 그룹 (암기 성능 추적용 — 일반화 test_texts와 분리 집계)
    train_list_path: Path = args.train_list or (path_config.dataset_root / model_name / "train.list")  # fmt: skip
    if args.num_train > 0 and train_list_path.exists():
        train_texts = load_train_texts(train_list_path, args.num_train, args.seed)
        logger.info(f"Sampled {len(train_texts)} train sentences from {train_list_path}")
    else:
        train_texts = []
        if args.num_train > 0:
            logger.warning(f"train.list not found at {train_list_path} — train group is skipped")  # fmt: skip

    # Whisper モデルの読み込み (全チェックポイントで共有)
    from faster_whisper import WhisperModel

    whisper_device: str = args.whisper_device or device
    logger.info(f"Loading faster-whisper model ({args.whisper_model}) on {whisper_device}")
    try:
        whisper = WhisperModel(args.whisper_model, device=whisper_device, compute_type=args.compute_type)  # fmt: skip
    except ValueError as e:
        logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
        whisper = WhisperModel(args.whisper_model, device=whisper_device)

    def transcribe_ko(wav_path: str) -> str:
        # vad_filter는 환각을 줄일 뿐 없애지 못해 남은 것은 발화 속도로 걸러낸다
        segments, _ = whisper.transcribe(wav_path, language="ko", beam_size=1, vad_filter=True)
        return "".join(segment.text for segment in drop_hallucinated_segments(segments))

    model_path = path_config.assets_root / model_name
    safetensors_files = list(model_path.glob("*.safetensors"))
    logger.info(f"There are {len(safetensors_files)} models.")

    results = []
    for model_file in tqdm(safetensors_files, dynamic_ncols=True):
        # `test_e10_s1000.safetensors` -> 1000 を取り出す
        match = re.search(r"_s(\d+)\.safetensors$", model_file.name)
        if match:
            step = int(match.group(1))
        else:
            logger.warning(f"Step count not found in {model_file.name}, so skip it.")
            continue
        model = TTSModel(
            model_path=model_file,
            config_path=model_file.parent / "config.json",
            style_vec_path=model_file.parent / "style_vectors.npy",
            device=device,
        )
        cers = []
        for group, text in [("train", t) for t in train_texts] + [("test", t) for t in test_texts]:  # fmt: skip
            sr, audio = model.infer(text, language=Languages.KO)
            # faster-whisper にはファイルパスで渡す (内部でリサンプリングされる)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                import wave

                with wave.open(tmp_path, "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(sr)
                    f.writeframes(audio.astype(np.int16).tobytes())
                hypothesis = transcribe_ko(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            cer = korean_cer(text, hypothesis, unit=unit)
            cers.append(cer)
            logger.info(f"CER {cer:.3f} [{group}]: {text} -> {hypothesis}")
        results.append((model_file.name, step, cers))
        del model

    logger.success("All models have been evaluated:")
    # 平均を計算し、CER が低い順にソート
    n_train = len(train_texts)
    results = [
        (model_file, step, cers + [float(np.mean(c)) if c else float("nan") for c in (cers[:n_train], cers[n_train:], cers)])  # fmt: skip
        for model_file, step, cers in results
    ]
    results = sorted(results, key=lambda x: x[2][-2])  # 일반화 성능(mean_test) 기준 정렬
    for model_file, step, cers in results:
        logger.info(f"{model_file}: mean CER = {cers[-1]:.3f} (train {cers[-3]:.3f} / test {cers[-2]:.3f})")  # fmt: skip

    with open(
        cer_result_dir / f"cer_{model_name}.csv", "w", encoding="utf_8_sig", newline=""
    ) as f:
        writer = csv.writer(f)
        text_cols = [f"[train] {t}" for t in train_texts] + [f"[test] {t}" for t in test_texts]  # fmt: skip
        writer.writerow(["model_path", "step"] + text_cols + ["mean_train", "mean_test", "mean"])  # fmt: skip
        for model_file, step, cers in results:
            writer.writerow([model_file] + [step] + cers)
    logger.info(f"cer_{model_name}.csv has been saved.")

    # ステップごとの CER 推移グラフ
    import matplotlib.pyplot as plt
    import pandas as pd

    steps = [step for _, step, _ in results]
    cer_values = [cers for _, _, cers in results]
    df = pd.DataFrame(cer_values, index=steps).sort_index()

    plt.figure(figsize=(10, 5))
    for col in range(len(df.columns) - 3):
        group = "train" if col < n_train else "test"
        idx = col + 1 if col < n_train else col - n_train + 1
        plt.plot(df.index, df.iloc[:, col], label=f"{group} {idx}", alpha=0.3)
    if n_train > 0:
        plt.plot(df.index, df.iloc[:, -3], label="Mean (train)", color="tab:blue", linewidth=2)  # fmt: skip
    plt.plot(df.index, df.iloc[:, -2], label="Mean (test)", color="tab:orange", linewidth=2)  # fmt: skip
    plt.plot(df.index, df.iloc[:, -1], label="Mean", color="black", linewidth=2)
    plt.title(f"TTS Round-trip CER ({unit} level, lower is better)")
    plt.xlabel("Step Count")
    plt.ylabel("CER")
    plt.grid(True, axis="x")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(cer_result_dir / f"cer_{model_name}.png")
    plt.show()

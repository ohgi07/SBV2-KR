# Style-Bert-VITS2 한국어 지원 (SBV2-KR)

Style-Bert-VITS2에 한국어(`KO`) 학습·추론을 추가한 포트입니다. 일본어 전용으로 설계된 Style-Bert-VITS2에서 언어 의존적인 두 구성요소(G2P와 BERT)만 한국어에 맞게 교체하고, 나머지 VITS2 아키텍처는 그대로 재사용합니다.

구현은 NDC26 발표 「SBV2 오픈소스를 활용한 한국어/일본어 TTS 만들기」(넥슨게임즈 김명지)에서 소개된 접근 방식을 기반으로 했습니다.

## 지원 현황

| 기능 | 상태 |
|------|------|
| 한국어 학습 | ✅ |
| 한국어 추론 | ✅ |
| WebUI | ✅ |
| JP-Extra 아키텍처 | ✅ |
| 기존 JP 모델 호환 | ✅ |
| ONNX 추론 | ❌ |
| 악센트(음높이) 제어 | ❌ |

## 주요 특징

- 한국어 G2P 자체 구현 (외부 G2P 라이브러리 불필요)
- 표준발음법 기반 발음 변환
- 숫자·단위·알파벳 정규화
- [`klue/roberta-large`](https://huggingface.co/klue/roberta-large) 기반 한국어 BERT 지원
- JP-Extra 아키텍처 그대로 사용
- 기존 JP 모델과 체크포인트 호환
- CER 기반 자동 평가 도구 포함

## 설치

### 1. 환경 준비

Python 3.10 기준입니다. 가상환경을 만들고 의존성을 설치합니다.

```bash
# 저장소 루트에서
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows bash / Linux / macOS:  source .venv/bin/activate   (Windows bash는 .venv/Scripts/activate)

pip install -r requirements.txt
```

`requirements.txt`에 한국어에 필요한 `kiwipiepy`, `transformers`, `torch` 등이 모두 포함되어 있습니다. GPU 학습·추론에는 CUDA용 torch가 필요합니다 (예: `torch<2.6+cu118`, `torchaudio<2.6+cu118`).

> **한국어 G2P 의존성 요약**
> - 발음 변환은 내장 규칙 엔진이라 **추가 G2P 라이브러리·네트워크가 필요 없습니다** (`g2pkk`/`eunjeon`/`mecab`/nltk `cmudict` 전부 불필요).
> - [kiwipiepy](https://github.com/bab2min/kiwipiepy)는 형태소 정보가 필요한 규칙(ㄴ첨가·의→에·일부 경음화)에 쓰입니다. 순수 wheel이라 Windows에서도 빌드 도구 없이 설치됩니다. 없어도 동작하지만 정확도를 위해 설치를 권장합니다.
> - 발음 규칙만 쓰거나 테스트만 돌릴 때는 torch 없이 `kiwipiepy`만으로 G2P가 동작합니다.

### 2. BERT 모델 다운로드

한국어 문맥 임베딩용 BERT를 `bert/` 아래에 내려받습니다. `bert/bert_models.json`에 `klue-roberta-large`가 등록되어 있어 아래 명령으로 자동 준비됩니다.

```bash
python initialize.py
```

## 사용 방법

### 데이터셋

`esd.list`의 언어 컬럼에 `KO`를 지정합니다:

```
wavs/speaker_001.wav|speaker|KO|3일 전, 배가 고팠다.
wavs/speaker_002.wav|speaker|KO|안녕하세요! 오늘은 날씨가 좋네요.
```

대본에는 문장 기호(`! ? … , .`)를 최대한 다양하게 포함하세요. **기반 모델 코퍼스에 등장하지 않는 기호·단어는 합성 품질이 떨어집니다.** 특히 치찰음 ㅅ/ㅆ/ㅈ/ㅊ 포함 단어는 다양한 문맥으로 포함하기를 권장합니다.

### 코퍼스 검사 (학습 전 권장)

```bash
python analyze_corpus.py --model_name YourModel
# 또는
python analyze_corpus.py --esd_path path/to/esd.list --json report.json
```

데이터셋의 음소·문장기호 커버리지, 치찰음 문맥 다양성, 텍스트·음성 길이 통계를 검사합니다. 흔한 문제 패턴(미등장 기호, 치찰음 문맥 부족, 지나치게 긴 평균 발화 길이)을 경고합니다.

### 학습

WebUI(`python app.py`)의 학습 탭에서 **JP-Extra판 사용**을 켜고 진행하거나, CLI로:

```bash
python preprocess_all.py --use_jp_extra ...
python train_ms_jp_extra.py ...
```

가중치 고정(freeze) 옵션 중 「일본어 bert 부분을 고정」(CLI `--freeze_JP_bert`)은 한국어에도 그대로 적용됩니다. KO는 JP-Extra의 단일 BERT 슬롯을 공유하므로 별도의 한국어용 플래그가 없습니다.

경험상 도움이 되는 팁:
- 기반 모델을 처음부터 만들 경우 **판별자(D) 오버피팅** 주의 — 생성자(G) 선행 학습 구간을 두고, D의 학습률을 G보다 낮게 설정
- 데이터셋에 노이즈가 많거나 평균 발화 길이가 지나치게 길면 수렴하지 않을 수 있음
- SBV2 파인튜닝은 기본 제공 파라미터를 그대로 쓰는 것이 안정적
- 그래프 수치가 낮다고 반드시 품질이 좋은 것은 아님 — 직접 청취 병행

### 추론

```python
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

bert_models.load_model(Languages.KO, "bert/klue-roberta-large")
bert_models.load_tokenizer(Languages.KO, "bert/klue-roberta-large")

model = TTSModel(model_path=..., config_path=..., style_vec_path=..., device="cuda")
sr, audio = model.infer(text="안녕하세요!", language=Languages.KO, style="Neutral")
```

웹 UI 음성 합성 탭에서는 「언어」 드롭다운에서 `KO`를 선택하면 됩니다.

### 웹 UI

`python app.py`로 실행하는 웹 UI는 탭 제목과 주요 3개 탭(음성 합성·데이터셋 생성·학습)의 표시 텍스트가 한국어화되어 있습니다. 스타일 생성·머지·ONNX 변환 탭 내부는 아직 일본어입니다. 일본어 합성 예문, whisper 초기 프롬프트 기본값, 기본 제공 모델의 크레딧 표기 원문 등 콘텐츠성 텍스트는 의도적으로 원문을 유지합니다.

### 자동 평가 (Whisper 왕복 CER)

```bash
python speech_cer.py --model_name YourModel [--whisper_model large-v3] [--unit jamo|syllable]
```

각 체크포인트(`*_s{step}.safetensors`)로 테스트 문장을 합성하고, faster-whisper(ko)로 재인식한 뒤 원문과의 CER을 계산합니다. loss 그래프가 수렴해도 품질이 나쁜 경우를 직접 청취 없이 걸러낼 수 있습니다. 결과는 `cer_results/cer_{모델명}.csv`와 스텝별 추이 그래프 `.png`로 저장됩니다.

문장은 두 그룹으로 나뉘어 집계됩니다:
- **test**: 스크립트에 내장된 고정 일반화 문장 (문장 기호·숫자·치찰음·격음/경음 포함)
- **train**: 모델의 `train.list`에서 자동 샘플링한 학습 문장 (`--num_train`, 기본 4개) — 암기 성능 추적용으로, 학습이 정상이라면 test보다 CER이 먼저 떨어집니다

CER은 **발음형 자모 기준**으로 계산됩니다 (`style_bert_vits2/nlp/korean/cer.py`):
- 양쪽 텍스트를 정규화·발음 변환 후 비교합니다. ASR의 표기 흔들림(`맛있다`/`마싣따`), 숫자 표기 차이(`3개`/`세 개`), 띄어쓰기·구두점 차이는 오류로 계산되지 않습니다.
- 자모 단위(기본)는 받침 하나 오류를 1/3 음절로 계산해 음절 단위보다 완만한 신호를 제공합니다 (`--unit syllable`로 변경 가능).

## 설계 개요

Style-Bert-VITS2에서 언어 의존적인 부분은 텍스트를 음소로 변환하는 G2P와 문맥 임베딩을 생성하는 BERT뿐입니다. VITS2 본체, 스타일 벡터, 디코더, 판별자 등 나머지 구성은 언어와 무관하게 동작하므로, 한국어 지원은 이 두 모듈만 교체하는 방식으로 구현했습니다.

### 1. G2P 파이프라인 (`style_bert_vits2/nlp/korean/`)

기존 일본어 처리 과정에서 한국어에 불필요한 단계를 제거하고 교체했습니다:

| 일본어 처리 | 한국어 처리 | 비고 |
|---|---|---|
| Normalize (num2words) | Normalize (자체 구현) | 숫자→한자어 수사(만/억/조), 통화·퍼센트, 알파벳 음독 |
| 형태소 분석 (pyopenjtalk) | **제거** | 한자 발음 분석이 불필요 |
| 악센트 정보 추출 | **제거** | 한국어는 고저 악센트로 의미가 갈리지 않음 |
| 가타카나 음소 변환 | 표준발음법 발음 변환 + 자모 분해 | 표기≠발음이므로 발음 변환 필수 |

#### 발음 변환 (`pronounce.py`)

표준발음법을 규칙 기반으로 구현한 자체 엔진입니다. 외부 G2P 라이브러리에 의존하지 않으며, 다음 규칙을 적용합니다.

- 연음
- 구개음화
- ㅎ 탈락·격음화
- 자음군 단순화
- 종성 중화
- 경음화
- 비음화
- 유음화
- ㅢ 발음 규칙 (희망→[히망])

**어절 경계(공백을 넘는) 규칙**도 적용합니다. 공백 정확히 1개로 인접한 어절 쌍에 연음(§15: 몇 월→[며둴])·격음화(못 해→[모태])·경음화(몇 개→[멷깨])·비음화(§18 붙임: 밥 먹는다→[밤멍는다])·유음화(§20)를 적용합니다. ㄴ첨가(§29 붙임2: 한 일→[한닐])는 형태소 조건(1음절 실질형태소)이 필요하므로 `morph.py` 단계에서 처리합니다. 구두점 등 공백 이외의 경계는 휴지로 간주해 적용하지 않습니다.

모든 변환은 **음절 수(문자 수) 보존을 보장**합니다 (word2ph 정렬에 필수).

초기 버전은 g2pkk를 백엔드로 썼으나, 전수 비교로 동등 이상을 확인한 뒤 의존을 제거했습니다.

#### 그 외 구성요소

- **숫자 읽기**: 단위명사 앞의 1-99는 고유어 관형형으로 읽습니다 (`3개→세 개`, `20살→스무 살`, `1시 30분→한 시 삼십 분`, `1번째→첫 번째`). 100 이상·소수·한자어 단위(개월/분 등)는 한자어 수사로 읽습니다.
- **형태소 기반 보정** (`morph.py`): kiwipiepy가 설치되어 있으면 형태소 정보가 필요한 규칙을 추가 적용합니다 — 형태소 경계 ㄴ첨가(한여름→[한녀름]), 속격 조사 의→[에](나의→[나에]), 관형사형 -ㄹ 뒤 경음화(갈 데가→[갈 떼가]), 용언 어간말 ㄴ/ㅁ 뒤 경음화(신다→[신따]). 등재 합성어의 ㄴ첨가와 예외어는 내장 발음 예외 사전(`PRONUNCIATION_EXCEPTIONS`, 같은 글자 수 항목을 추가해 확장 가능)으로 처리합니다 (솜이불→[솜니불], 맛있다→[마싣따] 등). kiwipiepy 인스턴스는 `morph.get_kiwi()`로 싱글턴으로 공유됩니다.
- **음소 체계**: 발음형 한글을 자모(초성 18종 + 중성 21종 + 중화된 종성 7종 = 46개)로 분해해 심볼로 사용합니다. 초성 ㅇ(무음)은 음소를 내지 않습니다.
- **톤**: 한국어는 전부 0 (톤 미사용, `NUM_KO_TONES = 1`).
- **공백**: `SP` 음소로 매핑되어 모델이 어절 간 호흡(무음)을 학습할 수 있습니다.

예시:

```
"3일 전, 배가 고팠다."
→ 정규화: "삼일 전, 배가 고팠다."
→ 발음:   "사밀 전, 배가 고팓따."
→ phones: [_, ᄉ, ᅡ, ᄆ, ᅵ, ᆯ, SP, ᄌ, ᅥ, ᆫ, ",", SP, ᄇ, ᅢ, ᄀ, ᅡ, SP, ᄀ, ᅩ, ᄑ, ᅡ, ᆮ, ᄄ, ᅡ, ".", _]
→ tones:  전부 0
→ word2ph: [1, 2, 3, 1, 3, 1, 1, 2, 2, 1, 2, 3, 2, 1, 1]
```

### 2. BERT 모델 교체

일본어 DeBERTa 대신 [`klue/roberta-large`](https://huggingface.co/klue/roberta-large)를 기본으로 사용합니다. `klue/roberta-large`는 hidden size가 1024로 JP-Extra의 기존 BERT 인터페이스와 동일하므로 모델 구조를 수정할 필요가 없습니다. 모두의 말뭉치·위키 등 정제된 코퍼스로 학습되어 댓글 코퍼스 기반 모델(KcBERT)보다 편향·도메인 쏠림이 적고, 최대 입력 길이도 더 깁니다.

WordPiece 서브워드 토크나이저는 중국어처럼 토큰과 문자가 1:1로 대응하지 않습니다. 그래서 offset mapping으로 토큰 특징을 문자 단위로 전개한 뒤, word2ph에 따라 음소 단위로 확장합니다 (`korean/bert_feature.py`). 이 경로는 아키텍처 중립적이므로 hidden size 1024인 다른 한국어 모델(`beomi/kcbert-large` 등)로 교체해 A/B 비교할 수 있습니다 — `bert_models.load_model/load_tokenizer(Languages.KO, <경로>)`로 지정합니다. 단, **BERT 선택은 TTS 학습 시점에 고정**되므로 기반 모델 학습 전에 비교를 마쳐야 합니다.

### 3. 아키텍처: JP-Extra 경로 재사용

한국어 모델은 **JP-Extra 계열 아키텍처(단일 BERT 입력)** 로 학습/추론합니다. 텍스트 프런트엔드만 한국어로 바뀌고 모델 구조는 동일하므로, 학습 시 `train_ms_jp_extra.py`(WebUI의 "JP-Extra판 사용" 체크)를 그대로 사용하면 됩니다.

### 4. 기존 모델과의 호환성

한국어 심볼·톤·언어 ID는 모두 기존 테이블 뒤에 추가되므로, 기존 JP/EN/ZH 심볼의 인덱스는 변경되지 않습니다. KO 추가 이전에 학습된 체크포인트를 로드하면 임베딩 테이블의 기존 행을 그대로 복사하고 새 행만 초기값으로 두는 호환 처리가 자동으로 적용됩니다 (`checkpoints.py` / `safetensors.py`).

기존 일본어 모델은 그대로 동작합니다. 일본어 사전학습 모델에서 한국어를 파인튜닝하는 것도 가능합니다 (단, 한국어 음소 임베딩은 처음부터 학습됩니다).

## 제한 사항

1. **ONNX 추론 미지원**: `convert_bert_onnx.py`가 아직 한국어 BERT 변환을 지원하지 않습니다. ONNX 경로(`extract_bert_feature_onnx`)는 변환된 모델을 직접 준비한 경우에만 동작합니다. PyTorch 추론 경로를 사용하세요.
2. **음높낮이(악센트) 조절 불가**: 악센트 추출 단계를 제거했으므로 일본어에서 제공되던 학습 기반 음높이 조절은 사용할 수 없습니다. 장음 조절도 마찬가지입니다.
3. **kiwipiepy 미설치 시 일부 규칙 비활성**: 형태소 기반 규칙(ㄴ첨가·어절 경계 ㄴ첨가 등)은 kiwipiepy 설치 시에만 동작합니다. 미설치 시 내장 규칙 엔진 + 예외 사전만 적용됩니다.
4. **숫자 읽기 한계**: 숫자+단위의 동철이의어(3분 = 삼 분/세 분, 20대 = 이십 대/스무 대)는 규칙만으로 구분할 수 없어 더 흔한 읽기로 고정되어 있습니다 (분→시간 단위 한자어, 대→수량 고유어). 단, 서수 접두 `제N`(제3장, 제2회)은 항상 한자어로 읽습니다. 한자어 수사+단위의 ㄴ첨가는 불규칙하므로(삼일→[사밀]이나 십육→[심뉵]) 일반 규칙에서 수사(NR)를 제외하고, ㄴ첨가가 필요한 어휘만 예외 사전에 등록합니다.
5. **표준발음법 중 의도적으로 구현하지 않은 조항**: §6·7 모음의 장단(음소 체계에 길이 구분이 없음 — 현대 서울말에서 변별력 상실), §16 한글 자모 이름의 특례(디귿이→[디그시] — TTS 입력에 드물고 일반 어휘 오발동 위험이 큼). 복수 발음이 허용되는 조항(§5 어중 ㅢ→[ㅣ] 허용, §22 되어[되여] 허용, §30 사이시옷 [ㄷ] 허용 등)은 한쪽 표준 발음을 일관되게 채택합니다.
6. **BERT 최대 입력 길이**: 모델에 따라 다릅니다 (klue/roberta-large: 512 토큰, kcbert-large: 300 토큰 — 토크나이저에서 자동으로 읽어 적용됨). 초과분은 잘리므로 긴 문장은 나눠서 합성하세요 (기본 추론 설정은 줄 단위 분할이 켜져 있음).

## 테스트

```bash
pytest tests/test_korean.py           # 정규화·발음 규칙·G2P 불변식·BERT 정렬
pytest tests/test_korean_goldset.py   # 표준발음법 골드셋 154항목 회귀 + 퍼징
```

모델 가중치는 불필요합니다. 골드셋은 표준발음법 조항별 대표 예시(어절 경계 규칙·어휘 예외 포함)를 실제 G2P 경로로 검증합니다. 유일한 `xfail`은 문맥 없이 판별 불가능한 고립 동철이의어 `신고`(신다[신꼬]/申告[신고])입니다.

```bash
python tests/test_korean_goldset.py   # 카테고리별 정확도 리포트 출력
```

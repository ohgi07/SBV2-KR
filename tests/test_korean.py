"""
韓国語 (KO) サポートのテスト。

実行方法: pytest tests/test_korean.py
BERT モデルの重みは不要 (トークナイザーのみ利用する)。
"""

import pytest

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text, cleaned_text_to_sequence
from style_bert_vits2.nlp.korean.g2p import g2p
from style_bert_vits2.nlp.korean.normalizer import normalize_text, read_number
from style_bert_vits2.nlp.korean.pronounce import pronounce
from style_bert_vits2.nlp.symbols import (
    KO_SYMBOLS,
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    NUM_TONES,
    PUNCTUATION_SYMBOLS,
    SYMBOLS,
)


class TestSymbols:
    def test_ko_symbols_appended_after_existing(self):
        """韓国語シンボルは既存シンボルの末尾に追加され、既存のインデックスを変えない"""
        # KO_SYMBOLS は SYMBOLS の末尾に位置する
        assert SYMBOLS[-len(KO_SYMBOLS) :] == KO_SYMBOLS
        # 既存の並び ([PAD] + NORMAL + PUNCTUATION) が先頭に保持されている
        assert SYMBOLS[0] == "_"
        pun_start = len(SYMBOLS) - len(KO_SYMBOLS) - len(PUNCTUATION_SYMBOLS)
        assert SYMBOLS[pun_start : pun_start + len(PUNCTUATION_SYMBOLS)] == PUNCTUATION_SYMBOLS  # fmt: skip

    def test_ko_symbols_unique(self):
        assert len(SYMBOLS) == len(set(SYMBOLS))

    def test_language_maps(self):
        assert LANGUAGE_ID_MAP["KO"] == 3
        assert LANGUAGE_TONE_START_MAP["KO"] == NUM_TONES - 1


class TestNormalizer:
    def test_numbers(self):
        assert read_number("3") == "삼"
        assert read_number("2026") == "이천이십육"
        assert read_number("10000") == "만"
        assert read_number("110000") == "십일만"
        assert read_number("0") == "영"
        assert read_number("3.14") == "삼점일사"
        assert read_number("100000000") == "억"
        # 일 생략은 최상위 그룹만: 중간 그룹까지 생략하면 "억만"처럼 다른 수로 들린다
        assert read_number("100010000") == "억일만"
        assert read_number("10001") == "만일"
        assert read_number("20000") == "이만"

    def test_normalize_numbers_in_text(self):
        assert normalize_text("3일 전") == "삼일 전"
        assert normalize_text("1,000원") == "천원"

    @pytest.mark.parametrize(
        "inp, expected",
        [
            ("3개", "세개"),
            ("3 개", "세 개"),
            ("21개", "스물한개"),
            ("20살", "스무살"),
            ("100개", "백개"),  # 100 이상은 한자어
            ("3개월", "삼개월"),  # 개월은 한자어
            ("30분", "삼십분"),  # 분은 한자어 (시간 단위)
            ("1시 30분", "한시 삼십분"),
            ("3시간", "세시간"),
            ("1번째", "첫번째"),
            ("12시", "열두시"),
            ("1대1", "일대일"),  # 숫자 사이의 단위는 변환하지 않음
            ("4명이", "네명이"),  # 조사가 붙어도 변환
            ("3.5개", "삼점오개"),  # 소수는 한자어
            ("제3장", "제삼장"),  # 서수 접두 제N은 항상 한자어
            ("제1회", "제일회"),
            ("3장", "세장"),  # 제 없이는 고유어
        ],
    )
    def test_native_numbers(self, inp: str, expected: str):
        assert normalize_text(inp) == expected

    def test_normalize_currency_and_percent(self):
        assert normalize_text("₩500") == "오백원"
        assert normalize_text("50%") == "오십퍼센트"

    @pytest.mark.parametrize(
        "inp, expected",
        [
            ("5kg", "오킬로그램"),
            ("30g", "삼십그램"),
            ("500mg", "오백밀리그램"),
            ("2t", "이톤"),
            ("10km 달리기", "십킬로미터 달리기"),
            ("180m", "백팔십미터"),
            ("3cm", "삼센티미터"),
            ("7mm", "칠밀리미터"),
            ("1.5L", "일점오리터"),
            ("500ml", "오백밀리리터"),
            ("5 kg", "오킬로그램"),  # 숫자와 단위 사이 공백 허용 (퍼센트와 동일)
            ("80km/h", "시속 팔십킬로미터"),  # 속도는 어순을 바꿔 자연스럽게 읽는다
            ("10m/s", "초속 십미터"),
            ("5G 시대", "오지 시대"),  # 대문자 G는 단위가 아님 (알파벳 낱자 읽기 유지)
            ("3gb", "삼지비"),  # 단위 뒤에 알파벳이 이어지면 단위로 보지 않음
        ],
    )
    def test_normalize_metric_units(self, inp: str, expected: str):
        assert normalize_text(inp) == expected

    @pytest.mark.parametrize(
        "inp, expected",
        [
            # 월 이름 특례 (표준 관용 읽기)
            ("10월 2일", "시월 이일"),
            ("6월 25일", "유월 이십오일"),
            ("12월", "십이월"),  # 특례는 6월·10월뿐
            ("16월", "십육월"),  # 월이 아닌 숫자 꼬리에는 오발동하지 않음
            # 연령대 (십 단위 + 대)는 한자어
            ("20대 여성", "이십대 여성"),
            ("30대 초반", "삼십대 초반"),
            ("10대", "십대"),
            ("차 3대", "차 세대"),  # 십 단위가 아니면 기존 고유어 유지
            # 점수·비율의 「숫자 대 숫자」는 양쪽 다 한자어
            ("11 대 8", "십일 대 팔"),
            ("70 대 60", "칠십 대 육십"),
            # 안내번호·전화번호는 낱자 읽기
            ("112를 누르세요", "일일이를 누르세요"),
            ("114에 전화", "일일사에 전화"),
            ("119에 신고", "일일구에 신고"),
            ("110 미만", "백십 미만"),  # 안내번호 목록 외의 3자리는 자릿수 읽기
            ("010-1234-5678", "공일공 일이삼사 오육칠팔"),
            ("02-345-6789", "공이 삼사오 육칠팔구"),
            ("01012345678", "공일공일이삼사오육칠팔"),  # 0으로 시작하는 숫자열은 자릿수 읽기가 성립하지 않음
            ("0.5", "영점오"),  # 소수는 낱자 읽기 대상 아님
            ("10.05", "십점영오"),
            # 번호 문맥의 「N번」은 한자어 (후속 명사 allowlist·다이얼 동사만, 오발동 방지)
            ("3번 출구로 나가세요", "삼번 출구로 나가세요"),
            ("10번 버스를 타세요", "십번 버스를 타세요"),
            ("9번 문제의 답", "구번 문제의 답"),
            ("3번을 눌러 주세요", "삼번을 눌러 주세요"),
            ("9번 누르세요", "구번 누르세요"),
            ("3번 반복하세요", "세번 반복하세요"),  # 횟수 의미는 고유어 유지
            ("그 영화를 3번 봤어", "그 영화를 세번 봤어"),
            ("하루에 3번 약을 드세요", "하루에 세번 약을 드세요"),
            # 두 자리 연도의 년생·학번은 낱자 읽기
            ("78년생이에요", "칠팔년생이에요"),
            ("98학번", "구팔학번"),
            ("1978년생", "천구백칠십팔년생"),  # 네 자리 연도는 자릿수 읽기 유지
            # 알파벳 바로 뒤의 한 자리 숫자는 영어식 읽기
            ("F1 경기", "에프원 경기"),
            ("mp3 파일", "엠피쓰리 파일"),
            ("H2O", "에이치투오"),
            ("A4 용지", "에이포 용지"),
            ("F16 전투기", "에프십육 전투기"),  # 여러 자리는 한자어 읽기 유지
            ("코로나19", "코로나십구"),  # 한글 뒤 숫자는 대상 아님
        ],
    )
    def test_number_reading_refinements(self, inp: str, expected: str):
        assert normalize_text(inp) == expected

    def test_alphabet_c_reads_ssi(self):
        # 표기 규범상은 '시'지만 실제 낭독 관례는 '씨' (KSS 낭독 기준)
        assert normalize_text("비타민 C") == "비타민 씨"

    def test_normalize_punctuation(self):
        assert normalize_text("안녕하세요。") == "안녕하세요."
        assert normalize_text("뭐라고？！") == "뭐라고?!"
        assert normalize_text("그건…") == "그건..."
        assert normalize_text("「인용」") == "'인용'"

    def test_normalize_alphabet(self):
        assert normalize_text("AI") == "에이아이"

    def test_normalize_removes_unknown_chars(self):
        # 日本語・絵文字などは除去される
        result = normalize_text("안녕 こんにちは 😊 하세요")
        assert result == "안녕 하세요"

    def test_normalize_result_charset(self):
        """正規化結果はハングル・スペース・句読点のみからなる"""
        import re

        result = normalize_text("복잡한 텍스트! 123개, ABC… ㅋㅋ (테스트)")
        assert re.fullmatch(r"[가-힣 !?.,'\-]*", result), result


class TestPronounce:
    @pytest.mark.parametrize(
        "orig, expected",
        [
            # 연음 (連音)
            ("밥이", "바비"),
            ("옷을", "오슬"),
            ("삼일", "사밀"),  # NDC 発表の例: 3일 → 삼일 → [사밀]
            # 구개음화 (口蓋音化)
            ("같이", "가치"),
            ("굳이", "구지"),
            # ㅎ 脱落・激音化
            ("좋아", "조아"),
            ("좋다", "조타"),
            ("국화", "구콰"),
            ("많이", "마니"),
            # 겹받침 (二重終声)
            ("값", "갑"),
            ("값이", "갑씨"),
            ("닭", "닥"),
            ("앉다", "안따"),
            # 終声中和
            ("옷", "옫"),
            ("부엌", "부억"),
            ("숲", "숩"),
            # 경음화 (硬音化)
            ("국밥", "국빱"),
            ("학교", "학꾜"),
            # 비음화 (鼻音化)
            ("국물", "궁물"),
            ("십리", "심니"),
            ("종로", "종노"),
            ("독립", "동닙"),
            # 유음화 (流音化)
            ("신라", "실라"),
            ("칼날", "칼랄"),
        ],
    )
    def test_pronunciation_rules(self, orig: str, expected: str):
        assert pronounce(orig) == expected

    def test_length_preserved(self):
        text = "옛날 옛적에 호랑이가 살았어요. 참! 값진 이야기죠?"
        assert len(pronounce(text)) == len(text)

    def test_non_hangul_preserved(self):
        text = "안녕... 뭐, 해?"
        result = pronounce(text)
        for p, o in zip(result, text):
            if not ("가" <= o <= "힣"):
                assert p == o


class TestMorphRules:
    """형태소 기반 발음 보정 (예외 사전은 항상, Kiwi 규칙은 설치 시에만)"""

    @pytest.mark.parametrize(
        "orig, expected",
        [
            ("맛있다", "마싣따"),
            ("멋있다", "머싣따"),
            ("맛없다", "마덥따"),
            ("솜이불", "솜니불"),
            ("색연필", "생년필"),
            ("꽃잎", "꼰닙"),
            ("나뭇잎", "나문닙"),
            ("물약", "물략"),
            ("식용유", "시굥뉴"),
            ("서울역", "서울력"),
            ("홑이불", "혼니불"),  # 구개음화 오적용(호치불) 방지
            ("늑막염", "능망념"),
            ("솔잎", "솔립"),
            ("물엿", "물렫"),
            ("줄넘기", "줄럼끼"),
        ],
    )
    def test_exception_dictionary(self, orig: str, expected: str):
        from style_bert_vits2.nlp.korean.morph import apply_morph_rules

        assert pronounce(apply_morph_rules(orig)) == expected

    @pytest.mark.parametrize(
        "orig, expected",
        [
            ("희망", "히망"),  # 자음 + ㅢ → ㅣ (필수)
            ("무늬", "무니"),
            ("회의감", "회이감"),  # 어중 의 → 이
        ],
    )
    def test_ui_rules(self, orig: str, expected: str):
        assert pronounce(orig) == expected

    @pytest.mark.parametrize(
        "orig, expected",
        [
            ("한여름", "한녀름"),  # 형태소 경계 ㄴ첨가
            ("나의 회의감", "나에 회이감"),  # 속격 조사 의 → 에
            ("갈 데가 없다", "갈 떼가 업따"),  # 관형사형 ㄹ 뒤 경음화 (축약형 ᆯ)
            ("먹을 것", "머글 껃"),  # 관형사형 을 (완성형 음절 폼도 검출)
            ("먹을 수 있다", "머글 쑤 읻따"),
            ("받을 돈", "바들 똔"),
            ("먹은 밥", "머근 밥"),  # ETM 은/ㄴ은 경음화 없음
            ("신다", "신따"),  # 어간말 ㄴ 뒤 경음화
            ("안고", "안꼬"),
            ("감고", "감꼬"),  # 어간말 ㅁ 뒤 경음화
            ("젊다", "점따"),  # 어간말 ㄻ 뒤 경음화 (표준발음법 제24항)
            ("닮고", "담꼬"),
            ("할 수 있다", "할 쑤 읻따"),
            ("신발을 신고 갔다", "신바를 신꼬 갇따"),  # 문맥으로 용언 판별
            # 오탐 방지
            ("산도 좋다", "산도 조타"),  # 명사 + 조사는 경음화 없음
            ("간 사람", "간 사람"),  # 과거 관형형 ㄴ은 경음화 없음
            ("삶과 죽음", "삼과 주금"),  # 명사의 ㄻ은 경음화 없음
            ("삼일", "사밀"),  # 한자어 수사 + 단위는 ㄴ첨가 없이 연음 (삼닐 아님)
            ("삼월", "사뭘"),
            ("십육", "심뉵"),  # 단, 십육류는 예외 사전으로 ㄴ첨가 유지
        ],
    )
    def test_kiwi_rules(self, orig: str, expected: str):
        pytest.importorskip("kiwipiepy")
        from style_bert_vits2.nlp.korean.morph import apply_morph_rules

        assert pronounce(apply_morph_rules(orig)) == expected

    def test_exception_entries_preserve_length(self):
        from style_bert_vits2.nlp.korean.morph import PRONUNCIATION_EXCEPTIONS

        for orig, replaced in PRONUNCIATION_EXCEPTIONS.items():
            assert len(orig) == len(replaced), f"{orig} -> {replaced}"

    def test_morph_rules_preserve_length(self):
        from style_bert_vits2.nlp.korean.morph import apply_morph_rules

        text = "맛있는 김치찌개와 솜이불, 갈 데가 없는 한여름의 서울역!"
        assert len(apply_morph_rules(text)) == len(text)


class TestCheckpointOptimizerCompat:
    """
    シンボルテーブル拡張前の .pth (optimizer state 込み) から学習を再開できることの検証。
    モデル重みは expand_embedding_if_needed で拡張されるが、Adam の exp_avg 等も
    同様に拡張しないと optimizer.step() で形状不一致になる。
    """

    class _TinyModel(__import__("torch").nn.Module):
        def __init__(self, n_symbols: int):
            import torch

            super().__init__()
            self.emb = torch.nn.Embedding(n_symbols, 4)
            self.lin = torch.nn.Linear(4, 3)

    def _train_step(self, model):
        import torch

        idx = torch.tensor([0, 1, 2])
        return model.lin(model.emb(idx)).sum()

    def test_resume_from_smaller_embedding_with_optimizer(self, tmp_path):
        import torch

        from style_bert_vits2.models.utils.checkpoints import (
            load_checkpoint,
            save_checkpoint,
        )

        # 拡張前 (5 symbols) のモデルで 1 step 学習し optimizer state を作って保存
        old_model = self._TinyModel(5)
        old_opt = torch.optim.AdamW(old_model.parameters())
        self._train_step(old_model).backward()
        old_opt.step()
        path = tmp_path / "G_100.pth"
        save_checkpoint(old_model, old_opt, 1e-4, 1, path)
        old_exp_avg = old_opt.state_dict()["state"][0]["exp_avg"].clone()

        # 拡張後 (8 symbols) のモデル + 新しい optimizer で再開
        new_model = self._TinyModel(8)
        new_opt = torch.optim.AdamW(new_model.parameters())
        load_checkpoint(path, new_model, new_opt)

        # 旧行の momentum は保持され、新規行はゼロ (新規パラメータの初期 state)
        exp_avg = new_opt.state_dict()["state"][0]["exp_avg"]
        assert exp_avg.shape == (8, 4)
        assert torch.equal(exp_avg[:5], old_exp_avg)
        assert torch.equal(exp_avg[5:], torch.zeros(3, 4))

        # そのまま学習を継続できる
        self._train_step(new_model).backward()
        new_opt.step()


class TestBertModelDtype:
    """
    transformers 5.x は config の torch_dtype を既定で尊重するため、fp16 で保存された
    モデル (ku-nlp/deberta-v2-large-japanese-char-wwm など) が half でロードされ、
    下流の fp32 conv と dtype が衝突する。明示的に fp32 でロードすることの検証。
    """

    def test_jp_bert_loads_as_float32(self):
        from pathlib import Path

        import torch

        jp_dir = Path(__file__).parent.parent / "bert" / "deberta-v2-large-japanese-char-wwm"
        if not (jp_dir / "model.safetensors").exists():
            pytest.skip("JP BERT weights not found")
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models

        model = bert_models.load_model(Languages.JP, str(jp_dir))
        dtype = next(model.parameters()).dtype
        bert_models.unload_model(Languages.JP)
        assert dtype == torch.float32


class TestRobertaCompatibility:
    """
    KO の BERT スロットが RoBERTa 系モデル (klue/roberta-large など) を受け入れることの検証。

    このプロジェクトの「BERT」スロットは元々アーキテクチャ中立
    (JP/EN は DeBERTa、ZH は BERT を AutoModelForMaskedLM / AutoTokenizer 経由でロード)。
    ここでは KO 経路が RoBERTa クラスで実際に動作することをコードレベルで固定する。
    """

    KLUE_DIR = None  # set in setup

    @pytest.fixture()
    def klue_tokenizer_dir(self):
        from pathlib import Path

        path = Path(__file__).parent.parent / "bert" / "klue-roberta-large"
        if not (path / "vocab.txt").exists():
            pytest.skip("klue-roberta-large tokenizer files not found")
        return str(path)

    def _inject_ko_model(self, model, tokenizer):
        """bert_models のキャッシュに KO モデル/トークナイザーを直接注入する"""
        from style_bert_vits2.nlp import bert_models

        vars(bert_models)["__loaded_models"][Languages.KO] = model
        vars(bert_models)["__loaded_tokenizers"][Languages.KO] = tokenizer

    def _unload_ko(self):
        from style_bert_vits2.nlp import bert_models

        bert_models.unload_model(Languages.KO)
        bert_models.unload_tokenizer(Languages.KO)

    def test_klue_tokenizer_is_fast_wordpiece(self, klue_tokenizer_dir):
        """klue/roberta は BertTokenizerFast (WordPiece) を持ち、offset mapping が使える"""
        transformers = pytest.importorskip("transformers")

        tokenizer = transformers.AutoTokenizer.from_pretrained(klue_tokenizer_dir)
        assert tokenizer.is_fast
        inputs = tokenizer("삼일 전, 배가 고팠다.", return_offsets_mapping=True)
        assert "offset_mapping" in inputs
        # RoBERTa (type_vocab_size=1) には token_type_ids が全て 0 で渡る必要がある
        assert set(inputs.get("token_type_ids", [0])) == {0}

    def test_roberta_class_accepted_in_extract_path(self, klue_tokenizer_dir):
        """
        ランダム初期化の小型 RobertaForMaskedLM を KO スロットに注入し、
        実際の extract_bert_feature() コード経路が RoBERTa クラスで動作することを検証する
        (モデルの重みは不要)
        """
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("torch")
        from style_bert_vits2.nlp import extract_bert_feature

        tokenizer = transformers.AutoTokenizer.from_pretrained(klue_tokenizer_dir)
        config = transformers.RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=514,
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )
        model = transformers.AutoModelForMaskedLM.from_config(config)
        # AutoModelForMaskedLM が RoBERTa アーキテクチャとして解決されることを確認
        assert type(model).__name__ == "RobertaForMaskedLM"

        self._inject_ko_model(model, tokenizer)
        try:
            norm_text, phones, tones, word2ph = clean_text("3일 전, 배가 고팠다.", Languages.KO)  # fmt: skip
            feature = extract_bert_feature(norm_text, word2ph, Languages.KO, "cpu")
            assert tuple(feature.shape) == (config.hidden_size, len(phones))
            # assist_text (スタイル参照) 経路も RoBERTa で動作する
            feature2 = extract_bert_feature(
                norm_text, word2ph, Languages.KO, "cpu",
                assist_text="정말 신나!", assist_text_weight=0.7,
            )
            assert tuple(feature2.shape) == (config.hidden_size, len(phones))
        finally:
            self._unload_ko()

    def test_roberta_max_length_is_dynamic(self, klue_tokenizer_dir):
        """max_length がトークナイザーから取得され、klue の 512 が活かされる"""
        transformers = pytest.importorskip("transformers")
        import style_bert_vits2.nlp.korean.bert_feature as bf

        get_max_length = getattr(bf, "__get_max_length")
        klue_tokenizer = transformers.AutoTokenizer.from_pretrained(klue_tokenizer_dir)
        assert get_max_length(klue_tokenizer) == 512

        from pathlib import Path

        kcbert_dir = Path(__file__).parent.parent / "bert" / "kcbert-large"
        if (kcbert_dir / "vocab.txt").exists():
            kcbert_tokenizer = transformers.AutoTokenizer.from_pretrained(str(kcbert_dir))  # fmt: skip
            assert get_max_length(kcbert_tokenizer) == 300

        # model_max_length が異常値 (未設定プレースホルダー) の場合はフォールバック
        class FakeTokenizer:
            model_max_length = int(1e30)

        assert get_max_length(FakeTokenizer()) == 512

    def test_truncation_keeps_char_map_in_range(self, klue_tokenizer_dir):
        """入力上限を超えるテキストでも文字→トークン対応が範囲内に収まる"""
        transformers = pytest.importorskip("transformers")
        import style_bert_vits2.nlp.korean.bert_feature as bf

        build = getattr(bf, "__build_char_to_token_map")
        tokenizer = transformers.AutoTokenizer.from_pretrained(klue_tokenizer_dir)
        long_text = normalize_text("오늘은 정말 길고 긴 하루였다. " * 100)
        inputs = tokenizer(long_text, return_offsets_mapping=True, truncation=True, max_length=64)  # fmt: skip
        num_tokens = len(inputs["input_ids"])
        mapping = build(inputs["offset_mapping"], len(long_text))
        assert len(mapping) == len(long_text)
        assert all(0 <= t < num_tokens for t in mapping)

    def test_roberta_real_weights_end_to_end(self, klue_tokenizer_dir):
        """実際の klue/roberta-large の重みでの統合検証 (重みがある場合のみ)"""
        transformers = pytest.importorskip("transformers")
        from pathlib import Path

        weights = Path(klue_tokenizer_dir) / "pytorch_model.bin"
        if not weights.exists():
            pytest.skip("klue-roberta-large weights not found")

        from style_bert_vits2.nlp import bert_models, extract_bert_feature

        self._unload_ko()
        try:
            model = bert_models.load_model(Languages.KO, klue_tokenizer_dir)
            bert_models.load_tokenizer(Languages.KO, klue_tokenizer_dir)
            assert type(model).__name__ == "RobertaForMaskedLM"

            # KcBERT の上限 300 トークンを超える長文でも RoBERTa の 512 で処理できる
            long_text = "오늘은 정말 길고 긴 하루였다. " * 40
            norm_text, phones, tones, word2ph = clean_text(long_text, Languages.KO)
            feature = extract_bert_feature(norm_text, word2ph, Languages.KO, "cpu")
            assert tuple(feature.shape) == (1024, len(phones))
        finally:
            self._unload_ko()


class TestG2P:
    def test_g2p_invariants(self):
        norm = normalize_text("3일 전, 배가 고팠다.")
        phones, tones, word2ph = g2p(norm)
        # すべての音素がシンボルテーブルに存在する
        assert all(p in SYMBOLS for p in phones)
        # トーンはすべて 0
        assert all(t == 0 for t in tones)
        assert len(phones) == len(tones)
        # word2ph は正規化テキストの各文字 + 前後パディングに対応
        assert len(word2ph) == len(norm) + 2
        assert sum(word2ph) == len(phones)
        # 前後はパディング
        assert phones[0] == "_" and phones[-1] == "_"

    def test_clean_text_ko(self):
        norm_text, phones, tones, word2ph = clean_text("안녕하세요!", Languages.KO)
        assert norm_text == "안녕하세요!"
        assert all(p in SYMBOLS for p in phones)
        assert len(word2ph) == len(norm_text) + 2

    def test_cleaned_text_to_sequence_ko(self):
        norm_text, phones, tones, word2ph = clean_text("반갑습니다.", Languages.KO)
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, Languages.KO)  # fmt: skip
        assert len(phone_ids) == len(tone_ids) == len(lang_ids)
        assert all(l == LANGUAGE_ID_MAP["KO"] for l in lang_ids)
        assert all(t == LANGUAGE_TONE_START_MAP["KO"] for t in tone_ids)

    def test_space_maps_to_sp(self):
        phones, _, _ = g2p("가 나")
        assert "SP" in phones

    def test_empty_ish_input(self):
        phones, tones, word2ph = g2p(".")
        assert phones == ["_", ".", "_"]


def _pron_to_symbols(pron: str) -> str:
    """発音形の文字列を g2p の音素表記 (初声 ㅇ 省略・空白=SP) に変換するテスト用ヘルパ"""
    from style_bert_vits2.nlp.korean.pronounce import (
        CHOSEONG,
        JONGSEONG,
        JUNGSEONG,
        decompose,
        is_hangul_syllable,
    )

    out: list[str] = []
    for ch in pron:
        if is_hangul_syllable(ch):
            cho, jung, coda = decompose(ch)
            if cho != "ㅇ":
                out.append(chr(0x1100 + CHOSEONG.index(cho)))
            out.append(chr(0x1161 + JUNGSEONG.index(jung)))
            if coda:
                out.append(chr(0x11A7 + JONGSEONG.index(coda[0])))
        elif ch == " ":
            out.append("SP")
        else:
            out.append(ch)
    return "".join(out)


class TestG2PPronunciation:
    """
    g2p() パイプライン全体 (morph → 発音エンジン → 字母分解) の発音回帰テスト。

    形態素解析の縮約トークン (하/VV + ᆫ다/EC のような重なりスパン) の扱いに起因する
    回帰 (縮約 ㄴ다 の誤経音化: 간다→[간따] バグ) を検出する。
    """

    @pytest.mark.parametrize(
        "text, expected_pron",
        [
            # 母音語幹 + 縮約 ㄴ다 (誤経音化バグの回帰: アダプタの複合タグ処理)
            ("간다", "간다"),
            ("한다", "한다"),
            ("온다", "온다"),
            ("웃긴다", "욷낀다"),
            # 本物の語幹末 ㄴ/ㅁ 経音化は維持される
            ("신다", "신따"),
            ("안고", "안꼬"),
            ("감다", "감따"),
            ("신고 간다", "신꼬 간다"),
            # 縮約 ETM ㄹ の後の経音化
            ("할 것이다", "할 꺼시다"),
            ("어쩔 수 없지", "어쩔 쑤 업찌"),
            # 代表的な音韻規則
            ("같이", "가치"),
            ("신라면", "실라면"),
            ("됐다", "됃따"),
            ("먹는다", "멍는다"),
            ("맛있다", "마싣따"),
            ("솜이불", "솜니불"),
        ],
    )
    def test_pipeline_pronunciation(self, text: str, expected_pron: str):
        norm = normalize_text(text)
        phones, _, word2ph = g2p(norm)
        assert sum(word2ph) == len(phones)
        assert "".join(phones[1:-1]) == _pron_to_symbols(expected_pron)


class TestCER:
    def test_identity(self):
        from style_bert_vits2.nlp.korean.cer import korean_cer

        assert korean_cer("안녕하세요", "안녕하세요") == 0.0

    def test_orthography_vs_pronunciation_equivalence(self):
        """발음이 같으면 표기가 달라도 CER 0 (ASR 표기 흔들림 무시)"""
        from style_bert_vits2.nlp.korean.cer import korean_cer

        assert korean_cer("맛있다", "마싣따") == 0.0
        assert korean_cer("같이", "가치") == 0.0
        assert korean_cer("신라", "실라") == 0.0

    def test_number_normalization_equivalence(self):
        """숫자 표기와 한글 표기가 같은 읽기면 CER 0"""
        from style_bert_vits2.nlp.korean.cer import korean_cer

        assert korean_cer("사과 3개", "사과 세 개") == 0.0

    def test_spacing_and_punctuation_ignored(self):
        from style_bert_vits2.nlp.korean.cer import korean_cer

        assert korean_cer("안녕하세요!", "안녕 하세요") == 0.0

    def test_error_rates(self):
        from style_bert_vits2.nlp.korean.cer import korean_cer

        # 완전 불일치는 1.0 근처, 부분 오류는 0과 1 사이
        assert korean_cer("가나다", "가나라") > 0.0
        assert korean_cer("가나다", "가나다라") > 0.0
        # 자모 단위: 초성 하나 차이 → [ㄱㅏ] vs [ㄴㅏ] → 0.5
        assert korean_cer("가", "나", unit="jamo") == 0.5
        assert korean_cer("가", "나", unit="syllable") == 1.0
        # 자모 단위가 음절 단위보다 완만한 점수를 준다 (받침 하나 차이)
        jamo = korean_cer("강", "간", unit="jamo")
        syllable = korean_cer("강", "간", unit="syllable")
        assert jamo < syllable == 1.0

    def test_empty_reference(self):
        from style_bert_vits2.nlp.korean.cer import korean_cer

        assert korean_cer("", "") == 0.0
        assert korean_cer("...", "가나다") == 1.0

    def test_levenshtein(self):
        from style_bert_vits2.nlp.korean.cer import levenshtein

        assert levenshtein("abc", "abc") == 0
        assert levenshtein("abc", "abd") == 1
        assert levenshtein("abc", "ab") == 1
        assert levenshtein("", "abc") == 3
        assert levenshtein("kitten", "sitting") == 3


class TestCorpusAnalyzer:
    def test_analyze(self, tmp_path):
        import analyze_corpus

        esd = tmp_path / "esd.list"
        esd.write_text(
            "a.wav|spk|KO|3일 전, 배가 고팠다.\n"
            "b.wav|spk|KO|안녕하세요! 값이 얼마죠?\n"
            "c.wav|spk|JP|こんにちは\n"
            "broken line without pipes\n",
            encoding="utf-8",
        )
        report = analyze_corpus.analyze(esd, check_audio=False)
        assert report["total_lines"] == 4
        assert report["languages"]["KO"] == 2
        assert len(report["error_lines"]) == 1
        # 등장한 음소가 카운트되고, 미등장 음소가 감지된다
        counts = report["phoneme_coverage"]["counts"]
        assert sum(counts.values()) > 0
        assert len(report["phoneme_coverage"]["missing"]) > 0
        # 문장 기호 카운트
        assert report["punctuation"]["counts"]["!"] == 1
        assert report["punctuation"]["counts"]["?"] == 1
        # 치찰음 문맥
        assert "ㅅ" in report["sibilant_contexts"] or "ㅆ" in report["sibilant_contexts"]


class TestBertFeatureAlignment:
    def test_char_to_token_map(self):
        import style_bert_vits2.nlp.korean.bert_feature as bf

        build = getattr(bf, "__build_char_to_token_map")
        # トークン 1 が文字 0-1、トークン 2 が文字 3-4 をカバーし、文字 2 (スペース) は未カバー
        offsets = [(0, 0), (0, 2), (3, 5), (0, 0)]
        mapping = build(offsets, 5)
        assert mapping == [1, 1, 1, 2, 2]  # スペースは直前のトークンに割り当て

    def test_char_to_token_map_leading_gap(self):
        import style_bert_vits2.nlp.korean.bert_feature as bf

        build = getattr(bf, "__build_char_to_token_map")
        offsets = [(0, 0), (1, 3), (0, 0)]
        mapping = build(offsets, 3)
        # 先頭の未カバー文字は直後のトークンで埋められる
        assert mapping == [1, 1, 1]

    def test_kcbert_tokenizer_alignment(self):
        """実際の KcBERT トークナイザーで文字→トークン対応が構築できる"""
        transformers = pytest.importorskip("transformers")
        from pathlib import Path

        tokenizer_path = Path(__file__).parent.parent / "bert" / "kcbert-large"
        if not (tokenizer_path / "vocab.txt").exists():
            pytest.skip("kcbert-large tokenizer files not found")
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_path))

        import style_bert_vits2.nlp.korean.bert_feature as bf

        build = getattr(bf, "__build_char_to_token_map")

        text = normalize_text("삼일 전, 배가 고팠다.")
        inputs = tokenizer(text, return_offsets_mapping=True)
        num_tokens = len(inputs["input_ids"])
        mapping = build(inputs["offset_mapping"], len(text))
        assert len(mapping) == len(text)
        assert all(0 <= t < num_tokens for t in mapping)


class TestWordBoundaryRules:
    """어절 경계 음운 규칙의 적용/미적용 경계 조건 (내장 엔진 경로)"""

    def _pron(self, text: str) -> str:
        from style_bert_vits2.nlp.korean.morph import apply_morph_rules
        from style_bert_vits2.nlp.korean.pronounce import pronounce

        return pronounce(apply_morph_rules(text))

    def test_liaison_moves_coda_across_space(self):
        assert self._pron("오늘 아침") == "오느 라침"

    def test_ieung_coda_does_not_liaise(self):
        # ㅇ 받침 (/ŋ/) 은 연음하지 않음
        assert self._pron("사랑 안에서") == "사랑 아네서"

    def test_no_rules_across_punctuation(self):
        # 구두점 = 휴지: 경계 규칙 미적용
        assert self._pron("옷, 입다") == "옫, 입따"
        assert self._pron("밥. 먹는다") == "밥. 멍는다"

    def test_no_rules_across_double_space(self):
        # 공백 2개 이상은 휴지로 간주
        assert self._pron("밥  먹는다") == "밥  멍는다"

    def test_tensification_after_obstruent(self):
        assert self._pron("결국 승리") == "결국 씅니"

    def test_no_tensification_after_sonorant_coda(self):
        # 유성음 받침 뒤 평음은 경계에서 경음화하지 않음 (관형형 ㄹ은 morph 담당)
        assert self._pron("사람 사이") == "사람 사이"

    def test_word2ph_contract_preserved(self):
        # 연음이 일어나도 g2p 계약 (word2ph 길이/합) 은 유지된다
        from style_bert_vits2.nlp.korean.g2p import g2p

        for text in ["오늘 아침", "몇 월", "밥 먹는다", "할 일"]:
            phones, tones, word2ph = g2p(text)
            assert len(word2ph) == len(text) + 2
            assert sum(word2ph) == len(phones)

    def test_n_insertion_not_applied_to_multisyllable_word(self):
        # 다음절 실질형태소에는 첨가하지 않고 연음만 (과잉 적용 방지)
        assert self._pron("저는 야구") == "저느 냐구"  # [저는냐구]가 아님

    def test_n_insertion_not_applied_to_numeral(self):
        # 한자어 수사 (NR) 는 첨가 제외 유지 (삼일→[사밀] 과 일관)
        assert self._pron("삼 일") == "사 밀"

    def test_n_insertion_requires_left_coda(self):
        # 앞 어절이 모음으로 끝나면 첨가 없음
        assert self._pron("그 일") == "그 일"


# ============================================================
# warm_start (KO 임베딩 초기화 매핑)
# ============================================================


class TestWarmStartMap:
    def test_전_KO_심볼_커버(self):
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP
        from style_bert_vits2.nlp.symbols import KO_SYMBOLS

        assert set(KO_JP_INIT_MAP.keys()) == set(KO_SYMBOLS)
        assert len(KO_JP_INIT_MAP) == 46

    def test_가중치_합은_1(self):
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP

        for ko, sources in KO_JP_INIT_MAP.items():
            assert abs(sum(w for _, w in sources) - 1.0) < 1e-6, ko

    def test_소스는_전부_베이스_구간_JP_심볼(self):
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP, NUM_BASE_SYMBOLS, SYMBOL_TO_IDX

        for ko, sources in KO_JP_INIT_MAP.items():
            for jp, _ in sources:
                assert jp in SYMBOL_TO_IDX, f"{ko}: {jp}"
                assert SYMBOL_TO_IDX[jp] < NUM_BASE_SYMBOLS, f"{ko}: {jp}"

    def test_symbol_to_idx_캐시(self):
        from style_bert_vits2.nlp.korean.warm_start import NUM_BASE_SYMBOLS, SYMBOL_TO_IDX
        from style_bert_vits2.nlp.symbols import SYMBOLS

        assert SYMBOL_TO_IDX == {s: i for i, s in enumerate(SYMBOLS)}
        assert NUM_BASE_SYMBOLS == 112

    def test_확정_매핑_스팟체크(self):
        # 스펙 확정 테이블의 대표값 회귀망 (2026-07-22 확정)
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP

        assert KO_JP_INIT_MAP["ᄅ"] == [("r", 1.0)]
        assert KO_JP_INIT_MAP["ᅥ"] == [("o", 1.0)]
        assert KO_JP_INIT_MAP["ᅳ"] == [("u", 1.0)]
        assert KO_JP_INIT_MAP["ᅣ"] == [("y", 0.3), ("a", 0.7)]
        assert KO_JP_INIT_MAP["ᄉ"] == [("s", 0.8), ("sh", 0.2)]
        assert KO_JP_INIT_MAP["ᆼ"] == [("N", 1.0)]
        assert KO_JP_INIT_MAP["ᆨ"] == [("q", 1.0)]


class TestBuildEmbedding:
    def _base(self):
        import torch

        torch.manual_seed(0)
        return torch.randn(5, 4, dtype=torch.float32)

    def test_기존_행_보존(self):
        import torch
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        out = build_embedding(base, 7, {5: [(0, 1.0)], 6: [(1, 0.5), (2, 0.5)]})
        assert out.shape == (7, 4)
        assert torch.equal(out[:5], base)

    def test_단일_매핑은_소스와_정확히_일치(self):
        # identity 테스트: w=1.0이면 비트 단위로 동일해야 함 (인덱스 어긋남 회귀망)
        import torch
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        out = build_embedding(base, 6, {5: [(2, 1.0)]})
        assert torch.equal(out[5], base[2])

    def test_가중_결합_수치(self):
        import torch
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        out = build_embedding(base, 6, {5: [(1, 0.3), (3, 0.7)]})
        assert torch.allclose(out[5], 0.3 * base[1] + 0.7 * base[3])

    def test_신규_행_범위_밖_인덱스는_에러(self):
        import pytest
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        with pytest.raises(ValueError):
            build_embedding(base, 6, {4: [(0, 1.0)]})  # 4는 기존 행
        with pytest.raises(ValueError):
            build_embedding(base, 6, {6: [(0, 1.0)]})  # 6은 target_rows 밖

    def test_채워지지_않은_신규_행은_에러(self):
        import pytest
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        with pytest.raises(ValueError):
            build_embedding(base, 7, {5: [(0, 1.0)]})  # 행 6 누락

    def test_소스_행_범위_검증(self):
        # IndexError로 죽지 않고 명시적 ValueError를 내야 함
        import pytest
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        with pytest.raises(ValueError):
            build_embedding(base, 6, {5: [(5, 1.0)]})  # 소스 5는 기존 행(0~4) 아님
        with pytest.raises(ValueError):
            build_embedding(base, 6, {5: [(-1, 1.0)]})

    def test_target_rows가_기존보다_작으면_에러(self):
        import pytest
        from style_bert_vits2.nlp.korean.warm_start import build_embedding

        base = self._base()
        with pytest.raises(ValueError):
            build_embedding(base, 3, {})

    def test_KO_PHONEME_INIT_MAP_상수는_실제_심볼_인덱스(self):
        from style_bert_vits2.nlp.korean.warm_start import KO_PHONEME_INIT_MAP, NUM_BASE_SYMBOLS, SYMBOL_TO_IDX

        assert set(KO_PHONEME_INIT_MAP.keys()) == set(range(NUM_BASE_SYMBOLS, len(SYMBOL_TO_IDX)))
        assert KO_PHONEME_INIT_MAP[SYMBOL_TO_IDX["ᄅ"]] == [(SYMBOL_TO_IDX["r"], 1.0)]


class TestWarmStartConvert:
    def _fake_g0(self, tmp_path, num_symbols=112, num_tones=12, num_langs=3):
        # 실제 G_0의 임베딩 3키 + 무관 텐서 1개를 가진 최소 safetensors를 만든다
        import torch
        from safetensors.torch import save_file

        tensors = {
            "enc_p.emb.weight": torch.randn(num_symbols, 8),
            "enc_p.tone_emb.weight": torch.randn(num_tones, 8),
            "enc_p.language_emb.weight": torch.randn(num_langs, 8),
            "dec.conv_pre.weight": torch.randn(4, 4),
        }
        path = tmp_path / "G_0.safetensors"
        save_file(tensors, str(path), metadata={"iteration": "0"})
        return path, tensors

    def test_변환_결과_행수와_초기화(self):
        import torch
        from safetensors import safe_open
        from style_bert_vits2.nlp.korean.warm_start import SYMBOL_TO_IDX
        from style_bert_vits2.nlp.symbols import LANGUAGE_TONE_START_MAP, NUM_TONES, SYMBOLS  # fmt: skip
        from warm_start_ko import convert

        in_path, tensors = self._fake_g0(self.tmp_path)
        out_path = self.tmp_path / "G_0_ko.safetensors"
        convert(in_path, out_path)

        with safe_open(str(out_path), framework="pt") as f:
            emb = f.get_tensor("enc_p.emb.weight")
            tone = f.get_tensor("enc_p.tone_emb.weight")
            lang = f.get_tensor("enc_p.language_emb.weight")
            dec = f.get_tensor("dec.conv_pre.weight")
            assert f.metadata() == {"iteration": "0"}

        assert emb.shape[0] == len(SYMBOLS)  # 158
        assert tone.shape[0] == NUM_TONES  # 13
        assert lang.shape[0] == 4
        # identity: ᄅ 행 == r 행, 언어 KO 행 == JP 행
        assert torch.equal(emb[SYMBOL_TO_IDX["ᄅ"]], tensors["enc_p.emb.weight"][SYMBOL_TO_IDX["r"]])  # fmt: skip
        assert torch.equal(lang[3], tensors["enc_p.language_emb.weight"][1])
        # 가중: 톤 KO 행 == JP 두 행 평균
        jp_start = LANGUAGE_TONE_START_MAP["JP"]
        expected = 0.5 * tensors["enc_p.tone_emb.weight"][jp_start] + 0.5 * tensors["enc_p.tone_emb.weight"][jp_start + 1]  # fmt: skip
        assert torch.allclose(tone[LANGUAGE_TONE_START_MAP["KO"]], expected)
        # 무관 텐서는 그대로
        assert torch.equal(dec, tensors["dec.conv_pre.weight"])

    def test_이미_확장된_파일은_에러(self):
        import pytest
        from warm_start_ko import convert

        in_path, _ = self._fake_g0(self.tmp_path, num_symbols=158, num_tones=13, num_langs=4)  # fmt: skip
        with pytest.raises(ValueError, match="이미"):
            convert(in_path, self.tmp_path / "out.safetensors")

    def test_생성자_체크포인트가_아니면_에러(self):
        import pytest
        import torch
        from safetensors.torch import save_file
        from warm_start_ko import convert

        path = self.tmp_path / "D_0.safetensors"
        save_file({"disc.conv.weight": torch.randn(4, 4)}, str(path))
        with pytest.raises(ValueError, match="G_0"):
            convert(path, self.tmp_path / "out.safetensors")

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp_path = tmp_path

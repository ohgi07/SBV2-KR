"""
韓国語 (KO) サポートのテスト。

実行方法: pytest tests/test_korean.py
BERT モデルの重みは不要 (トークナイザーのみ利用する)。
"""

from pathlib import Path

import pytest

import analyze_corpus
from speech_cer import (
    drop_hallucinated_segments,
    korean_cer,
    levenshtein,
    text_to_pronounced_units,
)
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text, cleaned_text_to_sequence
from style_bert_vits2.nlp.korean.g2p import g2p, to_pronunciation
from style_bert_vits2.nlp.korean.morph import (
    apply_morph_rules,
    clause_boundary_spaces,
    tokenize,
)
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


def _bert_feature_private(name: str):
    """bert_feature.py의 모듈 전용 (__ 접두) 함수를 꺼낸다 (transformers 임포트를 테스트 안으로 미룬다)"""
    import style_bert_vits2.nlp.korean.bert_feature as bf

    return getattr(bf, name)


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
        # 일을 생략하는 단위는 만뿐 (억 이상에서 생략하면 [억]처럼 단위만 남아 뜻이 흐려진다)
        assert read_number("100000000") == "일억"
        assert read_number("1000000000000") == "일조"
        assert read_number("100010000") == "일억만"
        assert read_number("10001") == "만일"
        assert read_number("20000") == "이만"

    def test_numbers_beyond_the_group_units(self):
        """경을 넘는 자릿수는 낱자로 읽는다 (예전에는 IndexError로 전처리·추론이 통째로 죽었다)"""
        assert read_number("1" + "0" * 19) == "천경"  # 20자리까지는 그룹 단위로
        assert read_number("1" + "0" * 20) == "일" + "공" * 20  # 21자리부터 낱자
        assert read_number("12345678901234567890123") == "일이삼사오육칠팔구공일이삼사오육칠팔구공일이삼"
        # 계좌번호처럼 긴 숫자가 섞인 문장도 끝까지 정규화된다
        assert normalize_text("계좌 12345678901234567890123 입니다").startswith("계좌 일이삼사오육")

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
            ("물동이", "물똥이"),  # 분석이 원문 기준 (물똥 + 이로 갈리면 [물똥니])
            ("물동이를 머리에 이고", "물똥이를 머리에 이고"),
            ("송별연에 갔다", "송벼려네 갇따"),  # 재작성된 초성 ㄹ은 ㄴ첨가를 막는다
        ],
    )
    def test_kiwi_rules(self, orig: str, expected: str):
        pytest.importorskip("kiwipiepy")
        assert pronounce(apply_morph_rules(orig)) == expected

    def test_morph_rules_preserve_length(self):
        # 예외 사전 항목의 문자 수 보존은 morph.py 임포트 시점에 검증된다
        text = "맛있는 김치찌개와 솜이불, 갈 데가 없는 한여름의 서울역!"
        assert len(apply_morph_rules(text)) == len(text)

    def test_shared_tokens_match_internal_analysis(self):
        """분석 결과를 넘겨받아도 각자 분석할 때와 같은 결과여야 한다 (발화당 1회로 줄이는 경로)"""
        pytest.importorskip("kiwipiepy")
        text = "이곳에 들어오시면 안 됩니다. 갈 데가 없는 한여름의 서울역!"
        tokens = tokenize(text)
        assert apply_morph_rules(text, tokens) == apply_morph_rules(text)
        assert clause_boundary_spaces(text, tokens) == clause_boundary_spaces(text)


class TestHfBackupPruning:
    """
    HF 백업의 stale LFS 정리 검증. 트리 조회가 LFS를 하나도 못 돌려주면 전량이 stale로
    보여 방금 올린 백업까지 영구 삭제되므로, 그 경우에는 정리를 건너뛰어야 한다.
    """

    class _Lfs:
        def __init__(self, sha: str):
            self.sha256 = sha

    class _TreeFile:
        def __init__(self, sha):
            self.lfs = TestHfBackupPruning._Lfs(sha) if sha else None

    class _LfsFile:
        def __init__(self, oid: str):
            self.file_oid = oid

    class _FakeApi:
        def __init__(self, tree_shas, lfs_oids, upload_error=None):
            self.tree_shas, self.lfs_oids = tree_shas, lfs_oids
            self.upload_error = upload_error
            self.squashed = False
            self.deleted = None

        def upload_folder(self, **kwargs):
            from concurrent.futures import Future

            future = Future()
            if self.upload_error is not None:
                future.set_exception(self.upload_error)
            else:
                future.set_result(type("R", (), {"commit_url": "https://hf.co/x/commit/abc"}))
            return future

        def super_squash_history(self, repo_id):
            self.squashed = True

        def list_repo_tree(self, repo_id, recursive=False):
            return [TestHfBackupPruning._TreeFile(s) for s in self.tree_shas]

        def list_lfs_files(self, repo_id):
            return [TestHfBackupPruning._LfsFile(o) for o in self.lfs_oids]

        def permanently_delete_lfs_files(self, repo_id, files):
            self.deleted = [f.file_oid for f in files]

    def _run(self, monkeypatch, fake):
        import torch

        # 학습 스크립트는 임포트할 때 torch 스레드 수를 1로 바꾸므로 원래대로 되돌린다
        # (그대로 두면 뒤의 스트리밍 테스트에서 첫 디코더 호출이 0.5초에서 25초 이상으로 느려진다)
        num_threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
        import train_ms_jp_extra

        torch.set_num_threads(num_threads)
        torch.set_float32_matmul_precision(precision)
        monkeypatch.setattr(train_ms_jp_extra, "api", fake)
        train_ms_jp_extra.backup_to_hf("user/repo")
        return fake

    def test_deletes_only_blobs_missing_from_the_tree(self, monkeypatch):
        fake = self._run(monkeypatch, self._FakeApi(tree_shas=["a"], lfs_oids=["a", "old"]))
        assert fake.squashed is True
        assert fake.deleted == ["old"]

    def test_keeps_every_blob_when_the_tree_lists_no_lfs_file(self, monkeypatch):
        fake = self._run(monkeypatch, self._FakeApi(tree_shas=[], lfs_oids=["a", "b"]))
        assert fake.deleted is None  # 전량 삭제 대신 아무것도 지우지 않는다

    def test_skips_pruning_when_an_upload_failed(self, monkeypatch):
        fake = self._FakeApi(tree_shas=["a"], lfs_oids=["a", "old"], upload_error=OSError("boom"))
        self._run(monkeypatch, fake)
        assert fake.squashed is False
        assert fake.deleted is None


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
        import torch

        jp_dir = Path(__file__).parent.parent / "bert" / "deberta-v2-large-japanese-char-wwm"
        if not (jp_dir / "model.safetensors").exists():
            pytest.skip("JP BERT weights not found")
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

    @pytest.fixture()
    def klue_tokenizer_dir(self):
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
        get_max_length = _bert_feature_private("__get_max_length")
        klue_tokenizer = transformers.AutoTokenizer.from_pretrained(klue_tokenizer_dir)
        assert get_max_length(klue_tokenizer) == 512

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
        build = _bert_feature_private("__build_char_to_token_map")
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
    @pytest.mark.parametrize(
        "reference, hypothesis",
        [
            ("안녕하세요", "안녕하세요"),
            ("", ""),
            # 발음이 같으면 표기가 달라도 CER 0 (ASR 표기 흔들림 무시)
            ("맛있다", "마싣따"),
            ("같이", "가치"),
            ("신라", "실라"),
            # 숫자 표기와 한글 표기가 같은 읽기면 CER 0
            ("사과 3개", "사과 세 개"),
            # 띄어쓰기·구두점은 무시
            ("안녕하세요!", "안녕 하세요"),
        ],
    )
    def test_equivalent_texts_score_zero(self, reference: str, hypothesis: str):
        assert korean_cer(reference, hypothesis) == 0.0

    def test_error_rates(self):
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
        # 참조가 비어 있고 가설이 있으면 1.0
        assert korean_cer("...", "가나다") == 1.0

    def test_uses_g2p_pronunciation_path(self):
        """합성에 쓰인 것과 같은 발음으로 재야 한다 (절 경계에서 어절 경계 규칙이 멈춘 발음)"""
        pytest.importorskip("kiwipiepy")
        units = text_to_pronounced_units("이곳에 들어오시면 안 됩니다", unit="syllable")
        assert "".join(units) == "이고세드러오시면안됨니다"  # 절 경계를 무시하면 드러오시며난

    def test_levenshtein(self):
        assert levenshtein("abc", "abc") == 0
        assert levenshtein("abc", "abd") == 1
        assert levenshtein("abc", "ab") == 1
        assert levenshtein("", "abc") == 3
        assert levenshtein("kitten", "sitting") == 3


class TestHallucinationFilter:
    """Whisper 환각 세그먼트 제거 (실측 속도: 정상 4.24~8.33자/초, 환각 0.37~1.00자/초)"""

    def _seg(self, start: float, end: float, text: str):
        from types import SimpleNamespace

        return SimpleNamespace(start=start, end=end, text=text)

    def test_drops_hallucination(self):
        # 뒤의 둘은 문구가 서로 다르다 — 블랙리스트가 아니라 속도로 걸러야 둘 다 잡힌다
        segs = [
            self._seg(0.30, 3.98, "오래 쪼그리고 앉아 있었더니 다리에 쥐가 나요."),
            self._seg(3.98, 33.96, "자막 제공 및 자막 제공 및 광고를 포함하고 있습니다."),
            self._seg(0.00, 29.98, "한글자막 by 한효정"),
        ]
        assert [s.text for s in drop_hallucinated_segments(segs)] == [segs[0].text]

    def test_keeps_real_speech(self):
        segs = [
            self._seg(0.00, 2.36, "삶은 달걀 있어요?"),
            self._seg(2.36, 10.90, "어제 저녁에 친구를 만나서 이런저런 이야기를 나누다 보니 자정이 넘었습니다."),
            self._seg(10.90, 12.40, "네."),  # 짧으면 속도가 낮아도 길이 조건에 걸리지 않는다
        ]
        assert len(drop_hallucinated_segments(segs)) == 3
        assert drop_hallucinated_segments([]) == []


class TestCorpusAnalyzer:
    def test_analyze(self, tmp_path):
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

    def _write_wav(self, path, seconds: float, rate: int = 8000):
        import wave

        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(b"\x00\x00" * int(rate * seconds))

    def test_measures_audio_relative_to_the_dataset_dir(self, tmp_path):
        """esd.list의 경로는 데이터셋 기준이다 (CWD 기준으로 열면 아무것도 측정되지 않는다)"""
        esd = tmp_path / "esd.list"
        esd.write_text(
            "a.wav|spk|KO|안녕하세요\n"
            "sub/b.wav|spk|KO|반갑습니다\n"
            "missing.wav|spk|KO|없는 파일\n",
            encoding="utf-8",
        )
        self._write_wav(tmp_path / "wavs" / "a.wav", 2.0)  # 리샘플 후
        self._write_wav(tmp_path / "raw" / "sub" / "b.wav", 4.0)  # 리샘플 전 (raw 폴백)

        ad = analyze_corpus.analyze(esd, check_audio=True)["audio_duration_sec"]
        assert ad["files_measured"] == 2
        assert ad["mean"] == 3.0
        assert ad["files_unmeasured"] == 1  # 읽지 못한 파일은 조용히 사라지지 않는다

    def test_no_audio_section_when_measurement_is_skipped(self, tmp_path):
        esd = tmp_path / "esd.list"
        esd.write_text("a.wav|spk|KO|안녕하세요\n", encoding="utf-8")
        assert analyze_corpus.analyze(esd, check_audio=False)["audio_duration_sec"] == {}


class TestBertFeatureAlignment:
    def test_char_to_token_map(self):
        build = _bert_feature_private("__build_char_to_token_map")
        # トークン 1 が文字 0-1、トークン 2 が文字 3-4 をカバーし、文字 2 (スペース) は未カバー
        offsets = [(0, 0), (0, 2), (3, 5), (0, 0)]
        mapping = build(offsets, 5)
        assert mapping == [1, 1, 1, 2, 2]  # スペースは直前のトークンに割り当て

    def test_char_to_token_map_leading_gap(self):
        build = _bert_feature_private("__build_char_to_token_map")
        offsets = [(0, 0), (1, 3), (0, 0)]
        mapping = build(offsets, 3)
        # 先頭の未カバー文字は直後のトークンで埋められる
        assert mapping == [1, 1, 1]

    def test_kcbert_tokenizer_alignment(self):
        """実際の KcBERT トークナイザーで文字→トークン対応が構築できる"""
        transformers = pytest.importorskip("transformers")
        tokenizer_path = Path(__file__).parent.parent / "bert" / "kcbert-large"
        if not (tokenizer_path / "vocab.txt").exists():
            pytest.skip("kcbert-large tokenizer files not found")
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_path))

        build = _bert_feature_private("__build_char_to_token_map")
        text = normalize_text("삼일 전, 배가 고팠다.")
        inputs = tokenizer(text, return_offsets_mapping=True)
        num_tokens = len(inputs["input_ids"])
        mapping = build(inputs["offset_mapping"], len(text))
        assert len(mapping) == len(text)
        assert all(0 <= t < num_tokens for t in mapping)


class TestWordBoundaryRules:
    """어절 경계 음운 규칙의 적용/미적용 경계 조건 (내장 엔진 경로)"""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("오늘 아침", "오느 라침"),  # 받침이 공백을 넘어 연음된다
            ("사랑 안에서", "사랑 아네서"),  # ㅇ 받침 (/ŋ/) 은 연음하지 않음
            # 구두점 = 휴지: 경계 규칙 미적용
            ("옷, 입다", "옫, 입따"),
            ("밥. 먹는다", "밥. 멍는다"),
            ("밥  먹는다", "밥  멍는다"),  # 공백 2개 이상도 휴지로 간주
            ("결국 승리", "결국 씅니"),  # 장애음 받침 뒤 경음화
            # 유성음 받침 뒤 평음은 경계에서 경음화하지 않음 (관형형 ㄹ은 morph 담당)
            ("사람 사이", "사람 사이"),
            # 다음절 실질형태소에는 ㄴ 첨가 없이 연음만 ([저는냐구]가 아님, 과잉 적용 방지)
            ("저는 야구", "저느 냐구"),
            ("삼 일", "사 밀"),  # 한자어 수사 (NR) 는 첨가 제외 (삼일→[사밀] 과 일관)
            ("그 일", "그 일"),  # 앞 어절이 모음으로 끝나면 첨가 없음
        ],
    )
    def test_boundary_rules(self, text: str, expected: str):
        assert pronounce(apply_morph_rules(text)) == expected

    @pytest.mark.parametrize(
        "text",
        ["오늘 아침", "몇 월", "밥 먹는다", "할 일", "도착하면 알려 줘", "발생하지 않도록 조심해라"],
    )
    def test_word2ph_contract_preserved(self, text: str):
        # 연음이 일어나거나 절 경계에서 멈춰도 g2p 계약 (word2ph 길이/합) 은 유지된다
        phones, _, word2ph = g2p(text)
        assert len(word2ph) == len(text) + 2
        assert sum(word2ph) == len(phones)


class TestClauseBoundary:
    """
    절 경계 (연결어미·종결어미 뒤) 에서 어절 경계 규칙이 멈추는지 검증.
    적용되면 "안 됩니다"가 [난 됨니다], "알려 줘"가 [날려 줘]로 나가 뜻이 바뀐다.
    형태소 정보가 필요하므로 pronounce 단독이 아닌 g2p 경로로 확인한다.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            # 연결어미 뒤에서는 연음하지 않는다
            ("이곳에 들어오시면 안 됩니다", "이고세 드러오시면 안 됨니다"),
            ("도착하면 알려 줘", "도차카면 알려 줘"),
            ("길을 건너면 은행이 있어요", "기를 건너면 은행이 이써요"),
            # 경음화도 절 경계를 넘지 않는다 (조심 → [쪼심] 방지)
            ("발생하지 않도록 조심해라", "발생하지 안토록 조심해라"),
            # ㄴ 첨가도 절 경계를 넘지 않는다 (여덟 → [녀덜] 방지)
            ("깎아 주시면 여덟 개", "까까 주시면 여덜 개"),
            # 관형사형 어미는 뒤 명사와 한 마디이므로 제29항 붙임2가 그대로 적용된다
            ("먹은 엿", "머근 녇"),
            ("할 일", "할 릴"),
            ("먹을 엿", "머글 렫"),
            # 체언·조사로 끝나는 어절 뒤는 절 경계가 아니므로 기존 동작 유지
            ("오늘 아침", "오느 라침"),
            ("삼 일", "사 밀"),
            ("옷 입다", "온 닙따"),
        ],
    )
    def test_pronunciation(self, text: str, expected: str):
        assert to_pronunciation(text) == expected


# ============================================================
# warm-start (KO 임베딩 초기화 매핑). torch가 필요해 각 테스트 안에서 임포트한다
# ============================================================


class TestWarmStartMap:
    def test_covers_all_46_ko_symbols(self):
        # 가중치 합 = 1, 소스 = 베이스 구간 JP 심볼 조건은 warm_start 임포트 시점에 검증된다
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP

        assert set(KO_JP_INIT_MAP.keys()) == set(KO_SYMBOLS)
        assert len(KO_JP_INIT_MAP) == 46

    def test_spot_check_of_final_weights(self):
        # 스펙 확정 테이블의 대표값 회귀망
        from style_bert_vits2.nlp.korean.warm_start import KO_JP_INIT_MAP

        expected = {
            "ᄅ": [("r", 1.0)],
            "ᅥ": [("o", 1.0)],
            "ᅳ": [("u", 1.0)],
            "ᄀ": [("g", 1.0)],
            "ᄁ": [("k", 0.95), ("g", 0.05)],
            "ᄉ": [("s", 0.75), ("sh", 0.25)],
            "ᄊ": [("s", 0.85), ("sh", 0.15)],
            "ᅣ": [("y", 0.35), ("a", 0.65)],
            "ᅩ": [("o", 0.85), ("u", 0.15)],
            "ᅯ": [("w", 0.2), ("o", 0.8)],
            "ᆫ": [("N", 0.85), ("n", 0.15)],
            "ᆷ": [("N", 0.7), ("m", 0.3)],
            "ᆼ": [("N", 1.0)],
            "ᆨ": [("q", 0.9), ("k", 0.1)],
        }
        assert {ko: KO_JP_INIT_MAP[ko] for ko in expected} == expected

    def test_phoneme_map_covers_every_ko_row(self):
        from style_bert_vits2.nlp.korean.warm_start import KO_WARM_START_MAPS

        assert set(KO_WARM_START_MAPS["enc_p.emb.weight"].keys()) == set(range(112, len(SYMBOLS)))


class TestWarmStartExpansion:
    """KO 추가 전 체크포인트를 로드할 때 KO 임베딩 행이 JP 행의 가중 결합으로 채워지는지 검증"""

    IDX = {s: i for i, s in enumerate(SYMBOLS)}

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        from style_bert_vits2.models.utils.checkpoints import expand_embedding_if_needed

        torch.manual_seed(0)
        self.torch, self.expand = torch, expand_embedding_if_needed

    def _expand(self, key: str, num_saved: int, num_rows: int):
        saved, model = self.torch.randn(num_saved, 8), self.torch.randn(num_rows, 8)
        return saved, model, self.expand(key, saved, model)

    def test_ko_phoneme_rows_are_weighted_jp_rows(self):
        saved, _, out = self._expand("enc_p.emb.weight", 112, len(SYMBOLS))
        assert self.torch.equal(out[:112], saved)
        # 단일 매핑(w=1.0)은 소스 행과 비트 단위로 동일
        assert self.torch.equal(out[self.IDX["ᄅ"]], saved[self.IDX["r"]])
        expected = 0.75 * saved[self.IDX["s"]] + 0.25 * saved[self.IDX["sh"]]
        assert self.torch.allclose(out[self.IDX["ᄉ"]], expected)

    def test_ko_tone_row_is_mean_of_jp_low_and_high(self):
        saved, _, out = self._expand("enc_p.tone_emb.weight", NUM_TONES - 1, NUM_TONES)
        jp = LANGUAGE_TONE_START_MAP["JP"]
        assert self.torch.allclose(out[LANGUAGE_TONE_START_MAP["KO"]], 0.5 * saved[jp] + 0.5 * saved[jp + 1])

    def test_ko_language_row_copies_jp_row(self):
        saved, _, out = self._expand("enc_p.language_emb.weight", 3, 4)
        assert self.torch.equal(out[LANGUAGE_ID_MAP["KO"]], saved[LANGUAGE_ID_MAP["JP"]])

    def test_saved_ko_rows_are_not_overwritten(self):
        saved, _, out = self._expand("enc_p.emb.weight", 120, len(SYMBOLS))
        assert self.torch.equal(out[:120], saved)
        assert self.torch.equal(out[self.IDX["ᆼ"]], saved[self.IDX["N"]])

    def test_keeps_initial_values_without_jp_source_rows(self):
        # 언어 테이블에 ZH 행만 있으면 소스(JP=1)가 범위 밖
        _, model, out = self._expand("enc_p.language_emb.weight", 1, 4)
        assert self.torch.equal(out[1:], model[1:])

    def test_keeps_initial_values_for_unmapped_keys(self):
        _, model, out = self._expand("emb.weight", 112, len(SYMBOLS))
        assert self.torch.equal(out[112:], model[112:])


class TestWarmStartOnLoad:
    """실제 로드 경로(사전학습 G_0 safetensors, KO 추가 전 .pth 이어 학습)에서 자동 적용되는지 검증"""

    IDX = {s: i for i, s in enumerate(SYMBOLS)}

    @staticmethod
    def _model(n_symbols: int):
        import torch

        model = torch.nn.Module()
        model.enc_p = torch.nn.Module()
        model.enc_p.emb = torch.nn.Embedding(n_symbols, 4)
        return model

    def test_load_safetensors(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        from style_bert_vits2.models.utils.safetensors import load_safetensors

        saved = torch.randn(112, 4)
        path = tmp_path / "G_0.safetensors"
        save_file({"enc_p.emb.weight": saved}, str(path))
        model, _ = load_safetensors(path, self._model(len(SYMBOLS)))

        emb = model.enc_p.emb.weight.detach()
        assert torch.equal(emb[:112], saved)
        assert torch.equal(emb[self.IDX["ᄅ"]], saved[self.IDX["r"]])

    def test_resume_from_pth(self, tmp_path):
        import torch

        from style_bert_vits2.models.utils.checkpoints import load_checkpoint, save_checkpoint

        old_model = self._model(112)
        old_opt = torch.optim.AdamW(old_model.parameters())
        old_model.enc_p.emb(torch.tensor([0, 1, 2])).sum().backward()
        old_opt.step()
        path = tmp_path / "G_100.pth"
        save_checkpoint(old_model, old_opt, 1e-4, 1, path)
        saved = old_model.enc_p.emb.weight.detach().clone()

        new_model = self._model(len(SYMBOLS))
        new_opt = torch.optim.AdamW(new_model.parameters())
        load_checkpoint(path, new_model, new_opt)

        emb = new_model.enc_p.emb.weight.detach()
        assert torch.equal(emb[self.IDX["ᄅ"]], saved[self.IDX["r"]])
        # optimizer state의 KO 행은 warm-start 대상이 아니라 0 (새 파라미터의 초기 state)
        exp_avg = new_opt.state_dict()["state"][0]["exp_avg"]
        assert torch.equal(exp_avg[112:], torch.zeros(len(SYMBOLS) - 112, 4))


# ============================================================
# 전사 초기 프롬프트 (transcribe.py). torch를 임포트하므로 테스트 안에서 임포트한다
# ============================================================


class TestTranscribeInitialPrompt:
    def test_ko_defaults_to_korean_example(self):
        from transcribe import get_initial_prompt

        prompt = get_initial_prompt("ko")
        assert any("가" <= c <= "힣" for c in prompt)
        assert not any("぀" <= c <= "ヿ" for c in prompt)  # 가나 섞임 금지

    def test_ja_keeps_japanese_example(self):
        from transcribe import get_initial_prompt

        assert get_initial_prompt("ja") == "こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！"

    def test_language_without_example_gets_empty_prompt(self):
        from transcribe import get_initial_prompt

        assert get_initial_prompt("en") == ""

    def test_explicit_prompt_wins_and_is_unquoted(self):
        from transcribe import get_initial_prompt

        assert get_initial_prompt("ko", '"네, 그러네요."') == "네, 그러네요."


# ============================================================
# 학습 시작·재개 체크포인트 선택. 파일이 빠졌을 때 조용히 처음부터 학습하지 않는지 검증
# ============================================================


class TestTrainingCheckpointSelection:
    PREFIXES = ["G", "D", "WD"]

    @staticmethod
    def _touch(directory: Path, *names: str):
        for name in names:
            (directory / name).write_bytes(b"")

    def test_returns_pretrained_paths_when_all_present(self, tmp_path):
        from style_bert_vits2.models.utils.checkpoints import find_pretrained_paths

        self._touch(tmp_path, "G_0.safetensors", "D_0.safetensors", "WD_0.safetensors")
        paths = find_pretrained_paths(tmp_path, self.PREFIXES)
        assert paths == {p: tmp_path / f"{p}_0.safetensors" for p in self.PREFIXES}

    def test_missing_pretrained_files_raise_with_names(self, tmp_path):
        from style_bert_vits2.models.utils.checkpoints import find_pretrained_paths

        self._touch(tmp_path, "D_0.safetensors")
        with pytest.raises(FileNotFoundError, match="G_0.safetensors, WD_0.safetensors"):
            find_pretrained_paths(tmp_path, self.PREFIXES)

    def test_resumes_from_latest_complete_step(self, tmp_path):
        from style_bert_vits2.models.utils.checkpoints import find_resume_checkpoints

        self._touch(tmp_path, *[f"{p}_{s}.pth" for p in self.PREFIXES for s in (1000, 2000)])
        step, paths = find_resume_checkpoints(tmp_path, self.PREFIXES)
        assert step == 2000
        assert paths == {p: tmp_path / f"{p}_2000.pth" for p in self.PREFIXES}

    def test_incomplete_latest_step_warns_and_falls_back(self, tmp_path):
        from style_bert_vits2.logging import logger
        from style_bert_vits2.models.utils.checkpoints import find_resume_checkpoints

        self._touch(tmp_path, *[f"{p}_1000.pth" for p in self.PREFIXES], "G_2000.pth", "WD_2000.pth")
        messages = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            step, paths = find_resume_checkpoints(tmp_path, self.PREFIXES)
        finally:
            logger.remove(sink)
        assert step == 1000
        assert paths["G"] == tmp_path / "G_1000.pth"
        assert any("D_2000.pth" in m for m in messages)

    def test_raises_without_complete_step(self, tmp_path):
        from style_bert_vits2.models.utils.checkpoints import find_resume_checkpoints

        self._touch(tmp_path, "G_1000.pth", "D_2000.pth", "WD_1000.pth")
        with pytest.raises(FileNotFoundError):
            find_resume_checkpoints(tmp_path, self.PREFIXES)

    def test_other_prefixes_are_not_mistaken_for_d(self, tmp_path):
        from style_bert_vits2.models.utils.checkpoints import find_resume_checkpoints

        # WD_·DUR_ 파일은 D_ 파일이 아니다
        self._touch(tmp_path, "G_1000.pth", "WD_1000.pth", "DUR_1000.pth")
        with pytest.raises(FileNotFoundError):
            find_resume_checkpoints(tmp_path, ["G", "D"])


# ============================================================
# 학습 언어 기록과 추론 기본 언어. 언어를 넘기지 않으면 모델이 학습한 언어로 합성한다
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestTrainedLanguage:
    @staticmethod
    def _model(languages: list[str]):
        import numpy as np

        from style_bert_vits2.models.hyper_parameters import HyperParameters
        from style_bert_vits2.tts_model import TTSModel

        hps = HyperParameters.model_validate({"data": {"languages": languages}})
        return TTSModel(Path("dummy.safetensors"), hps, np.zeros((1, 256), dtype=np.float32), device="cpu")

    def test_preprocess_text_records_training_languages(self, tmp_path, monkeypatch):
        # preprocess_text.py는 임포트할 때 pyopenjtalk 워커를 띄우고 사용자 사전을 적용한다.
        # 한국어 줄에는 둘 다 필요 없으므로 막고 같은 프로세스에서 실행한다 (서브프로세스로 돌리면 10초 걸린다).
        # 초기화를 막은 모듈이 sys.modules 에 남지 않도록 import 대신 run_path 로 읽는다.
        # 일본어 줄을 섞으면 일본어 G2P 초기화로 25초가 더 걸려 한국어만 쓴다
        import json
        import runpy
        import shutil

        from style_bert_vits2.nlp.japanese import pyopenjtalk_worker, user_dict

        monkeypatch.setattr(pyopenjtalk_worker, "initialize_worker", lambda: None)
        monkeypatch.setattr(user_dict, "update_dict", lambda: None)
        preprocess = runpy.run_path(str(REPO_ROOT / "preprocess_text.py"))["preprocess"]

        lines = []
        for i, text in enumerate(["안녕하세요.", "네, 그러네요."]):
            (tmp_path / f"{i}.wav").write_bytes(b"")
            lines.append(f"{tmp_path / f'{i}.wav'}|spk|KO|{text}")
        (tmp_path / "esd.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
        config_path = tmp_path / "config.json"
        shutil.copy(REPO_ROOT / "configs" / "config_jp_extra.json", config_path)
        preprocess(
            transcription_path=tmp_path / "esd.list", cleaned_path=None, train_path=tmp_path / "train.list", val_path=tmp_path / "val.list",
            config_path=config_path, val_per_lang=0, max_val_total=0, use_jp_extra=True, yomi_error="raise", correct_path=False,
        )
        assert json.loads(config_path.read_text(encoding="utf-8"))["data"]["languages"] == ["KO"]

    def test_trained_language_is_the_first_recorded_language(self):
        assert self._model(["KO", "JP"]).trained_language == Languages.KO

    def test_model_without_record_has_no_trained_language(self):
        assert self._model([]).trained_language is None

    def test_infer_defaults_to_trained_language(self, monkeypatch):
        import numpy as np

        import style_bert_vits2.models.infer as infer_module

        used = []
        monkeypatch.setattr(infer_module, "infer", lambda **kw: used.append(kw["language"]) or np.zeros(100, dtype=np.float32))
        model = self._model(["KO"])
        model.net_g = object()  # 실제 모델 로드 생략
        model.infer("안녕하세요.")
        assert used == [Languages.KO]

    def test_infer_defaults_to_jp_without_record(self, monkeypatch):
        import numpy as np

        import style_bert_vits2.models.infer as infer_module

        used = []
        monkeypatch.setattr(infer_module, "infer", lambda **kw: used.append(kw["language"]) or np.zeros(100, dtype=np.float32))
        model = self._model([])
        model.net_g = object()
        model.infer("こんにちは。")
        assert used == [Languages.JP]

    def test_infer_stream_defaults_to_trained_language(self, monkeypatch):
        import style_bert_vits2.models.infer as infer_module

        used = []

        def fake_prepare_latent(*args, **kw):
            used.append(kw["language"])
            raise RuntimeError("stop")

        monkeypatch.setattr(infer_module, "prepare_latent", fake_prepare_latent)
        model = self._model(["KO"])
        model.net_g = object()
        with pytest.raises(RuntimeError, match="stop"):
            model.infer_stream("안녕하세요.")
        assert used == [Languages.KO]

    def test_webui_load_selects_trained_language(self, tmp_path):
        import json

        import numpy as np

        from style_bert_vits2.tts_model import TTSModelHolder

        model_dir = tmp_path / "kss"
        model_dir.mkdir()
        (model_dir / "kss_e1_s1.safetensors").write_bytes(b"")
        np.save(model_dir / "style_vectors.npy", np.zeros((1, 256), dtype=np.float32))
        (model_dir / "config.json").write_text(json.dumps({"data": {"languages": ["KO"]}}), encoding="utf-8")
        holder = TTSModelHolder(tmp_path, "cpu", [], ignore_onnx=True)
        updates = holder.get_model_for_gradio("kss", str(model_dir / "kss_e1_s1.safetensors"))
        assert updates[3]["value"] == "KO"


# ============================================================
# 전처리 캐시(.bert.pt·.spec.pt) 무효화. 입력이 바뀌었는데 예전 특징량으로 학습하지 않는지 검증
# ============================================================


class TestPreprocessCache:
    PHONES = ["_", "ᄀ", "ᅡ", "_"]
    WORD2PH = [1, 2, 1]

    def _loader(self, tmp_path):
        import wave

        from data_utils import TextAudioSpeakerLoader
        from style_bert_vits2.models.hyper_parameters import HyperParametersData

        wav = tmp_path / "a.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\0\0" * 22050)
        fields = [str(wav), "spk", "KO", "가", " ".join(self.PHONES), " ".join(["0"] * len(self.PHONES))]
        (tmp_path / "train.list").write_text("|".join(fields + [" ".join(map(str, self.WORD2PH))]) + "\n", encoding="utf-8")
        hps = HyperParametersData(use_jp_extra=True, spk2id={"spk": 0})
        return TextAudioSpeakerLoader(str(tmp_path / "train.list"), hps), str(wav)

    def _get_text(self, loader, wav):
        return loader.get_text("가", list(self.WORD2PH), list(self.PHONES), [0] * len(self.PHONES), "KO", wav)

    def _current_key(self):
        from data_utils import bert_feature_key

        # add_blank 후의 음소 수와 word2ph (bert_gen과 학습 로더가 같은 값으로 지문을 만든다)
        word2ph = [w * 2 for w in self.WORD2PH]
        word2ph[0] += 1
        return bert_feature_key("가", word2ph, 2 * len(self.PHONES) + 1, "KO")

    def test_bert_feature_key_changes_with_text_and_alignment(self):
        from data_utils import bert_feature_key

        key = bert_feature_key("안녕", [1, 2, 2, 1], 6, "KO")
        assert key == bert_feature_key("안녕", [1, 2, 2, 1], 6, "KO")
        assert key != bert_feature_key("안녕!", [1, 2, 2, 1], 6, "KO")
        assert key != bert_feature_key("안녕", [1, 3, 1, 1], 6, "KO")

    def test_saved_bert_feature_loads_only_with_matching_key_and_length(self, tmp_path):
        import torch

        from data_utils import load_bert_feature, save_bert_feature

        path = str(tmp_path / "a.bert.pt")
        bert = torch.randn(1024, 5)
        save_bert_feature(path, bert, "k1")
        assert torch.equal(load_bert_feature(path, "k1", 5), bert)
        assert load_bert_feature(path, "k2", 5) is None
        assert load_bert_feature(path, "k1", 6) is None

    def test_legacy_bert_feature_is_regenerated_but_still_trainable(self, tmp_path):
        import torch

        from data_utils import load_bert_feature

        path = str(tmp_path / "a.bert.pt")
        bert = torch.randn(1024, 5)
        torch.save(bert, path)  # 지문이 없는 예전 형식
        assert load_bert_feature(path, "k1", 5) is None
        assert torch.equal(load_bert_feature(path, "k1", 5, allow_legacy=True), bert)

    def test_unreadable_bert_feature_is_none(self, tmp_path):
        from data_utils import load_bert_feature

        path = tmp_path / "a.bert.pt"
        path.write_bytes(b"broken")
        assert load_bert_feature(str(path), "k1", 5, allow_legacy=True) is None

    def test_training_loader_uses_current_bert_feature(self, tmp_path):
        import torch

        from data_utils import save_bert_feature

        loader, wav = self._loader(tmp_path)
        save_bert_feature(wav.replace(".wav", ".bert.pt"), torch.ones(1024, 9), self._current_key())
        _, ja_bert, *_ = self._get_text(loader, wav)
        assert torch.equal(ja_bert, torch.ones(1024, 9))

    def test_training_loader_rejects_outdated_bert_feature(self, tmp_path):
        import torch

        from data_utils import save_bert_feature

        loader, wav = self._loader(tmp_path)
        save_bert_feature(wav.replace(".wav", ".bert.pt"), torch.ones(1024, 9), "outdated")
        with pytest.raises(RuntimeError, match="bert_gen"):
            self._get_text(loader, wav)

    def test_training_loader_reports_missing_bert_feature(self, tmp_path):
        loader, wav = self._loader(tmp_path)
        with pytest.raises(RuntimeError, match="bert_gen"):
            self._get_text(loader, wav)

    def test_training_loader_recomputes_spec_older_than_audio(self, tmp_path):
        import os

        import torch

        loader, wav = self._loader(tmp_path)
        spec_path = wav.replace(".wav", ".spec.pt")
        torch.save(torch.zeros(3), spec_path)
        os.utime(spec_path, (1000, 1000))
        os.utime(wav, (2000, 2000))
        spec, _ = loader.get_audio(wav)
        assert spec.shape[0] == loader.filter_length // 2 + 1

    def test_training_loader_reuses_spec_newer_than_audio(self, tmp_path):
        import os

        import torch

        loader, wav = self._loader(tmp_path)
        spec_path = wav.replace(".wav", ".spec.pt")
        torch.save(torch.zeros(3), spec_path)
        os.utime(wav, (1000, 1000))
        os.utime(spec_path, (2000, 2000))
        spec, _ = loader.get_audio(wav)
        assert torch.equal(spec, torch.zeros(3))

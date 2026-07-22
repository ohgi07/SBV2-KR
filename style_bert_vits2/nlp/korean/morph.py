"""
형태소 정보에 기반한 발음 보정.

pronounce.py의 음운 규칙 엔진은 형태소 경계 정보가 필요한 규칙을 다룰 수 없으므로,
이 모듈에서 발음 변환 전에 텍스트를 고쳐 써서 보정한다.
모든 재작성은 문자 수를 보존한다 (word2ph 정렬에 필수).

구현된 보정:
1. 발음 예외 사전: 어휘화된 합성어의 ㄴ 첨가 (솜이불→솜니불)나 예외어 (맛있다)의
   재작성. ㄴ만 삽입하면 나머지 음운 변화 (비음화·유음화 등)는
   pronounce.py가 올바르게 유도한다.
2. kiwipiepy 형태소 분석 (설치되어 있는 경우에만):
   - 속격 조사 의 → 에 (나의→나에)
   - 형태소 경계의 ㄴ 첨가 (한+여름→한녀름, 헛+일→헛닐)
   - 어절 경계의 ㄴ 첨가 (제29항 붙임2: 한 일→한 닐 → [한닐])
   - 관형사형 어미 -ㄹ 뒤의 경음화 (갈 데가→갈 떼가)
   - 용언 어간말 ㄴ/ㅁ 뒤 어미의 경음화 (신다→신따, 안고→안꼬)

kiwipiepy가 없으면 예외 사전만 적용된다.
"""

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.pronounce import (
    TENSIFICATION_MAP,
    compose,
    decompose,
    is_hangul_syllable,
)


# 발음 예외 사전 (표기 → 발음을 유도할 수 있는 형태로의 재작성)
## 값은 반드시 키와 같은 문자 수여야 한다
## ㄴ 첨가어는 ㄴ을 삽입한 형태로만 만들면 된다 (후속 음운 규칙이 나머지를 유도)
PRONUNCIATION_EXCEPTIONS: dict[str, str] = {
    # 예외어 (합성어 경계의 절음 + 연음)
    "맛있": "마싯",  # 맛있다 → [마싣따]
    "멋있": "머싯",  # 멋있다 → [머싣따]
    "맛없": "마덦",  # 맛없다 → [마덥따]
    "멋없": "머덦",  # 멋없다 → [머덥따]
    "끝없": "끄덦",  # 끝없다 → [끄덥따], 끝없이 → [끄덥씨]
    # 제15항 붙임: 받침 + ㅏㅓㅗㅜㅟ 시작 실질 형태소의 합성어는 대표음으로 바꾸어서
    # 연음. kiwi가 단일 토큰으로 등재한 어휘라 규칙으로는 경계를 얻을 수 없어
    # 앞 음절 받침을 중화형으로 미리 재작성한다 (연음은 엔진이 수행).
    "겉옷": "걷옷",  # → [거돋]
    "헛웃": "헏웃",  # 헛웃음 → [허두슴]
    "젖어미": "젇어미",  # → [저더미]
    "값어치": "갑어치",  # → [가버치]
    "팥알": "팓알",  # → [파달]
    "홑옷": "혿옷",  # → [호돋]
    "웃어른": "욷어른",  # → [우더른]
    "첫아": "첟아",  # 첫아들 → [처다들], 첫아이 → [처다이]
    # ㄴ 첨가 (어휘화된 합성어)
    "솜이불": "솜니불",  # → [솜니불]
    "색연필": "색년필",  # → [생년필]
    "꽃잎": "꽃닙",  # → [꼰닙]
    "나뭇잎": "나뭇닙",  # → [나문닙]
    "깻잎": "깻닙",  # → [깬닙]
    "담요": "담뇨",  # → [담뇨]
    "물약": "물냑",  # → [물략]
    "알약": "알냑",  # → [알략]
    "식용유": "식용뉴",  # → [시굥뉴]
    "휘발유": "휘발뉴",  # → [휘발류]
    "서울역": "서울녁",  # → [서울력]
    "막일": "막닐",  # → [망닐]
    "눈요기": "눈뇨기",  # → [눈뇨기]
    "영업용": "영업뇽",  # → [영엄뇽]
    "홑이불": "홑니불",  # → [혼니불] (구개음화 오적용 방지)
    "늑막염": "늑막념",  # → [능망념]
    "솔잎": "솔닙",  # → [솔립]
    "물엿": "물녓",  # → [물렫]
    "콩엿": "콩녓",  # → [콩녇]
    "내복약": "내복냑",  # → [내봉냑]
    # 합성어 내부의 어간말 ㅁ 경음화 (형태소 분석으로는 경계를 얻을 수 없는 등재어)
    "줄넘기": "줄넘끼",  # → [줄럼끼]
    # 한자어 수사의 ㄴ 첨가는 불규칙해 어휘화되어 있다 (삼일→[사밀]이지만 십육→[심뉵]).
    # 일반 규칙에서는 수사 (NR)를 제외하고 (아래 참조), 첨가하는 단어만 여기서 다룬다.
    "십육": "십뉵",  # → [심뉵] (예: 십육, 육십육)
    # 구개음화의 어휘 예외 (형태소 경계 정보 없이는 유도 불가)
    "벼훑이": "벼훌치",  # → [벼훌치] (표준발음법 §17)
    # ㅎ 받침 + ㅊ: 규칙 엔진은 ㅎ 탈락 [노치다]로 유도하지만 표준은 [녿치다]
    "놓치": "녿치",  # 놓치다 → [녿치다]
    "놓쳐": "녿쳐",  # → [녿처]
    "놓쳤": "녿쳗",  # → [녿첟따]
    "놓친": "녿친",  # → [녿친]
    "놓칠": "녿칠",  # → [녿칠]
    # 합성어 굶주리다: 경음화 없이 [굼주리다]가 표준
    "굶주": "굼주",  # 굶주리다 → [굼주리다]
    # 밟- 의 모음 어미 활용형: ㄼ→[ㅂ] 어휘 예외 + 연음 (표준발음법 제10항 다만)
    "밟아": "발바",  # 밟아 → [발바]
    "밟으": "발브",  # 밟으면 → [발브면]
    "밟았": "발밨",  # 밟았다 → [발받따]
    # 밟- 의 자음 어미 활용형: ㄼ→[ㅂ] 어휘 예외 (표준발음법 제10항 다만)
    # (ㅎ 어미 (밟히다→[발피다])는 내장 엔진의 격음화 규칙이 올바르게 유도하므로 제외)
    "밟은": "발븐",  # → [발븐]
    "밟을": "발블",  # → [발블]
    "밟다": "밥다",  # → [밥따]
    "밟고": "밥고",  # → [밥꼬]
    "밟지": "밥지",  # → [밥찌]
    "밟는": "밥는",  # → [밤는]
    "밟게": "밥게",  # → [밥께]
    "밟기": "밥기",  # → [밥끼]
    "밟습": "밥습",  # 밟습니다 → [밥씀니다]
    # 넓- 파생어의 ㄼ→[ㅂ] (표준발음법 제10항 다만)
    "넓죽": "넙쭉",  # 넓죽하다 → [넙쭈카다]
    "넓둥": "넙뚱",  # 넓둥글다 → [넙뚱글다]
    # ㄴ+ㄹ→[ㄴㄴ] 한자어 예외 (표준발음법 제20항 다만)
    "의견란": "의견난",  # → [의견난]
    "생산량": "생산냥",  # → [생산냥]
    "결단력": "결딴녁",  # → [결딴녁]
    # 부사로 어휘화된 -ㄹ수록 (형태소 경계가 없어 규칙으로 도출 불가)
    "갈수록": "갈쑤록",  # → [갈쑤록]
    # 한자어 경음화·사잇소리 등재어 (규칙 도출 불가 — BERT vocab 전수 diff에서 수집,
    # 표준국어대사전 발음 확인분만. 복수 표준 (금융·깃발·햇살 등)은 추가하지 않음)
    "갈등": "갈뜽",  # 葛藤 → [갈뜽]
    "발전": "발쩐",  # 發展/發電 → [발쩐] (발전소·선발전 등 파생 포함)
    "물질": "물찔",  # 物質 → [물찔]
    "갈증": "갈쯩",  # 渴症 → [갈쯩]
    "절도": "절또",  # 竊盜/節度 → [절또]
    "말살": "말쌀",  # 抹殺 → [말쌀]
    "동일시": "동일씨",  # → [동일씨]
    "등불": "등뿔",  # → [등뿔]
    "술잔": "술짠",  # → [술짠]
    "길가": "길까",  # → [길까]
    "강가": "강까",  # → [강까]
    "눈동자": "눈똥자",  # → [눈똥자]
    "문고리": "문꼬리",  # → [문꼬리]
    "발걸음": "발껄음",  # → [발꺼름]
    "발바닥": "발빠닥",  # → [발빠닥]
    "발동": "발똥",  # 發動 → [발똥]
    "신바람": "신빠람",  # → [신빠람]
    "아침밥": "아침빱",  # → [아침빱]
    "상견례": "상견녜",  # 相見禮 → [상견녜] (ㄴ+ㄹ→[ㄴㄴ] 예외)
    "공권력": "공꿘녁",  # 公權力 → [공꿘녁]
    # 제26항 공식 예시 (한자어 ㄹ 받침 + ㄷㅅㅈ 경음화 — 어휘 판별이 필요해 사전 처리)
    "불소": "불쏘",  # 弗素 → [불쏘]
    "일시": "일씨",  # 日時 → [일씨] (일시적·일시불 포함)
    "몰상식": "몰쌍식",  # → [몰쌍식]
    "불세출": "불쎄출",  # → [불쎄출]
    "문법": "문뻡",  # 文法 → [문뻡]
    # 제28항 공식 예시 (사이시옷 없는 관형격 합성어 경음화)
    "산새": "산쌔",  # → [산쌔]
    "손재주": "손째주",  # → [손째주]
    "물동이": "물똥이",  # → [물똥이]
    "굴속": "굴쏙",  # → [굴쏙]
    "바람결": "바람껼",  # → [바람껼]
    "그믐달": "그믐딸",  # → [그믐딸]
    "초승달": "초승딸",  # → [초승딸]
    "창살": "창쌀",  # → [창쌀]
    "강줄기": "강쭐기",  # → [강쭐기]
    # 제29항 다만: ㄴ첨가 없이 연음하는 어휘 (형태소 경계가 있어도 첨가 금지)
    "송별연": "송벼련",  # → [송벼련]
    # 제30항 3: 사이시옷 뒤 '이' 음 → [ㄴㄴ] (ㄴ만 삽입하면 비음화가 나머지를 유도)
    "베갯잇": "베갯닛",  # → [베갠닏]
    "도리깻열": "도리깻녈",  # → [도리깬녈]
    # 받아들이다: 들+일 은 어간 내부이므로 ㄴ 첨가 없음 ([바다드릴])
    "받아들일": "바다드릴",  # → [바다드릴]
    "받아들인": "바다드린",  # → [바다드린]
}

# ㄴ 첨가의 조건이 되는 모음 (이, 야, 여, 요, 유)
__N_INSERTION_VOWELS = {"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ"}

# ㄴ 첨가가 일어나는 형태소 경계의 태그 (앞쪽 / 뒤쪽)
# 수사 (NR)는 한자어 수사의 ㄴ 첨가가 불규칙하므로 일반 규칙에서 제외하고
# (삼일→[사밀]이지 [삼닐]이 아님, 육쩜이오→[육쩌미오]),
# 첨가하는 단어는 PRONUNCIATION_EXCEPTIONS에서 다룬다.
__N_INSERTION_LEFT_TAGS = {"NNG", "NNP", "NNB", "XPN", "XSN", "MM"}
__N_INSERTION_RIGHT_TAGS = {"NNG", "NNP", "NNB", "XSN"}

# 어절 경계 (공백을 사이에 둔) ㄴ 첨가에서 오른쪽에 설 수 있는 실질 형태소 태그 (제29항 붙임2).
# 공식 예 (한 일, 옷 입다, 먹은 엿 등)의 오른쪽은 모두 1음절 실질 형태소이며,
# 다음절어 (야구, 이유 등)로의 첨가는 과잉 적용이 되므로 1음절로 한정한다.
__N_INSERTION_CROSS_WORD_RIGHT_TAGS = {"NNG", "NNP", "NP", "VV", "VA", "VV-I", "VV-R"}

# 용언 어간의 태그 (어간말 ㄴ/ㅁ 뒤의 경음화에 사용)
__VERB_STEM_TAGS = {"VV", "VA", "VX", "VV-I", "VV-R", "VA-I", "VA-R"}

# 어미의 태그
__ENDING_TAGS = {"EC", "EF", "ETM", "ETN", "EP"}

# kiwipiepy (선택 의존성)
__kiwi_instance = None
__kiwi_unavailable = False


def get_kiwi():
    """
    kiwipiepy가 설치되어 있으면 그 인스턴스 (싱글턴)를 반환한다.
    미설치라면 None을 반환한다.
    """
    global __kiwi_instance, __kiwi_unavailable
    if __kiwi_unavailable:
        return None
    if __kiwi_instance is None:
        try:
            from kiwipiepy import Kiwi

            __kiwi_instance = Kiwi()
            logger.info("Using kiwipiepy for morphology-aware Korean pronunciation")
        except Exception as e:
            __kiwi_unavailable = True
            logger.info(
                f"kiwipiepy is not available ({e}), "
                "morphology-aware pronunciation rules are disabled"
            )
            return None
    return __kiwi_instance


# 예외 사전 항목은 문자 수 보존이 필수 (word2ph 정렬이 깨지기 때문)
# 사용자가 사전을 확장하는 것을 상정해, 임포트 시 한 번만 명시적으로 검증한다
for __orig, __replaced in PRONUNCIATION_EXCEPTIONS.items():
    if len(__orig) != len(__replaced):
        raise ValueError(
            f"PRONUNCIATION_EXCEPTIONS entry must preserve length: "
            f"{__orig!r} ({len(__orig)}) -> {__replaced!r} ({len(__replaced)})"
        )


def apply_exceptions(text: str) -> str:
    """발음 예외 사전에 의한 재작성 (문자 수 보존)"""
    for orig, replaced in PRONUNCIATION_EXCEPTIONS.items():
        if orig in text:
            text = text.replace(orig, replaced)
    return text


def __palatalize_d_hyeo(text: str) -> str:
    """
    ㄷ 받침 + 혀/혔의 구개음화 (표준발음법 §17 붙임: 닫혀→[다처], 갇혔다→[가첟따]).

    표기상 이 조합은 -히- 의 축약 (닫히+어)에서만 생기므로 무조건 적용할 수 있다.
    발음 변환 전에 구개음화를 선반영한 형태 (다쳐)로 고쳐 써 둔다
    (쳐→처의 단모음화는 pronounce.py가 수행). 문자 수는 보존된다.
    """
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i + 1] in ("혀", "혔") and is_hangul_syllable(chars[i]):
            cho, jung, coda = decompose(chars[i])
            if coda == ["ㄷ"]:
                chars[i] = compose(cho, jung, [])
                chars[i + 1] = "쳐" if chars[i + 1] == "혀" else "쳤"
    return "".join(chars)


def __set_cho(char: str, new_cho: str) -> str:
    cho, jung, coda = decompose(char)
    return compose(new_cho, jung, coda)


def apply_morph_rules(text: str) -> str:
    """
    형태소 정보에 기반한 발음 보정을 텍스트에 적용한다.
    출력은 입력과 같은 문자 수임이 보장된다.
    """
    text = apply_exceptions(text)
    text = __palatalize_d_hyeo(text)

    kiwi = get_kiwi()
    if kiwi is None:
        return text

    try:
        tokens = list(kiwi.tokenize(text))
    except Exception as e:
        logger.warning(f"kiwipiepy failed ({e}), skipping morphology-aware rules")
        return text

    chars = list(text)

    prev_token = None
    for token in tokens:
        start = token.start
        # 1. 속격 조사 의 → 에
        if token.tag == "JKG" and token.form == "의" and start < len(chars) and chars[start] == "의":  # fmt: skip
            chars[start] = "에"

        # 인접한 형태소 (같은 어절 안에서 직전 형태소와 틈 없이 이어짐)인 경우에만
        adjacent = prev_token is not None and prev_token.start + prev_token.len == start and start > 0  # fmt: skip

        # 2. 형태소 경계의 ㄴ 첨가 (한+여름 → 한녀름)
        if (
            adjacent
            and prev_token.tag in __N_INSERTION_LEFT_TAGS
            and token.tag in __N_INSERTION_RIGHT_TAGS
            and start < len(chars)
            and is_hangul_syllable(chars[start])
            and is_hangul_syllable(chars[start - 1])
        ):
            cho, jung, _ = decompose(chars[start])
            _, _, prev_coda = decompose(chars[start - 1])
            if cho == "ㅇ" and jung in __N_INSERTION_VOWELS and prev_coda:
                chars[start] = __set_cho(chars[start], "ㄴ")

        # 2b. 어절 경계의 ㄴ 첨가 (제29항 붙임2: 한 일→한 닐, 옷 입다→옷 닙다)
        #     직전 형태소와 공백 1개를 사이에 두고, 앞 어절이 자음 받침으로 끝나며
        #     1음절 실질형태소가 이/야/여/요/유로 시작하는 경우.
        #     첨가 후의 유도 (비음화·유음화)는 pronounce.py의 경계 패스가 수행한다.
        cross_word = (
            prev_token is not None
            and prev_token.start + prev_token.len + 1 == start
            and start >= 2
            and chars[start - 1] == " "
        )
        if (
            cross_word
            and token.tag in __N_INSERTION_CROSS_WORD_RIGHT_TAGS
            and token.len == 1
            and start < len(chars)
            and is_hangul_syllable(chars[start])
            and is_hangul_syllable(chars[start - 2])
        ):
            cho, jung, _ = decompose(chars[start])
            _, _, prev_coda = decompose(chars[start - 2])
            if cho == "ㅇ" and jung in __N_INSERTION_VOWELS and prev_coda:
                chars[start] = __set_cho(chars[start], "ㄴ")

        # 3. 용언 어간말 ㄴ/ㅁ 뒤 어미의 경음화 (신다 → 신따)
        if (
            adjacent
            and prev_token.tag in __VERB_STEM_TAGS
            and token.tag in __ENDING_TAGS
            and start < len(chars)
            and is_hangul_syllable(chars[start])
            and is_hangul_syllable(chars[start - 1])
        ):
            _, _, stem_coda = decompose(chars[start - 1])
            cho, _, _ = decompose(chars[start])
            # 어간 받침 ㄴ(ㄵ)/ㅁ(ㄻ) 뒤의 어미는 경음화 (표준발음법 제24항)
            # ㄵ은 pronounce.py의 장애음 규칙에서 처리되므로, 여기서는 ㄴ/ㅁ/ㄻ을 다룬다
            # (명사의 ㄻ (삶과 등)은 경음화하지 않으므로, 용언 어간에 한정하는 이곳이 맞는 위치)
            if stem_coda in (["ㄴ"], ["ㅁ"], ["ㄹ", "ㅁ"]) and cho in ("ㄱ", "ㄷ", "ㅅ", "ㅈ"):  # fmt: skip
                chars[start] = __set_cho(chars[start], TENSIFICATION_MAP[cho])
            # 용언 어간 말음 ㄺ은 ㄱ 어미 앞에서 [ㄹ] + 어미 경음화 (표준발음법 제11항 다만:
            # 맑고→[말꼬]). 명사 (닭과 등)는 [ㄱ]이므로 용언 어간에 한정하는 여기가 맞다.
            if stem_coda == ["ㄹ", "ㄱ"] and cho == "ㄱ":
                pcho, pjung, _ = decompose(chars[start - 1])
                chars[start - 1] = compose(pcho, pjung, ["ㄹ"])
                chars[start] = __set_cho(chars[start], "ㄲ")

        # 4. 관형사형 어미 -(으)ㄹ 뒤의 경음화 (갈 데가 → 갈 떼가, 먹을 것 → 먹을 껏)
        ## form 문자열로 판정하면 축약형의 "ᆯ" (자모)은 잡히지만 "을/를" 같은
        ## 완성형 음절을 놓치므로, 토큰 마지막 음절의 종성으로 판정한다
        etm_ends_with_rieul = False
        last_pos = start + token.len - 1
        if token.tag == "ETM" and 0 <= last_pos < len(chars) and is_hangul_syllable(chars[last_pos]):  # fmt: skip
            _, _, etm_coda = decompose(chars[last_pos])
            etm_ends_with_rieul = bool(etm_coda) and etm_coda[-1] == "ㄹ"
        if etm_ends_with_rieul:
            next_pos = start + token.len
            # 공백 1개까지 건너뛰고 다음 문자를 찾는다
            if next_pos < len(chars) and chars[next_pos] == " ":
                next_pos += 1
            if next_pos < len(chars) and is_hangul_syllable(chars[next_pos]):
                cho, _, _ = decompose(chars[next_pos])
                if cho in ("ㄱ", "ㄷ", "ㅂ", "ㅅ", "ㅈ"):
                    chars[next_pos] = __set_cho(chars[next_pos], TENSIFICATION_MAP[cho])

        # 5. -(으)ㄹ로 시작하는 축약 어미의 경음화 (표준발음법 제27항 붙임: 할수록→[할쑤록],
        #    할지→[할찌], 할게→[할께]). kiwi는 축약 어미를 ᆯ수록 같은 자모 ᆯ로 시작하는
        #    겹침 스팬으로 반환하므로, 어미 2음절째 (start+1)가 경음화 대상이 된다.
        if token.tag in __ENDING_TAGS and len(token.form) >= 2 and token.form[0] == "ᆯ" and token.len >= 2:  # fmt: skip
            pos2 = start + 1
            if pos2 < len(chars) and is_hangul_syllable(chars[pos2]):
                cho, _, _ = decompose(chars[pos2])
                if cho in ("ㄱ", "ㄷ", "ㅂ", "ㅅ", "ㅈ"):
                    chars[pos2] = __set_cho(chars[pos2], TENSIFICATION_MAP[cho])

        prev_token = token

    result = "".join(chars)
    assert len(result) == len(text)
    return result

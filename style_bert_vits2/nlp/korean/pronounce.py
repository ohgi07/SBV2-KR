"""
표준 발음법에 기반한 규칙 기반 발음 변환 엔진.

표기상의 한글을 실제로 발음되는 한글로 변환한다 (음절 수는 반드시 보존됨).
외부 의존성 없이 동작하는 유일한 발음 변환 엔진이며, morph.py의 형태소 기반 보정과
결합되어 g2p.py에서 사용된다. 어절 내부 규칙에 더해, 공백 하나로 인접한 어절
쌍에 대한 어절 경계 규칙(연음·격음화·경음화·비음화·유음화)도 적용한다.

구현된 규칙:
- 져/쪄/쳐의 단모음화 (제5항 다만1): 가져→가저, 가르쳐→가르처
- 구개음화: 굳이→구지, 같이→가치, 닫히다→다치다
- ㅎ 탈락·격음화: 좋아→조아, 좋다→조타, 국화→구콰, 못하다→모타다
- 연음: 밥이→바비, 옷을→오슬, 값이→갑씨
- 자음군 단순화: 값→갑, 닭→닥
- 음절의 끝소리 규칙 (종성 중화): 옷→옫, 부엌→부억
- 경음화: 국밥→국빱, 학교→학꾜
- 비음화: 국물→궁물, 십리→심니, 종로→종노
- 유음화: 신라→실라, 칼날→칼랄

제한 사항 (형태소 분석이 필요해 미구현):
- ㄴ 첨가 (솜이불→솜니불)
- 어휘 의미에 따른 예외 (밟다→밥따 등)
- 한자어 ㄹ 관련의 예외적 경음화
"""

# 초성 (choseong) 19개: 유니코드 조합 순서
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]  # fmt: skip

# 중성 (jungseong) 21개: 유니코드 조합 순서
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]  # fmt: skip

# 종성 (jongseong) 27개 (인덱스 1-27, 0은 종성 없음): 유니코드 조합 순서
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]  # fmt: skip

# 겹받침 → 구성 자음 분해 맵
DOUBLE_CODA_SPLIT = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}

# 음절의 끝소리 규칙 (종성 중화): 발음상 존재할 수 있는 7종성으로의 사상
CODA_NEUTRALIZATION = {
    "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ",
    "ㄴ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ",
    "ㅁ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ",
    "ㅇ": "ㅇ",
}  # fmt: skip

# 격음화: 평음 + ㅎ / ㅎ + 평음 → 격음
ASPIRATION_MAP = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}

# 경음화: 장애음 받침 뒤의 평음 → 경음
TENSIFICATION_MAP = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}

# 비음화: 장애음 받침 → 대응하는 비음
NASALIZATION_MAP = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}

__CHOSEONG_INDEX = {c: i for i, c in enumerate(CHOSEONG)}
__JUNGSEONG_INDEX = {c: i for i, c in enumerate(JUNGSEONG)}
__JONGSEONG_INDEX = {c: i for i, c in enumerate(JONGSEONG)}

HANGUL_BASE = 0xAC00


def is_hangul_syllable(char: str) -> bool:
    """완성형 한글 음절 (가-힣) 인지 여부를 반환한다"""
    return 0xAC00 <= ord(char) <= 0xD7A3


def decompose(char: str) -> tuple[str, str, list[str]]:
    """
    한글 음절을 (초성, 중성, 종성 리스트)로 분해한다.
    종성은 겹받침이면 2개 요소, 없으면 빈 리스트.
    """
    code = ord(char) - HANGUL_BASE
    cho = CHOSEONG[code // 588]
    jung = JUNGSEONG[(code % 588) // 28]
    jong = JONGSEONG[code % 28]
    if jong == "":
        coda: list[str] = []
    elif jong in DOUBLE_CODA_SPLIT:
        coda = list(DOUBLE_CODA_SPLIT[jong])
    else:
        coda = [jong]
    return cho, jung, coda


def compose(cho: str, jung: str, coda: list[str]) -> str:
    """(초성, 중성, 종성 리스트)에서 한글 음절을 조합한다"""
    if len(coda) == 0:
        jong = ""
    elif len(coda) == 1:
        jong = coda[0]
    else:
        # 겹받침 조합 (규칙 적용 후에는 보통 여기 오지 않음)
        for double, parts in DOUBLE_CODA_SPLIT.items():
            if parts == tuple(coda):
                jong = double
                break
        else:
            # decompose() 에서 나온 종성이라면 반드시 위에서 조합 가능해야 한다.
            # 조용히 자모를 버리면 원인 추적이 어려워지므로 명시적으로 실패시킨다
            raise ValueError(f"Cannot compose invalid double coda: {coda}")
    code = (
        HANGUL_BASE
        + __CHOSEONG_INDEX[cho] * 588
        + __JUNGSEONG_INDEX[jung] * 28
        + __JONGSEONG_INDEX[jong]
    )
    return chr(code)


def __apply_jyeo_rule(syls: list[list]) -> None:
    """
    져/쪄/쳐의 단모음화 (표준 발음법 제5항 다만1):
    용언 활용형의 '져, 쪄, 쳐'는 [저, 쩌, 처]로 발음한다 (가져→가저, 다쳐→다처).
    ㅈ 계열 초성은 이미 구개음이라 ㅕ의 [j]가 실현되지 않으므로 무조건 적용해도 된다.
    """
    for syl in syls:
        if syl[0] in ("ㅈ", "ㅉ", "ㅊ") and syl[1] == "ㅕ":
            syl[1] = "ㅓ"


def __apply_ui_rules(syls: list[list]) -> None:
    """
    ㅢ의 발음 규칙 (표준 발음법 제5항):
    - 자음을 초성으로 가진 ㅢ는 반드시 [ㅣ] (희망→히망, 무늬→무니)
    - 어중 (첫음절·끝음절 제외)의 의는 [이] (회의감→회이감)
      어말의 의는 속격 조사([에])일 가능성이 있어 형태소 계층에 맡기고 그대로 둔다
    """
    for i, syl in enumerate(syls):
        if syl[1] != "ㅢ":
            continue
        if syl[0] != "ㅇ":
            syl[1] = "ㅣ"
        elif 0 < i < len(syls) - 1:
            syl[1] = "ㅣ"


def __apply_palatalization(syls: list[list]) -> None:
    """
    구개음화: 받침 ㄷ/ㅌ + 이 → 지/치, ㄷ + 히 → 치.
    ㅕ에도 적용한다 (붙여 = 붙이+어 → [부처], 닫혀 = 닫히+어 → [다처]).
    표기상 ㄷ/ㅌ 받침 + 여/혀는 '이' 계열 접미사의 축약에서만 생기므로 무조건 적용해도 된다.
    생성된 쳐는 뒤따르는 __apply_jyeo_rule이 [처]로 단모음화한다.
    """
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2] or nxt[1] not in ("ㅣ", "ㅕ"):
            continue
        # ㅕ는 '이' 계열 축약 음절인 여/였/혀/혔 (받침 없음 또는 ㅆ)에 한정한다.
        # 받침을 가진 형/열/염 등은 축약이 아닌 실질 음절 (맏형[마텽]은 격음화가 맞음)
        if nxt[1] == "ㅕ" and nxt[2] not in ([], ["ㅆ"]):
            continue
        last = cur[2][-1]
        if nxt[0] == "ㅇ":
            if last == "ㄷ":
                cur[2] = cur[2][:-1]
                nxt[0] = "ㅈ"
            elif last == "ㅌ":
                cur[2] = cur[2][:-1]
                nxt[0] = "ㅊ"
        elif nxt[0] == "ㅎ" and last == "ㄷ":
            cur[2] = cur[2][:-1]
            nxt[0] = "ㅊ"


def __apply_h_rules(syls: list[list]) -> None:
    """ㅎ 탈락·격음화"""
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2]:
            continue
        last = cur[2][-1]
        # 받침 쪽의 ㅎ (ㅎ, ㄶ, ㅀ)
        if last == "ㅎ":
            if nxt[0] in ASPIRATION_MAP:
                # 좋다→조타, 많고→만코
                cur[2] = cur[2][:-1]
                nxt[0] = ASPIRATION_MAP[nxt[0]]
            elif nxt[0] == "ㅅ":
                # 닿소→다쏘
                cur[2] = cur[2][:-1]
                nxt[0] = "ㅆ"
            elif nxt[0] == "ㅇ":
                # 좋아→조아, 많이→마니 (남은 ㄴ/ㄹ은 이후 연음에서 이동)
                cur[2] = cur[2][:-1]
            elif nxt[0] == "ㄴ":
                # 놓는→논는 (ㅎ→ㄷ→비음화로 ㄴ이 되지만 바로 ㄴ으로 처리)
                if len(cur[2]) == 1:
                    cur[2] = ["ㄴ"]
                else:
                    cur[2] = cur[2][:-1]
        # 초성 쪽의 ㅎ: 받침의 장애음과 결합해 격음화 (국화→구콰, 앉히다→안치다)
        elif nxt[0] == "ㅎ":
            if last in ASPIRATION_MAP:
                cur[2] = cur[2][:-1]
                nxt[0] = ASPIRATION_MAP[last]
            elif last in ("ㅅ", "ㅆ", "ㅊ", "ㅌ"):
                # 제12항 붙임2: ㄷ으로 중화되는 받침 + ㅎ → [ㅌ] (못하다→모타다, 깨끗하다→깨끄타다)
                cur[2] = cur[2][:-1]
                nxt[0] = "ㅌ"


def __apply_liaison(syls: list[list]) -> None:
    """연음: 받침 + 모음으로 시작하는 음절 → 받침이 다음 초성으로 이동"""
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2] or nxt[0] != "ㅇ":
            continue
        last = cur[2][-1]
        # ㅇ 받침은 이동하지 않음 (강이→강이)
        if last == "ㅇ":
            continue
        cur[2] = cur[2][:-1]
        # 겹받침에서 옮겨 간 ㅅ은 경음화한다 (값이→갑씨, 넋이→넉씨)
        if last == "ㅅ" and len(cur[2]) > 0:
            nxt[0] = "ㅆ"
        else:
            nxt[0] = last


def __apply_coda_simplification(syls: list[list]) -> None:
    """자음군 단순화 + 음절의 끝소리 규칙 (종성 중화)"""
    for syl in syls:
        if not syl[2]:
            continue
        if len(syl[2]) == 2:
            first, second = syl[2]
            # 대표음 선택: ㄺ→ㄱ, ㄻ→ㅁ, ㄿ→ㅂ은 뒤 자음, 그 외에는 앞 자음
            if first + second in ("ㄹㄱ", "ㄹㅁ", "ㄹㅍ"):
                syl[2] = [second]
            else:
                syl[2] = [first]
        syl[2] = [CODA_NEUTRALIZATION[syl[2][0]]]


def __apply_tensification(syls: list[list]) -> None:
    """
    경음화: 장애음 받침 + 평음 → 경음 (국밥→국빱, 앉다→안따)
    자음군 단순화로 장애음 정보가 사라지기 전에, 중화 후의 값으로 판정해야 한다
    """
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2]:
            continue
        last = CODA_NEUTRALIZATION[cur[2][-1]]
        if last in ("ㄱ", "ㄷ", "ㅂ") and nxt[0] in TENSIFICATION_MAP:
            nxt[0] = TENSIFICATION_MAP[nxt[0]]


def __apply_nasalization(syls: list[list]) -> None:
    """비음화: 국물→궁물, 십리→심니, 종로→종노, 독립→동닙"""
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2]:
            continue
        coda = cur[2][-1]
        if coda in NASALIZATION_MAP and nxt[0] in ("ㄴ", "ㅁ"):
            cur[2] = cur[2][:-1] + [NASALIZATION_MAP[coda]]
        elif coda in NASALIZATION_MAP and nxt[0] == "ㄹ":
            # 장애음 + ㄹ → 비음 + ㄴ (십리→심니)
            cur[2] = cur[2][:-1] + [NASALIZATION_MAP[coda]]
            nxt[0] = "ㄴ"
        elif coda in ("ㅁ", "ㅇ") and nxt[0] == "ㄹ":
            # 비음 + ㄹ → 비음 + ㄴ (종로→종노, 담력→담녁)
            nxt[0] = "ㄴ"


def __apply_lateralization(syls: list[list]) -> None:
    """유음화: ㄴ + ㄹ / ㄹ + ㄴ → ㄹㄹ (신라→실라, 칼날→칼랄)"""
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if not cur[2]:
            continue
        coda = cur[2][-1]
        if coda == "ㄴ" and nxt[0] == "ㄹ":
            cur[2] = cur[2][:-1] + ["ㄹ"]
        elif coda == "ㄹ" and nxt[0] == "ㄴ":
            nxt[0] = "ㄹ"


def pronounce_word(word: str) -> str:
    """
    공백·문장 기호를 포함하지 않는 한글 음절열을 발음형으로 변환한다.
    입력과 출력의 음절 수는 반드시 일치한다.
    """
    if not word:
        return word
    # [초성, 중성, 종성 리스트]의 가변 리스트로 분해
    syls: list[list] = [list(decompose(char)) for char in word]

    __apply_ui_rules(syls)
    __apply_palatalization(syls)
    __apply_h_rules(syls)
    __apply_liaison(syls)
    __apply_tensification(syls)
    __apply_coda_simplification(syls)
    __apply_nasalization(syls)
    __apply_lateralization(syls)
    # jyeo 규칙은 마지막에 적용한다: 표기상의 져/쳐 (가져)뿐 아니라, 구개음화 (붙여→부쳐)나
    # 격음화 (젖혀→저쳐)가 생성한 쳐까지 한꺼번에 단모음화하기 위함
    __apply_jyeo_rule(syls)

    result = "".join(compose(cho, jung, coda) for cho, jung, coda in syls)
    assert len(result) == len(word), f"Syllable count changed: {word} -> {result}"
    return result


def __apply_word_boundary(left: str, right: str) -> tuple[str, str]:
    """
    어절 경계 (공백 하나로 인접한 두 어절)의 음운 규칙을, 왼쪽 어절의 마지막 음절과
    오른쪽 어절의 첫 음절에 적용한다. 두 어절 모두 어절 내부 규칙이 적용된 상태
    (종성은 중화 완료·단일)임을 전제로 한다. 표준 발음법 제15항이
    "대표음으로 바꾸어서 연음한다"고 규정하므로 이 순서가 규정과 일치한다.
    음절 수는 보존된다.
    """
    lcho, ljung, lcoda = decompose(left)
    rcho, rjung, rcoda = decompose(right)
    if not lcoda:
        return left, right
    lc = lcoda[-1]
    # 1. 연음 (제15항): 받침 + 모음 시작 → 대표음을 다음 초성으로 이동
    #    ㅇ 받침 (/ŋ/)은 이동하지 않음
    if rcho == "ㅇ":
        if lc != "ㅇ":
            return compose(lcho, ljung, []), compose(lc, rjung, rcoda)
        return left, right
    # 2. 격음화 (제12항 붙임 준용): 장애음 받침 + ㅎ → 격음 합류 (못 해→모 태)
    if rcho == "ㅎ" and lc in ASPIRATION_MAP:
        return compose(lcho, ljung, []), compose(ASPIRATION_MAP[lc], rjung, rcoda)
    if lc in ("ㄱ", "ㄷ", "ㅂ"):
        # 3. 경음화 (제23항 준용): 장애음 받침 + 평음 (몇 개→멷 깨)
        if rcho in TENSIFICATION_MAP:
            return left, compose(TENSIFICATION_MAP[rcho], rjung, rcoda)
        # 4. 비음화 (제18항 붙임): 장애음 받침 + 비음 (밥 먹는다→밤 멍는다)
        if rcho in ("ㄴ", "ㅁ"):
            return compose(lcho, ljung, [NASALIZATION_MAP[lc]]), right
    # 5. 유음화 (제20항): ㄹ+ㄴ / ㄴ+ㄹ → ㄹㄹ (할 닐→할 릴)
    if lc == "ㄹ" and rcho == "ㄴ":
        return left, compose("ㄹ", rjung, rcoda)
    if lc == "ㄴ" and rcho == "ㄹ":
        return compose(lcho, ljung, ["ㄹ"]), right
    return left, right


def pronounce(text: str) -> str:
    """
    임의의 텍스트 안의 한글 음절열 (공백·문장 기호로 구분되는 단위별)을 발음형으로 변환한다.
    공백 하나로 인접한 어절 쌍에는 어절 경계 음운 규칙 (연음·격음화·경음화·
    비음화·유음화)도 적용된다. 문장 기호 등 공백 이외의 경계에서는 적용되지 않는다 (휴지).
    한글 이외의 문자는 위치·내용 모두 그대로 유지되며, 문자열 길이는 반드시 일치한다.
    """
    # (is_hangul_word, segment) 리스트로 분할하고, 어절별로 내부 규칙을 적용
    segments: list[list] = []
    word_buffer: list[str] = []
    for char in text:
        if is_hangul_syllable(char):
            word_buffer.append(char)
        else:
            if word_buffer:
                segments.append([True, pronounce_word("".join(word_buffer))])
                word_buffer = []
            segments.append([False, char])
    if word_buffer:
        segments.append([True, pronounce_word("".join(word_buffer))])

    # 어절 경계 규칙: 공백 정확히 하나로 인접한 어절 쌍에 왼쪽부터 순서대로 적용
    for i in range(len(segments) - 2):
        if segments[i][0] and not segments[i + 1][0] and segments[i + 1][1] == " " and segments[i + 2][0]:  # fmt: skip
            left_word, right_word = segments[i][1], segments[i + 2][1]
            new_left, new_right = __apply_word_boundary(left_word[-1], right_word[0])
            segments[i][1] = left_word[:-1] + new_left
            segments[i + 2][1] = new_right + right_word[1:]

    pronounced = "".join(seg for _, seg in segments)
    assert len(pronounced) == len(text)
    return pronounced

"""
한국어 텍스트 정규화.

정규화 후의 텍스트는 정확히 다음 문자들로만 구성된다:
- 한글 음절 (가-힣)
- 반각 스페이스
- `.` `,` `?` `!` `'` `-` (문장 기호, `…`는 `...`로 변환됨)

숫자는 그룹 단위 (만/억/조)를 고려한 한자어 수사 읽기의 한글로 변환된다.
알파벳은 한 글자씩 한국어 이름 (에이, 비, ...)으로 변환된다.
"""

import re
import unicodedata

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


# 기호류 정규화 맵 (일본어 구현을 따르되 한국어에 맞게 조정)
__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "⋯": "...",
    "·": ",",
    "、": ",",
    "„": "'",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "「": "'",
    "」": "'",
    "~": "-",
    "～": "-",
    "〜": "-",
    # 하이픈·대시류를 반각 하이픈으로 통일
    "˗": "-",
    "‐": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "⁃": "-",
    "−": "-",
    "⎯": "-",
    "⏤": "-",
    "─": "-",
    "━": "-",
    "⸺": "-",
    "⸻": "-",
}
__REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP))

# 알파벳 → 한국어 이름
__ALPHABET_MAP = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프",
    "g": "지", "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘",
    "m": "엠", "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알",
    "s": "에스", "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스",
    "y": "와이", "z": "제트",
}  # fmt: skip

# 숫자 읽기 (한자어 수사)
__SINO_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
__SMALL_UNITS = ["", "십", "백", "천"]
__GROUP_UNITS = ["", "만", "억", "조", "경"]

# 숫자 읽기 (고유어 수사)
## 단위명사 (조수사) 앞에서는 1-99를 고유어 관형형으로 읽는다: 3개 → 세 개
__NATIVE_TENS = {
    1: "열", 2: "스물", 3: "서른", 4: "마흔", 5: "쉰",
    6: "예순", 7: "일흔", 8: "여든", 9: "아흔",
}  # fmt: skip
## 관형형 (단위명사 바로 앞에서 쓰는 형태): 하나→한, 둘→두, 셋→세, 넷→네
__NATIVE_ONES_DET = {
    1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯",
    6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉",
}  # fmt: skip

# 고유어 수사로 읽는 단위명사 (조수사)
## 긴 것부터 먼저 매치해야 한다 (시간 → 시, 번째 → 번 보다 앞에)
__NATIVE_COUNTERS = [
    "번째", "시간", "가지", "군데", "그루", "마리", "사람", "송이",
    "켤레", "봉지", "상자", "조각", "포기", "그릇", "자루", "다발",
    "마디", "모금", "접시", "스푼", "갈래", "발짝",
    "개", "명", "살", "시", "번", "잔", "병", "대", "장", "권",
    "채", "척", "판", "곡", "줄", "벌", "알", "곳", "근", "발", "축", "쾌",
]  # fmt: skip

# 고유어로 읽을 것처럼 보이지만 한자어 수사로 읽는 단위 (「개」 등의 오매치 방지를 위해 먼저 매치)
__SINO_COUNTERS = ["개월", "개국", "개소", "개년", "인분", "번지", "분"]

__COUNTER_PATTERN = re.compile(
    # 서수 접두사 "제" (제3장, 제2회)가 붙는 숫자는 항상 한자어 수사로 읽으므로 제외한다
    r"(?<![0-9.제])([0-9]+)( ?)("
    + "|".join(__SINO_COUNTERS + __NATIVE_COUNTERS)
    + r")(?![0-9])"
)

# 통화·단위 기호
__CURRENCY_MAP = {"₩": "원", "$": "달러", "¥": "엔", "£": "파운드", "€": "유로"}
__CURRENCY_PATTERN = re.compile(r"([₩$¥£€])([0-9,.]*[0-9])")
__PERCENT_PATTERN = re.compile(r"([0-9,.]*[0-9])\s*%")

# 숫자 뒤에 붙는 단위 약어
## 대문자 약어는 단위가 아닌 경우가 흔하므로 (5G 통신 등) 리터 이외는 소문자만 인정한다
__UNIT_MAP = {
    "kg": "킬로그램", "mg": "밀리그램", "g": "그램", "t": "톤",
    "km": "킬로미터", "cm": "센티미터", "mm": "밀리미터", "m": "미터",
    "mL": "밀리리터", "ml": "밀리리터", "L": "리터", "l": "리터",
}  # fmt: skip
## 속도는 「시속/초속 N 단위」로 어순을 바꿔 읽는다
__SPEED_UNIT_MAP = {"km/h": ("시속", "킬로미터"), "m/s": ("초속", "미터")}
__UNIT_PATTERN = re.compile(
    # 알파벳 뒤의 숫자 (mp3 등)나 알파벳·숫자가 이어지는 약어 (3gb 등)는 단위로 보지 않는다
    r"(?<![A-Za-z.])([0-9]+(?:\.[0-9]+)?)\s*("
    + "|".join(sorted(list(__SPEED_UNIT_MAP) + list(__UNIT_MAP), key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


def __convert_unit(match: "re.Match[str]") -> str:
    """숫자 + 단위 약어 매치를 한글 단위명으로 변환한다"""
    number, unit = match.group(1), match.group(2)
    if unit in __SPEED_UNIT_MAP:
        prefix, unit_name = __SPEED_UNIT_MAP[unit]
        return prefix + " " + number + unit_name
    return number + __UNIT_MAP[unit]


# 월 이름 특례 (표준 관용 읽기): 6월 → 유월, 10월 → 시월
__MONTH_SPECIAL_PATTERN = re.compile(r"(?<![0-9.])(6|10)월")
__MONTH_SPECIAL_MAP = {"6": "유월", "10": "시월"}

# 알파벳 바로 뒤의 한 자리 숫자는 영어식으로 읽는 관례를 따른다: F1 → 에프원, mp3 → 엠피쓰리
## 숫자 뒤에 알파벳이 오는 경우(5G → 오지)와 여러 자리(F16 → 에프십육)는 한자어 읽기가 관례
__ENGLISH_DIGIT_PATTERN = re.compile(r"(?<=[A-Za-z])[0-9](?![0-9])")
__ENGLISH_DIGIT_NAMES = ["제로", "원", "투", "쓰리", "포", "파이브", "식스", "세븐", "에이트", "나인"]  # fmt: skip

# 전화번호 (하이픈 구분 숫자열)·안내번호는 낱자로 읽는다: 010-1234-5678 → 공일공 일이삼사 오육칠팔
## 15-20 같은 범위 표기를 피하기 위해 하이픈 뒤 그룹은 3-4자리만 인정한다
__PHONE_PATTERN = re.compile(r"(?<![0-9.-])[0-9]{2,4}(?:-[0-9]{3,4})+(?![0-9.-])")
__HOTLINE_PATTERN = re.compile(r"(?<![0-9.-])(112|114|119)(?![0-9.-])")
__DIGIT_NAMES = "공일이삼사오육칠팔구"

# 두 자리 연도의 년생·학번은 낱자로 읽는다: 78년생 → 칠팔년생 (네 자리 연도는 자릿수 읽기)
__TWO_DIGIT_YEAR_PATTERN = re.compile(r"(?<![0-9.])([0-9]{2})( ?)(년생|학번)")

# 0으로 시작하는 숫자열은 자릿수 읽기가 성립하지 않으므로 (전화번호·코드류) 낱자로 읽는다
__LEADING_ZERO_PATTERN = re.compile(r"(?<![0-9.])0[0-9]+(?![0-9.])")

# 점수·비율의 「숫자 대 숫자」는 양쪽 다 한자어로 읽는다: 11 대 8 → 십일 대 팔
__SCORE_PATTERN = re.compile(r"(?<![0-9.])([0-9]+)( ?대 ?)([0-9]+)(?![0-9.])")

# 번호 문맥의 「N번」은 한자어로 읽는다: 3번 출구 → 삼번 출구 (횟수 의미면 고유어: 세 번)
## 동철이의어라 확실한 문맥만 처리: 번호가 확실한 후속 명사 allowlist + 다이얼 동사 (누르다).
## KSS 실험 (2026-07-22): kiwi 품사 기반 판별은 「3번 반복」 같은 하다-명사에서 오발동해 채택하지 않음
__BEON_NUMBER_PATTERN = re.compile(
    r"(?<![0-9.제])([0-9]+)( ?번 ?)"
    r"(?=출구|버스|채널|터미널|창구|게이트|승강장|플랫폼|문제|노선|좌석|테이블|트랙|[을이]? ?누르|[을이]? ?눌러)"
)


def __read_digits(digits: str) -> str:
    """숫자열을 낱자 읽기로 변환한다 (0은 공): "112" → "일일이" """
    return "".join(__DIGIT_NAMES[int(d)] for d in digits)
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

# 정규화 후에 남기는 것이 허용된 문자 이외를 제거하는 패턴
__CLEANUP_PATTERN = re.compile(r"[^가-힣 " + "".join(re.escape(p) for p in PUNCTUATIONS) + r"]+")  # fmt: skip


def __read_four_digits(num: int) -> str:
    """0-9999 정수를 한자어 수사로 읽는다 (그룹 내부에서는 일을 생략: 1234→천이백삼십사)"""
    if num == 0:
        return ""
    result = ""
    for i, digit in enumerate(reversed(str(num))):
        d = int(digit)
        if d == 0:
            continue
        unit = __SMALL_UNITS[i]
        # 십/백/천 앞의 일은 읽지 않는다
        digit_reading = "" if (d == 1 and i > 0) else __SINO_DIGITS[d]
        result = digit_reading + unit + result
    return result


def read_number(num_str: str) -> str:
    """
    숫자 문자열 (정수 또는 소수)을 한자어 수사 읽기의 한글로 변환한다.
    예: "2026" → "이천이십육", "3.14" → "삼점일사", "10000" → "만"
    """
    if "." in num_str:
        int_part, frac_part = num_str.split(".", 1)
    else:
        int_part, frac_part = num_str, ""

    int_value = int(int_part) if int_part else 0
    if int_value == 0:
        int_reading = "영"
    else:
        # (그룹 값, 그룹 단위 인덱스)를 하위부터 수집
        raw_groups: list[tuple[int, int]] = []
        group_index = 0
        while int_value > 0:
            group = int_value % 10000
            if group > 0:
                raw_groups.append((group, group_index))
            int_value //= 10000
            group_index += 1
        groups: list[str] = []
        top_index = raw_groups[-1][1]
        for group, index in raw_groups:
            reading = __read_four_digits(group)
            # 일의 생략은 최상위 그룹만 (10000→만, 100010000→억일만)
            # 중간 그룹까지 생략하면 "억만"처럼 다른 수로 들려버린다
            if group == 1 and index > 0 and index == top_index:
                reading = ""
            groups.append(reading + __GROUP_UNITS[index])
        int_reading = "".join(reversed(groups))

    if frac_part:
        frac_reading = "점" + "".join(__SINO_DIGITS[int(d)] if d != "0" else "영" for d in frac_part)  # fmt: skip
    else:
        frac_reading = ""

    return int_reading + frac_reading


def read_number_native(n: int) -> str:
    """
    1-99 정수를 고유어 수사의 관형형으로 읽는다.
    예: 1 → "한", 3 → "세", 20 → "스무", 21 → "스물한", 35 → "서른다섯"
    """
    assert 1 <= n <= 99
    if n == 20:
        return "스무"
    tens, ones = divmod(n, 10)
    result = ""
    if tens:
        result += __NATIVE_TENS[tens]
    if ones:
        result += __NATIVE_ONES_DET[ones]
    return result


def __convert_counter(match: "re.Match[str]") -> str:
    """숫자 + 단위명사 매치를 적절한 읽기로 변환한다"""
    number, space, counter = match.group(1), match.group(2), match.group(3)
    n = int(number)
    # 한자어 수사로 읽는 단위는 그대로 남긴다 (후단의 범용 변환에서 한자어 수사가 됨)
    if counter in __SINO_COUNTERS:
        return match.group(0)
    # 고유어로 읽는 것은 1-99만 (100 이상은 한자어: 100개 → 백 개)
    if not (1 <= n <= 99):
        return match.group(0)
    # 십 단위 + 「대」는 연령대(20대 여성 → 이십대)로 읽는 것이 일반적이므로 한자어로 남긴다
    if counter == "대" and n % 10 == 0:
        return match.group(0)
    # 1번째 → 첫 번째
    if counter == "번째" and n == 1:
        return "첫" + space + counter
    return read_number_native(n) + space + counter


def normalize_text(text: str) -> str:
    """
    한국어 텍스트를 정규화한다.
    결과는 한글 음절·반각 스페이스·문장 기호 (`!?.,'-`)로만 구성된다.
    """
    # 유니코드 정규화 (전각 숫자→반각 등. 한글 음절은 조합된 상태로 유지됨)
    res = unicodedata.normalize("NFKC", text)

    # 기호류 정규화
    res = __REPLACE_PATTERN.sub(lambda m: __REPLACE_MAP[m.group()], res)

    # 자릿수 구분 쉼표 제거 (1,000 → 1000)
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m.group().replace(",", ""), res)

    # 통화·퍼센트 표기 변환 (₩1000 → 1000원, 50% → 50퍼센트)
    res = __CURRENCY_PATTERN.sub(lambda m: m.group(2) + __CURRENCY_MAP[m.group(1)], res)
    res = __PERCENT_PATTERN.sub(lambda m: m.group(1) + "퍼센트", res)

    # 단위 약어 변환 (5kg → 5킬로그램, 80km/h → 시속 80킬로미터)
    res = __UNIT_PATTERN.sub(__convert_unit, res)

    # 월 이름 특례 (6월 → 유월, 10월 → 시월)
    res = __MONTH_SPECIAL_PATTERN.sub(lambda m: __MONTH_SPECIAL_MAP[m.group(1)], res)

    # 알파벳 뒤의 한 자리 숫자는 영어식으로 (F1 → F원, 뒤의 알파벳 변환을 거쳐 에프원)
    res = __ENGLISH_DIGIT_PATTERN.sub(lambda m: __ENGLISH_DIGIT_NAMES[int(m.group())], res)

    # 전화번호·안내번호는 낱자 읽기로 (010-1234-5678 → 공일공 일이삼사 오육칠팔, 112 → 일일이)
    res = __PHONE_PATTERN.sub(lambda m: " ".join(__read_digits(g) for g in m.group().split("-")), res)
    res = __HOTLINE_PATTERN.sub(lambda m: __read_digits(m.group(1)), res)
    res = __TWO_DIGIT_YEAR_PATTERN.sub(lambda m: __read_digits(m.group(1)) + m.group(2) + m.group(3), res)  # fmt: skip
    res = __LEADING_ZERO_PATTERN.sub(lambda m: __read_digits(m.group()), res)

    # 점수의 「숫자 대 숫자」는 양쪽 다 한자어로 (11 대 8 → 십일 대 팔)
    res = __SCORE_PATTERN.sub(lambda m: read_number(m.group(1)) + m.group(2) + read_number(m.group(3)), res)  # fmt: skip

    # 번호 문맥의 「N번」은 한자어로 (3번 출구 → 삼번 출구)
    res = __BEON_NUMBER_PATTERN.sub(lambda m: read_number(m.group(1)) + m.group(2), res)

    # 단위명사 앞의 숫자를 고유어 수사로 변환 (3개 → 세개, 3시 → 세시)
    res = __COUNTER_PATTERN.sub(__convert_counter, res)

    # 남은 숫자를 한자어 수사 읽기의 한글로 변환
    res = __NUMBER_PATTERN.sub(lambda m: read_number(m.group()), res)

    # 알파벳을 한국어 이름으로 변환
    res = re.sub(r"[A-Za-z]", lambda m: __ALPHABET_MAP[m.group().lower()], res)

    # 허용되지 않은 문자 제거
    res = __CLEANUP_PATTERN.sub("", res)

    # 연속 스페이스를 하나로 합치고 앞뒤 공백 제거
    res = re.sub(r" +", " ", res).strip()

    # 연속 문장 기호 정리 (예: "....." → "...")
    res = re.sub(r"[.]{4,}", "...", res)
    res = re.sub(r"([!?,\-'])\1+", r"\1", res)

    return res

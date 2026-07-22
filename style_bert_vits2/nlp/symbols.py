# Punctuations
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# Punctuations and special tokens
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

# Padding
PAD = "_"

# Chinese symbols
ZH_SYMBOLS = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
NUM_ZH_TONES = 6

# Japanese
JP_SYMBOLS = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
NUM_JP_TONES = 2

# English
EN_SYMBOLS = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
NUM_EN_TONES = 4

# Korean
## 音素は Unicode Hangul Jamo (初声 U+1100-, 中声 U+1161-, 終声 U+11A8-) をそのまま記号として用いる
## 初声の "ㅇ" (無音) は音素として出力されないため含めない
## 終声は標準発音法の中和 (음절의 끝소리 규칙) 適用後の 7 종성のみ
KO_INITIALS = [
    "ᄀ",  # ᄀ
    "ᄁ",  # ᄁ
    "ᄂ",  # ᄂ
    "ᄃ",  # ᄃ
    "ᄄ",  # ᄄ
    "ᄅ",  # ᄅ
    "ᄆ",  # ᄆ
    "ᄇ",  # ᄇ
    "ᄈ",  # ᄈ
    "ᄉ",  # ᄉ
    "ᄊ",  # ᄊ
    "ᄌ",  # ᄌ
    "ᄍ",  # ᄍ
    "ᄎ",  # ᄎ
    "ᄏ",  # ᄏ
    "ᄐ",  # ᄐ
    "ᄑ",  # ᄑ
    "ᄒ",  # ᄒ
]
KO_VOWELS = [chr(code) for code in range(0x1161, 0x1176)]  # ᅡ-ᅵ (21 vowels)
KO_FINALS = [
    "ᆨ",  # ᆨ
    "ᆫ",  # ᆫ
    "ᆮ",  # ᆮ
    "ᆯ",  # ᆯ
    "ᆷ",  # ᆷ
    "ᆸ",  # ᆸ
    "ᆼ",  # ᆼ
]
KO_SYMBOLS = KO_INITIALS + KO_VOWELS + KO_FINALS
NUM_KO_TONES = 1

# Combine all symbols
## 既存モデルとの互換性を保つため、韓国語の音素は既存のシンボルテーブルの末尾に追加する
## (NORMAL_SYMBOLS に混ぜてソートすると既存シンボルのインデックスがずれてしまう)
NORMAL_SYMBOLS = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
SYMBOLS = [PAD] + NORMAL_SYMBOLS + PUNCTUATION_SYMBOLS + KO_SYMBOLS
SIL_PHONEMES_IDS = [SYMBOLS.index(i) for i in PUNCTUATION_SYMBOLS]

# Combine all tones
## 韓国語も末尾に追加することで既存言語のトーン ID を保持する
NUM_TONES = NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES + NUM_KO_TONES

# Language maps
LANGUAGE_ID_MAP = {"ZH": 0, "JP": 1, "EN": 2, "KO": 3}
NUM_LANGUAGES = len(LANGUAGE_ID_MAP.keys())

# Language tone start map
LANGUAGE_TONE_START_MAP = {
    "ZH": 0,
    "JP": NUM_ZH_TONES,
    "EN": NUM_ZH_TONES + NUM_JP_TONES,
    "KO": NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES,
}


if __name__ == "__main__":
    a = set(ZH_SYMBOLS)
    b = set(EN_SYMBOLS)
    print(sorted(a & b))

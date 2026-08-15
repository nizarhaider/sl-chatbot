"""Language detection and text normalization used only at the speech boundary."""

from __future__ import annotations

import re

SINHALA_ONES = (
    "බිංදුව",
    "එක",
    "දෙක",
    "තුන",
    "හතර",
    "පහ",
    "හය",
    "හත",
    "අට",
    "නවය",
)
SINHALA_SMALL = {
    10: "දහය",
    11: "එකොළහ",
    12: "දොළහ",
    13: "දහතුන",
    14: "දාහතර",
    15: "පහළොව",
    16: "දහසය",
    17: "දාහත",
    18: "දහඅට",
    19: "දහනවය",
}
SINHALA_TENS = {
    2: "විසි",
    3: "තිස්",
    4: "හතළිස්",
    5: "පනස්",
    6: "හැට",
    7: "හැත්තෑ",
    8: "අසූ",
    9: "අනූ",
}
SINHALA_HUNDREDS = (
    "",
    "එකසිය",
    "දෙසිය",
    "තුන්සිය",
    "හාරසිය",
    "පන්සිය",
    "හයසිය",
    "හත්සිය",
    "අටසිය",
    "නවසිය",
)

PLACE_NAMES = {
    "si": {
        "Athurugiriya": "අතුරුගිරිය",
        "Battaramulla": "බත්තරමුල්ල",
        "Colombo": "කොළඹ",
        "Dehiwala": "දෙහිවල",
        "Galle": "ගාල්ල",
        "Homagama": "හෝමාගම",
        "Ja-Ela": "ජා ඇල",
        "Kadawatha": "කඩවත",
        "Kandy": "මහනුවර",
        "Kottawa": "කොට්ටාව",
        "Kurunegala": "කුරුණෑගල",
        "Maharagama": "මහරගම",
        "Malabe": "මාලබේ",
        "Matara": "මාතර",
        "Mount Lavinia": "ගල්කිස්ස",
        "Negombo": "මීගමුව",
        "Nugegoda": "නුගේගොඩ",
        "Piliyandala": "පිළියන්දල",
        "Rajagiriya": "රාජගිරිය",
    },
    "ta": {
        "Battaramulla": "பத்தரமுல்லை",
        "Colombo": "கொழும்பு",
        "Dehiwala": "தெஹிவளை",
        "Galle": "காலி",
        "Kandy": "கண்டி",
        "Kurunegala": "குருநாகல்",
        "Malabe": "மாலபே",
        "Matara": "மாத்தறை",
        "Negombo": "நீர்கொழும்பு",
        "Nugegoda": "நுகேகொட",
        "Piliyandala": "பிலியந்தலை",
    },
}

PROPERTY_WORDS = re.compile(
    r"\b(?:properties|listings|proyes)\b|ප්‍රොපටි|පොපටි|පොබ්ඩිස්|ப்ராப்பர்ட்டிகள்|சொத்துகள்",
    re.IGNORECASE,
)
PROPERTY_SPECIFICS = re.compile(
    r"\b(?:apartment|house|villa|land|bedrooms?|budget|price|near|with|in|where|locations?|areas?)\b|"
    + "|".join(
        re.escape(name)
        for language_names in PLACE_NAMES.values()
        for pair in language_names.items()
        for name in pair
    )
    + r"|නිදන|කාමර|මිල|ළඟ|ප්‍රදේශ|කොහෙද|வீடு|வில்லா|காணி|படுக்கையறை|விலை|பகுதி|எங்கே",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    if re.search(r"[\u0D80-\u0DFF]", text):
        return "si"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    choice = text.strip().casefold()
    if choice in {"sinhala", "sinhalese"}:
        return "si"
    if choice in {"tamil", "தமிழ்"}:
        return "ta"
    return "en"


def selected_language(text: str) -> str | None:
    """Return a language only when the caller is clearly selecting it."""
    words = re.findall(r"[A-Za-z\u0B80-\u0BFF\u0D80-\u0DFF]+", text.casefold())
    if not words:
        return None
    choices = {
        "en": {"english"},
        "si": {"sinhala", "sinhalese", "සිංහල"},
        "ta": {"tamil", "தமிழ்"},
    }
    for language, names in choices.items():
        if all(word in names for word in words):
            return language
    return None


def is_broad_property_request(text: str) -> bool:
    """Return true when inventory was requested without a useful preference."""
    return bool(PROPERTY_WORDS.search(text)) and not PROPERTY_SPECIFICS.search(text)


def normalize_for_tts(text: str, language: str | None = None) -> tuple[str, str]:
    language = language or detect_language(text)
    spoken = re.sub(r"\s+", " ", text).strip().rstrip(",;:")
    for written, pronunciation in PLACE_NAMES.get(language, {}).items():
        spoken = re.sub(
            rf"\b{re.escape(written)}\b", pronunciation, spoken, flags=re.IGNORECASE
        )
    if language == "si":
        spoken = re.sub(
            r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?",
            lambda match: sinhala_number(match.group()),
            spoken,
        )
    # A neutral full stop gives the cloned voice a falling cadence. The displayed
    # transcript remains unchanged, so this affects delivery rather than meaning.
    spoken = spoken.rstrip(".!?") + "."
    return spoken, language


def sinhala_number(raw: str) -> str:
    cleaned = raw.replace(",", "")
    whole, dot, fraction = cleaned.partition(".")
    number = int(whole)
    if number >= 1_000_000 and number % 100_000 == 0:
        millions = number // 1_000_000
        remainder = number % 1_000_000
        words = f"මිලියන {_sinhala_under_thousand(millions)}"
        if remainder:
            words += f" දශම {SINHALA_ONES[remainder // 100_000]}"
    elif number < 1_000:
        words = _sinhala_under_thousand(number)
    else:
        words = " ".join(SINHALA_ONES[int(digit)] for digit in str(number))
    if dot and fraction:
        words += " දශම " + " ".join(SINHALA_ONES[int(digit)] for digit in fraction)
    return words


def _sinhala_under_thousand(number: int) -> str:
    if number < 10:
        return SINHALA_ONES[number]
    if number < 20:
        return SINHALA_SMALL[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return SINHALA_TENS[tens] + (f" {SINHALA_ONES[ones]}" if ones else "")
    hundreds, remainder = divmod(number, 100)
    return SINHALA_HUNDREDS[hundreds] + (
        f" {_sinhala_under_thousand(remainder)}" if remainder else ""
    )

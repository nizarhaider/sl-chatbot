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
PROPERTY_LOCATION_REQUEST = re.compile(
    r"\b(?:where|locations?|which\s+areas?)\b|මොන\s+(?:locations?|ප්‍රදේශ)|"
    r"ප්‍රදේශ.*කොහෙද|எந்த\s+(?:இட|பகுதி)|எங்கே",
    re.IGNORECASE,
)
PROPERTY_BROAD_INVENTORY = re.compile(r"ඔයාලා\s+ළඟ\s+තියෙන|තියෙන\s+ඒවා\s+පෙන්නන්න")
PLACE_ALIASES = {
    "කුරුණෑග": "Kurunegala",
    "රාජිය": "Rajagiriya",
    # Frequent Whisper renderings of spoken "Piliyandala" from production calls.
    "පිලියන්දල": "Piliyandala",
    "පෙන්නැඳිලා": "Piliyandala",
    "කෙළියන්ද": "Piliyandala",
}


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
    matches = [
        language
        for language, names in choices.items()
        if any(word in names for word in words)
    ]
    if len(matches) != 1:
        return None
    language = matches[0]
    if all(word in choices[language] for word in words):
        return language
    if (
        len(words) <= 7
        and not PROPERTY_WORDS.search(text)
        and not PROPERTY_SPECIFICS.search(text)
    ):
        return language
    return None


def is_broad_property_request(text: str) -> bool:
    """Return true when inventory was requested without a useful preference."""
    return bool(PROPERTY_WORDS.search(text)) and bool(
        PROPERTY_BROAD_INVENTORY.search(text) or not PROPERTY_SPECIFICS.search(text)
    )


def is_property_location_request(text: str) -> bool:
    """Return true for an explicit request to list inventory locations."""
    return bool(PROPERTY_WORDS.search(text) and PROPERTY_LOCATION_REQUEST.search(text))


def known_location(text: str) -> str | None:
    """Return the English inventory location named by the caller."""
    folded = text.casefold()
    for names in PLACE_NAMES.values():
        for english, localized in names.items():
            if english.casefold() in folded or localized in text:
                return english
    return next(
        (place for alias, place in PLACE_ALIASES.items() if alias in text), None
    )


def tool_acknowledgement(language: str, name: str, arguments: dict) -> str:
    """Create one contextual acknowledgement for an actual tool call."""
    if name == "list_property_locations":
        return {
            "en": "Sure, let me check the available locations.",
            "si": "හරි, properties තියෙන ප්‍රදේශ බලලා කියන්නම්.",
            "ta": "சரி, properties உள்ள பகுதிகளைப் பார்த்துச் சொல்கிறேன்.",
        }[language]
    if name == "book_appointment":
        return {
            "en": "Sure, I'll book that viewing now.",
            "si": "හරි, ඒ viewing එක දැන් book කරන්නම්.",
            "ta": "சரி, அந்த viewing-ஐ இப்போது book செய்கிறேன்.",
        }[language]

    location = str(arguments.get("location") or "").strip()
    location = PLACE_NAMES.get(language, {}).get(location, location)
    kind = str(arguments.get("property_type") or "").strip().casefold()
    if language == "si":
        if location and kind:
            return f"හරි, {location} ප්‍රදේශයේ {kind} තියෙනවද බලන්නම්."
        if location:
            return f"හරි, {location} ප්‍රදේශයේ තියෙන properties බලන්නම්."
        return f"හරි, තියෙන {kind or 'property'} විස්තර බලලා කියන්නම්."
    if language == "ta":
        if location and kind:
            return f"சரி, {location} பகுதியில் உள்ள {kind} விவரங்களைப் பார்க்கிறேன்."
        if location:
            return f"சரி, {location} பகுதியில் உள்ள properties-ஐ பார்க்கிறேன்."
        return f"சரி, உள்ள {kind or 'property'} விவரங்களைப் பார்க்கிறேன்."
    plural = {"apartment": "apartments", "house": "houses", "villa": "villas"}.get(
        kind, kind
    )
    if location and plural:
        return f"Sure, let me check for {plural} in {location}."
    if location:
        return f"Sure, I'll check what's available in {location}."
    return (
        f"Give me a moment to check the available {plural or 'property information'}."
    )


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

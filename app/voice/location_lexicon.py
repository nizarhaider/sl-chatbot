"""Sri Lankan place names used at the speech and property-search boundaries."""

from __future__ import annotations

import re


# Keep this deliberately small and curated. These are common service areas and
# the aliases used by the current property inventory; an uncertain location must
# still be clarified rather than fuzzy-matched to a different town.
LOCATION_LEXICON: dict[str, tuple[str, tuple[str, ...]]] = {
    "Battaramulla": ("බත්තරමුල්ල", ("බත්තරමුල්ල", "බත්තරමුල්ලේ", "battaramulla")),
    "Malabe": ("මාලඹේ", ("මාලඹේ", "මාලබේ", "malabe")),
    "Dehiwala": ("දෙහිවල", ("දෙහිවල", "දෙහිවලේ", "dehiwala")),
    "Nugegoda": ("නුගේගොඩ", ("නුගේගොඩ", "නුගේගොඩේ", "nugegoda")),
    "Rajagiriya": ("රාජගිරිය", ("රාජගිරිය", "rajagiriya")),
    "Kotte": ("කෝට්ටේ", ("කෝට්ටේ", "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ", "kotte")),
    "Maharagama": ("මහරගම", ("මහරගම", "maharagama")),
    "Kottawa": ("කොට්ටාව", ("කොට්ටාව", "kottawa")),
    "Piliyandala": ("පිළියන්දල", ("පිළියන්දල", "piliyandala")),
    "Boralesgamuwa": ("බොරලැස්ගමුව", ("බොරලැස්ගමුව", "boralesgamuwa")),
    "Kadawatha": ("කඩවත", ("කඩවත", "kadawatha")),
    "Kelaniya": ("කැලණිය", ("කැලණිය", "kelaniya")),
    "Wattala": ("වත්තල", ("වත්තල", "wattala")),
    "Ja-Ela": ("ජා ඇල", ("ජා ඇල", "ජාඇල", "ja-ela", "ja ela")),
    "Negombo": ("මීගමුව", ("මීගමුව", "negombo")),
    "Gampaha": ("ගම්පහ", ("ගම්පහ", "gampaha")),
    "Colombo": ("කොළඹ", ("කොළඹ", "colombo")),
    "Kandy": ("මහනුවර", ("මහනුවර", "kandy")),
    "Galle": ("ගාල්ල", ("ගාල්ල", "galle")),
    "Matara": ("මාතර", ("මාතර", "matara")),
    "Kurunegala": ("කුරුණෑගල", ("කුරුණෑගල", "kurunegala")),
    "Anuradhapura": ("අනුරාධපුර", ("අනුරාධපුර", "anuradhapura")),
}


def canonicalize_location_mentions(text: str) -> str:
    """Use canonical inventory names for known Sinhala/Latin place aliases."""
    normalized = text
    aliases = [
        (alias, canonical)
        for canonical, (_, values) in LOCATION_LEXICON.items()
        for alias in values
    ]
    for alias, canonical in sorted(aliases, key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(re.escape(alias), canonical, normalized, flags=re.IGNORECASE)
    return normalized


def spoken_location_text(text: str) -> str:
    """Replace known English place names with their Sinhala spoken forms."""
    spoken = text
    for canonical, (sinhala, _) in sorted(LOCATION_LEXICON.items(), key=lambda item: len(item[0]), reverse=True):
        spoken = re.sub(rf"\b{re.escape(canonical)}\b", sinhala, spoken, flags=re.IGNORECASE)
    return spoken


def location_search_terms(location: str) -> str:
    """Return canonical and Sinhala aliases to embed with a property record."""
    sinhala, aliases = LOCATION_LEXICON.get(location, (location, (location,)))
    return " ".join(dict.fromkeys((location, sinhala, *aliases)))


ASR_LOCATION_HINTS = ", ".join(
    value
    for canonical, (sinhala, _) in LOCATION_LEXICON.items()
    for value in (canonical, sinhala)
)

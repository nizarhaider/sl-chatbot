from app.voice.agent import VoiceAgent
from app.voice.location_lexicon import (
    canonicalize_location_mentions,
    location_search_terms,
    spoken_location_text,
)
from app.voice.pinecone_store import _property_record


def test_known_sinhala_location_is_canonicalized_for_search() -> None:
    assert canonicalize_location_mentions("මට බත්තරමුල්ලේ apartment එකක් ඕනේ") == (
        "මට Battaramulla apartment එකක් ඕනේ"
    )
    assert canonicalize_location_mentions("ආඤ්ඤුගය ගොඩ පැත්තෙන්") == "Angoda පැත්තෙන්"


def test_tts_uses_sinhala_pronunciation_for_known_locations() -> None:
    agent = VoiceAgent.__new__(VoiceAgent)
    assert agent._prepare_tts_text("Capital Heights is in Battaramulla.") == (
        "Capital Heights is in බත්තරමුල්ල."
    )
    assert spoken_location_text("Dehiwala and Malabe") == "දෙහිවල and මාලඹේ"


def test_property_embeddings_include_sinhala_location_aliases() -> None:
    record = _property_record({
        "property_id": "property-1",
        "name": "Capital Heights",
        "location": "Battaramulla",
        "property_type": "apartment",
        "bedrooms": 2,
        "price_lkr": 35_000_000,
        "price_millions": 35,
        "price_label": "LKR 35 million",
        "details": "Near Parliament Road.",
    })

    assert "බත්තරමුල්ල" in location_search_terms("Battaramulla")
    assert "බත්තරමුල්ල" in record["content"]

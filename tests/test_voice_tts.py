from app.voice.tts import _language_for_text


def test_tts_language_tracks_each_sentence_script() -> None:
    assert _language_for_text("Please say English.") == "en"
    assert _language_for_text("කරුණාකර සිංහල කියන්න.") == "si"
    assert _language_for_text("தயவுசெய்து தமிழ் என்று சொல்லுங்கள்.") == "ta"


def test_tts_uses_language_agnostic_mode_for_code_switching() -> None:
    assert _language_for_text("මම Homelands Properties වෙනුවෙන් කතා කරනවා.") is None


def test_tts_language_falls_back_for_non_words() -> None:
    assert _language_for_text("123...", default="si") == "si"

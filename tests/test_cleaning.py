from prepare_data import clean_brand_text, clean_customer_text, is_mostly_english


def test_customer_text_masks_urls_and_handles():
    assert clean_customer_text("@SpotifyCares @123 help https://t.co/abc  now") == "@user help <url> now"


def test_brand_reply_strips_leading_handles_signature_and_dm_link():
    raw = "@123456 Hey there! Can you DM us your email? We'll take a look backstage /NQ https://t.co/ldFdZRiNAt"
    assert clean_brand_text(raw) == "Hey there! Can you DM us your email? We'll take a look backstage"


def test_brand_reply_keeps_content_links():
    raw = "@1 Hey! There's info about Spotify content here: https://t.co/x"
    assert clean_brand_text(raw) == "Hey! There's info about Spotify content here: <url>"


def test_english_filter():
    assert is_mostly_english("my songs keep skipping on android")
    assert not is_mostly_english("音楽が再生されません助けてください")
    assert not is_mostly_english("ok")

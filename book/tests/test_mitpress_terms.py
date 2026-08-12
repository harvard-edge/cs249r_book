from book.cli.checks.mitpress_terms import find_in_text


def test_limited_data_set_preserves_hipaa_legal_term():
    assert find_in_text("HIPAA permits a limited data set with an agreement.") == []


def test_general_data_set_still_uses_canonical_spelling():
    hits = find_in_text("The training data set is versioned.")

    assert [(hit.match, hit.replacement) for hit in hits] == [("data set", "dataset")]

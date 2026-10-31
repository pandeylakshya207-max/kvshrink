import pytest

from kvshrink.longbench import (bootstrap_ci, middle_truncate, normalize_answer, qa_f1,
                                retrieval_score)


def test_normalize_answer():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("An apple, a pear.") == "apple pear"


def test_qa_f1_exact_partial_and_max_over_answers():
    assert qa_f1("The Eiffel Tower", ["eiffel tower"]) == 1.0
    assert qa_f1("Paris", ["London"]) == 0.0
    # pred {paris, france} vs gold {paris}: p=1/2, r=1 -> f1=2/3
    assert abs(qa_f1("Paris, France", ["Paris"]) - 2 / 3) < 1e-9
    assert qa_f1("Paris", ["London", "Paris"]) == 1.0
    assert qa_f1("", ["Paris"]) == 0.0


def test_retrieval_score():
    assert retrieval_score("Paragraph 12", ["Paragraph 12"]) == 1.0
    assert retrieval_score("Paragraph 3", ["Paragraph 12"]) == 0.0
    assert retrieval_score("Paragraph 12 or Paragraph 3", ["Paragraph 12"]) == 0.5
    assert retrieval_score("no idea", ["Paragraph 12"]) == 0.0
    with pytest.raises(ValueError):
        retrieval_score("1", ["twelve"])


def test_middle_truncate():
    ids = list(range(10))
    assert middle_truncate(ids, 20) == ids
    assert middle_truncate(ids, 4) == [0, 1, 8, 9]
    assert middle_truncate(ids, 5) == [0, 1, 7, 8, 9]


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci([1.0] * 20)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    m, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert m == 0.5 and lo < 0.5 < hi and hi - lo < 0.3
    assert bootstrap_ci([0.0, 1.0] * 50) == bootstrap_ci([0.0, 1.0] * 50)  # deterministic

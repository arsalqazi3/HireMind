from evaluation.baselines import tfidf_logreg_category


def test_tfidf_baseline_learns_categories():
    texts = ["python sql software", "python developer software", "baking cakes kitchen", "chef kitchen baking"]
    labels = ["IT", "IT", "CHEF", "CHEF"]
    assert tfidf_logreg_category(texts, labels, ["python software", "chef baking"]) == ["IT", "CHEF"]
    assert tfidf_logreg_category(texts, labels, []) == []

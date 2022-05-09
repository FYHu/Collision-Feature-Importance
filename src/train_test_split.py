import sklearn.model_selection


# Useful reference:
# `sklearn.model_selection.train_test_split`:
# <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
def train_test_split(features, labels, test_ratio=0.2):
    return sklearn.model_selection.train_test_split(
        features,
        labels,
        random_state=42,
        shuffle=True,
        stratify=None,
        test_size=test_ratio,
    )

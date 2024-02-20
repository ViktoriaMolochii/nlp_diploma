from ua_gec import Corpus
import random
from sklearn.model_selection import train_test_split

from usefull_methods import get_real_labels, list_of_classes, calc_accuracy_precision


class RandomBaselineModel:
    def __init__(self, list_of_classes):
        self.list_of_classes = list_of_classes

    def predict(self, ann):
        return random.choice(self.list_of_classes)


def main():
    corpus = Corpus(partition="train", annotation_layer="gec-only")
    train_corpus, test_corpus = train_test_split(corpus.get_documents(), test_size=0.25, random_state=42)

    modelRandom = RandomBaselineModel(list_of_classes)

    y_test = get_real_labels(test_corpus, list_of_classes)

    y_pred = []
    for doc in test_corpus:
        for ann in doc.annotated.iter_annotations():
            y_pred.append(list_of_classes.index(modelRandom.predict(ann)))

    acc, prec = calc_accuracy_precision(y_test, y_pred, zero_division=True)


if __name__ == "__main__":
    main()
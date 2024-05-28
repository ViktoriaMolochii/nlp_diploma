import random
from usefull_methods import get_real_labels, list_of_classes, calc_metrics
from corpus import Corpus
import pandas as pd
class RandomBaselineModel:
    def __init__(self, list_of_classes):
        self.list_of_classes = list_of_classes

    def predict(self, ann):
        return random.choice(self.list_of_classes)


def main():
    train_corpus = Corpus(partition="train", annotation_layer="gec-only")
    test_corpus = Corpus(partition="test", annotation_layer="gec-only")
    modelRandom = RandomBaselineModel(list_of_classes)
    y_test = get_real_labels(test_corpus, list_of_classes)

    y_pred = []
    for doc in test_corpus:
        for ann in doc.annotated.iter_annotations():
            y_pred.append(list_of_classes.index(modelRandom.predict(ann)))

    acc, prec, recall, f1 = calc_metrics(y_test, y_pred, zero_division=True)


if __name__ == "__main__":
    main()
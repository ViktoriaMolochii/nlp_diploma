from ua_gec import Corpus
from sklearn.model_selection import train_test_split

from usefull_methods import list_of_classes, calc_accuracy_precision, get_real_labels


class MajorityBaselineModel:
    def __init__(self, list_of_classes):
        self.list_of_classes = list_of_classes
        self.majority_class = None

    def get_majority_class(self):
        if self.majority_class:
            return self.majority_class
        else:
            print("Majority class is not yet known.")

    def fit(self, train):
        class_counts = {}
        for doc in train:
            for annotation in doc.annotated.iter_annotations():
                error_type = annotation.meta.get("error_type")
                if error_type in class_counts:
                    class_counts[error_type] += 1
                else:
                    class_counts[error_type] = 1

        self.majority_class = max(class_counts, key=class_counts.get)

    def predict(self, test):
        predicted_labels = []
        for doc in test:
            for ann in doc.annotated.iter_annotations():
                predicted_labels.append(self.list_of_classes.index(self.majority_class))
        return predicted_labels



def main():
    corpus = Corpus(partition="train", annotation_layer="gec-only")
    train_corpus, test_corpus = train_test_split(corpus.get_documents(), test_size=0.25, random_state=42)

    modelMajority = MajorityBaselineModel(list_of_classes)
    modelMajority.fit(train_corpus)
    y_pred = modelMajority.predict(test_corpus)

    y_test = get_real_labels(test_corpus, list_of_classes)

    acc, prec = calc_accuracy_precision(y_test, y_pred, zero_division=True)


if __name__ == "__main__":
    main()

from ua_gec import Corpus, AnnotatedText
from majorityBaseline import MajorityBaselineModel

from usefull_methods import get_real_labels, plot_unique_y, list_of_classes, calc_metrics

class MemorizingBaselineModel:
    def __init__(self, train_corpus, majority_class, list_of_classes):
        self.majority_class = majority_class
        self.train_corpus = train_corpus
        self.annotations_map = {}
        self.list_of_classes = list_of_classes

    def fit(self):
        train_data = []
        for doc in self.train_corpus:
            for ann in doc.annotated.iter_annotations():
                train_data.append((ann.source_text, ann.top_suggestion, ann.meta["error_type"]))
        for source_text, target_text, error_type in train_data:
            self.annotations_map[(source_text, target_text)] = error_type

    def predict(self, test_data):
        '''
            :param test_data:
            :return: list of indexes of classes
        '''
        predicted_classes = []
        for doc in test_data:
            annotations = self._predict_annotation(doc.annotated)
            predicted_classes.extend(annotations)
        return predicted_classes

    def _predict_annotation(self, anotated_text):
        annotations = []
        for ann in anotated_text.iter_annotations():
            source_text = ann.source_text
            target_text = ann.top_suggestion
            if (source_text, target_text) in self.annotations_map:
                error_type = self.annotations_map[(source_text, target_text)]
                error_type_idx = self._get_error_type_index(error_type)
                annotations.append(error_type_idx)
            elif source_text in self.annotations_map:
                error_type = self.annotations_map[source_text]
                error_type_idx = self._get_error_type_index(error_type)
                annotations.append(error_type_idx)
            else:
                annotations.append(self._get_error_type_index(self.majority_class))
        return annotations

    def predict_for_sentence(self, text):
        test_sentence = AnnotatedText(text)
        test_sentence_indxs = self._predict_annotation(test_sentence)
        result_classes = []
        for class_idx in test_sentence_indxs:
            result_classes.append(self.list_of_classes[class_idx])
        for i, ann in enumerate(test_sentence.iter_annotations()):
            print((f"Annotation: {{{ann.source_text}}}=>{{{ann.top_suggestion}}} ::: {result_classes[i]}"))
        return result_classes

    def _get_error_type_index(self, error_type):
        return self.list_of_classes.index(error_type)


def main():
    train_corpus = Corpus(partition="train", annotation_layer="gec-only")
    test_corpus = Corpus(partition="test", annotation_layer="gec-only")

    model_majority = MajorityBaselineModel(list_of_classes)
    model_majority.fit(train_corpus)
    majority_class = model_majority.get_majority_class()

    modelMemorizing = MemorizingBaselineModel(train_corpus, majority_class, list_of_classes)
    modelMemorizing.fit()

    y_pred = modelMemorizing.predict(test_corpus)
    y_test = get_real_labels(test_corpus, list_of_classes)

    plot_unique_y(y_test, y_pred)

    acc, prec, recall, f1 = calc_metrics(y_test, y_pred, zero_division=True)
    #  in y_pred there is no predicted 6 class so we get warning:
    #  Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
    #  Use `zero_division` parameter to control this behavior.

    test_sentence = "Вітаю тебе {вже=>уже} з Днем Народження {,=>.}"
    test_sentence_classes = modelMemorizing.predict_for_sentence(test_sentence)


if __name__ == "__main__":
    main()
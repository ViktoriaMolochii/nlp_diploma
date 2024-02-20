from collections import Counter
from sklearn.metrics import precision_score, accuracy_score

list_of_classes = ["Spelling", "Punctuation", "G/Case", "G/Gender", "G/Number", "G/Aspect", "G/Tense",
                   "G/VerbVoice", "G/PartVoice", "G/VerbAForm", "G/Prep", "G/Participle",
                   "G/UngrammaticalStructure", "G/Comparison", "G/Conjunction", "G/Particle", "G/Other",
                   "F/Style", "F/Calque", "F/Collocation", "F/PoorFlow", "F/Repetition", "F/Other"]


def get_real_labels(corpus, list_of_classes):
    real_labels = []
    for doc in corpus:
        for ann in doc.annotated.iter_annotations():
            error_type = ann.meta.get("error_type")
            real_labels.append(list_of_classes.index(error_type))
    return real_labels


def calc_accuracy_precision(y_test, y_pred, zero_division=False):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn' if zero_division else None)
    print("Precision:", precision)
    return accuracy, precision


def plot_unique_y(y_test, y_pred):
    counts_y_test = Counter(y_test)
    sorted_elements_y_test = sorted(counts_y_test.items(), key=lambda x: x[1], reverse=True)

    counts_y_pred = Counter(y_pred)
    sorted_elements_y_pred = sorted(counts_y_pred.items(), key=lambda x: x[1], reverse=True)

    print("| Class | count y_test |")
    for number, count in sorted_elements_y_test:
        print(f"| {number:5d}  | {count:5d}  |")

    print("| Class | count y_pred |")
    for number, count in sorted_elements_y_pred:
        print(f"| {number:5d}  | {count:5d}  |")


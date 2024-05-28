from collections import Counter
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import os
from sklearn.model_selection import train_test_split
import pandas as pd

list_of_classes = ["Spelling", "Punctuation", "G/Case", "G/Gender", "G/Number", "G/Aspect", "G/Tense",
                   "G/VerbVoice", "G/PartVoice", "G/VerbAForm", "G/Prep", "G/Participle",
                   "G/UngrammaticalStructure", "G/Comparison", "G/Conjunction", "G/Particle", "G/Other", "Other"]


def get_real_labels(corpus, list_of_classes):
    real_labels = []
    for doc in corpus:
        for ann in doc.annotated.iter_annotations():
            error_type = ann.meta.get("error_type")
            real_labels.append(list_of_classes.index(error_type))
    return real_labels


def calc_metrics(y_test, y_pred, zero_division=False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn' if zero_division else None)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division='warn' if zero_division else None)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division='warn' if zero_division else None)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1


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

def merge_datasets(features_train_path, features_test_path, output_path):
    if os.path.exists(output_path):
        print("File exists!")
        return
    else:
        features_train = pd.read_csv(features_train_path)
        features_test = pd.read_csv(features_test_path)
        merged_dataset = pd.concat([features_train, features_test], ignore_index=True)
        merged_dataset.to_csv(output_path, index=False)
        return merged_dataset

def split_dataset_by_error_class(dataset, class_column, test_ratio=0.1, seed=42):
    class_labels = sorted(dataset[class_column].unique())
    class_instances = {label: dataset[dataset[class_column] == label] for label in class_labels}
    train_datasets_list = []
    test_datasets_list = []

    for label, instances in class_instances.items():
        train_samples, test_samples = train_test_split(instances, test_size=test_ratio, random_state=seed)
        train_datasets_list.append(train_samples)
        test_datasets_list.append(test_samples)

    train_dataset = pd.concat(train_datasets_list)
    test_dataset = pd.concat(test_datasets_list)

    train_dataset = train_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_dataset = test_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_dataset, test_dataset


def balanced_datasets(features_train_path, features_test_path, output_balanced_train_path, output_balanced_test_path,
                      class_column, test_ratio=0.1, seed=42):

    if os.path.exists(output_balanced_train_path) and os.path.exists(output_balanced_test_path):
        print('Files exist!')
        balanced_train_data = pd.read_csv(output_balanced_train_path)
        balanced_test_data = pd.read_csv(output_balanced_test_path)
        return balanced_train_data, balanced_test_data

    output_file = "data/merged_features_vector3.csv"
    merge_datasets(features_train_path, features_test_path, output_file)
    merged_data = pd.read_csv(output_file)
    merged_data = merged_data[merged_data["error_class"] != -1]
    merged_data = merged_data.drop(merged_data[merged_data["error_class"] == 11.0].index)

    balanced_train_data, balanced_test_data = split_dataset_by_error_class(merged_data, class_column, test_ratio, seed)
    balanced_train_data.to_csv(output_balanced_train_path, index=False)
    balanced_test_data.to_csv(output_balanced_test_path, index=False)
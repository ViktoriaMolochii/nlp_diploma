#!/usr/bin/env python3
import matplotlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ua_gec import AnnotatedText
from corpus import Corpus
from usefull_methods import list_of_classes, calc_metrics, balanced_datasets
from featuresVectorCreator import FeaturesVectorCreator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.model_selection import GridSearchCV

# param_grid = {
        #     'C': [0.01, 0.1, 10, 100],  # значення для параметру C
        #     'solver': ['newton-cg'],  # значення для параметру solver
        #     'max_iter': [300, 500, 700, 1000],  # значення для параметру max_iter
        # }
        # grid_search = GridSearchCV(estimator=LogisticRegression(class_weight='balanced', random_state=16),
        #                            param_grid=param_grid, cv=5, scoring='accuracy')

        # grid_search.fit(X_train, y_train)
        # print("Найкращі параметри:", grid_search.best_params_)
        # self.lr = grid_search.best_estimator_
        # self.lr.fit(X_train, y_train)

# cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list_of_classes, yticklabels=list_of_classes)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.tight_layout()
        # plt.savefig('confusion_matrix.png')
        # plt.close()

class LogisticRegressionModel:
    def __init__(self):
        self.features_creator = FeaturesVectorCreator(list_of_classes)
        self.lr = None

    def train(self):
        train_corpus = Corpus(partition="train", annotation_layer="gec-only")
        # features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "features_vector3/balanced_trainFinal.csv")
        # features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "features_vector2/balanced_train2.csv")
        features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "features_vector2/balanced_train2.csv")
        features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator,
                                                                        "features_vector1/balanced_train1.csv")

        # feature_cols = self.features_creator.features_list_no_err + self.features_creator.morphosyntactic_features_list
        feature_cols = self.features_creator.features_list_no_err

        X_train = features_df_train[feature_cols]
        y_train = features_df_train.error_class


        self.lr = LogisticRegression(C=0.01, solver='newton-cg', max_iter=500, random_state=16)
        self.lr.fit(X_train, y_train)

    def test(self):
        test_corpus = Corpus(partition="test", annotation_layer="gec-only")
        # features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator, "features_vector3/balanced_testFinal.csv")
        # features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator, "features_vector2/balanced_test2.csv")
        features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator,
                                                                       "features_vector1/balanced_test1.csv")

        # feature_cols = self.features_creator.features_list_no_err + self.features_creator.morphosyntactic_features_list
        feature_cols = self.features_creator.features_list_no_err

        X_test = features_df_test[feature_cols]
        y_test = features_df_test.error_class

        y_pred = self.lr.predict(X_test)
        acc, prec, recall, f1 = calc_metrics(y_test, y_pred, zero_division=True)

        return acc, prec, recall, f1

    def predict_for_each_ann_test_example(self, example):
        features = self.features_creator.features_for_text(example)
        predictions = []
        for one_ann_features in features:
            # X_ann = pd.DataFrame([one_ann_features], columns=self.features_creator.features_list + self.features_creator.morphosyntactic_features_list)
            X_ann = pd.DataFrame([one_ann_features], columns=self.features_creator.features_list)
            X_ann = X_ann.drop('error_class', axis=1)
            y_pred_ann = self.lr.predict(X_ann)
            y_pred_ann = int(y_pred_ann[0])
            predictions.append(list_of_classes[y_pred_ann])
            print("Predicted class:", list_of_classes[y_pred_ann])
        return predictions

    #
    # def plot_confusion_matrix(self):
    #     cm = confusion_matrix(y_test, y_pred)
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list_of_classes, yticklabels=list_of_classes)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.title('Confusion Matrix')
    #     plt.tight_layout()
    #     plt.savefig('confusion_matrix.png')
    #     plt.close()
    #
    # accuracy_scores = cross_val_score(lr, X_train, y_train, cv=5)
    # print(accuracy_scores.mean()) 0.8020755358965636
    # print(accuracy_scores.std()) 0.0005549338696937577

    def test_example(self, example_text):
        # example = AnnotatedText(example_text)
        return self.predict_for_each_ann_test_example(example_text)


def main():
    # balanced_datasets("features_vector3/features_trainFinal.csv", "features_vector3/features_testFinal.csv",
    #                   "features_vector3/balanced_trainFinal.csv",
    #                   "features_vector3/balanced_testfinal.csv",
    #                   "error_class", test_ratio=0.1, seed=42)

    # balanced_datasets("features_vector1/features_train1.csv", "features_vector2/features_test.csv",
    #                   "features_vector2/balanced_train2.csv",
    #                   "features_vector2/balanced_test2.csv",
    #                   "error_class", test_ratio=0.1, seed=42)

    balanced_datasets("features_vector1/features_train1.csv", "features_vector1/features_test1.csv",
                      "features_vector1/balanced_train1.csv",
                      "features_vector1/balanced_test1.csv",
                      "error_class", test_ratio=0.1, seed=42)
    model = LogisticRegressionModel()
    model.train()

    acc, prec, recall, f1 = model.test()
    # example_predictions = model.test_example("Вітаю {тбе=>тебе} з Днем Народження {?=>!}")

if __name__ == "__main__":
    main()

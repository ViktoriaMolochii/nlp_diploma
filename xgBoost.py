import pandas as pd
from xgboost import XGBClassifier
from corpus import Corpus
from usefull_methods import list_of_classes, calc_metrics, balanced_datasets
from featuresVectorCreator import FeaturesVectorCreator
import matplotlib
matplotlib.use('agg')
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

class XGBoostModel:
    def __init__(self):
        self.features_creator = FeaturesVectorCreator(list_of_classes)
        self.xgb = None

    def train(self):
        train_corpus = Corpus(partition="train", annotation_layer="gec-only")
        features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "data/balanced_trainFinal.csv")
        # features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "features_vector1/balanced_train1.csv")
        # features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator, "features_vector2/balanced_train2.csv")


        feature_cols = self.features_creator.features_list_no_err + self.features_creator.morphosyntactic_features_list
        # feature_cols = self.features_creator.features_list_no_err

        X_train = features_df_train[feature_cols]
        y_train = features_df_train.error_class

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        self.xgb = XGBClassifier(
            objective='multi:softmax',
            n_estimators=100,
            max_depth=7,
            learning_rate=0.01,
            random_state=16
        )

        self.xgb.fit(X_train, y_train)

        # param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [3, 5, 7],
        #     'learning_rate': [0.1, 0.01]
        # }
        # grid_search = GridSearchCV(estimator=self.xgb, param_grid=param_grid, cv=5, scoring='accuracy')
        # grid_search.fit(X_train, y_train)
        # print("Найкращі параметри:", grid_search.best_params_)

        # self.xgb = grid_search.best_estimator_
        # self.xgb.fit(X_train, y_train)
        # self.xgb.fit(X_train, y_train)


    def test(self):
        test_corpus = Corpus(partition="test", annotation_layer="gec-only")
        features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator, "data/balanced_testFinal.csv")
        # features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator, "features_vector1/balanced_test1.csv")
        # features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator, "features_vector2/balanced_test2.csv")


        feature_cols = self.features_creator.features_list_no_err + self.features_creator.morphosyntactic_features_list
        # feature_cols = self.features_creator.features_list_no_err


        X_test = features_df_test[feature_cols]
        y_test = features_df_test.error_class

        y_pred = self.xgb.predict(X_test)

        acc, prec, recall, f1 = calc_metrics(y_test, y_pred, zero_division=True)

        return acc, prec, recall, f1

    def predict_for_each_ann_test_example(self, example):
        features = self.features_creator.features_for_text(example)
        predictions = []
        for one_ann_features in features:
            X_ann = pd.DataFrame([one_ann_features], columns=self.features_creator.features_list + self.features_creator.morphosyntactic_features_list)
            X_ann = X_ann.drop('error_class', axis=1)
            y_pred_ann = self.xgb.predict(X_ann)
            y_pred_ann = int(y_pred_ann[0])
            predictions.append(list_of_classes[y_pred_ann])
            print("Predicted class:", list_of_classes[y_pred_ann])
        return predictions

    def test_example(self, example_text):
        # example = AnnotatedText(example_text)
        return self.predict_for_each_ann_test_example(example_text)

def main():
    balanced_datasets("data/features_trainFinal.csv", "data/features_testFinal.csv", "data/balanced_trainFinal.csv",
                      "data/balanced_testFinal.csv",
                      "error_class", test_ratio=0.1, seed=42)

    # balanced_datasets("features_vector1/features_train1.csv", "features_vector1/features_test1.csv",
    #                   "features_vector1/balanced_train1.csv",
    #                   "features_vector1/balanced_test1.csv",
    #                   "error_class", test_ratio=0.1, seed=42)

    # balanced_datasets("features_vector2/features_train.csv", "features_vector2/features_test.csv",
    #                   "features_vector2/balanced_train2.csv",
    #                   "features_vector2/balanced_test2.csv",
    #                   "error_class", test_ratio=0.1, seed=42)

    model = XGBoostModel()
    model.train()

    acc, prec, recall, f1 = model.test()

if __name__ == "__main__":
    main()

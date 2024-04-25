from xgboost import XGBClassifier
import pandas as pd
from featuresVectorCreator import FeaturesVectorCreator
from ua_gec import Corpus, AnnotatedText
from usefull_methods import list_of_classes, calc_metrics
from sklearn.preprocessing import LabelEncoder
class XGBoostModel:
    def __init__(self):
        self.features_creator = FeaturesVectorCreator(list_of_classes)
        self.XGBoost = None

    def train(self):
        train_corpus = Corpus(partition="train", annotation_layer="gec-only")
        features_df_train = self.features_creator.fit_and_save_features(train_corpus, self.features_creator,
                                                                        "data/features_train.csv")

        feature_cols = self.features_creator.features_list_no_err

        X_train = features_df_train[feature_cols]
        y_train = features_df_train.error_class
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        self.XGBoost =XGBClassifier()
        self.XGBoost.fit(X_train, y_train)

    def predict_for_each_ann_test_example(self, example):
        features = self.features_creator.features_for_text(example)
        predictions = []
        for one_ann_features in features:
            X_ann = pd.DataFrame([one_ann_features], columns=self.features_creator.features_list)
            X_ann = X_ann.drop('error_class', axis=1)
            y_pred_ann = self.XGBoost.predict(X_ann)
            predictions.append(list_of_classes[y_pred_ann[0]])
            print("Predicted class:", list_of_classes[y_pred_ann[0]])
        return predictions

    def test(self):
        test_corpus = Corpus(partition="test", annotation_layer="gec-only")
        features_df_test = self.features_creator.fit_and_save_features(test_corpus, self.features_creator,
                                                                       "data/features_test.csv")

        feature_cols = self.features_creator.features_list_no_err

        X_test = features_df_test[feature_cols]
        y_test = features_df_test.error_class

        y_pred = self.XGBoost.predict(X_test)

        acc, prec, recall, f1 = calc_metrics(y_test, y_pred, zero_division=True)

        return acc, prec, recall, f1

    def test_example(self, example_text):
        example = AnnotatedText(example_text)
        return self.predict_for_each_ann_test_example(example)

def main():
    model = XGBoostModel()
    model.train()
    acc, prec, recall, f1 = model.test()
    # example_predictions = model.test_example("Вітаю {тбе=>тебе} з Днем Народження {?=>!}")
    example_predictions = model.test_example("Який вчора {БУВ=>був} гарний день! Просто супер, хіба не так {:=>?} ")


if __name__ == "__main__":
    main()
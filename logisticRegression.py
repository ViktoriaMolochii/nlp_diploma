from ua_gec import Corpus
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ua_gec import AnnotatedText
from usefull_methods import list_of_classes, calc_accuracy_precision
from featuresVectorCreator import FeaturesVectorCreator


def main():
    train_corpus = Corpus(partition="train", annotation_layer="gec-only")
    test_corpus = Corpus(partition="test", annotation_layer="gec-only")

    features_creator = FeaturesVectorCreator(list_of_classes)

    features_df_train = features_creator.fit_and_save_features(train_corpus, features_creator, "data/features_train.csv")
    features_df_test = features_creator.fit_and_save_features(test_corpus, features_creator, "data/features_test.csv")

    # print(features_df_train.head(5))
    # print(features_df_test.head(5))

    feature_cols = features_creator.features_list_no_err

    X_train = features_df_train[feature_cols]
    y_train = features_df_train.error_class

    X_test = features_df_test[feature_cols]
    y_test = features_df_test.error_class

    lr = LogisticRegression(C=100, solver='lbfgs', max_iter=1000, random_state=16)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    acc, prec = calc_accuracy_precision(y_test, y_pred, zero_division=True)

    # from sklearn.model_selection import cross_val_score
    # accuracy_scores = cross_val_score(lr, X_train, y_train, cv=5)
    # print(accuracy_scores.mean()) 0.8020755358965636
    # print(accuracy_scores.std()) 0.0005549338696937577

    example = AnnotatedText("Вітаю {тбе=>тебе} з Днем Народження {?=>!}")
    features_list = features_creator.features_vector_example(example)

    for features in features_list:
        X_ann = pd.DataFrame(features, index=[0])
        y_pred_ann = lr.predict(X_ann)
        print("Predicted class:", list_of_classes[y_pred_ann[0]])


if __name__ == "__main__":
    main()

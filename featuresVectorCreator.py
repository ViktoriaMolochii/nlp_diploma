import os
import editdistance
import string
import pandas as pd


class FeaturesVectorCreator:
    def __init__(self, list_of_classes):
        self.list_of_classes = list_of_classes
        self.features_methods_map = {
            'is_source_empty': self._is_empty,
            'is_source_punctuation_only': self._is_punctuation_only,
            'num_source_tokens': self._num_tokens,
            # 'is_source_dictionary_words': self._your_dictionary_words_method,
            'is_target_empty': self._is_empty,
            'is_target_punctuation_only': self._is_punctuation_only,
            'num_target_tokens': self._num_tokens,
            'edit_distance': self._normalize_levenshtein_distance,
            'error_class': self._error_class
        }
        self.features_list = list(self.features_methods_map.keys())
        features_list_error_removed = self.features_list.copy()
        features_list_error_removed.remove("error_class")
        self.features_list_no_err = features_list_error_removed

    def _is_empty(self, ann):
        text = ann.source_text
        return 1 if not text else 0

    def _is_punctuation_only(self, ann):
        text = ann.source_text
        if text.strip():
            return 1 if all(char in string.punctuation + '–'+'—' for char in text.strip()) else 0
        else:
            return 0

    def _num_tokens(self, ann):
        text = ann.source_text
        return len(text.split())

    def _normalize_levenshtein_distance(self, ann):
        source, target = ann.source_text, ann.top_suggestion
        distance = editdistance.eval(source, target)
        max_length = max(len(source), len(target))

        normalized_distance = float(max_length - distance) / float(max_length) if max_length != 0 else 0.0

        return normalized_distance

    def _error_class(self, ann):
        error = ann.meta["error_type"]
        return self.list_of_classes.index(error)

    def _normalize_num_tokens(self, dataframe_column):
        max_tokens_source = dataframe_column.max()
        dataframe_column_upd = dataframe_column.div(max_tokens_source)
        return dataframe_column_upd

    def features_vector_example(self, text):
        features_vector = []
        for ann in text.iter_annotations():
            features_vector.append({
                feature: self.features_methods_map[feature](ann) for feature in self.features_list_no_err
            })
        return features_vector

    def fit(self, corpus):
        features_df = pd.DataFrame(columns=self.features_list)
        for doc in corpus:
            for ann in doc.annotated.iter_annotations():
                temp_df = pd.DataFrame({
                    feature: self.features_methods_map[feature](ann) for feature in self.features_list
                }, index=[0])
                features_df = pd.concat([features_df, temp_df], ignore_index=True)
                self._normalize_num_tokens(features_df['num_source_tokens'])
                self._normalize_num_tokens(features_df['num_target_tokens'])
        return features_df

    def save_df(self, features_df, path):
        features_df.to_csv(path, index=False)

    def fit_and_save_features(self, corpus, features_creator, output_file):
        if os.path.exists(output_file):
            print(f"Features file '{output_file}' already exists. Skipping fitting and saving.")
            return pd.read_csv(output_file)

        features_df = features_creator.fit(corpus)
        features_creator.save_df(features_df, output_file)
        return features_df
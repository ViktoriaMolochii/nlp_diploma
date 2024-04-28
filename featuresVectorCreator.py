import os
import copy
import editdistance
import string
import pandas as pd
import stanza
import tqdm
import phunspell
import numpy as np

class FeaturesVectorCreator:
    """Generates features vectors for annotated texts.

    Args:
        list_of_classes (list): List of error types.
        include_ud_features (bool): Whether to include Universal Dependencies features.
    """

    def __init__(self, list_of_classes, include_ud_features=True):
        self.list_of_classes = list_of_classes
        self.features_methods_map = {
            'is_source_empty': self._is_empty,
            'is_source_punctuation_only': self._is_punctuation_only,
            'num_source_tokens': self._num_tokens,
            'is_source_dictionary_words': self._is_source_dictionary_words,
            'is_target_empty': self._is_empty,
            'is_target_punctuation_only': self._is_punctuation_only,
            'num_target_tokens': self._num_tokens,
            'edit_distance': self._normalize_levenshtein_distance,
            'error_class': self._error_class
        }

        if include_ud_features:
            self._stanza = stanza.Pipeline(lang='uk' , download_method=None)
            self.features_methods_map.update({
                'morphosyntactic_feats_changed': self._morphosyntactic_feats_changed,
            })

        self.feature_index_map = ["Gender", "Case", "Number", "Animacy", "NameType", "PronType", "Tense", "Person"]
        self.morphosyntactic_features_list = [f"is_{f.lower()}_changed" for f in self.feature_index_map]

        self.features_list = list(self.features_methods_map.keys())
        features_list_error_removed = self.features_list.copy()
        features_list_error_removed.remove("error_class")
        self.features_list_no_err = features_list_error_removed

        self.spellchecker = phunspell.Phunspell('uk_UA')

    def _is_empty(self, ann, doc):
        text = ann.source_text
        return 1 if not text else 0

    def _is_punctuation_only(self, ann, doc):
        text = ann.source_text
        if text.strip():
            return 1 if all(char in string.punctuation + '–'+'—' for char in text.strip()) else 0
        else:
            return 0

    def _num_tokens(self, ann, doc):
        text = ann.source_text
        return len(text.split())

    def _is_source_dictionary_words(self, ann, doc):
        text = ann.source_text
        for word in text.split():
            if not self.spellchecker.lookup(word):
                return 0
        return 1

    def _normalize_levenshtein_distance(self, ann, doc):
        source, target = ann.source_text, ann.top_suggestion
        distance = editdistance.eval(source, target)
        max_length = max(len(source), len(target))

        normalized_distance = float(max_length - distance) / float(max_length) if max_length != 0 else 0.0

        return normalized_distance

    def _error_class(self, ann, doc):
        error = ann.meta.get("error_type")
        if error is None:
            return -1
        try:
            return self.list_of_classes.index(error)
        except ValueError:
            return -1

    def _normalize_num_tokens(self, dataframe_column):
        max_tokens_source = dataframe_column.max()
        dataframe_column_upd = dataframe_column.div(max_tokens_source)
        return dataframe_column_upd

    def get_source_target_feats(self, ann, doc):

        source = self._stanza(ann.source_text)
        target = self._stanza(ann.top_suggestion)

        src_toks = list(source.iter_tokens())
        tgt_toks = list(target.iter_tokens())

        return src_toks, tgt_toks

    def _morphosyntactic_feats_changed(self, ann, doc):
        # TODO: This implementation is very slow for long texts.
        #       We should prepare UA-GEC so that we have annotated
        #       sentences, and work with that.

        src_toks, tgt_toks = self.get_source_target_feats(ann, doc)

        labels_to_return = list(np.zeros(len(self.feature_index_map) + 1, dtype=float))
        feats_index_map = self.feature_index_map

        if len(src_toks) != len(tgt_toks):  # !!!!!!! it's better to fix this part, cuz it's unlogical
            return [1.0, *list(np.zeros(len(feats_index_map), dtype=float))]

        for tok_src, tok_tgt in zip(src_toks, tgt_toks):
            feats_src = tok_src.to_dict()[0].get('feats', '')
            feats_tgt = tok_tgt.to_dict()[0].get('feats', '')
            if feats_src != feats_tgt:
                changed_feats = self._what_was_changed(feats_src, feats_tgt, feats_index_map, labels_to_return[1:])
                labels_to_return = [float(changed_feats.sum() > 1), *list(changed_feats)]
                # labels_to_return = [1.0, *list(changed_feats)] # заходження в іфку могло бути спровоковано пунктуацією, що не є морфол
                # return labels_to_return # 'if' can be called more than once
        return labels_to_return

    def _get_present_features(self, feats_string):
        present_feats = {}
        if feats_string:
            feats_parts = feats_string.split("|")
            for part in feats_parts:
                if '=' in part:
                    feat, value = part.split('=')
                    present_feats[feat] = value
        return present_feats

    def _what_was_changed(self, feats_src, feats_tgt, feat_idx_map, resulted_changed_feats):
        src_feats_dict = self._get_present_features(feats_src)
        tgt_feats_dict = self._get_present_features(feats_tgt)

        result_keys = []

        set_1 = set(src_feats_dict.keys())
        set_2 = set(tgt_feats_dict.keys())

        common_keys = list(set(set_1 & set_2))

        for key in common_keys:
            if src_feats_dict[key] != tgt_feats_dict[key]:
                result_keys.append(key)

        unique_keys_from_dict_1 = list(set_1 - set_2)
        unique_keys_from_dict_2 = list(set_2 - set_1)

        result_keys.extend(unique_keys_from_dict_1)
        result_keys.extend(unique_keys_from_dict_2)

        for key in result_keys:
            if key not in feat_idx_map:
                continue
            resulted_changed_feats[feat_idx_map.index(key)] = 1
        return np.array(resulted_changed_feats)

    def features_for_text(self, annotated_text):
        feature_matrix = []
        for ann in annotated_text.iter_annotations():
            features = []
            for feature_method in self.features_methods_map.values():
                labels = feature_method(ann, annotated_text)
                if type(labels) == list:
                    features.extend(labels)
                else:
                    features.append(float(labels))

            feature_matrix.append(features)
        return feature_matrix

    def fit(self, corpus):
        feature_matrix = []
        for doc in tqdm.tqdm(corpus):
            feature_matrix += self.features_for_text(doc.annotated)
        columns_names = self.features_list + self.morphosyntactic_features_list
        features_df = pd.DataFrame(feature_matrix, columns=columns_names)
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
        print(features_df.head(10))
        features_creator.save_df(features_df, output_file)
        return features_df

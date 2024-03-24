import os
import copy
import editdistance
import string
import pandas as pd
import stanza
import tqdm
import phunspell

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
            self._stanza = stanza.Pipeline(lang='uk', download_method=None)
            self.features_methods_map.update({
                'morphosyntactic_feats_changed': self._morphosyntactic_feats_changed,
            })

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

    def _morphosyntactic_feats_changed(self, ann, doc):

        # TODO: This implementation is very slow for long texts.
        #       We should prepare UA-GEC so that we have annotated
        #       sentences, and work with that.

        # Remove annotations other than `ann`
        doc = copy.deepcopy(doc)
        for ann_ in doc.iter_annotations():
            if ann_ != ann:
                doc.apply_correction(ann_)

        # Parse source and target texts
        source = self._stanza(doc.get_original_text())
        target = self._stanza(doc.get_corrected_text())

        # Check features change. Ideally, we should look at the correction
        # span only. However, to make implementation simpler, we compare
        # the whole texts.
        src_toks = list(source.iter_tokens())
        tgt_toks = list(target.iter_tokens())
        if len(src_toks) != len(tgt_toks):
            return 1

        for tok_src, tok_tgt in zip(src_toks, tgt_toks):
            feats_src = tok_src.to_dict()[0].get('feats', '')
            feats_tgt = tok_tgt.to_dict()[0].get('feats', '')
            if feats_src != feats_tgt:
                return 1
        
        return 0

    def features_for_text(self, annotated_text):
        feature_matrix = []
        for ann in annotated_text.iter_annotations():
            features = [feature_method(ann, annotated_text)
                        for feature_method in self.features_methods_map.values()]
            feature_matrix.append(features)
        return feature_matrix

    def fit(self, corpus):
        #features_df = pd.DataFrame(columns=self.features_list)
        feature_matrix = []
        for doc in tqdm.tqdm(corpus):
            feature_matrix += self.features_for_text(doc.annotated)
        features_df = pd.DataFrame(feature_matrix, columns=self.features_list)
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

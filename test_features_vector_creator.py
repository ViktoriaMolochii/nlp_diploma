import pytest
from featuresVectorCreator import FeaturesVectorCreator
from ua_gec import AnnotatedText


class TestMorphosyntacticFeatsChanged:
    @pytest.fixture(scope="class")
    def fvc(self):
        # Specific classes are not important for these tests
        classes = ["Spelling", "Punctuation", "Grammar", "Fluency"]
        return FeaturesVectorCreator(classes, include_ud_features=True)

    def test_changed(self, fvc):
        # Граматична форма дієслова змінилася, ознака має бути 1
        doc = AnnotatedText("Я хочу {їхав=>їхати} в Київ.")
        ann = doc.get_annotations()[0]
        assert fvc._morphosyntactic_feats_changed(ann, doc) == 1

    def test_unchanged(self, fvc):
        # Граматична форма дієслова не змінилася, ознака має бути 0
        doc = AnnotatedText("Я хочу {їхав=>плив} в Київ.")
        ann = doc.get_annotations()[0]
        assert fvc._morphosyntactic_feats_changed(ann, doc) == 0

    def test_multiple_annotations(self, fvc):
        # Речення має дві анотації, ознака має бути визначена для кожної окремо
        doc = AnnotatedText("Я {хочу=>хотів} {їхав=>плив} в Київ.")
        ann_0 = doc.get_annotations()[0]
        ann_1 = doc.get_annotations()[1]
        assert fvc._morphosyntactic_feats_changed(ann_0, doc) == 1
        assert fvc._morphosyntactic_feats_changed(ann_1, doc) == 0



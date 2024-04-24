import pytest
import errant
from align_errant import align

class TestAlign:

    def test_insert_word(self, errant_):
        src = "Привіт!"
        tgt = "Привіт світ!"
        annotated = align(errant_, src, tgt)
        assert str(annotated) == 'Привіт {=>світ}!'

    @pytest.fixture(scope='class')
    def errant_(self):
        return errant.load('en')

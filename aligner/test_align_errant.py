import pytest
import errant
from ua_gec import AnnotatedText
from align_errant import align

class TestAlign:

    def test_single_token_changes(self, errant_):
        cases = [

            # sl = source token has whitespace on the left
            # sr = source token has whitespace on the right
            # tl = target token has whitespace on the left
            # tr = target token has whitespace on the right

            # deletions                 # sl  sr  tl  tr
            "Привіт{ світ=>}!",         # 1   0   0   0
            "{Привіт =>}світ!",         # 0   1   0   0 
            "Привіт світ{!=>}",         # 0   0   0   0
            "Привіт{,=>} світ!",        # 0   1   0   0

            # insertions
            "Привіт{=> світ}!",         # 0   0   1   0
            "{=>Привіт }світ!",         # 0   0   0   1
            "Привіт{=>,} світ!",        # 0   0   0   0

            # substitutions
            "Привіт {світ=>світе}!",    # 0   0   0   0
            "Привіт{;=>,} світе!",      # 0   0   0   0
        ]
        for case in cases:
            annotated = AnnotatedText(case)
            src = annotated.get_original_text()
            tgt = annotated.get_corrected_text()
            reannotated = align(errant_, src, tgt)
            assert str(reannotated) == case

    @pytest.fixture(scope='class')
    def errant_(self):
        return errant.load('en')

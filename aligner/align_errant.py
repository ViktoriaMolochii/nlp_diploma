#!/usr/bin/env python3
"""Create a ua_gec.AnnotatedText object from a source and target sentence using Errant.

This is useful for aligning parallel sentences for grammatical error correction tasks.

Example:
    >>> src = "Крім того я дізнався та розповів про багато цікавих речей та подій."
    >>> tgt = "Крім того, я дізнався і розповів про багато цікавих речей та подій."
    >>> errant_ = errant.load('en')   # but we will use it for Ukrainian
    >>> annotated = align(errant_, src, tgt)
    'Крім того {=>, }я дізнався {та=>і} розповів про багато цікавих речей та подій.'



"""

import errant
import ua_gec

def align(errant_, source_sentence, target_sentence):

    # Align source and target with Errant
    src_toks = errant_.parse(source_sentence, tokenise=True)
    tgt_toks = errant_.parse(target_sentence, tokenise=True)
    alignment = errant_.align(src_toks, tgt_toks)
    edits = errant_.merge(alignment)

    # Create an AnnotatedText object
    annotated = ua_gec.AnnotatedText(source_sentence)
    for edit in edits:
        # Errant uses token-level indices
        # AnnotatedText uses character-level indices
        src_start = src_toks[edit.o_start].idx if edit.o_start < len(src_toks) else len(source_sentence)
        src_end = src_toks[edit.o_end - 1].idx + len(src_toks[edit.o_end - 1].text)
        src_end = max(src_end, src_start)
        annotated.annotate(src_start, src_end, edit.c_str)

    return annotated


def align_corpus():
    result = []
    corpus = ua_gec.Corpus()
    errant_ = errant.load('en')
    for doc in corpus:
        for src, tgt in zip(doc.source_sentences, doc.target_sentences):
            annotated = align(errant_, src, tgt)
            result.append(annotated)
        break

    return result


def example():
    # Example usage
    source_sentence = "This is the source sentence."
    target_sentence = "This is a different sentence structure."

    errant_ = errant.load('en')
    annotated = align(errant_, source_sentence, target_sentence)
    print(annotated)



if __name__ == "__main__":
    for annotated in align_corpus():
        print(annotated)


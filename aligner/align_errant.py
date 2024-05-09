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
    #import ipdb; ipdb.set_trace()
    for edit in edits:
        # (1) Errant uses token-level indices
        # (2) AnnotatedText uses character-level indices
        # Convert (1) to (2).
        # Be careful to insert whitespace where necessary.
        src_has_left_space, src_has_right_space = surrounding_space(src_toks, edit.o_start)
        tgt_has_left_space, tgt_has_right_space = surrounding_space(tgt_toks, edit.c_start)

        src_start = src_toks[edit.o_start].idx if edit.o_start < len(src_toks) else len(source_sentence)
        src_end = src_toks[edit.o_end - 1].idx + len(src_toks[edit.o_end - 1].text)
        src_end = max(src_end, src_start)
        tgt_text = edit.c_str

        # Add left/right spaces if necessary
        if src_has_right_space and not tgt_has_right_space:
            src_end += 1
        if src_has_left_space and not tgt_has_left_space:
            src_start -= 1
        if tgt_has_right_space and not src_has_right_space:
            tgt_text = tgt_text + " "
        if tgt_has_left_space and not src_has_left_space:
            tgt_text = " " + tgt_text

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


def surrounding_space(toks, idx):
    """Returns (is_left_space, is_right_space) for the token at idx."""

    left_space = False
    right_space = False
    if idx > 0:
        left_space = bool(toks[idx - 1].whitespace_)
    if idx < len(toks) - 1:
        right_space = bool(toks[idx].whitespace_)

    return left_space, right_space


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


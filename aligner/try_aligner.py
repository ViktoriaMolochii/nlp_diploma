
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
# Оголошення функції align з двома параметрами: source_sentence та target_sentence





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


def main():
    # Example usage
    src = "Привіт, Настя!"
    tgt = "Привіт, Насте!"

    # src = 'Вона йшов до школи дуже рано'
    # tgt = 'Вона йшла до школи дуже рано'

    # src = 'Такий гарний вечір, промовив Степан і пішовши додому'
    # tgt = 'Такий гарний вечір, - промовив Степан, і пішов додому'

    # src = 'Як же автору вдалося передбачити майбутнє ?'
    # tgt = 'Як же авторові вдалося передбачити майбутнє ?'

    # src = 'Мені завжди було цікаво читати статті про життя закордоном .'
    # tgt = 'Мені завжди було цікаво читати статті про життя за кордоном .'

    # src = 'Ми відвідували багато музеїв , парків , подорожували по іншим містам та країнами .'
    # tgt = 'Ми відвідували багато музеїв , парків , подорожували іншими містами та країнами .'

    print(src)
    print(tgt)
    # source_sentence = "This is the source sentence."
    # target_sentence = "This is a different sentence structure."
    errant_ = errant.load('en')
    # annotated = align(errant_, source_sentence, target_sentence)
    annotated = align(errant_,src, tgt)
    print(annotated)

if __name__ == "__main__":
    main()
    # for annotated in align_corpus():
    #     print(annotated)

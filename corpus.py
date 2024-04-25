from dataclasses import dataclass
from ua_gec import AnnotatedText


class Corpus:
    """This class mimicks ua_gec.Corpus, but works with local data files that
    store sentence-split annotated texts."""

    def __init__(self, partition, annotation_layer):
        self.partition = partition
        self.annotation_layer = annotation_layer
        self.path = f"data/{annotation_layer}.{partition}.annotated"

    def __iter__(self):
        # Pretend that each sentence is a document
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                annotated = AnnotatedText(line)
                yield Document(annotated)


@dataclass
class Document:
    annotated: AnnotatedText

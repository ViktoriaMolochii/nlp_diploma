# !pip install datasets evaluate transformers[sentencepiece]

from transformers import BertTokenizer, BertModel, BertTokenizerFast
import numpy as np
import torch
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
import pickle

from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')


def encode_sentence(text: str, tokenizer):
    return tokenizer.encode_plus(text, return_tensors='pt', padding=True, truncation=True)

list_of_classes = ["Spelling", "Punctuation", "G/Case", "G/Gender", "G/Number", "G/Aspect", "G/Tense",
                   "G/VerbVoice", "G/PartVoice", "G/VerbAForm", "G/Prep", "G/Participle",
                   "G/UngrammaticalStructure", "G/Comparison", "G/Conjunction", "G/Particle", "Other", "G/Other",
                   "F/Style", "F/Calque", "F/Collocation", "F/PoorFlow", "F/Repetition", "F/Other"]


# індекси токенів, що відповідають зазначеному слову у sentence_word_ids
def tokens_ids(sentence_word_ids: list, pos: list):
    positions = []
    for p in pos:
        positions.extend(np.where(np.array(sentence_word_ids) == p)[0])
    return positions

def concatenate_vectors(vector1, vector2):
    return torch.cat((vector1, vector2), dim=1)

def char_to_word_index(s: str, char_index: int) -> int:
    """Return index of a word corresponding to a character position. """
    pos = -1
    word_index = -1
    while (pos := s.find(" ", pos + 1)) != -1:
        word_index += 1
        if pos >= char_index:
            return word_index
    return word_index + 1

def get_annotation_source_indexes(annotated, annotation) -> list[int]:
    text = annotated.get_original_text()
    start_word_index = char_to_word_index(text, annotation.start)
    end_word_index = char_to_word_index(text, annotation.end)
    return list(range(start_word_index, end_word_index + 1))


def get_annotation_target_indexes(annotated, annotation) -> list[int]:
    assert len(annotated.get_annotations()) == 1, \
        ("This implemention assumes there is only one annotation per text. "
         "Otherwise it won't work correctly.")
    text = annotated.get_corrected_text()
    char_start = annotation.start
    char_end = char_start + len(annotation.top_suggestion)
    start_word_index = char_to_word_index(text, char_start)
    end_word_index = char_to_word_index(text, char_end)
    return list(range(start_word_index, end_word_index + 1))


def map_error_type_to_error_label(list_of_classes, error_type):
  #print(error_type)
  return list_of_classes.index(error_type)


def form_dataset_for_training(annotated_text_flatten, tokenizer):
  input_ids_source_res = []
  attention_mask_source_res = []
  input_ids_target_res = []
  attention_mask_target_res = []
  source_tokens_idxs_res = []
  target_tokens_idxs_res = []
  labels = []


  for i, example_annotated in tqdm(enumerate(annotated_text_flatten)):
      if i == 1:
        print("ddd")

      # Дістаємо єдину анотацію
      example_annotation = example_annotated.get_annotations()[0]

      # Індекси source/target слова в original/corrected реченні (відповідно)
      example_pos_source = get_annotation_source_indexes(example_annotated, example_annotation)
      example_pos_target = get_annotation_target_indexes(example_annotated, example_annotation)

      # Токенізуємо original/corrected речення
      encoded_example_source = encode_sentence(example_annotated.get_original_text(), tokenizer)
      encoded_example_target = encode_sentence(example_annotated.get_corrected_text(), tokenizer)

      # Дістаємо індекси ТОКЕНІВ source/target слова
      source_tokens_idxs = tokens_ids(encoded_example_source.word_ids(), example_pos_source)
      target_tokens_idxs = tokens_ids(encoded_example_target.word_ids(), example_pos_target)

      # Взнаємо тип помилки
      error_label = map_error_type_to_error_label(list_of_classes, example_annotation.meta['error_type'])

      def pad_tensors_to_equal_length(tensor1, tensor2, pad_token_id):
          max_length = max(len(tensor1), len(tensor2))
          padded_tensor1 = torch.nn.functional.pad(tensor1, (0, max_length - len(tensor1)), value=pad_token_id)
          padded_tensor2 = torch.nn.functional.pad(tensor2, (0, max_length - len(tensor2)), value=pad_token_id)
          return padded_tensor1, padded_tensor2

      # Example usage:
      # Assuming you have two tensors tensor1 and tensor2 and tokenizer.pad_token_id is the value you want to pad with
      padded_input_ids_source, padded_input_ids_target = pad_tensors_to_equal_length(encoded_example_source["input_ids"][0],
                                                                            encoded_example_target["input_ids"][0],
                                                                            tokenizer.pad_token_id)

      padded_attention_mask_source, padded_attention_mask_target = pad_tensors_to_equal_length(encoded_example_source["attention_mask"][0],
                                                                            encoded_example_target["attention_mask"][0],
                                                                            0)

      # input_ids_source_res.append(encoded_example_source["input_ids"][0])
      input_ids_source_res.append(padded_input_ids_source)

      #attention_mask_source_res.append(encoded_example_source["attention_mask"][0])
      attention_mask_source_res.append(padded_attention_mask_source)


      #input_ids_target_res.append(encoded_example_target["input_ids"][0])
      input_ids_target_res.append(padded_input_ids_target)


      #attention_mask_target_res.append(encoded_example_target["attention_mask"][0])
      attention_mask_target_res.append(padded_attention_mask_target)

      source_tokens_idxs_res.append(source_tokens_idxs)
      target_tokens_idxs_res.append(target_tokens_idxs)

      labels.append(error_label)

  final_df = Dataset.from_dict({"input_ids": input_ids_source_res,
                                "input_ids_target": input_ids_target_res,
                                "attention_mask": attention_mask_source_res,
                                "attention_mask_target": attention_mask_target_res,
                                "source_tokens_idxs": source_tokens_idxs_res,
                                "target_tokens_idxs": target_tokens_idxs_res,
                                "labels": labels,
                                })

  return final_df


def main():
    # Завантаження набору даних з файлу
    with open('annotated_text_flatten.pkl', 'rb') as f:
        loaded_data = pickle.load(f)


    print("DEBUG !!!!!!")
    df_no_mask = form_dataset_for_training(loaded_data[:10], tokenizer)



if __name__ == "__main__":
    main()

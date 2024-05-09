import pickle

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

# %%
list_of_classes = [
    "Spelling",
    "Punctuation",
    "G/Case",
    "G/Gender",
    "G/Number",
    "G/Aspect",
    "G/Tense",
    "G/VerbVoice",
    "G/PartVoice",
    "G/VerbAForm",
    "G/Prep",  # "G/Participle",
    "G/UngrammaticalStructure",
    "G/Comparison",
    "G/Conjunction",
    "G/Particle",
    "Other",
    "G/Other",
    "F/Style",
    "F/Calque",
    "F/Collocation",
    "F/PoorFlow",
    "F/Repetition",
    "F/Other",
]


def map_error_type_to_error_label(list_of_classes, error_type):
    # print(error_type)
    return list_of_classes.index(error_type)


def encode_sentence(text: str, tokenizer):
    return tokenizer.encode_plus(
        text, return_tensors="pt", padding=True, truncation=True
    )


# індекси токенів, що відповідають зазначеному слову у sentence_word_ids
def tokens_ids(sentence_word_ids: list, pos: list):
    positions = []
    for p in pos:
        positions.extend(np.where(np.array(sentence_word_ids) == p)[0])
    return positions


def concatenate_vectors(vector1, vector2):
    return torch.cat((vector1, vector2), dim=1)


def char_to_word_index(s: str, char_index: int) -> int:
    """Return index of a word corresponding to a character position."""
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
    assert len(annotated.get_annotations()) == 1, (
        "This implemention assumes there is only one annotation per text. "
        "Otherwise it won't work correctly."
    )
    text = annotated.get_corrected_text()
    char_start = annotation.start
    char_end = char_start + len(annotation.top_suggestion)
    start_word_index = char_to_word_index(text, char_start)
    end_word_index = char_to_word_index(text, char_end)
    return list(range(start_word_index, end_word_index + 1))


def form_dataset_for_training(annotated_text_flatten, tokenizer):
    input_ids_source_res = []
    attention_mask_source_res = []
    input_ids_target_res = []
    attention_mask_target_res = []
    source_tokens_idxs_res = []
    target_tokens_idxs_res = []
    labels = []

    for i, example_annotated in tqdm(enumerate(annotated_text_flatten)):
        # Дістаємо єдину анотацію
        example_annotation = example_annotated.get_annotations()[0]

        # Індекси source/target слова в original/corrected реченні (відповідно)
        example_pos_source = get_annotation_source_indexes(
            example_annotated, example_annotation
        )
        example_pos_target = get_annotation_target_indexes(
            example_annotated, example_annotation
        )

        # Токенізуємо original/corrected речення
        encoded_example_source = encode_sentence(
            example_annotated.get_original_text(), tokenizer
        )
        encoded_example_target = encode_sentence(
            example_annotated.get_corrected_text(), tokenizer
        )

        # Дістаємо індекси ТОКЕНІВ source/target слова
        source_tokens_idxs = tokens_ids(
            encoded_example_source.word_ids(), example_pos_source
        )
        target_tokens_idxs = tokens_ids(
            encoded_example_target.word_ids(), example_pos_target
        )

        # Взнаємо тип помилки
        error_label = map_error_type_to_error_label(
            list_of_classes, example_annotation.meta["error_type"]
        )

        def pad_tensors_to_equal_length(tensor1, tensor2, pad_token_id):
            if len(tensor1) == len(tensor2):
                return tensor1, tensor2
            max_length = max(len(tensor1), len(tensor2))
            padded_tensor1 = torch.nn.functional.pad(
                tensor1, (0, max_length - len(tensor1)), value=pad_token_id
            )
            padded_tensor2 = torch.nn.functional.pad(
                tensor2, (0, max_length - len(tensor2)), value=pad_token_id
            )
            return padded_tensor1, padded_tensor2

        # Example usage:
        # Assuming you have two tensors tensor1 and tensor2 and tokenizer.pad_token_id is the value you want to pad with
        padded_input_ids_source, padded_input_ids_target = pad_tensors_to_equal_length(
            encoded_example_source["input_ids"][0],
            encoded_example_target["input_ids"][0],
            tokenizer.pad_token_id,
        )

        padded_attention_mask_source, padded_attention_mask_target = (
            pad_tensors_to_equal_length(
                encoded_example_source["attention_mask"][0],
                encoded_example_target["attention_mask"][0],
                0,
            )
        )

        # print(tokenizer.convert_ids_to_tokens(padded_input_ids_source))
        # print(tokenizer.convert_ids_to_tokens(padded_input_ids_target))

        # input_ids_source_res.append(encoded_example_source["input_ids"][0])
        input_ids_source_res.append(padded_input_ids_source)

        # attention_mask_source_res.append(encoded_example_source["attention_mask"][0])
        attention_mask_source_res.append(padded_attention_mask_source)

        # input_ids_target_res.append(encoded_example_target["input_ids"][0])
        input_ids_target_res.append(padded_input_ids_target)

        # attention_mask_target_res.append(encoded_example_target["attention_mask"][0])
        attention_mask_target_res.append(padded_attention_mask_target)

        source_tokens_idxs_res.append(source_tokens_idxs)
        target_tokens_idxs_res.append(target_tokens_idxs)

        labels.append(error_label)

    final_df = Dataset.from_dict(
        {
            "input_ids": input_ids_source_res,
            "input_ids_target": input_ids_target_res,
            "attention_mask": attention_mask_source_res,
            "attention_mask_target": attention_mask_target_res,
            "source_tokens_idxs": source_tokens_idxs_res,
            "target_tokens_idxs": target_tokens_idxs_res,
            "labels": labels,
        }
    )
    final_df.set_format("torch")

    return final_df


def make_collate_fn(tokenizer):
    def collate_function(batch_examples, return_tensors="pt"):
        def select_columns(batch_examples, needed_columns):
            filtered_batch = []  # len(filtered_batch) = batch_size
            for example in batch_examples:
                filtered_batch.append(
                    {column: example[column] for column in needed_columns}
                )
            return filtered_batch

        def make_mask(batch, feature):
            mask = torch.zeros_like(batch["input_ids"])
            for i, xs in enumerate(batch_examples):
                mask[i][xs[feature]] = 1
            return mask

        inputs = select_columns(
            batch_examples,
            [
                "input_ids",
                #  "input_ids_target",
                "attention_mask",
                #  "attention_mask_target"
            ],
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        batch_inputs = data_collator(inputs)

        batch_inputs["source_tokens_mask"] = make_mask(
            batch_inputs, "source_tokens_idxs"
        )
        # print(batch_inputs["source_tokens_mask"])

        batch_inputs["target_tokens_mask"] = make_mask(
            batch_inputs, "target_tokens_idxs"
        )

        def get_feature(batch_inputs, feature, pad_token=tokenizer.pad_token_id):
            feature_values = torch.zeros_like(batch_inputs["input_ids"])
            max_length = feature_values.shape[1]

            # batch_examples - list of dictionaries, where each dict is a one instance from batch
            for i, xs in enumerate(batch_examples):
                feature_length = xs[feature].shape[0]

                # Pad xs[feature] to max_length if it's less than max_length
                if feature_length < max_length:
                    padded_feature = F.pad(
                        xs[feature], (0, max_length - feature_length), value=pad_token
                    )
                    feature_values[i] = padded_feature
                elif feature_length > max_length:
                    print("⛔️ TRUNCATED")
                    # If feature_length is greater than or equal to max_length, truncate xs[feature] to max_length
                    feature_values[i] = xs[feature][:max_length]
                else:
                    feature_values[i] = xs[feature]

            return feature_values

        # print(batch_examples["input_ids_target"])
        batch_inputs["input_ids_target"] = get_feature(batch_inputs, "input_ids_target")
        batch_inputs["attention_mask_target"] = get_feature(
            batch_inputs, "attention_mask_target", pad_token=0
        )

        # Виділення міток з функції
        labels = [float(x["labels"]) for x in batch_examples]
        batch_labels = {"labels": torch.tensor(labels)}

        # Повернення окремих пакунків даних для бачів та міток
        return batch_inputs, batch_labels

    return collate_function


if __name__ == "__main__":
    with open("annotated_text_flatten.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    df_no_mask = form_dataset_for_training(loaded_data[:5000], tokenizer)
    print(df_no_mask)

    collate_function = make_collate_fn(tokenizer)
    dataloader = DataLoader(df_no_mask, batch_size=4, collate_fn=collate_function)

    for batch, batch_labels in dataloader:
        import ipdb; ipdb.set_trace()
        print(batch)
        break

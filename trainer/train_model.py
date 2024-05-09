import ua_gec
from ua_gec import Corpus, AnnotatedText

# %%
import matplotlib.pyplot as plt

# %%

from transformers import BertTokenizer, BertModel, BertTokenizerFast
import numpy as np
import torch
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import concatenate_datasets, load_dataset
import pickle

from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel

# %%
BATCH_SIZE = 32
EVALUTION_FREQ = 100
LEARNING_RATE = 0.00005
# LEARNING_RATE = 0.0001

# %%
BATCH_SIZE

# %%
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

# %%
list_of_classes = ["Spelling", "Punctuation", "G/Case", "G/Gender", "G/Number", "G/Aspect", "G/Tense",
                   "G/VerbVoice", "G/PartVoice", "G/VerbAForm", "G/Prep", # "G/Participle",
                   "G/UngrammaticalStructure", "G/Comparison", "G/Conjunction", "G/Particle", "Other", "G/Other",
                   "F/Style", "F/Calque", "F/Collocation", "F/PoorFlow", "F/Repetition", "F/Other"]



def map_error_type_to_error_label(list_of_classes, error_type):
  #print(error_type)
  return list_of_classes.index(error_type)

# %%

# %%
# %%
class SiameseErrorTypeClassificator(nn.Module):
    """Siameese network for error type classification.
    """
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        size_of_input = encoder.config.hidden_size * 2
        self.linear = nn.Linear(size_of_input, num_classes)

    def forward(self, batch):
        # outputs_1 = self.encoder(**inputs_1)
        # outputs_2 = self.encoder(**inputs_2)

        input_ids_source = batch["input_ids"]
        input_ids_target = batch["input_ids_target"]

        attention_mask_source = batch["attention_mask"]
        attention_mask_target = batch["attention_mask_target"]

        tokens_mask_source = batch["source_tokens_mask"]
        tokens_mask_target = batch["target_tokens_mask"]

        bert_source_outputs = self.encoder(
            input_ids=input_ids_source,
            attention_mask=attention_mask_source,
        )

        bert_target_outputs = self.encoder(
            input_ids=input_ids_target,
            attention_mask=attention_mask_target,
        )

        target_vector_1 = self._span_vector(bert_source_outputs, tokens_mask_source)
        target_vector_2 = self._span_vector(bert_target_outputs, tokens_mask_target)

        features = concatenate_vectors(target_vector_1, target_vector_2)
        logits = self.linear(features)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs


    def _span_vector(self, outputs, span_tokens):
        """Compute mean output vector for span_tokens. """

        # hidden_states = outputs.last_hidden_state[:, span_tokens, :]
        # mean = hidden_states.mean(dim=1)

        hidden_state = outputs.last_hidden_state   # (batch_size, seq_len, hidden_size)

        # Compute average hidden state of target word in each sentence
        sum_target = (span_tokens.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        avg_target = sum_target / span_tokens.sum(dim=1).unsqueeze(-1)    # (batch_size, hidden_size)
        return avg_target


# %%
def train(model, train_dataloader, device, optimizer, criterion, num_epochs, eval_dataloader=None):

    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    eval_losses = []
    eval_accuracies = []

    batches_since_eval = 0
    batches_since_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the training dataset
        for batch, batch_labels in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            # print("*")
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            optimizer.zero_grad()

            log_probs = model(batch)


            # class_indices = torch.tensor(batch_labels["labels"], dtype=torch.long)
            class_indices = batch_labels["labels"].long()
            loss = criterion(log_probs, class_indices)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            _, predicted = torch.max(log_probs, 1)
            correct_predictions += (predicted == class_indices).sum().item()


            total_samples += len(batch_labels["labels"])

            # Evaluate the model every EVALUTION_FREQ batches
            batches_since_eval += 1
            batches_since_epoch += 1
            if eval_dataloader is not None and batches_since_eval >= EVALUTION_FREQ:
                epoch_loss = running_loss / batches_since_epoch
                epoch_accuracy = correct_predictions / total_samples
                eval_loss, eval_accuracy = evaluate_model(model, device, eval_dataloader, criterion)
                print(f"\nEpoch {epoch+1}, Train_Loss: {epoch_loss:.2f}, Train_Accuracy: {epoch_accuracy:.2f}, "
                      f"Eval_Loss: {eval_loss:.2f}, Eval_Accuracy: {eval_accuracy:.2f}\n")
                batches_since_eval = 0

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_samples

        # Append loss and accuracy values to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Compute the evaluation loss and accuracy
        eval_loss, eval_accuracy = evaluate_model(model, device, eval_dataloader, criterion)

        # Append loss and accuracy values to lists
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

        # Print the average loss and accuracy for this epoch
        print(f"\nEpoch {epoch+1}, Train_Loss: {epoch_loss:.2f}, Train_Accuracy: {epoch_accuracy:.2f} "
              f"Eval_Loss: {eval_loss:.2f}, Eval_Accuracy: {eval_accuracy:.2f} \n")

    return model, train_losses, train_accuracies, eval_losses, eval_accuracies

# %%
def evaluate_model(model, device, eval_dataloader, criterion):
    if eval_dataloader is None:
        return None, None

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch, batch_labels in eval_dataloader:
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            log_probs = model(batch)

            # class_indices = torch.tensor(batch_labels["labels"], dtype=torch.long)
            class_indices = batch_labels["labels"].long()
            loss = criterion(log_probs, class_indices)

            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            _, predicted = torch.max(log_probs, 1)
            correct_predictions += (predicted == class_indices).sum().item()

            total_samples += len(batch_labels["labels"])

    # Calculate average loss and accuracy
    eval_loss = running_loss / len(eval_dataloader)
    eval_accuracy = correct_predictions / total_samples

    return eval_loss, eval_accuracy


# %%
def calc_class_weights(class_frequencies, epsilon=1e-7):
    # Total number of instances
    total_instances = class_frequencies.sum()

    # Calculate class frequencies
    class_freq = class_frequencies / total_instances

    # Handle division by zero by adding epsilon to the denominator
    class_freq_denom = class_freq.clone()
    class_freq_denom[class_freq_denom == 0] = epsilon

    # Calculate class weights (inverse of class frequencies)
    class_weights = 1 / class_freq_denom

    # Normalize class weights
    class_weights /= class_weights.sum()
    return class_weights


# %%
def split_df(df):
    # from datasets import Dataset
    # from sklearn.model_selection import train_test_split

    split_ratio = 0.1  # Train-test split ratio
    class_labels = sorted(df.unique('labels'))

    # Separate instances of each class
    class_instances = {label: df.filter(lambda example: example['labels'] == label) for label in
                       class_labels}

    # Initialize empty datasets for train and test
    train_datasets_list = []
    test_datasets_list = []

    # Split each class into train and test according to the desired ratio
    for label, instances in class_instances.items():
        dataset_dict = instances.train_test_split(test_size=split_ratio, seed=42)
        train_datasets_list.append(dataset_dict['train'])
        test_datasets_list.append(dataset_dict['test'])

    train_dataset = concatenate_datasets(train_datasets_list)
    test_dataset = concatenate_datasets(test_datasets_list)

    # Shuffle train and test datasets
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    return train_dataset, test_dataset

# %%

# %%
with open('annotated_text_flatten.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# %%
df_no_mask = form_dataset_for_training(loaded_data[:5000], tokenizer)
# df_no_mask = form_dataset_for_training(loaded_data, tokenizer)

#dfs_no_mask = df_no_mask.train_test_split(test_size=0.1)
df_no_mask.set_format("torch")

# %%
labels_column = df_no_mask['labels']
labels_tensor = torch.tensor(labels_column)
# Count occurrences of each label
label_counts = torch.bincount(labels_tensor)
# Print number of entries for each class label
for label, count in enumerate(label_counts):
    print(f"Label {label}: {count} entries")

class_weights = calc_class_weights(label_counts)

dfs_no_mask_train, dfs_no_mask_test = split_df(df_no_mask)

# %%
dfs_no_mask_train.set_format("torch")
dfs_no_mask_test.set_format("torch")

# %%
train_dataloader = DataLoader(
    dfs_no_mask_train,
    # dfs_no_mask["train"],
    # dfs_no_mask_train_50,
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=collate_function,
)

eval_dataloader = DataLoader(
    dfs_no_mask_test,
    # dfs_no_mask["test"],
    batch_size=BATCH_SIZE,
    collate_fn=collate_function
)

# for batch, labels in train_dataloader:
#     print(batch)
#     print(labels)
#     break

# %%
model = SiameseErrorTypeClassificator(encoder, 17) # len(list_of_classes)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %%
class_weights

# %%
device

# %%
# Training ...
model, train_losses, train_accuracies, eval_losses, eval_accuracies = train(
    model,
    train_dataloader,
    device,
    optimizer,
    criterion,
    num_epochs=5,
    eval_dataloader=eval_dataloader
)

# %%
def plot_loss_accuracy(losses, accuracies, df_name="Train"):
    # Plot loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label=f"{df_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{df_name} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label=f"{df_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{df_name} Accuracy")
    plt.legend()
    plt.show()

# %%
plot_loss_accuracy(train_losses, train_accuracies, "Train")
plot_loss_accuracy(eval_losses, eval_accuracies, "Eval")




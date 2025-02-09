import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

poetry_data = [
    "The whispering winds in twilights glow",
    "Carry secrets only shadows know.",
    "The silver moon with solemn grace",
    "Casts dreams upon the water's face.",
    "Through valleys deep and mountains high,",
    "The rivers weave, the echoes sigh.",
    "A golden sun ignites the sky",
    "Yet time moves on, and days drift by.",
    "The rose that blooms in morning's breath",
    "Knows not the touch of looming death.",
    "Yet petals fall and stems decay",
    "As nature sings its sweet ballet.",
    "A traveler walks through paths untold",
    "In search of wisdom, lost and old.",
    "Through forests thick and deserts wide",
    "With hope and sorrow side by side.",
]

word_to_index = {}
index_to_word = {}

# Create the dictionary

for line in poetry_data:
    for word in line.split():
        if word not in word_to_index:
            index = len(word_to_index)
            word_to_index[word] = index
            index_to_word[index] = word

# Print the dictionary for the poetry data
print(word_to_index)

# Map the poetry dataset with the created dictionary indexes
indexed_dataset = [
    [word_to_index[word] for word in line.split()] for line in poetry_data
]

# Split the data for training and testing
n = int(len(indexed_dataset)*0.9)
train_data = indexed_dataset[:n]  # 90% will be used for training
valid_data = indexed_dataset[n:]  # Remaining 10% will be used for validation


class poetry_dataset(Dataset):
    def __init__(self, indexed_dataset):
        self.data = []

        for seq in indexed_dataset:
            if len(seq) > 1:
                input_seq = seq[:-1]
                output_seq = seq[1:]
                self.data.append((torch.tensor(input_seq), torch.tensor(output_seq)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Create the collate function for padding the sequences
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    padded_input = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_output = pad_sequence(outputs, batch_first=True, padding_value=0)
    return padded_input, padded_output

# Create the dataset and data loader
train_dataset = poetry_dataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

valid_dataset = poetry_dataset(valid_data)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for input, output in train_loader:
    print('padded input sequence :', input)
    print('padded output sequence :', output)
    break
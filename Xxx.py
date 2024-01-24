import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import os
from torch.nn import functional as F
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        а 2
        б 3
        в 4
        г 5
        д 6
        е 7
        ё 8
        ж 9
        з 10
        и 11
        й 12
        к 13
        л 14
        м 15
        н 16
        о 17
        п 18
        р 19
        с 20
        т 21
        у 22
        ф 23
        х 24
        ц 25
        ч 26
        ш 27
        щ 28
        ъ 29
        ы 30
        ь 31
        э 32
        ю 33
        я 34
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = '<SPACE>'
            else:
                ch = c
            # Handle unknown characters
            int_sequence.append(self.char_map.get(ch, 0))  # Use 0 for unknown characters
        return int_sequence

    def int_to_text(self, int_sequence):
        """Use an index map and convert an integer sequence to text"""
        string = []
        for i in int_sequence:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

text_transform = TextTransform()

class SpeechDataset(Dataset):
    def __init__(self, file_paths, transcriptions, text_transform):
        self.file_paths = file_paths
        self.transcriptions = transcriptions
        self.text_transform = text_transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        # Convert to mono by averaging across channels if not already mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel_spectrogram_transform = MelSpectrogram(sample_rate, n_fft=800, hop_length=160, n_mels=80)
        feature = mel_spectrogram_transform(waveform)
        label = torch.Tensor(self.text_transform.text_to_int(self.transcriptions[idx].lower()))

        # Padding
        max_len_feature = max(feature.shape[2], len(label))
        feature = F.pad(feature, (0, max_len_feature - feature.shape[2]))
        label = F.pad(label, (0, max_len_feature - len(label)))

        return feature, label

def prepare_datasets(dataset_path):
    with open(os.path.join(dataset_path, 'transcriptions.txt'), 'r') as f:
        lines = f.readlines()

    file_paths = []
    transcriptions = []
    for line in lines:
        filename, transcription = line.strip().split(maxsplit=1)
        file_paths.append(os.path.join(dataset_path, filename))
        transcriptions.append(transcription)

    train_files, val_files, train_transcriptions, val_transcriptions = train_test_split(
        file_paths, transcriptions, test_size=0.2, random_state=42
    )

    train_dataset = SpeechDataset(train_files, train_transcriptions, text_transform)
    val_dataset = SpeechDataset(val_files, val_transcriptions, text_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, len(text_transform.char_map)

class SpeechToTextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SpeechToTextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension
        x, _ = self.lstm(x)  # LSTM outputs hidden states and cell states
        x = self.fc(x)

        x = F.log_softmax(x, dim=2)  # Apply log_softmax to logits
        return x

def train_model(model, train_dataloader, val_dataloader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # CTC loss function

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (audio, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_lengths, target_lengths = torch.full((audio.size(0),), audio.size(2), dtype=torch.long), torch.full((targets.size(0),), targets.size(1), dtype=torch.long)
            # Forward pass
            outputs = model(audio)  # (batch, time, n_class)
            outputs = outputs.transpose(0, 1)  # (time, batch, n_class)
            loss = criterion(outputs, targets.int(), input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (audio, targets) in enumerate(val_dataloader):
                input_lengths, target_lengths = torch.full((audio.size(0),), audio.size(2), dtype=torch.long), torch.full((targets.size(0),), targets.size(1), dtype=torch.long)
                outputs = model(audio)
                outputs = outputs.transpose(0, 1)
                loss = criterion(outputs, targets.int(), input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    return model

def save_model(model, model_path='/content/drive/MyDrive/x/www'):
    torch.save(model.state_dict(), model_path)

# Usage
dataset_path = '/content/drive/MyDrive/x/www'
train_dataloader, val_dataloader, output_size = prepare_datasets(dataset_path)
model = SpeechToTextLSTM(input_size=80, hidden_size=256, num_layers=3, output_size=output_size)
trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs=10)
save_model(trained_model, model_path='/content/drive/MyDrive/m')

from torch import nn

class LanguagePredictor(nn.Module):
    def __init__(self, num_chars, embedding_dim=64, lstm_hidden_dim=128, num_languages=37):
        super(LanguagePredictor, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, num_languages)
        self.fc2 = nn.Linear(lstm_hidden_dim, num_languages)
        self.fc3 = nn.Linear(lstm_hidden_dim, num_languages)
    
    def forward(self, x, lengths):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)  
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        out1 = self.fc1(h_n)
        out2 = self.fc2(h_n)
        out3 = self.fc3(h_n)
        return out1, out2, out3
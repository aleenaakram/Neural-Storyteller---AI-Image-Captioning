"""
Model Architecture - Fixed for Streamlit
Save this as: model_architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """Encoder: Projects 2048-dim image features to hidden_size"""
    
    def __init__(self, feature_size=2048, hidden_size=512, dropout=0.5):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        self.fc = nn.Linear(feature_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, features):
        encoded = self.fc(features)
        encoded = self.bn(encoded)
        encoded = self.relu(encoded)
        encoded = self.dropout(encoded)
        return encoded


class Decoder(nn.Module):
    """Decoder: LSTM-based caption generator"""
    
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, 
                 num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, encoded_features, captions, lengths=None):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        
        batch_size = encoded_features.size(0)
        h0 = encoded_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if lengths is not None:
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        outputs = self.fc(self.dropout(lstm_out))
        return outputs
    
    def generate_greedy(self, encoded_features, vocab, max_length=50):
        batch_size = encoded_features.size(0)
        device = encoded_features.device
        
        h = encoded_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        inputs = torch.tensor([vocab.word2idx[vocab.start_token]] * batch_size).to(device)
        inputs = inputs.unsqueeze(1)
        
        generated_ids = []
        
        for _ in range(max_length):
            embeddings = self.embed(inputs)
            lstm_out, (h, c) = self.lstm(embeddings, (h, c))
            outputs = self.fc(lstm_out.squeeze(1))
            predicted = outputs.argmax(dim=1)
            generated_ids.append(predicted.cpu().numpy())
            
            if (predicted == vocab.word2idx[vocab.end_token]).all():
                break
            
            inputs = predicted.unsqueeze(1)
        
        generated_ids = np.array(generated_ids).T
        
        captions = []
        for ids in generated_ids:
            caption = vocab.decode(ids)
            captions.append(caption)
        
        return captions
    
    def generate_beam_search(self, encoded_features, vocab, beam_width=3, max_length=50):
        device = encoded_features.device
        
        h = encoded_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        start_idx = vocab.word2idx[vocab.start_token]
        end_idx = vocab.word2idx[vocab.end_token]
        
        beams = [([start_idx], 0.0, h, c)]
        completed = []
        
        for step in range(max_length):
            candidates = []
            
            for seq, score, h_state, c_state in beams:
                if seq[-1] == end_idx:
                    completed.append((seq, score))
                    continue
                
                inputs = torch.tensor([seq[-1]]).to(device).unsqueeze(0)
                embeddings = self.embed(inputs)
                lstm_out, (h_new, c_new) = self.lstm(embeddings, (h_state, c_state))
                outputs = self.fc(lstm_out.squeeze(1))
                
                log_probs = F.log_softmax(outputs, dim=1)
                top_probs, top_indices = log_probs.topk(beam_width, dim=1)
                
                for i in range(beam_width):
                    word_idx = top_indices[0, i].item()
                    word_prob = top_probs[0, i].item()
                    new_seq = seq + [word_idx]
                    new_score = score + word_prob
                    candidates.append((new_seq, new_score, h_new, c_new))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            if len(beams) == 0:
                break
        
        completed.extend(beams)
        
        if completed:
            completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            best_seq = completed[0][0]
        else:
            best_seq = beams[0][0] if beams else [start_idx, end_idx]
        
        caption = vocab.decode(best_seq)
        return caption


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    
    def __init__(self, vocab_size, feature_size=2048, embed_size=300, 
                 hidden_size=512, num_layers=1, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = Encoder(feature_size, hidden_size, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, 
                              num_layers, dropout)
        
    def forward(self, features, captions, lengths=None):
        encoded = self.encoder(features)
        outputs = self.decoder(encoded, captions, lengths)
        return outputs
    
    def generate_caption(self, features, vocab, method='greedy', beam_width=3, max_length=50):
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            encoded = self.encoder(features)
            
            if method == 'greedy':
                captions = self.decoder.generate_greedy(encoded, vocab, max_length)
                return captions[0]
            elif method == 'beam':
                return self.decoder.generate_beam_search(encoded, vocab, beam_width, max_length)
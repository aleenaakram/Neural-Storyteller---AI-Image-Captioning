"""
Vocabulary Class - Fixed for Streamlit Import
Save this as: vocabulary.py
"""

import re
from collections import Counter


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"
        
        # Initialize with special tokens
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = 4  # Next available index
        
    def build_vocabulary(self, captions_list):
        """Build vocabulary from list of captions"""
        # Count word frequencies
        for caption in captions_list:
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)
        
        # Add words that meet frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
        
    def tokenize(self, text):
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = text.split()
        return tokens
    
    def numericalize(self, text):
        """Convert text to list of indices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) 
                for token in tokens]
    
    def decode(self, indices):
        """Convert list of indices back to text"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if word == self.end_token:
                break
            if word not in [self.pad_token, self.start_token]:
                words.append(word)
        return " ".join(words)
    
    def __len__(self):
        return len(self.word2idx)
import re
import os
import json
import emoji
from pyvi import ViTokenizer

class VietnameseTextCleaner:
    def __init__(self, dict_dir="data/dictionaries"):
        teencode_path = os.path.join(dict_dir, "teencode.json")
        emoji_path = os.path.join(dict_dir, "emoji_vi.json")
        
        # Load Teencode dictionary
        with open(teencode_path, 'r', encoding='utf-8') as f:
            self.teencode_dict = json.load(f)
            
        # Load Emoji dictionary
        with open(emoji_path, 'r', encoding='utf-8') as f:
            self.emoji_dict = json.load(f)

    def remove_noise(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s\d_]', ' ', text)
        return text

    def normalize_repeated_chars(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def translate_emoji(self, text):
        for emo, meaning in self.emoji_dict.items():
            text = text.replace(emo, f" {meaning} ")
        text = emoji.replace_emoji(text, replace='')
        return text

    def normalize_teencode(self, text):
        words = text.split()
        normalized_words = [self.teencode_dict.get(word, word) for word in words]
        return " ".join(normalized_words)

    def segment_words(self, text):
        return ViTokenizer.tokenize(text)

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = self.translate_emoji(text)
        text = self.remove_noise(text)
        text = self.normalize_repeated_chars(text)
        text = self.normalize_teencode(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.segment_words(text)
        return text
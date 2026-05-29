import re
import os
import json
import emoji

class EnglishTextCleaner:
    def __init__(self, dict_dir="data/dictionaries"):
        slang_path = os.path.join(dict_dir, "slang_en.json")
        emoji_path = os.path.join(dict_dir, "emoji_en.json")
        
        with open(slang_path, 'r', encoding='utf-8') as f:
            self.slang_dict = json.load(f)
        with open(emoji_path, 'r', encoding='utf-8') as f:
            self.emoji_dict = json.load(f)

    def remove_noise(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s\d_]', ' ', text)
        return text

    def translate_emoji(self, text):
        for emo, meaning in self.emoji_dict.items():
            text = text.replace(emo, f" {meaning} ")
        text = emoji.replace_emoji(text, replace='')
        return text

    def normalize_slang(self, text):
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return " ".join(normalized_words)

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = self.translate_emoji(text)
        text = self.remove_noise(text)
        text = self.normalize_slang(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
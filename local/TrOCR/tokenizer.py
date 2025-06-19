import string

# Lettres de base
lettres = string.ascii_letters  # a-z + A-Z
chiffres = string.digits        # 0-9
ponctuations = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# Lettres françaises supplémentaires
lettres_avec_accents = "àâäçéèêëîïôöùûüÿœæÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒÆ"

# Combinaison finale
all_tokens = lettres + lettres_avec_accents + chiffres + ponctuations+' '
unk_token = '¤'
all_tokens = all_tokens + unk_token 

class charTokenizer:
    def __init__(self):
        self.vocab = all_tokens

    def __len__(self):
        return len(all_tokens)+1 # for padding
    
    def get_maximum_length(self,df):
        max_length = 0
        for text in df['text']:
            length = len(self.encode(text))
            if length > max_length:
                max_length = length
        return max_length
    
    def encode(self, text):
        stoi = { ch: i for i, ch in enumerate(self.vocab) }
        stoi['<pad>']=  len(self.vocab)
        return [stoi.get(ch, stoi[unk_token]) for ch in text]
    
    def encode_with_padding(self, text, max_length):
        tokens = self.encode(text)
        if len(tokens) < max_length:
            tokens += [len(self.vocab)] * (max_length - len(tokens))
        return tokens
    
    def decode(self, tokens):
        itos = { i: ch for i, ch in enumerate(self.vocab) }
        itos[len(self.vocab)] = '<pad>'
        return ''.join([itos.get(token, unk_token) for token in tokens])

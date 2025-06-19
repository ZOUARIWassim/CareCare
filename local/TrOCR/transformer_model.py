import torch
import torch.nn as nn
import torch.nn.functional as F

#define the transformer model 


block_size=100
n_layer=6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, n_embed):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_embed = n_embed
        # Taille du patch (exemple : 16x16)
        self.conv = nn.Conv2d(3, n_embed, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch_size,channels,height,width) with height=384, width=384
        x = self.conv(x)  # Applique la convolution pour découper en patches et créer les embeddings
        x = x.flatten(2)  # Aplatissement des patches pour chaque position
        x = x.transpose(1, 2)  # Conversion pour que les patches soient en forme (batch_size, n_patches, n_embed)
        return x # (batch_size, N, n_embed)


class single_head(nn.Module):
    def __init__(self, head_size, n_embed):
        super().__init__()
        self.Key=nn.Linear(n_embed,head_size)
        self.Query=nn.Linear(n_embed,head_size)
        self.Value=nn.Linear(n_embed,head_size)
    def forward(self,x):
        # x.shape = (batch_size,N,n_embed)    N: number of patches
        Key=self.Key(x) # x * K ----->(batch_size,N,head_size)
        Query=self.Query(x) # x * Q ----->(batch_size,N,head_size)
        Value=self.Value(x) # x * V ----->(batch_size,N,head_size)
        out=Query@Key.transpose(2,1)/Key.shape[-1]**0.5 # (batch_size,N,N)
        out=F.softmax(out,dim=-1)
        return out @ Value # (batch_size,N,head_size)

class MultiHeadAttention(nn.Module):
    

    def __init__(self, n_embed, head_size):
        super().__init__()
        self.num_heads=n_embed//head_size
        self.heads = nn.ModuleList([single_head(head_size,n_embed) for _ in range(self.num_heads)])
        self.proj = nn.Linear(head_size * self.num_heads, n_embed)

    def forward(self, x):
        # x.shape = (batch_size,N,head_size)    
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (batch_size,N,n_embed)
        out = self.proj(out) # (batch_size,N,n_embed)
        return out 

class FeedFoward(nn.Module):
    

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x) # (batch_size,N,n_embed)

class Block(nn.Module):

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x.shape = (batch_size,N,n_embed)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (batch_size,N,n_embed)

# define the encoder class
class encoder(nn.Module):

    def __init__(self,n_embed,head_size):
        super().__init__()
        self.patch_embed=PatchEmbedding(16,n_embed) # 16x16 patches
        self.position_embedding_table = nn.Embedding(576, n_embed)# 576=384*2/16**2
        self.blocks = nn.Sequential(*[Block(n_embed, head_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        

        
        patch_embed = self.patch_embed(idx) # (Batch_size,N,n_embed)
        pos_emb = self.position_embedding_table(torch.arange(patch_embed.shape[1], device=device)) # (N,n_embed)
        x = patch_embed + pos_emb # (Batch_siz,N,n_embed)
        x = self.blocks(x) # (Batch_size,N,n_embed)
        x = self.ln_f(x) # (Batch_size,N,n_embed)
        return x

# the decoder part
class Head_decoder_1(nn.Module):

    def __init__(self, n_embed,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        Batch_size,len_seq,n_embed = x.shape # x.shape = (Batch_size,len_seq,n_embed)
        k = self.key(x)   # (Batch_size,len_seq,head_size)
        q = self.query(x) # (Batch_size,len_seq,head_size)
        v = self.value(x) # (Batch_size,len_seq,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:len_seq, :len_seq] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
         
        out = wei @ v # (Batch_size,len_seq,head_size)
        return out # (Batch_size,len_seq,head_size)

class MultiHeadAttention_1(nn.Module):

    def __init__(self, n_embed, head_size):
        super().__init__()
        num_heads = n_embed // head_size

        self.heads = nn.ModuleList([Head_decoder_1(n_embed,head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        

    def forward(self, x):
        # x.shape = (batch_size,len_seq,n_embed)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # (batch_size, len_seq, n_embed)
        return out # (batch_size, len_seq, n_embed)

class Head_decoder_2(nn.Module):

    def __init__(self, n_embed,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.multihead = MultiHeadAttention_1(n_embed,head_size)

    def forward(self, x,enc_out):
        # x.shape = (Batch_size,len_seq,n_embed)    
        # enc_out.shape = (Batch_size,N,n_embed)
        attention_x=self.multihead(x) # attention_x.shape = (Batch_size,len_seq,n_embed)
        k = self.key(enc_out)   # enc_out * K ----->(Batch_size,N,head_size)
        q = self.query(attention_x) # x * Q ----->(Batch_size,len_seq,head_size)
        v = self.value(enc_out) # enc_out * V ----->(Batch_size,N,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = F.softmax(wei, dim=-1) 
        
         
        out = wei @ v # (Batch_size,len_seq,head_size)
        return out

class MultiHeadAttention_2(nn.Module):

    def __init__(self, n_embed, head_size):
        super().__init__()
        num_heads = n_embed // head_size

        self.heads = nn.ModuleList([Head_decoder_2(n_embed,head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        

    def forward(self, x,enc_out):
        out = torch.cat([h(x,enc_out) for h in self.heads], dim=-1)
        out = self.proj(out) # (batch_size, len_seq, n_embed)
        return out

class FeedFoward_Decoder(nn.Module):
    

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x) # (batch_size,context_size,n_embed)

class Block_decoder(nn.Module):
    def __init__(self,n_embed,head_size):
        super().__init__()
        self.attention=MultiHeadAttention_1(n_embed,head_size)
        self.attention_2=MultiHeadAttention_2(n_embed,head_size)
        self.ffwd = FeedFoward_Decoder(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)
    def forward(self,x,enc_out):
        # x.shape = (batch_size,len_seq,n_embed)
        # enc_out.shape = (batch_size,N,n_embed)
        x= x + self.attention(self.ln1(x))
        out_1=self.attention_2(self.ln2(x),self.ln2(enc_out))
        x= x + out_1
        x= x + self.ffwd(self.ln3(x))
        return x # (batch_size,len_seq,n_embed)

# define the decoder class

class decoder(nn.Module):
    def __init__(self,n_embed,head_size,vocab_size):
        super().__init__()
        self.blocks = nn.ModuleList([Block_decoder(n_embed, head_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx,enc_out,targets=None):
        #idx.shape = (batch_size,len_seq)
        #enc_out.shape = (batch_size,N,n_embed)
        idx = idx.long()
        token_embed = self.token_embedding(idx) # (batch_size,len_seq,n_embed)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        
        x = token_embed + pos_emb # (batch_size,len_seq,n_embed)
      
        for block in self.blocks:
            x = block(x, enc_out) # (batch_size,len_seq,n_embed)
        x = self.ln_f(x) #  (batch_size,len_seq,n_embed)
        logits = self.lm_head(x) # (batch_size,len_seq,vocab_size)
        if targets is None:
            loss = None
        else:
            
            Batch_size, len_seq, v_size = logits.shape
            logits = logits.view(Batch_size*len_seq, v_size) # (Batch_size*len_seq,vocab_size)
            targets = targets.view(Batch_size*len_seq)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# define the transformer model

class Transformer(nn.Module):
    def __init__(self, n_embed, head_size,vocab_size):
        super().__init__()
        self.encoder = encoder(n_embed, head_size)  # Ton encoder que tu as défini avant
        self.decoder = decoder(n_embed, head_size,vocab_size)

    def forward(self, idx,pixels, targets=None):
        enc_out = self.encoder(pixels)  # (B, context_len, n_embed)

        logits, loss = self.decoder(idx, enc_out, targets)  # (B, target_len, vocab_size), loss

        
        return logits,loss
    def generate(self, idx,pixels, max_new_tokens=20):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond,pixels)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

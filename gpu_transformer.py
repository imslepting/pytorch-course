import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

raw_data = [
    ("I love AI", "我愛人工智慧"),
    ("Deep learning is fun","深度學習很有趣" ),
    ("transformers is powerful","transformer 很強大" ),
    ("This is a ling sequence to test padding mechanism", "這是一個用來測試填充機制的句子"),
    ("Seq2Seq model is cool", "序列對序列模型很酷"),
]

class SimpleTokenizer:
    def __init__(self, data, lang_idx):
        self.word2idx = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0:"<PAD>", 1:"<BOS>", 2:"<EOS>", 3:"<UNK>"}
        vocab = set()
        for pair in data:
            sentences = pair[lang_idx]
            if lang_idx == 0:
                words = sentences.split()
            else:
                words = list(sentences)
            vocab.update(words)

        for i , word in enumerate(vocab):
            self.word2idx[word] = i +4 
            self.idx2word[i+4] = word
    def encode(self, text, lang_type='en'):
        word = text.split() if lang_type =='en' else list(text)
        return [self.word2idx.get(w,3) for w in word]
    def decode(self, indices):
        return "".join([self.idx2word.get(idx, "") for idx in indices if idx not in [0,1,2]])
    
scr_tokenizer = SimpleTokenizer(raw_data, 0)
tgt_tokenizer = SimpleTokenizer(raw_data, 1)

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_ids = [1] + self.src_tokenizer.encode(src_text, 'en') + [2]
        tgt_ids = [1] + self.tgt_tokenizer.encode(tgt_text, 'ch') + [2]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
    
def collate_fn(batch):
    src_batch, tgt_batch = [],[]
    for src_item , tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded
#============定義模型============#

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                d_model=512,
                nhead=8,
                num_layers=3,
                dropout=0.1
                ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True)
        
        self.fc_out = nn.Linear(d_model,tgt_vocab_size)
    def forward(self,src,tgt):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

        src_padding_mask = (src ==0)
        tgt_padding_mask = (tgt ==0)

        outs=self.transformer(
            src = src_emb, 
            tgt = tgt_emb,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask,
        )
        return self.fc_out(outs)
#============訓練模型============#
def train():
    BATCH_SIZE = 2
    EPOCHS = 100
    LR = 0.0001

    dataset = TranslationDataset(raw_data, scr_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        src_vocab_size = len(scr_tokenizer.word2idx),
        tgt_vocab_size = len(tgt_tokenizer.word2idx),
        d_model = 128,
        nhead = 4,
        num_layers = 2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    print("Start training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for src , tgt in dataloader:
            src , tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model
        
# ===========測試模型============#
def translate(model,src_sentence):
    model.eval()

    src_ids = [1] + scr_tokenizer.encode(src_sentence, 'en') + [2]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    tgt_ids = [1]

    print(f"\n原文: {src_sentence}")
    print("翻譯結果: ", end="")

    for i in range(20):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
        next_token_id = logits[0, -1, :].argmax().item()
        tgt_ids.append(next_token_id)
        if next_token_id == 2:
            break
    result = tgt_tokenizer.decode(tgt_ids)
    print(f"結果：{result}")

if __name__ == "__main__":
    trained_model = train()
    test_sentences = [
        "I love AI",
        "This is a long sequence to test padding mechanism",
        "Seq2Seq model is cool"
    ]
    for sentence in test_sentences:
        translate(trained_model, sentence)
    print("Translation completed.")

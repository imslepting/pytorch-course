import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 檢查是否有可用的 GPU，並設定運算設備
# 如果有 GPU 可用，則使用 GPU，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定義原始資料，包含英文句子和對應的中文翻譯
raw_data = [
    ("I love AI", "我愛人工智慧"),
    ("Deep learning is fun","深度學習很有趣" ),
    ("transformers is powerful","transformer 很強大" ),
    ("This is a ling sequence to test padding mechanism", "這是一個用來測試填充機制的句子"),
    ("Seq2Seq model is cool", "序列對序列模型很酷"),
]

# 定義一個簡單的分詞器，用於將句子轉換為詞或字的索引
class SimpleTokenizer:
    def __init__(self, data, lang_idx):
        # 初始化詞彙表，包含特殊符號
        self.word2idx = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3} # padding, begin of sentence, end of sentence, unknown
        self.idx2word = {0:"<PAD>", 1:"<BOS>", 2:"<EOS>", 3:"<UNK>"}
        vocab = set()
        for pair in data:
            sentences = pair[lang_idx]
            if lang_idx == 0: 
                words = sentences.split() # 英文用空格分詞
            else:
                words = list(sentences) # 中文直接分字
            vocab.update(words)

        # 將詞彙加入詞彙表
        for i , word in enumerate(vocab):
            self.word2idx[word] = i +4 # 0-3 已被特殊符號佔用
            self.idx2word[i+4] = word

    # 將文字編碼為索引序列
    def encode(self, text, lang_type='en'):
        word = text.split() if lang_type =='en' else list(text) # 英文用空格切詞，中文直接切字 
        return [self.word2idx.get(w,3) for w in word] # 找不到的字用<UNK>代替

    # 將索引序列解碼為文字
    def decode(self, indices):
        # 因為中文字是單字，所以直接將索引對應的字串連接起來
        return "".join([self.idx2word.get(idx, "") for idx in indices if idx not in [0,1,2]]) # 忽略padding, bos, eos

# 初始化英文和中文的分詞器
scr_tokenizer = SimpleTokenizer(raw_data, 0) # 英文分詞器 source  
tgt_tokenizer = SimpleTokenizer(raw_data, 1) # 中文分詞器 target

# 定義翻譯數據集類別
class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 將原始句子轉換為索引序列，並在開頭和結尾加入特殊符號
        src_text, tgt_text = self.data[idx]
        src_ids = [1] + self.src_tokenizer.encode(src_text, 'en') + [2]
        tgt_ids = [1] + self.tgt_tokenizer.encode(tgt_text, 'ch') + [2]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# 定義自定義的 collate 函數，用於處理 batch 中的 padding
def collate_fn(batch):
    src_batch, tgt_batch = [],[]
    for src_item , tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)

    # 將序列 padding 到相同長度，形狀為 (batch_size, seq_len)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded

# 定義 Seq2Seq Transformer 模型
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
        # 定義源語言和目標語言的嵌入層
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)  # embedding 之後形狀為 (batch_size, seq_len, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)  # 這裡決定 transformer 層的輸出形狀
        # 定義 Transformer 模型
        self.transformer = nn.Transformer( 
            d_model=d_model, # embedding 維度
            nhead=nhead, # 多頭注意力機制頭數
            num_encoder_layers=num_layers, # 3 層 encoder
            num_decoder_layers=num_layers, # 3 層 decoder
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True)
        
        # 定義全連接層，將 Transformer 的輸出形狀從 (batch_size, tgt_seq_len, d_model)
        # 映射為 (batch_size, tgt_seq_len, tgt_vocab_size)
        self.fc_out = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt):
        # 將輸入嵌入並進行縮放
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) 
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # 生成目標序列的遮罩
        tgt_seq_len = tgt.size(1) # size(1) 是序列長度
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device) 

        # 生成源序列和目標序列的 padding 遮罩
        src_padding_mask = (src ==0)
        tgt_padding_mask = (tgt ==0)

        # 通過 Transformer 模型進行前向傳播
        outs=self.transformer(
            src = src_emb, 
            tgt = tgt_emb,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask,
        )
        return self.fc_out(outs)

# 定義訓練函數
def train():
    BATCH_SIZE = 2
    EPOCHS = 100
    LR = 0.0001

    # 初始化數據集和數據加載器
    dataset = TranslationDataset(raw_data, scr_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 初始化模型
    model = Seq2SeqTransformer(
        src_vocab_size = len(scr_tokenizer.word2idx),
        tgt_vocab_size = len(tgt_tokenizer.word2idx),
        d_model = 128,
        nhead = 4,
        num_layers = 2,
    ).to(device)

    # 定義優化器和損失函數
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    print("Start training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for src , tgt in dataloader:
            src , tgt = src.to(device), tgt.to(device)
            # 假設 tgt = {<BOS> ... <EOS>} 輸入給 decoder
            tgt_input = tgt[:, :-1] # 去掉 <EOS> 當作輸入
            tgt_output = tgt[:, 1:] # 去掉 <BOS> 當作標籤，形狀為 {batch_size, seq_len-1}

            optimizer.zero_grad()
            logits = model(src, tgt_input)  # {batch_size, seq_len-1, tgt_vocab_size}
                                            # logits.size(-1) = {tgt_vocab_size}
            
            # (batch_size * (tgt_seq_len - 1), vocab_size) vs (batch_size * (tgt_seq_len - 1))
            # nn.CrossEntropyLoss 的輸入要求：
            # 預測值（logits）：形狀為 (N, C)，其中 N 是樣本數，C 是類別數（即 vocab_size）。
            # 目標值（tgt_output）：形狀為 (N)，其中 N 是樣本數。
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model
        
# 定義翻譯函數
def translate(model,src_sentence):
    model.eval()

    # 將輸入句子編碼為索引序列
    src_ids = [1] + scr_tokenizer.encode(src_sentence, 'en') + [2]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    tgt_ids = [1] # 初始化目標序列，僅包含 <BOS>

    print(f"\n原文: {src_sentence}")
    print("翻譯結果: ", end="")

    for i in range(20):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
        next_token_id = logits[0, -1, :].argmax().item() # (batch_size, tgt_seq_len, vocab_size)
        tgt_ids.append(next_token_id)
        if next_token_id == 2: # 遇到 <EOS> 就停止
            break
    result = tgt_tokenizer.decode(tgt_ids)
    print(f"結果：{result}")

if __name__ == "__main__":
    # 訓練模型
    trained_model = train()
    # 測試句子
    test_sentences = [
        "I love AI",
        "This is a long sequence to test padding mechanism",
        "Seq2Seq model is cool"
    ]
    for sentence in test_sentences:
        translate(trained_model, sentence)
    print("Translation completed.")

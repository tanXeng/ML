from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from model import Transformer
import config as config
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("t5-small")

train_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='train')
val_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='validation')
test_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='test')

print(f"Train dataset: {train_dataset}")
print(f"Validation dataset: {val_dataset}")
print(f"Test dataset: {test_dataset}")

def prepare_data():
    dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')
    train_df = pd.DataFrame(dataset['train']['translation'])
    valid_df = pd.DataFrame(dataset['validation']['translation'])
    test_df = pd.DataFrame(dataset['test']['translation'])
    return train_df, valid_df, test_df

train_df, val_df, test_df = prepare_data()

def load_data(batch_size=32):
    dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    train_df = pd.DataFrame(dataset['train']['translation'])
    val_df = pd.DataFrame(dataset['validation']['translation'])
    test_df = pd.DataFrame(dataset['test']['translation'])
    
    # train_df['de'] = '<s> ' + train_df['de'] + ' </s>'
    # val_df['de'] = '<s> ' + val_df['de'] + ' </s>'
    # test_df['de'] = '<s> ' + test_df['de'] + ' </s>'
    
    train_ds = TranslationDataset(train_df, tokenizer, 128)
    val_ds = TranslationDataset(val_df, tokenizer, 128)
    test_ds = TranslationDataset(test_df, tokenizer, 128)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, tokenizer

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['en']
        tgt_text = self.data.iloc[idx]['de']
        
        src_enc = self.tokenizer(
            src_text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'src_ids': src_enc['input_ids'].squeeze(),
            'tgt_ids': tgt_enc['input_ids'].squeeze(),
            'src_mask': src_enc['attention_mask'].squeeze(),
            'tgt_mask': tgt_enc['attention_mask'].squeeze()
        }

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-5):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    i=1
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            outputs = model(src_ids, tgt_ids[:, :-1])
            loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), 
                            tgt_ids[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f'     Batch {i}, Train Loss: {loss.item()}')
            i += 1

        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch['src_ids'].to(device)
                tgt_ids = batch['tgt_ids'].to(device)
                
                outputs = model(src_ids, tgt_ids[:, :-1])
                loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), 
                                tgt_ids[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

            val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}\n')
        
        scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },
            './translator.pth')
            print('Saving model...')
          

# main training loop
if __name__ == '__main__':
    train_loader, val_loader, _, tokenizer = load_data()
    
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=128
    ).to(device)
    
    train_model(model, train_loader, val_loader)

    

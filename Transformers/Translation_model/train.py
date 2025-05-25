from datasets import load_dataset
import pandas as pd
from model import Transformer
from config import config
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau  

# train_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='train')
# val_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='validation')
# test_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True, split='test')

# print(f"Train dataset: {train_dataset}")
# print(f"Validation dataset: {val_dataset}")
# print(f"Test dataset: {test_dataset}")

def load_data(train_df, val_df, test_df, tokenizer, batch_size=32, sampler=None):

    # tokenizer = AutoTokenizer.from_pretrained('t5-small')
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    
    train_ds = TranslationDataset(train_df, tokenizer, 128)
    val_ds = TranslationDataset(val_df, tokenizer, 128)
    test_ds = TranslationDataset(test_df, tokenizer, 128)
    
    if sampler:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

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
            return_tensors='pt',
            add_special_tokens=True
        )
        
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True, 
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'src_ids': src_enc['input_ids'].squeeze(),
            'tgt_ids': tgt_enc['input_ids'].squeeze(),
            'src_mask': src_enc['attention_mask'].squeeze(),
            'tgt_mask': tgt_enc['attention_mask'].squeeze()
        }

def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device,
    best_val_loss=float('inf'), num_epochs=50, lr=1e-4, clip_grad_norm=1.0
):  
    pad_token_id = model.tokenizer.pad_token_id 
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        i=1
        for batch in train_loader:
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            outputs = model(src_ids, tgt_ids[:, :-1])
            loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), 
                            tgt_ids[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
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
        print(f'Validation Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss

            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_validation_loss': best_val_loss
            },
            './translator.pth')
            print('Saved model...')

# main training loop
if __name__ == '__main__':

    TOKENIZER = config['tokenizer']
    D_MODEL = config['d_model']
    NUM_HEADS = config['num_heads']
    NUM_LAYERS = config['num_layers']
    D_FF = config['d_ff']
    MAX_SEQ_LEN = config['max_seq_len']
    DROPOUT = config['dropout']
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LR = config['learning_rate']
    CLIP_GRAD_NORM = config['clip_grad_norm']
    DEVICE = config['device']

    dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')
    # dataset = load_dataset("wmt14", "de-en")
    print(dataset)

    train_df = pd.DataFrame(dataset['train']['translation'])
    val_df = pd.DataFrame(dataset['validation']['translation'])
    test_df = pd.DataFrame(dataset['test']['translation'])

    # sampler = RandomSampler(train_df, replacement=True, num_samples=32)
    train_loader, val_loader, test_loader = load_data(
        train_df,
        val_df,
        test_df, 
        TOKENIZER,
        batch_size=32, 
        # sampler=sampler
    )
    
    model = Transformer(
        TOKENIZER,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1)

    checkpoint = torch.load('./translator.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_validation_loss']
    print('Starting the training...')
    # print(device)
    train_model(
        model,
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler,
        DEVICE,
        best_val_loss=best_val_loss,
        num_epochs=200,
        clip_grad_norm=CLIP_GRAD_NORM
    )

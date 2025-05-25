import sys
from model import Transformer
from train import TranslationDataset
import torch
import torch.nn.functional as F
from config import config

def translate(input, tokenizer, model, device, max_len=128, k=5, temperature=1):

    # Different tokenizers might have different beginning of sentence tokens
    start_token_id = tokenizer.pad_token_id 
    end_token_id = tokenizer.eos_token_id  
    
    decoder_inputs_ids = torch.tensor(
        start_token_id,
        dtype=torch.long,
        device=device
    ).unsqueeze(0).unsqueeze(1)
    encoder_inputs_ids = input['input_ids']

    for i in range(max_len):
        with torch.no_grad():
            outputs = model(
                encoder_inputs_ids, 
                decoder_inputs_ids
            )

        next_token_logits = outputs[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        if k:
            topk_values, topk_indices = torch.topk(next_token_logits, k, largest=True)
            probs = F.softmax(topk_values, dim=-1)
            next_token_id = topk_indices.gather(-1, torch.multinomial(probs, num_samples=1))
        

        decoder_inputs_ids = torch.cat(
            [decoder_inputs_ids, next_token_id],
            dim=-1
        )
        
        if next_token_id.item() == end_token_id:
            break

    translated_text = tokenizer.decode(
        decoder_inputs_ids[0], 
        skip_special_tokens=True
    )

    return translated_text

if __name__=='__main__':

    TOKENIZER = config['tokenizer']
    D_MODEL = config['d_model']
    NUM_HEADS = config['num_heads']
    NUM_LAYERS = config['num_layers']
    D_FF = config['d_ff']
    MAX_SEQ_LEN = config['max_seq_len']
    DEVICE = config['device']

    # tokenizer = AutoTokenizer.from_pretrained("t5-small")
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    vocab = TOKENIZER.get_vocab()

    inputs = str(sys.argv[1])
    # temperature = float(sys.argv[2])
    inputs = TOKENIZER(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    model = Transformer(
            TOKENIZER,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            d_ff=D_FF,
            max_seq_len=MAX_SEQ_LEN
        ).to(DEVICE)

    checkpoint = torch.load('./translator.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    translated_text = translate(inputs, TOKENIZER, model, DEVICE, max_len=MAX_SEQ_LEN)

    print("German Translated:", translated_text)

import sys
from model import Transformer
from train import TranslationDataset
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

def translate(input, tokenizer, model, max_len=128, k=None, temperature=1):
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
        if k:
            next_token_logits = torch.topk(next_token_logits, k, largest=True)[0]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    vocab = tokenizer.get_vocab()

    inputs = str(sys.argv[1])
    # temperature = float(sys.argv[2])
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model = Transformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=128
        ).to(device)

    checkpoint = torch.load('./translator.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    translated_text = translate(inputs, tokenizer, model, max_len=128)

    print("German Translated:", translated_text)

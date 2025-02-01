from attention_with_RoPE import *

class TranslationTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        pad_token_id = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def train_step(self, enc_input, dec_input, target):
        """
        enc_input: 영어 문장의 토큰 시퀀스 [batch_size, enc_seq_len]
        dec_input: 한글 문장의 토큰 시퀀스 (시작 토큰 포함, 마지막 토큰 제외)
                  [batch_size, dec_seq_len]
        target: 한글 문장의 토큰 시퀀스 (시작 토큰 제외)
                [batch_size, dec_seq_len]
        """



        self.optimizer.zero_grad()

        output = self.model(enc_input, dec_input)

        # output = torch.Size([batch, decoder_input_length, vocab_size])

        output = output.contiguous().view(-1, output.size(-1))  # [batch_size * dec_seq_len, vocab_size]
        target = target.contiguous().view(-1)  # [batch_size * dec_seq_len]

        # loss 계산
        loss = self.criterion(output, target)

        # 역전파
        loss.backward()
        self.optimizer.step()

        return loss.item()
from datasets import load_dataset

ds = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, eng_tokenizer, kor_tokenizer, max_length=128):
        """
        hf_dataset: HuggingFace dataset (train/validation/test split)
        eng_tokenizer: 영어 토크나이저
        kor_tokenizer: 한글 토크나이저
        max_length: 최대 시퀀스 길이
        """
        self.dataset = hf_dataset
        self.eng_tokenizer = eng_tokenizer
        self.kor_tokenizer = kor_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # HF dataset에서 데이터 가져오기
        item = self.dataset[idx]
        eng_sent = item['english']
        kor_sent = item['korean']

        # 토큰화

        eng_tokens = self.eng_tokenizer(eng_sent,
                                        max_length = self.max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors = "pt")
        kor_tokens = self.kor_tokenizer(kor_sent,
                                        max_length = self.max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors = "pt")
        return {
            'eng_tokens': eng_tokens['input_ids'].squeeze(0),  # [seq_len]
            'eng_attention_mask': eng_tokens['attention_mask'].squeeze(0),  # [seq_len]
            'kor_tokens': kor_tokens['input_ids'].squeeze(0),  # [seq_len]
            'kor_attention_mask': kor_tokens['attention_mask'].squeeze(0)  # [seq_len]
        }
    
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
eng_tokenizer = tokenizer
kor_tokenizer = tokenizer
datasetloader = TranslationDataset(ds, tokenizer, tokenizer)

def collate_fn(batch):
    """
    BERT 토크나이저는 이미 패딩을 처리하므로 단순히 배치로 묶어주기만 하면 됩니다
    """
    return {
        'eng_tokens': torch.stack([item['eng_tokens'] for item in batch]),
        'eng_attention_mask': torch.stack([item['eng_attention_mask'] for item in batch]),
        'kor_tokens': torch.stack([item['kor_tokens'] for item in batch]),
        'kor_attention_mask': torch.stack([item['kor_attention_mask'] for item in batch])
    }

from torch.utils.data import DataLoader

def create_dataloaders(dataset_dict, eng_tokenizer, kor_tokenizer,
                      batch_size=32, max_length=64):
    dataloaders = {}

    for split in ['train', 'validation', 'test']:
        dataset = TranslationDataset(
            dataset_dict[split],
            eng_tokenizer=eng_tokenizer,
            kor_tokenizer=kor_tokenizer,
            max_length=max_length
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate_fn,
            num_workers=4
        )

    return dataloaders

num_epochs = 5
batch_size = 64

config.intermediate_size = 768*4
config.num_attention_heads = 6
config.num_hidden_layers = 8
#config.hidden_dropout_prob = 0.2  # 기존보다 높게 설정
#config.attention_probs_dropout_prob = 0.2


model = Transformer(config)

model = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
trainer = TranslationTrainer(model, optimizer, nn.CrossEntropyLoss(label_smoothing=0.1))
dataloader = create_dataloaders(ds, eng_tokenizer, kor_tokenizer, batch_size=batch_size)
val_dataloader = dataloader['validation']
train_dataloader = dataloader['train']

from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_dataloader),
    epochs=num_epochs
)

def validate(model, val_dataloader, tokenizer, num_examples=5):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

    model.eval()
    total_val_loss = 0
   
    examples = []
    example_count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            enc_input = batch['eng_tokens'].to(device)
            dec_input = batch['kor_tokens'][:, :-1].to(device)
            target = batch['kor_tokens'][:, 1:].to(device)
            
            output = model(enc_input, dec_input)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_val_loss += loss.item()
        if example_count < num_examples:
            pred_tokens = output[0].argmax(dim=-1)
            src_text = tokenizer.decode(enc_input[0], skip_special_tokens = True)
            target_full = batch['kor_tokens'][0].to(device)
            tgt_text = tokenizer.decode(target_full, skip_special_tokens = True)

            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)

            examples.append({
                'source': src_text,
                'target' : tgt_text,
                'prediction' : pred_text
            })
            for i, example in enumerate(examples, 1):
                print(f"\nExample {i}:")
                print(f"Source    : {example['source']}")
                print(f"Target    : {example['target']}")
                print(f"Prediction: {example['prediction']}")
                print("-" * 50)
            example_count +=1
            
    return total_val_loss / len(val_dataloader)

from tqdm import tqdm
import wandb
total_steps = len(train_dataloader) * num_epochs
progress_bar = tqdm(total=total_steps, desc="Training")
run = wandb.init(project="transformer")

run.watch(model)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        enc_input = batch['eng_tokens'].to(device)
        dec_input = batch['kor_tokens'][:, :-1].to(device)
        target = batch['kor_tokens'][:, 1:].to(device)
        
        loss = trainer.train_step(enc_input, dec_input, target)
        
        # Gradient Clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update learning rate
        #scheduler.step()
        if batch_idx % 10 == 0:
            val_loss = validate(model, val_dataloader=val_dataloader, tokenizer=tokenizer)
            run.log({"val_loss": val_loss})
        if batch_idx:
            run.log({"loss": loss})

        total_loss += loss
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'test_model.pth')
model.eval()

text = "time flies like an arrow"
inputs2 = tokenizer(text2, return_tensors = "pt", add_special_tokens=False)

decoder_text = "시간이"
inputs3 = tokenizer(decoder_text, return_tensors = "pt", add_special_tokens=False)


class Translator:
    def __init__(self, model, eng_tokenizer, kor_tokenizer, device, max_length=128):
        self.model = model
        self.eng_tokenizer = eng_tokenizer
        self.kor_tokenizer = kor_tokenizer
        self.device = device
        self.max_length = max_length
        
    def translate_greedy(self, text):
        self.model.eval()
        
        # Tokenize English input
        enc_input = self.eng_tokenizer(text, 
                                     return_tensors='pt',
                                     max_length=self.max_length,
                                     truncation=True,
                                     padding='max_length')
        
        # Move to device
        enc_input = enc_input['input_ids'].to(self.device)
        
        # Initialize decoder input with start token
        dec_input = torch.tensor([[self.kor_tokenizer.cls_token_id]]).to(self.device)
        
        output_tokens = []
        
        with torch.no_grad():
            for _ in range(self.max_length):
                # Get model output
                output = self.model(enc_input, dec_input)
                
                # Get the next token prediction
                next_token_logits = output[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
                
                # Break if we predict the end token
                if next_token.item() == self.kor_tokenizer.sep_token_id:
                    break
                    
                output_tokens.append(next_token.item())
                # Update decoder input
                dec_input = torch.cat([dec_input, next_token], dim=1)
        
        # Decode the generated tokens
        translated_text = self.kor_tokenizer.decode(output_tokens, skip_special_tokens=True)
        return translated_text

    def translate_beam_search(self, text, beam_size=3):
        self.model.eval()
        
        # Tokenize English input
        enc_input = self.eng_tokenizer(text, 
                                     return_tensors='pt',
                                     max_length=self.max_length,
                                     truncation=True,
                                     padding='max_length')
        enc_input = enc_input['input_ids'].to(self.device)
        
        # Initialize with start token
        start_token = torch.tensor([[self.kor_tokenizer.cls_token_id]]).to(self.device)
        beams = [(start_token, 0.0)]
        completed_sequences = []
        
        with torch.no_grad():
            for _ in range(self.max_length):
                candidates = []
                
                for dec_input, score in beams:
                    if dec_input[0, -1].item() == self.kor_tokenizer.sep_token_id:
                        completed_sequences.append((dec_input, score))
                        continue
                        
                    # Get model output
                    output = self.model(enc_input, dec_input)
                    next_token_logits = output[:, -1, :]
                    next_token_probs = torch.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top k tokens
                    values, indices = next_token_probs[0].topk(beam_size)
                    
                    for value, token_id in zip(values, indices):
                        new_dec_input = torch.cat([dec_input, 
                                                 token_id.unsqueeze(0).unsqueeze(0)], 
                                                dim=1)
                        candidates.append((new_dec_input, score + value.item()))
                
                # Keep top beam_size candidates
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
                
                # Early stopping if we have enough completed sequences
                if len(completed_sequences) >= beam_size:
                    break
        
        # Choose the sequence with highest score
        if completed_sequences:
            best_sequence = max(completed_sequences, key=lambda x: x[1])[0]
        else:
            best_sequence = max(beams, key=lambda x: x[1])[0]
        
        # Decode tokens (exclude start token)
        output_tokens = best_sequence[0][1:].cpu().numpy().tolist()
        translated_text = self.kor_tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        return translated_text

# Example usage:
def test_translation(model, test_sentence):
    translator = Translator(model, eng_tokenizer, kor_tokenizer, device)
    print("Input:", test_sentence)
    print("Greedy translation:", translator.translate_greedy(test_sentence))
    print("Beam search translation:", translator.translate_beam_search(test_sentence, beam_size=3))



# Example usage:
def test_translation(model_path, test_sentence):
    # Load the saved model
    model = Transformer(config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Create translator instance
    translator = Translator(model, eng_tokenizer, kor_tokenizer, device)
    
    # Test greedy search
    print("Input:", test_sentence)
    print("Greedy translation:", translator.translate_greedy(test_sentence))
    print("Beam search translation:", translator.translate_beam_search(test_sentence, beam_size=3))

# Test the model
test_sentence = "these gorillas were murdered"
test_translation("test_model.pth", test_sentence)
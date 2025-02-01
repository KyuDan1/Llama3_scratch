from transformers import AutoTokenizer
import torch 
from attention_with_RoPE import Transformer
from transformers import AutoConfig

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
eng_tokenizer = kor_tokenizer = tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = AutoConfig.from_pretrained(model_ckpt)

config.intermediate_size = 768*4
config.num_attention_heads = 6
config.num_hidden_layers = 8
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
test_sentence = "Language is more than a probability distribution over words or phonemes"
test_translation("test_model.pth", test_sentence)
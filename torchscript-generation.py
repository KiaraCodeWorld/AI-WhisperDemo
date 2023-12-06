from transformers import AutoModelForCausalLM
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large")

# Input text to encode
#text = "Hello world"
#inputs = tokenizer(text, return_tensors="pt")

dummy_waveform = torch.randint(-32768, 32767, (1, 40000))  # Assuming 40,000 samples per audio clip


# Put model in evaluation mode and trace
model.eval()
#traced_model = torch.jit.trace(model, dummy_waveform)

# Save model
#traced_model.save("whisper-large.pt")


"""
import torch

model = AutoModelForCausalLM.from_pretrained("openai/whisper-large-v3")


model.config

# Create some dummy input tensors
batch_size = 1
sequence_length = 128
input_ids = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
attention_mask = torch.ones((batch_size, sequence_length))

# Create the trace
#traced_model = torch.jit.trace(model, (input_ids, attention_mask))

# Save the trace as a .pt file
#torch.jit.save(traced_model, "whisper-large.pt")



model = WhisperingTinyForCausalLM.from_pretrained("facebook/whisper-tiny")
tokenizer = WhisperingTinyTokenizer.from_pretrained("facebook/whisper-tiny")


model_id = "/Users/abhijeetrajput/myfiles/huggigface-whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperPreprocessor.from_pretrained("openai/whisper-tiny")

model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 512))
dummy_attention_mask = torch.ones((1, 512))

traced_model = torch.jit.trace(model, (dummy_input_ids, dummy_attention_mask))
"""
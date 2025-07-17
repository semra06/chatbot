from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
import gradio as gr

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

def generate_text(text):
    encoded_input = tokenizer(text, return_tensors="tf")
    output = model.generate(
        encoded_input["input_ids"],
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Bir şeyler yazın..."),
    outputs="text",
    title="GPT-2 Demo",
    description="Bir metin girin, GPT-2 devam ettirsin."
)

iface.launch()

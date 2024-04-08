import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vietnamese to English Translator")
        
        # Initialize tokenizer and model
        self.tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
        
        # Define colors
        self.bg_color = "#f0f0f0"  # Light gray
        self.text_color = "#333333"  # Dark gray
        self.button_bg = "#4CAF50"  # Green
        self.button_fg = "white"  # White

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Configure root window background
        self.root.configure(bg=self.bg_color)

        # Input label and text area
        self.label_input = tk.Label(self.root, text="Enter Vietnamese text:", bg=self.bg_color, fg=self.text_color)
        self.label_input.pack()
        self.text_input = scrolledtext.ScrolledText(self.root, width=40, height=5, bg="white", fg=self.text_color)
        self.text_input.pack()

        # Output label and text area
        self.label_output = tk.Label(self.root, text="Translated English text:", bg=self.bg_color, fg=self.text_color)
        self.label_output.pack()
        self.text_output = scrolledtext.ScrolledText(self.root, width=40, height=5, bg="white", fg=self.text_color, state=tk.DISABLED)
        self.text_output.pack()

        # Translate button
        self.button_translate = tk.Button(self.root, text="Translate", command=self.translate_vi2en, bg=self.button_bg, fg=self.button_fg)
        self.button_translate.pack()

    def translate_vi2en(self):
        # Get input text
        vi_text = self.text_input.get("1.0", tk.END).strip()
        if vi_text:
            # Translate input text to English
            input_ids = self.tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
            output_ids = self.model_vi2en.generate(
                input_ids,
                decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
                num_return_sequences=1,
                num_beams=5,
                early_stopping=True
            )
            en_text = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
            en_text = " ".join(en_text)
            
            # Display translated text
            self.text_output.config(state=tk.NORMAL)
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, en_text)
            self.text_output.config(state=tk.DISABLED)

# Create the Tkinter application
root = tk.Tk()
app = TranslatorApp(root)
root.mainloop()

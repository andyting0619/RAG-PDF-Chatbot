import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings


warnings.filterwarnings('ignore')


class Chatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(
            'all-MiniLM-L12-v2', device=self.device)

        print("Loading language model...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.chunks = []
        self.embeddings = None
        self.pdf_text = ""

        print("Models loaded successfully.")

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                num_pages = len(pdf_reader.pages)
                print(f"Extracting text from {num_pages} pages...")

                for i, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + "\n"
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{num_pages} pages...")

                return text, num_pages
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return f"Error: {str(e)}", 0

    def chunk_text(self, text, chunk_size=200, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def process_pdf(self, pdf_file):
        if pdf_file is None:
            return "Please upload a PDF file first."
        try:
            print(f"Processing PDF: {pdf_file}")
            self.pdf_text, num_pages = self.extract_text_from_pdf(pdf_file)

            if self.pdf_text.startswith("Error"):
                return self.pdf_text

            if not self.pdf_text.strip():
                return "No text found in PDF. The PDF might be image-based or empty."

            print("Creating text chunks...")
            self.chunks = self.chunk_text(self.pdf_text)

            if not self.chunks:
                return "Could not create chunks from PDF text."

            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            self.embeddings = self.embedder.encode(
                self.chunks,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )

            return f" PDF processed successfully.\n Pages: {num_pages}\n Chunks: {len(self.chunks)}\n Ready to chat."

        except Exception as e:
            print(f"Error: {str(e)}")
            return f"Error: {str(e)}"

    def retrieve_relevant_chunks(self, query, top_k=4):
        if self.embeddings is None:
            return []

        query_embedding = self.embedder.encode([query], convert_to_tensor=True)

        similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            self.embeddings.cpu().numpy()
        )[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices]

        return relevant_chunks

    def generate_response(self, query, context):
        prompt = f"""<|system|>
        You are a helpful assistant. Answer based on the context provided. Be concise.</|system|>
        <|user|>
        Context: {context[:2000]}

        Question: {query}</|user|>
        <|assistant|>"""

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1800).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response

    def chat(self, message, history):
        if not self.chunks:
            return history + [[message, "Please upload and process a PDF first."]]

        if not message or message.strip() == "":
            return history

        try:
            relevant_chunks = self.retrieve_relevant_chunks(message, top_k=4)
            context = "\n\n".join(relevant_chunks)

            response = self.generate_response(message, context)

            history = history + [[message, response]]
            return history

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [[message, error_msg]]
            return history

    def summarize_pdf(self):
        if not self.pdf_text or not self.chunks:
            return "Please upload and process a PDF first."

        try:
            num_chunks = min(len(self.chunks), 8)
            summary_text = " ".join(self.chunks[:num_chunks])[:1200]

            prompt = f"""<|system|>
            You are a helpful assistant. Provide a brief summary of the text.</|system|>
            <|user|>
            Summarize this text in 3-4 sentences:

            {summary_text}</|user|>
            <|assistant|>"""

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

            summary = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            if "<|assistant|>" in summary:
                summary = summary.split("<|assistant|>")[-1].strip()

            return summary

        except Exception as e:
            return f"Error: {str(e)}"


print("Initializing RAG Chatbot...")
bot = Chatbot()

custom_css = """
* {
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

.gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}

h1 {
    text-align: center !important;
    color: #58a6ff !important;
    font-size: 2em !important;
    font-weight: 600 !important;
    margin: 20px 0 !important;
}

.subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 1.1em;
    margin-bottom: 25px;
}

.developer-credit {
    text-align: center;
    color: #6e7681;
    font-size: 0.95em;
    margin-top: 20px;
    padding-top: 15px;
    border-top: 1px solid #21262d;
}

button {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: white !important;
    font-size: 1em !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
}

button:hover {
    background: #2ea043 !important;
}

.chatbot {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    background: #161b22 !important;
}

.message {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 12px !important;
    color: #c9d1d9 !important;
    font-size: 1em !important;
}

input, textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-size: 1em !important;
    border-radius: 6px !important;
}

input:focus, textarea:focus {
    border-color: #58a6ff !important;
}

.file-upload {
    border: 2px dashed #30363d !important;
    background: #161b22 !important;
    border-radius: 6px !important;
}

label {
    color: #c9d1d9 !important;
    font-size: 1em !important;
    font-weight: 500 !important;
}

.output-text {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    padding: 12px !important;
    font-size: 1em !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1>RAG PDF Chatbot</h1>")
    gr.HTML("<div class='subtitle'>AI-Powered PDF Q&A System</div>")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status_output = gr.Textbox(
                label="Status", lines=4, interactive=False)
            summarize_btn = gr.Button("Summarize PDF", variant="secondary")
            summary_output = gr.Textbox(
                label="Summary", lines=8, interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500
            )
            msg = gr.Textbox(
                placeholder="Ask a question about the PDF:",
                show_label=False,
                container=False
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", scale=3)
                clear = gr.Button("Clear", scale=1)

    gr.HTML("<div class='developer-credit'>Developed by Andy Ting</div>")

    process_btn.click(
        fn=bot.process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )

    summarize_btn.click(
        fn=bot.summarize_pdf,
        outputs=[summary_output]
    )

    msg.submit(
        fn=bot.chat,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", None, [msg])

    submit_btn.click(
        fn=bot.chat,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", None, [msg])

    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    print("Starting server...")
    demo.launch(share=True)

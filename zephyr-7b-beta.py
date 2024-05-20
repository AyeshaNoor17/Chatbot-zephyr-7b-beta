import gradio as gr
from huggingface_hub import InferenceClient

# Markdown description
DESCRIPTION = '''
<div>
<h1 style="text-align: center;">zephyr-7b-beta</h1>
<p>This Space demonstrates the instruction-tuned model <b>zephyr-7b-beta by Hugging face</b>. zephyr-7b-beta is the new open 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.<br><i>It can work as a summarizer, sentiment analyzer and Q/A chatbot with multiple other facilities.</i> 
<br>Feel free to play with it, or duplicate to run privately!</p>
</div>
'''

# License markdown
LICENSE = """
<p/>
---
Built with zephyr-7b-beta
"""

# Placeholder HTML
PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="data:image/png;base64,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" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">zephyr-7b-beta</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""

# CSS styles
css = """
h1 {
  text-align: center;
  display:block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

# Initialize InferenceClient
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Function to respond to user messages
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

# Create a Chatbot
chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

# Define the interface layout
with gr.Blocks(fill_height=True,css=css) as demo:
    # Add description markdown
    gr.Markdown(DESCRIPTION)
    # Add duplicate button
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    # Add chat interface
    gr.ChatInterface(
        fill_height=True,
        fn=respond,
        chatbot=chatbot,
        examples=[
            ['How to setup a human base on Mars? Give short answer.'],
            ['Explain theory of relativity to me like I’m 8 years old.'],
            ['What is 9,000 * 9,000?'],
            ['Write a pun-filled happy birthday message to my friend Alex.'],
            ['Justify why a penguin might make a good king of the jungle.']
        ],
        cache_examples=False,
        additional_inputs_accordion = gr.Accordion(label="⚙️ Parameters", open=False, render= False),
            additional_inputs = [
                gr.Textbox(value="You are a friendly Chatbot.", label="System message",render= False),
                gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens",render= False),
                gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature",render= False),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling)",
                    render= False
                ),
            ]
        
    )
        # Add license markdown
    gr.Markdown(LICENSE)

# Launch the interface
if __name__ == "__main__":
    demo.launch()

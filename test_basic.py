import gradio as gr

def test_function(text):
    return f"Hello {text}!"

with gr.Blocks() as demo:
    gr.Markdown("# Test Interface")
    text_input = gr.Textbox(label="Enter text")
    text_output = gr.Textbox(label="Output")
    btn = gr.Button("Test")
    btn.click(test_function, inputs=text_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
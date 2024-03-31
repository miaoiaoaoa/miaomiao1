# coding = utf-8
import json
import os
import time
os.environ["HF_ENDPOINT"]="http://hf-mirror.com"
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from options import parser

history = []
readable_history = []
cmd_opts = parser.parse_args()
model_path="D:\\SSDOWN\\ChatGLM-webui\\ChatGLM-webui\\model\\chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained("D:\SSDOWN\ChatGLM-webui\ChatGLM-webui\model\chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\SSDOWN\ChatGLM-webui\ChatGLM-webui\model\chatglm-6b", trust_remote_code=True)
url =" jetbrains://pycharm/navigate/reference?project=webui.py&path=miao.png"

path = ['background.jpg']

_css = """

#del-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
    margin: 1.5em 0;
}

.gradio-container {
background-color: #4096ff;
background-image : url('file=background.jpg');
}

footer {
display:none!important;
}


:root {
  --button-secondary-background: #4096ff;
  --button-secondary-text-color: white;
  --button-secondary-border-color: #4096ff;
  --panel-background:#red;
  --block-border-color:black;
  --block-background:#FBFBFB;
  --block-label-background: #892828;
  --button-secondary-background-hover:pink;
}

h1 {
margin-top: 30px;
font-size: 64px;
background:-webkit-linear-gradient(top, black, grey);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}

.icon-buttons.svelte-tsr9e2 {
display:none;
}

.svelte-1b6s6s float {
display:none;
}


label.svelte-1b6s6s  {
display:none;
}



button.svelte-196hf5y {
    display: none;
    }



"""

def prepare_model():
    global model
    if cmd_opts.cpu:
        model = model.float()
    else:
        if cmd_opts.precision == "fp16":
            model = model.half().cuda()
        elif cmd_opts.precision == "int4":
            model = model.half().quantize(4).cuda()
        elif cmd_opts.precision == "int8":
            model = model.half().quantize(8).cuda()
    model = model.eval()
prepare_model()


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


def predict(query):
    global history
    output, history = model.chat(
        tokenizer, query=query, history=history,

    )
    readable_history.append((query, parse_codeblock(output)))
    print(output)
    return readable_history


def save_history():
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    s = [{"q": i[0], "o": i[1]} for i in history]
    filename = f"save-{int(time.time())}.json"
    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        f.write(json.dumps(s, ensure_ascii=False))


def load_history(file):
    global history, readable_history
    try:
        with open(file.name, "r", encoding='utf-8') as f:
            j = json.load(f)
            _hist = [(i["q"], i["o"]) for i in j]
            _readable_hist = [(i["q"], parse_codeblock(i["o"])) for i in j]
    except Exception as e:
        print(e)
        return readable_history
    history = _hist.copy()
    readable_history = _readable_hist.copy()
    return readable_history


def clear_history():
    history.clear()
    readable_history.clear()
    return gr.update(value=[])

def clear_message(message):
    message.label = ""


def create_ui():
    with gr.Blocks(css=_css) as demo:
        prompt = "请输入你遇到的代码问题..."
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>喵喵苗的代码助手</center></h2>""")
                

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button(elem_id="_Button",value="清空对话（上下文）")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')
                        image_path='miao.png'
                        gr.Image(image_path)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=500)
                with gr.Row():
                    message = gr.Textbox(placeholder=prompt, show_label=False, lines=2)#这是》》》
                    print(message.label)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")
                with gr.Row():
                    submit = gr.Button("发送")


        submit.click(predict, inputs=[
            message,

        ], outputs=[
            chatbot
        ]).then(clear_message(message))
        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[message], outputs=[message])

        save_his_btn.click(save_history)
        load_his_btn.upload(load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])

    return demo


ui = create_ui()
ui.launch(
    file_directories=path,
    server_name="0.0.0.0" if cmd_opts.listen else None,
    server_port=cmd_opts.port,
    share=True

)

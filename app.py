import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization")
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def resumer_texte(texte):
    res = summarizer(texte, max_length=120, min_length=30, do_sample=False)
    return res[0]['summary_text']

def generer_qcm(texte):
    texte_hl = f"<hl> {texte} <hl>"
    question = qg_pipeline(texte_hl)[0]['generated_text']
    return question

tabs = gr.TabbedInterface(
    [
        gr.Interface(fn=resumer_texte, inputs=gr.Textbox(lines=10), outputs="text", title="Résumé automatique"),
        gr.Interface(fn=generer_qcm, inputs=gr.Textbox(lines=10), outputs="text", title="Génération de QCM")
    ],
    tab_names=["Résumé", "QCM"]
)

tabs.launch()

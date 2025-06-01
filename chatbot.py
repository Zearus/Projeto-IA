import pandas as pd
import warnings
import torch
import gradio as gr
import requests
from deep_translator import GoogleTranslator
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===============================
# Função de Tradução
# ===============================
def traduzir_texto(texto, to_lang='en'):
    try:
        return GoogleTranslator(source='auto', target=to_lang).translate(texto)
    except:
        return texto

# ===============================
# Carregamento e preparação dos dados da planilha CSV
# ===============================
df = pd.read_csv('/content/remedios.csv', sep=';').dropna(subset=['Indicação'])
colunas = ['Remédio', 'Indicação', 'Posologia (Como usar)', 'Efeitos Colaterais Principais',
           'Contraindicações Principais', 'Observações Úteis']

for col in colunas:
    df[col] = df[col].astype(str).apply(lambda x: traduzir_texto(x, 'en'))

# ===============================
# Busca na API pública da OpenFDA (fallback)
# ===============================
def buscar_na_openfda(pergunta_en):
    url = "https://api.fda.gov/drug/label.json"
    query = f"?search=indications_and_usage:{pergunta_en}&limit=5"

    try:
        resposta = requests.get(url + query)
        dados = resposta.json().get("results", [])
        if not dados:
            return None

        resultados = []
        for item in dados:
            nome = item.get("openfda", {}).get("brand_name", ["Desconhecido"])[0]
            uso = item.get("indications_and_usage", ["Sem informação"])[0][:400]
            efeitos = item.get("adverse_reactions", ["Não informado"])[0][:300]
            resultados.append(f"💊 **{nome}**\n- Indicação: {uso}\n- Efeitos colaterais: {efeitos}\n")

        return traduzir_texto("\n\n".join(resultados), to_lang='pt')

    except Exception as e:
        return None

# ===============================
# Indexação semântica com FAISS e MiniLM
# ===============================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Concatenar contexto com menos ruído

def montar_contexto_vetorial(row):
    return (
        f"{row['Remédio']} - Indication: {row['Indicação']} - Usage: {row['Posologia (Como usar)']}"
    )

documentos = [
    Document(page_content=montar_contexto_vetorial(row), metadata={"linha_completa": row.to_dict()})
    for _, row in df.iterrows()
]

index = FAISS.from_documents(documentos, embeddings)

# ===============================
# Carregamento do modelo TinyLLaMA
# ===============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, truncation=True)

# ===============================
# Montagem do prompt para o modelo
# ===============================
def gerar_prompt(remedios, pergunta_en):
    contexto = ""
    for i, r in enumerate(remedios):
        contexto += (
            f"Remedy {i+1}: {r['Remédio']}\n"
            f"Indication: {r['Indicação']}\n"
            f"Usage: {r['Posologia (Como usar)']}\n"
            f"Side effects: {r['Efeitos Colaterais Principais']}\n"
            f"Contraindications: {r['Contraindicações Principais']}\n"
            f"Notes: {r['Observações Úteis']}\n---\n"
        )

    prompt = f"""
You are a highly accurate medical assistant specialized in medication leaflets.
Only consider the medicines listed below. Recommend only if there's a clear match with the patient's question.
If none matches directly, say so politely. Do not overgeneralize or guess.

{contexto}

Patient question:
{pergunta_en}

Answer clearly and accurately:
"""
    return prompt.strip()

# ===============================
# Função principal que integra tudo
# ===============================
def responder(pergunta_usuario):
    warnings.filterwarnings("ignore")
    pergunta_en = traduzir_texto(pergunta_usuario, 'en')

    # Busca exata por nome
    for nome in df['Remédio'].unique():
        if nome.lower() in pergunta_en.lower():
            dados = df[df['Remédio'] == nome].iloc[0].to_dict()
            resposta = (
                f"**{dados['Remédio']}**\n\n"
                f"- Indication: {dados['Indicação']}\n"
                f"- Usage: {dados['Posologia (Como usar)']}\n"
                f"- Side effects: {dados['Efeitos Colaterais Principais']}\n"
                f"- Contraindications: {dados['Contraindicações Principais']}\n"
                f"- Notes: {dados['Observações Úteis']}"
            )
            return traduzir_texto(resposta, 'pt')

    # Busca semântica com corte mais restrito (score >= 0.9)
    resultados = index.similarity_search_with_score(pergunta_en, k=5)
    remedios_relevantes = [r.metadata["linha_completa"] for r, score in resultados if score >= 0.9]

    if remedios_relevantes:
        prompt = gerar_prompt(remedios_relevantes, pergunta_en)
        output = pipe(prompt, max_new_tokens=800, temperature=0.3, do_sample=False)[0]["generated_text"]
        resposta_en = output.split("Answer clearly and accurately:")[-1].strip()
        return traduzir_texto(resposta_en, 'pt')
    # 🔄 Fallback: buscar na OpenFDA
    resposta_fallback = buscar_na_openfda(pergunta_en)
    if resposta_fallback:
        return resposta_fallback
    return "Desculpe, não encontrei medicamentos que se encaixem claramente na sua descrição."


# ===============================
# Interface do usuário com Gradio
# ===============================
gr.Interface(
    fn=responder,
    inputs=gr.Textbox(label="Pergunta sobre medicamentos"),
    outputs=gr.Textbox(label="Resposta do assistente"),
    title="💊 Chatbot Médico",
    description="Digite sua dúvida. O modelo responderá com base em dados reais de bulas."
).launch()

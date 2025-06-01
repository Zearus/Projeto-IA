import streamlit as st
import pandas as pd
import torch
import requests
import warnings
from deep_translator import GoogleTranslator
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =======================
# Função de tradução
# =======================
def traduzir_texto(texto, to_lang='en'):
    try:
        return GoogleTranslator(source='auto', target=to_lang).translate(texto)
    except:
        return texto

# =======================
# Carregar base traduzida
# =======================
@st.cache_data
def carregar_base_traduzida():
    df = pd.read_csv('remedios.csv', sep=';').dropna(subset=['Indicação'])
    colunas = ['Remédio', 'Indicação', 'Posologia (Como usar)', 'Efeitos Colaterais Principais',
               'Contraindicações Principais', 'Observações Úteis']
    for col in colunas:
        df[col] = df[col].astype(str).apply(lambda x: traduzir_texto(x, 'en'))
    return df

df = carregar_base_traduzida()

# =======================
# Vetorização semântica
# =======================
@st.cache_resource
def criar_indexacao(df):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documentos = [
        Document(
            page_content=f"{row['Remédio']} - Indication: {row['Indicação']} - Usage: {row['Posologia (Como usar)']}",
            metadata={"linha_completa": row.to_dict()}
        )
        for _, row in df.iterrows()
    ]
    return FAISS.from_documents(documentos, embeddings)

index = criar_indexacao(df)

# =======================
# Carregar modelo TinyLLaMA
# =======================
@st.cache_resource
def carregar_modelo():
    model_name = 'TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0'
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, truncation=True)
    return pipe

pipe = carregar_modelo()

# =======================
# Fallback via OpenFDA
# =======================
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
    except:
        return None

# =======================
# Geração de prompt
# =======================
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
    return f"""
You are a medical assistant specialized in drug leaflets.
Answer clearly and based only on the information below. Do not guess. Be precise.

{contexto}

Patient question:
{pergunta_en}

Answer:""".strip()

# =======================
# Função principal
# =======================
def responder(pergunta_usuario):
    pergunta_en = traduzir_texto(pergunta_usuario, 'en')

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

    resultados = index.similarity_search_with_score(pergunta_en, k=5)
    remedios_relevantes = [r.metadata["linha_completa"] for r, score in resultados if score >= 0.9]

    if remedios_relevantes:
        prompt = gerar_prompt(remedios_relevantes, pergunta_en)
        output = pipe(prompt, max_new_tokens=800, temperature=0.3, do_sample=False)[0]["generated_text"]
        resposta_en = output.split("Answer:")[-1].strip()
        return traduzir_texto(resposta_en, 'pt')

    fallback = buscar_na_openfda(pergunta_en)
    if fallback:
        return fallback

    return "Desculpe, não encontrei medicamentos compatíveis com sua descrição."

# =======================
# Interface com Streamlit
# =======================
st.title("💊 Chatbot Médico")
st.markdown("Digite sua dúvida sobre medicamentos e receba uma resposta baseada em dados reais de bulas.")

pergunta = st.text_input("Pergunta:")

if st.button("Consultar"):
    if pergunta.strip() != "":
        with st.spinner("Consultando modelo..."):
            resposta = responder(pergunta)
            st.markdown(resposta)
    else:
        st.warning("Por favor, digite uma pergunta válida.")

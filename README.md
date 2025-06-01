# Projeto-IA

## Chat Médico com RAG e TinyLLaMA

Um chatbot inteligente para recomendação de medicamentos com base em bulas reais, utilizando Recuperação de Informação com Geração (RAG), embeddings semânticos, e um modelo leve de linguagem natural (TinyLLaMA). O sistema responde perguntas médicas com precisão e pode recorrer a fontes oficiais como a base OpenFDA caso a resposta não esteja na base local.

---

##  Objetivo

Desenvolver um assistente conversacional que responda dúvidas sobre medicamentos com base em dados reais de bulas, evitando respostas genéricas ou imprecisas. O modelo busca entender a pergunta do usuário, recuperar contextos relevantes e gerar uma resposta adequada.

---

##  Dados Utilizados

- Arquivo `remedios.csv` com colunas como:
  - Remédio
  - Indicação
  - Posologia (Como usar)
  - Efeitos Colaterais
  - Contraindicações
  - Observações

- Tradução automática do conteúdo para inglês com `deep_translator` para compatibilidade com embeddings e modelos de linguagem.

---

##  Ferramentas e Bibliotecas

- [`pandas`](https://pandas.pydata.org/): Manipulação de dados tabulares.
- [`deep-translator`](https://pypi.org/project/deep-translator/): Tradução automática multilíngue.
- [`HuggingFace Transformers`](https://huggingface.co/transformers/): Uso do modelo TinyLLaMA.
- [`FAISS`](https://github.com/facebookresearch/faiss): Busca semântica eficiente.
- [`LangChain`](https://www.langchain.com/): Integração entre embeddings, vetores e LLMs.
- [`Gradio`](https://www.gradio.app/): Interface interativa para perguntas e respostas.
- [`Sentence Transformers`](https://www.sbert.net/): Geração de embeddings semânticos.

---

##  Arquitetura da Solução

1. **Pré-processamento e tradução** do dataset `remedios.csv`.
2. **Criação de embeddings** com modelo `all-MiniLM-L6-v2`.
3. **Indexação FAISS** com os documentos vetorizados.
4. **RAG**: Dado uma pergunta, o sistema:
   - Traduz para inglês.
   - Busca vetorial por remédios relevantes (score ≥ 0.9).
   - Gera um prompt com o contexto relevante.
   - Usa o modelo **TinyLLaMA** para gerar a resposta.
   - Traduz a resposta final para o português.
5. **Fallback**: Caso não encontre correspondência, consulta a **API OpenFDA**.

---

##  Exemplos de Perguntas

- "Estou com dor de cabeça, qual remédio posso tomar?"
- "Qual o efeito colateral do Paracetamol?"
- "Existe contraindicação para ibuprofeno?"
- "Como devo usar o Omeprazol?"
- "Qual remédio serve para dor nas costas?"

---

##  Avaliação do Modelo

O sistema inclui uma função de avaliação automática com 5 perguntas de teste para medir:

- Qualidade da resposta
- Tempo médio de geração
- Capacidade de recuperação semântica

---

##  Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seunome/chat-medico-tinyllama.git
   cd chat-medico-tinyllama

2. Instale as dependências:
   pip install -r requirements.txt
   
3. Execute o código chatbot.py para rodar.

4. Para rodar a aplicação streamlit faça "streamlit run streamlit.py".

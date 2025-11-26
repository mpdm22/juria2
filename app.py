import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import List
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

# ------------ CONFIGURATION PAGE (mode clair) ------------
st.set_page_config(page_title="Chatbot Juridique SN", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
        .stTextInput>div>input {
            border-radius: 25px;
            padding: 16px;
            font-size: 18px;
            border: 1px solid #ccc;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #1a1a1a !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------ TITRE & PRÉSENTATION ------------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("drapeau justicesn.jpg", width=120)
with col2:
    st.markdown("""
        <h1 style='margin-top: 30px; font-size: 36px;'>
            LexSen : VOTRE ASSISTANT JURIDIQUE SÉNÉGALAIS
        </h1>
    """, unsafe_allow_html=True)

st.divider()

# ------------ SIDEBAR INFO ------------
with st.sidebar:
    st.title("ℹ️ Informations")
    st.markdown("""
    **📚 Domaines de droit prises en charge :**
    - Droit civil et procédure civile
    - Droit pénal et procédure pénale
    - Droit social 
    - Organisation judiciaire
    - Organisation de l’administration
    - Droit OHADA

    **📞 Assistance technique :**
    - WhatsApp : +221 77 339 76 94

    ℹ️ **Dernière mise à jour des textes : Avril 2025**
    """)

st.markdown("""
    <div style='
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        font-size: 20px;
        line-height: 1.6;
        margin-bottom: 25px;
    '>
        👋 Bonjour ! Je suis <strong>LexSen</strong>, votre assistant juridique spécialisé dans le droit sénégalais.<br><br>
        Posez-moi vos questions sur :
        <ul>
            <li>📘 Le code de la famille ou du travail </li>
            <li>⚖️ Le code pénal et la procédure pénale</li>
            <li>📄 Les lois, décrets, arrêtés etc.</li>
    </div>
""", unsafe_allow_html=True)

# ------------ CLASSE LLM ------------
class GroqLLM(LLM):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    api_key: str = os.environ.get("GROQ_API_KEY")

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

# ------------ CHARGE LA CHAINE RAG ------------
@st.cache_resource
def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    prompt_template = PromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans les textes de loi du Sénégal (Code de la famille, Code pénal, décrets, lois, etc).

Ta mission est de répondre de manière claire, concise et fiable à des questions posées par un utilisateur en t'appuyant exclusivement sur les extraits de documents juridiques suivants :

{context}

Consignes strictes :

- Réponds uniquement à partir du contenu fourni dans les extraits ci-dessus.
- Ne fais aucune supposition ni déduction en dehors des textes.
- N'invente jamais de références, de lois, ni de liens.
- Si l’information n’est pas présente, dis simplement : « Je suis désolé, mais aucun extrait de document en ma possession ne semble contenir une réponse claire à cette question. »
- Si la question est une salutation (bonjour, salut, etc.), réponds simplement avec une formule de politesse adaptée.
- Utilise un ton neutre, factuel et professionnel mais des réponses longues et explicatives.
- Réponds dans la langue de la question posée : français ou anglais.

---

Question : {question}

Réponse :
    """)

    llm = GroqLLM()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="mmr", k=2),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

qa_chain = load_qa_chain()

# ------------ AFFICHAGE DES MESSAGES ------------
USER_ICON = ""
BOT_ICON = "⚖️"

if "messages" not in st.session_state:
    st.session_state.messages = []

def message_bulle(texte, role="user"):
    icon = USER_ICON if role == "user" else BOT_ICON
    bubble_color = "#DCF8C6" if role == "user" else "#E6E6E6"
    st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
            <div style='font-size: 30px; margin-right: 10px;'>{icon}</div>
            <div style='background-color:{bubble_color}; padding:15px; border-radius:12px; max-width: 85%; font-size: 20px; color: #000;'>
                {texte}
            </div>
        </div>
    """, unsafe_allow_html=True)

for m in st.session_state.messages:
    message_bulle(m["content"], m["role"])

# ------------ DÉTECTION DES CAS SANS SOURCES ------------
def should_show_sources(question: str, response: str) -> bool:
    """Détermine si les sources doivent être affichées."""
    phrases_absence = [
        "Je suis désolé, mais aucun extrait de document en ma possession ne semble contenir une réponse claire à cette question.",
        "I'm sorry, but none of the excerpts in my possession appear to contain a clear answer to this question."
    ]

    salutations = ["bonjour", "salut", "bonsoir", "hello", "hi", "hey"]

    # Cas 1 : la réponse indique une absence d'information
    for phrase in phrases_absence:
        if phrase.lower() in response.lower():
            return False

    # Cas 2 : la question est une salutation
    if question.strip().lower() in salutations:
        return False

    return True

# ------------ ZONE INPUT UTILISATEUR ------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Posez votre question juridique :", "", placeholder="Ex: Quels sont les droits des femmes dans le code de la famille ?")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Recherche juridique en cours..."):
        result = qa_chain.invoke(user_input)
        reponse = result["result"]
        sources = result["source_documents"]

    st.session_state.messages.append({"role": "assistant", "content": reponse})

    if should_show_sources(user_input, reponse):
        unique_seen = set()
        limited_sources_list = []
        for doc in sources:
            meta = doc.metadata
            source_id = meta.get("document_title", "") + meta.get("chunk_title", "")
            if source_id not in unique_seen:
                folder = meta.get("folder", "Sans dossier")
                title = meta.get("chunk_title", "Sans titre")
                source = meta.get("document_title", "Document inconnu")
                url = meta.get("source_url", "")
                label = f"📚 {folder} / {source} / {title}\n→ {url}"
                limited_sources_list.append(label)
                unique_seen.add(source_id)
            if len(limited_sources_list) == 2:
                break

        st.session_state.messages.append({
            "role": "assistant",
            "content": "🔎 Sources utilisées :\n\n" + "\n\n".join(limited_sources_list)
        })

    st.rerun()

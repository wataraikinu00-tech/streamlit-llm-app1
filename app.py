import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

from dotenv import load_dotenv

load_dotenv()



# =========================================
# API キーの確認
# =========================================
def check_api_key():
    # Streamlit Cloud の Secrets 優先
    api_key = st.secrets.get("OPENAI_API_KEY", None)

    # ローカル環境 (.env) にも対応
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", None)

    return api_key


# =========================================
# LLM 応答用の関数
# =========================================
def get_llm_response(user_text: str, expert: str, chat_history):
    """
    user_text: ユーザーの質問
    expert: 専門家の種類
    chat_history: Streamlit の session_state["history"]
    """

    expert_prompts = {
        "法律": "あなたは優秀な法律専門家です。法律の観点から、正確で分かりやすく説明してください。",
        "スポーツ": "あなたはスポーツ科学の専門家です。運動生理学やスポーツ理論を踏まえて答えてください。",
        "栄養学": "あなたは栄養学の専門家です。食事・栄養の観点から科学的に回答してください。",
        "医学": "あなたは医師です。医学的根拠に基づいて専門的に回答してください。",
        "心理学": "あなたは心理学者です。心理学理論に基づいて分かりやすく説明してください。",
        "IT": "あなたはITエンジニアです。技術的な視点で丁寧に回答してください。",
    }

    api_key = check_api_key()
    if api_key is None:
        return "❌ **API キーが設定されていません。**\n\nStreamlit Cloud の Secrets または `.env` に `OPENAI_API_KEY` を設定してください。"

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=api_key
    )

    # メッセージ
    messages = [SystemMessage(content=expert_prompts[expert])]
    messages.extend(chat_history)  # 連続会話の履歴
    messages.append(HumanMessage(content=user_text))

    # LLM 応答
    response = llm.invoke(messages)

    return response.content


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="AI チャットアプリ", page_icon="🤖")
st.title("🤖 LangChain × Streamlit 連続チャットアプリ")
st.write("専門家を選び、質問すると AI がその専門家として回答します。")


# =========================================
# セッションステート（履歴）
# =========================================
if "history" not in st.session_state:
    st.session_state["history"] = []  # System / Human / AI メッセージ


# =========================================
# UI：専門家選択
# =========================================
expert = st.selectbox(
    "AI にどの専門家として回答させますか？",
    ["法律", "スポーツ", "栄養学", "医学", "心理学", "IT"]
)

st.write("---")

# =========================================
# UI：入力フォーム
# =========================================
user_input = st.text_area("質問を入力してください：", height=120)

if st.button("送信"):
    if user_input.strip() == "":
        st.warning("テキストが空です。入力してください。")
    else:
        with st.spinner("AI が回答を生成中..."):
            ai_response = get_llm_response(
                user_input,
                expert,
                st.session_state["history"]
            )

            # 履歴に追加（連続チャット）
            st.session_state["history"].append(HumanMessage(content=user_input))
            st.session_state["history"].append(AIMessage(content=ai_response))


st.write("---")
st.subheader("📜 回答履歴")

# =========================================
# 履歴表示
# =========================================
for msg in st.session_state["history"]:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**🧑‍💬 あなた:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**🤖 AI ({expert}):** {msg.content}")
    st.write("---")


# =========================================
# フッター
# =========================================
st.caption("Powered by Streamlit × LangChain × OpenAI")



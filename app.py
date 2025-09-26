# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import torch  # для работы с тензорами
import numpy as np
import faiss  # pip install faiss-cpu

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data
def get_data():
    df = load_all_excels()  # Уже включает Faiss индекс
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

# 📂 Фразы по выбранным тематикам
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(
                f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: 600; color: #333;">📝 {row.phrase_full}</div>
                    <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>""",
                unsafe_allow_html=True
            )
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")
if query:
    try:
        search_df = df
        if filter_search_by_topics and selected_topics:
            mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
            search_df = df[mask].copy()  # Копируем, чтобы не менять оригинал
            
            # Строим Faiss индекс для фильтрованного DF (он маленький, так что быстро)
            if not search_df.empty:
                model = get_model()
                phrase_embs_tensor = model.encode(search_df["phrase_proc"].tolist(), convert_to_tensor=True)
                phrase_embs_np = phrase_embs_tensor.cpu().numpy()
                d = phrase_embs_np.shape[1]
                index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(phrase_embs_np)
                index.add(phrase_embs_np)
                search_df.attrs["faiss_index"] = index
                search_df.attrs["phrase_embs"] = phrase_embs_tensor  # Для совместимости
                print(f"✅ Faiss индекс для фильтрованного DF: {index.ntotal} векторов")
            else:
                search_df.attrs["faiss_index"] = None
        
        if search_df.empty:
            st.warning("Нет данных для поиска по выбранным тематикам.")
        else:
            results = semantic_search(query, search_df)
            if results:
                st.markdown("### 🔍 Результаты умного поиска:")
                for score, phrase_full, topics, comment in results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">🧠 {phrase_full}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                <div style="margin-top: 2px; font-size: 13px; color: #999;">🎯 Релевантность: {score:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.warning("Совпадений не найдено в умном поиске.")
            
            exact_results = keyword_search(query, search_df)
            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics, comment in exact_results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">📌 {phrase}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.info("Ничего не найдено в точном поиске.")
    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")

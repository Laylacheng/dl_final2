import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import urllib

# Page Config
st.set_page_config(
    page_title="Wiki Caption Matcher (RapidFuzz)",
    layout="wide" ## 使用寬版配置，方便左右對照
)

st.title("影像標題配對（文字相似度基準）")
st.caption("Baseline method using RapidFuzz token_set_ratio")

@st.cache_data
def load_test_data():
    # 讀取測試集 TSV 檔
    df = pd.read_csv("test.tsv", sep="\t")

    #影像資料分析
    # 從 image_url 解析出文字
    df["txt_from_url"] = df["image_url"].apply(
        lambda x: urllib.parse.unquote(x)
        .split("/")[-1][:-4]
        .replace("_", " ")
    )
    return df

@st.cache_data
def load_captions():
    df = pd.read_csv("test_caption_list.csv")
    return df["caption_title_and_reference_description"].astype(str).tolist()

df_test = load_test_data()
captions = load_captions()


# 介面分左右
left, right = st.columns([1.1, 1])

with left:
    st.subheader("選擇測試影像")
    # 讓使用者手動輸入索引值選擇特定的測試資料
    sample_idx = st.number_input(
        "選擇測試資料 index",
        min_value=0,
        max_value=len(df_test) - 1,
        value=0
    )

    ## 調整 Top-K 結果數量的滑桿(1~10)
    topk = st.slider(
        "Top-K 結果數",
        min_value=1,
        max_value=10,
        value=5
    )

    run_btn = st.button("開始文字比對")

    st.markdown("---")
    st.markdown("### 從 image_url 解析出的文字")
    st.code(df_test.loc[sample_idx, "txt_from_url"])

with right:
    st.subheader("匹配結果")
    ## 按鈕被點擊後執行的運算
    if run_btn:
        query_text = df_test.loc[sample_idx, "txt_from_url"]

        with st.spinner("RapidFuzz 比對中..."):
            results = process.extract(
                query_text,
                captions,
                scorer=fuzz.token_set_ratio, # 核心演算法：計算詞彙集合重合度
                processor=None,              # 維持原始解析文字
                limit=topk                   # 限制回傳數量
            )

        st.success("比對完成")

        for rank, (caption, score, _) in enumerate(results, start=1):
            st.markdown(
                f"""
                **{rank}. {caption}**  
                <span style="color:gray">Similarity score: {score}</span>
                """,
                unsafe_allow_html=True
            )

    if not run_btn:
        st.info("請選擇一筆測試資料，然後點擊「開始文字比對」")



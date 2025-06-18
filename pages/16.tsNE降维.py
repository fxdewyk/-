import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 设置页面标题
st.set_page_config(page_title="t-SNE 可视化", layout="wide")
st.title("t-SNE 降维可视化")


# 数据预处理函数
def data_preprocessing(df):
    st.session_state.preprocess_params = {
        'scale_numeric': True
    }

    if st.session_state.preprocess_params['scale_numeric']:
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])
    return df


# 主程序流
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed_df = data_preprocessing(df.copy())

    # 特征选择界面
    with st.sidebar:
        st.markdown("## 🎯 目标变量设置")
        target_col = st.selectbox("选择目标列", processed_df.columns)

        st.markdown("## 📊 特征选择")
        feature_cols = st.multiselect(
            "选择预测特征",
            [col for col in processed_df.columns if col != target_col]
        )

    if feature_cols:
        # 使用 t-SNE 进行降维
        X = processed_df[feature_cols]
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        # 可视化 t-SNE 降维结果
        st.subheader("📊 t-SNE 可视化")
        df_tsne = pd.DataFrame(X_tsne, columns=["x", "y"])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="x", y="y", data=df_tsne, ax=ax)
        ax.set_title("t-SNE Result")
        st.pyplot(fig)

else:
    st.info("📥 请上传CSV格式的数据文件")

import streamlit as st
import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🌟 MeanShift 聚类", layout="wide")
st.title("🌟 MeanShift 聚类模型分析")

# 上传数据
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### ⚙️ 聚类配置")
        feature_cols = st.multiselect("📊 选择特征列", df.columns)

    if feature_cols:
        X = df[feature_cols]

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练MeanShift模型
        model = MeanShift()
        model.fit(X_scaled)
        labels = model.labels_

        # 添加聚类标签到原始数据
        df['Cluster'] = labels

        st.subheader("✅ 聚类结果")
        st.markdown(f"**聚类数目**：`{len(set(labels))}`")

        st.subheader("📊 聚类结果可视化")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="deep", ax=ax)
        ax.set_title("MeanShift 聚类结果")
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
        st.pyplot(fig)

    else:
        st.warning("⚠️ 请至少选择两个特征列进行聚类")

else:
    st.info("📥 请上传CSV格式的数据文件。")

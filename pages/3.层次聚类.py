import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🌟 层次聚类分析", layout="wide")
st.title("🌟 层次聚类模型分析")

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

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用层次聚类
        Z = linkage(X_scaled, method='ward')  # 'ward' 方法最小化类内平方和

        # 树形图可视化
        st.subheader("📊 树形图 (Dendrogram) 可视化")
        fig, ax = plt.subplots(figsize=(12, 8))
        dendrogram(Z, ax=ax)
        ax.set_title("层次聚类树形图 (Dendrogram)")
        ax.set_xlabel("样本索引")
        ax.set_ylabel("距离")
        st.pyplot(fig)

        # 选择聚类数目
        num_clusters = st.slider("选择聚类数目", min_value=2, max_value=10, value=3)

        from scipy.cluster.hierarchy import fcluster

        # 根据树形图切割聚类
        clusters = fcluster(Z, num_clusters, criterion='maxclust')

        # 将聚类标签添加到原数据中
        df['Cluster'] = clusters

        st.subheader("✅ 聚类结果")
        st.markdown(f"**聚类数目**：`{num_clusters}`")
        st.dataframe(df.head(10))

        # 聚类结果可视化
        st.subheader("📊 聚类结果可视化")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="deep", ax=ax2)
        ax2.set_title("层次聚类结果")
        ax2.set_xlabel(feature_cols[0])
        ax2.set_ylabel(feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
        st.pyplot(fig2)

    else:
        st.warning("⚠️ 请至少选择两个特征列进行聚类")

else:
    st.info("📥 请上传CSV格式的数据文件。")

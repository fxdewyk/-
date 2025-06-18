# 数据挖掘入侵检测软件.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="KMeans 聚类原型软件", layout="wide")

st.title("🧠 数据挖掘算法原型：KMeans 聚类")

# 上传数据
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])

# 设置聚类参数
st.sidebar.header("⚙️ 参数设置")
n_clusters = st.sidebar.slider("聚类簇数 (K)", min_value=2, max_value=10, value=4)

# 处理数据
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 数据预览")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("❗ 请上传包含至少两个数值列的CSV文件。")
    else:
        # 🎯 用户选择用于聚类的列
        selected_features = st.multiselect("选择用于聚类的特征列：", numeric_cols, default=numeric_cols)

        if len(selected_features) < 2:
            st.warning("请至少选择两个特征用于聚类。")
        else:
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[selected_features])

            # 聚类
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(X_scaled)
            df["Cluster"] = clusters

            st.subheader("🔍 聚类结果展示")
            st.write(f"总共分为 **{n_clusters}** 个簇")
            st.dataframe(df.head())

            # 聚类中心展示
            st.subheader("📌 每个簇的中心（均值）特征")
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=selected_features
            )
            cluster_centers.index = [f"Cluster {i}" for i in range(n_clusters)]
            st.dataframe(cluster_centers)

            # 类别分布统计图
            st.subheader("📊 各簇样本数量分布")
            cluster_counts = df["Cluster"].value_counts().sort_index()
            fig1, ax1 = plt.subplots()
            ax1.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
                    autopct="%1.1f%%", startangle=90, colors=sns.color_palette("tab10"))
            ax1.axis("equal")
            st.pyplot(fig1)

            # 可视化（PCA 降维展示）
            st.subheader("📈 聚类二维可视化（PCA 降维）")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
            pca_df["Cluster"] = clusters
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax2)
            st.pyplot(fig2)

            # 💾 下载结果按钮
            st.subheader("📥 下载聚类结果")
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="点击下载聚类结果CSV",
                data=csv,
                file_name="kmeans_cluster_result.csv",
                mime="text/csv"
            )

            # Elbow方法辅助选择K
            st.subheader("📐 K值辅助选择工具（肘部法）")
            wcss = []
            for k in range(1, 11):
                kmeans_temp = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                wcss.append(kmeans_temp.inertia_)
            fig3, ax3 = plt.subplots()
            ax3.plot(range(1, 11), wcss, marker='o')
            ax3.set_xlabel("K 值")
            ax3.set_ylabel("WCSS（组内平方和）")
            ax3.set_title("Elbow 方法图示")
            st.pyplot(fig3)

else:
    st.info("请上传一个包含数值特征的CSV文件。")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="DBSCAN 聚类原型软件", layout="wide")
st.title("🧠 数据挖掘算法原型：DBSCAN 聚类")

# 上传数据
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])

# 设置参数
st.sidebar.header("⚙️ 参数设置")
eps = st.sidebar.slider("邻域半径 eps", 0.1, 5.0, step=0.1, value=0.5)
min_samples = st.sidebar.slider("最小样本数 min_samples", 1, 20, value=5)

# 是否绘制辅助选eps图
show_eps_helper = st.sidebar.checkbox("📐 显示 K-距离图辅助选择 eps", value=False)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 数据预览")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("❗ 请上传包含至少两个数值列的CSV文件。")
    else:
        selected_features = st.multiselect("选择用于聚类的特征列：", numeric_cols, default=numeric_cols)

        if len(selected_features) < 2:
            st.warning("请至少选择两个特征列。")
        else:
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[selected_features])

            if show_eps_helper:
                st.subheader("📐 K-距离图（辅助选择 eps）")
                neighbors = NearestNeighbors(n_neighbors=min_samples)
                neighbors_fit = neighbors.fit(X_scaled)
                distances, indices = neighbors_fit.kneighbors(X_scaled)

                k_distances = np.sort(distances[:, min_samples-1])

                fig_eps, ax_eps = plt.subplots(figsize=(8, 6))
                ax_eps.plot(k_distances)
                ax_eps.set_xlabel("样本点索引", fontsize=14)
                ax_eps.set_ylabel(f"{min_samples}-近邻距离", fontsize=14)
                ax_eps.set_title("K-距离图", fontsize=16)
                ax_eps.grid(True)
                st.pyplot(fig_eps)

            # 聚类
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            df["Cluster"] = labels

            st.subheader("🔍 聚类结果展示")
            st.write(f"共识别出 **{len(set(labels)) - (1 if -1 in labels else 0)}** 个簇，噪声点数：**{list(labels).count(-1)}**")
            st.dataframe(df.head())

            # 类别分布图
            st.subheader("📊 各簇样本数量分布")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
                       autopct="%1.1f%%", startangle=90, colors=sns.color_palette("tab10"))
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

            # 可视化（PCA降维）
            st.subheader("📈 聚类二维可视化（PCA降维）")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
            pca_df["Cluster"] = labels
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax_scatter)
            st.pyplot(fig_scatter)

            # 下载聚类结果
            st.subheader("📥 下载聚类结果")
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="点击下载聚类结果CSV",
                data=csv,
                file_name="dbscan_cluster_result.csv",
                mime="text/csv"
            )
else:
    st.info("请上传一个包含数值特征的CSV文件。")

# -*- coding: utf-8 -*-
# spectral_clustering_optimized.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib as mpl

# 中文可视化配置
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def main():
    st.set_page_config(page_title="谱聚类分析系统", layout="wide")
    st.title('📊 谱聚类分析系统')

    # ==================== 文件上传模块 ====================
    with st.expander("📁 数据上传", expanded=True):
        uploaded_file = st.file_uploader("请上传CSV格式数据文件", type="csv",
                                         help="建议使用UCI数据集，如Iris、Wine等")

    if uploaded_file is not None:
        try:
            # ==================== 数据加载与校验 ====================
            data = pd.read_csv(uploaded_file)

            # 自动过滤非数值列
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            non_numeric = data.columns.difference(numeric_cols).tolist()

            if len(numeric_cols) < 2:
                st.error("错误：有效数值列不足2个，无法进行聚类分析")
                return
            if len(numeric_cols) < data.shape[1]:
                st.warning(f"已自动忽略非数值列：{', '.join(non_numeric)}")

            # ==================== 高维数据处理 ====================
            if len(numeric_cols) > 20:
                st.warning("检测到高维数据(>20列)，建议先进行降维处理")
                if st.checkbox("启用自动PCA降维"):
                    pca = PCA(n_components=0.95)
                    data_pca = pca.fit_transform(data[numeric_cols])
                    data = pd.DataFrame(data_pca, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
                    numeric_cols = data.columns.tolist()
                    st.success(f"降维后保留主成分：{pca.n_components_}")

            # ==================== 界面布局 ====================
            col1, col2 = st.columns([0.7, 0.3], gap="large")

            with col1:
                # 数据预览
                with st.expander("🔍 数据预览", expanded=True):
                    cols = st.columns(2)
                    cols[0].markdown(f"**维度信息**\n\n- 总样本数：`{len(data)}`\n- 数值特征：`{len(numeric_cols)}`")
                    cols[1].dataframe(data[numeric_cols].head(3), height=150)

                # 可视化展示区域
                viz_tabs = st.tabs(["📈 二维散点图", "🌐 平行坐标图", "📊 特征分布"])

            with col2:
                # ==================== 参数设置 ====================
                with st.expander("⚙️ 算法参数", expanded=True):
                    n_clusters = st.slider(
                        "聚类数量",
                        min_value=2,
                        max_value=10,
                        value=3,
                        help="根据轮廓系数选择最佳聚类数"
                    )

                    affinity = st.selectbox(
                        "相似度矩阵算法",
                        options=["rbf", "nearest_neighbors"],
                        index=0,
                        format_func=lambda x: "RBF核" if x == "rbf" else "K近邻"
                    )

                    gamma = st.slider(
                        "RBF核参数",
                        min_value=0.01,
                        max_value=2.0,
                        value=1.0,
                        disabled=(affinity != "rbf"),
                        help="控制核函数的辐射范围"
                    )

            # ==================== 数据处理 ====================
            features = data[numeric_cols].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(features)

            # ==================== 算法执行 ====================
            spec = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                gamma=gamma,
                random_state=42
            )
            labels = spec.fit_predict(scaled_data)

            # ==================== 可视化 ====================
            with col1:
                # 二维散点图
                with viz_tabs[0]:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    scatter = ax1.scatter(
                        features.iloc[:, 0],
                        features.iloc[:, 1],
                        c=labels,
                        cmap='tab20',
                        s=60,
                        edgecolor='w',
                        alpha=0.8
                    )
                    plt.colorbar(scatter, ax=ax1).set_label('聚类类别', rotation=270, labelpad=20)
                    ax1.set_xlabel(numeric_cols[0], fontsize=10)
                    ax1.set_ylabel(numeric_cols[1], fontsize=10)
                    ax1.set_title(f"特征空间分布 - {n_clusters}个聚类", pad=15)
                    plt.grid(True, linestyle=':', alpha=0.4)
                    st.pyplot(fig1)

                # 平行坐标图
                with viz_tabs[1]:
                    if len(numeric_cols) > 2:
                        fig2 = parallel_plot(features, labels)
                        st.pyplot(fig2)
                    else:
                        st.warning("至少需要3个特征才能显示平行坐标图")

                # 特征分布图（已修复索引越界问题）
                with viz_tabs[2]:
                    n_cols = 2
                    n_rows = (len(numeric_cols) + 1) // n_cols  # 动态计算行数
                    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
                    axes = axes.flatten()

                    for i, col in enumerate(numeric_cols):
                        if i >= len(axes):  # 索引保护
                            break
                        axes[i].hist(features[col], bins=15, alpha=0.6, edgecolor='k')
                        axes[i].set_title(col, fontsize=9)
                        axes[i].tick_params(axis='both', labelsize=7)

                    # 隐藏多余子图
                    for j in range(len(numeric_cols), len(axes)):
                        axes[j].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig3)

            # ==================== 分析报告 ====================
            with col2:
                with st.expander("📝 分析报告", expanded=True):
                    silhouette = silhouette_score(scaled_data, labels)
                    st.metric("轮廓系数", f"{silhouette:.2f}",
                              help="[-1,1]区间，值越大表示聚类效果越好")
                    st.metric("聚类标准差", f"{np.std(labels):.2f}",
                              help="反映各类别样本分布的离散程度")

                    st.markdown(f"""
                    **核心参数配置**
                    - 使用特征：`{', '.join(numeric_cols)}`
                    - 样本数量：`{len(features)}`
                    - 相似度算法：`{"RBF核" if affinity == "rbf" else "K近邻"}`
                    {f"- Gamma参数：`{gamma}`" if affinity == "rbf" else ""}
                    """)

                    st.download_button(
                        label="下载报告",
                        data=generate_report(features, labels, numeric_cols, silhouette),
                        file_name="clustering_report.md"
                    )

        except Exception as e:
            st.error(f"运行时错误：{str(e)}")
            st.markdown("""
            **故障排除指南**
            1. 检查数据是否包含缺失值
            2. 尝试降低特征维度（使用PCA）
            3. 调整RBF核参数到合理范围
            4. 确保至少选择两个有效特征
            """)


def parallel_plot(data, labels):
    """生成平行坐标图"""
    fig = plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        pd.DataFrame(data).assign(Cluster=labels),
        'Cluster',
        colormap='tab20',
        alpha=0.5
    )
    plt.xticks(rotation=20)
    plt.grid(linestyle=':', alpha=0.6)
    plt.title("特征平行坐标分布", pad=15)
    return fig


def generate_report(data, labels, features, silhouette):
    """生成Markdown格式报告"""
    return f"""
    ## 谱聚类分析报告

    ### 基本信息
    - 分析时间：`{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}`
    - 样本数量：`{len(data)}`
    - 特征维度：`{len(features)}`

    ### 聚类结果
    - 轮廓系数：`{silhouette:.2f}`
    - 类别分布：\n{np.unique(labels, return_counts=True)[1]}

    ### 参数配置
    - 使用特征：`{', '.join(features)}`
    - 聚类数量：`{len(np.unique(labels))}`
    """.encode('utf-8')


if __name__ == "__main__":
    main()

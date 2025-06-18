import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 自定义PCA算法实现
def manual_pca(X, n_components):
    # 1. 中心化数据
    X_centered = X - np.mean(X, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. 排序特征向量
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]

    # 5. 选择主成分
    components = eigenvectors[:, :n_components]

    # 6. 转换数据
    projected = X_centered.dot(components)

    return projected, eigenvalues, eigenvectors

st.title("📉 PCA 主成分分析")

uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("原始数据预览")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("请上传包含至少两个数值特征的CSV文件")
    else:
        # 用户参数设置
        with st.sidebar.expander("⚙️ 参数设置"):
            n_components = st.slider("选择主成分数量", 2, min(10, len(numeric_cols)), 2)
            show_loadings = st.checkbox("显示特征载荷矩阵")
            use_sklearn = st.checkbox("使用scikit-learn实现", value=True)

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])

        # 执行PCA
        if use_sklearn:
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X_scaled)
            eigenvalues = pca.explained_variance_
        else:
            components, eigenvalues, eigenvectors = manual_pca(X_scaled, n_components)

        # 结果显示
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔢 分析结果")
            st.write(f"累计方差解释率: {sum(eigenvalues[:n_components])/sum(eigenvalues):.1%}")

            # 方差解释率表格
            var_df = pd.DataFrame({
                "主成分": [f"PC{i+1}" for i in range(n_components)],
                "方差解释率": eigenvalues[:n_components]/sum(eigenvalues),
                "累计解释率": np.cumsum(eigenvalues[:n_components])/sum(eigenvalues)
            })
            st.dataframe(var_df.style.format({
                "方差解释率": "{:.1%}",
                "累计解释率": "{:.1%}"
            }))

        with col2:
            st.subheader("📊 碎石图")
            fig1, ax1 = plt.subplots()
            ax1.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-')
            ax1.set_xlabel("主成分")
            ax1.set_ylabel("特征值")
            ax1.set_title("特征值衰减图")
            st.pyplot(fig1)

        # 主成分可视化
        st.subheader("🎨 主成分可视化")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        scatter = ax2.scatter(components[:, 0], components[:, 1], alpha=0.6)
        ax2.set_xlabel(f"PC1 ({var_df.iloc[0,1]:.1%})")
        ax2.set_ylabel(f"PC2 ({var_df.iloc[1,1]:.1%})")
        ax2.set_title("主成分空间分布")
        st.pyplot(fig2)

        # 特征载荷矩阵
        if show_loadings:
            st.subheader("🧮 特征载荷矩阵")
            if use_sklearn:
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            else:
                loadings = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])

            loadings_df = pd.DataFrame(
                loadings,
                index=numeric_cols,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=0))

else:
    st.info("👆 请上传一个CSV文件（建议包含数值型特征）")

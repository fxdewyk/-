import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 初始化会话状态
if 'preprocess_params' not in st.session_state:
    st.session_state.preprocess_params = {
        'scale_numeric': True
    }

st.set_page_config(page_title="AutoEncoder 异常检测", layout="wide")
st.title("AutoEncoder 异常检测（PCA代替）")


# ==================== 数据预处理模块 ====================
def data_preprocessing(df):
    with st.expander("🔧 数据预处理设置", expanded=True):
        st.session_state.preprocess_params['scale_numeric'] = st.checkbox(
            "标准化数值特征",
            value=True,
            key="scale_numeric"
        )

    # 选择需要标准化的数值特征
    if st.session_state.preprocess_params['scale_numeric']:
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])
    return df


# ==================== 主程序流程 ====================
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed_df = data_preprocessing(df.copy())

    # ==================== 特征选择界面 ====================
    with st.sidebar:
        st.markdown("## 🎯 目标变量设置")
        target_col = st.selectbox("选择目标列", processed_df.columns)

        st.markdown("## 📊 特征选择")
        feature_cols = st.multiselect(
            "选择预测特征",
            [col for col in processed_df.columns if col != target_col]
        )

    if feature_cols:
        # ==================== PCA 降维 ====================
        X = processed_df[feature_cols].values

        # 使用PCA进行降维（代替AutoEncoder）
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # ==================== 可视化 ====================
        st.subheader("📊 PCA 可视化")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA Result")
        st.pyplot(fig)

        # ==================== 异常检测 ====================
        st.subheader("🚨 异常检测")
        # 假设异常数据是在降维后距离原点较远的数据点
        threshold = st.slider("异常检测阈值", 0.0, 10.0, 2.0)
        distances = np.linalg.norm(X_pca, axis=1)  # 计算每个点到原点的距离
        anomalies = distances > threshold

        # 可视化异常点
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, label="Normal")
        ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], color="red", label="Abnormal", s=100, edgecolor='black')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA Result")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("📥 请上传CSV格式的数据文件")

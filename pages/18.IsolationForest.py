import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 设置页面标题
st.set_page_config(page_title="Isolation Forest 异常检测", layout="wide")
st.title("Isolation Forest 异常检测")


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
        # 使用 Isolation Forest 进行异常检测
        X = processed_df[feature_cols]

        # 训练 Isolation Forest 模型
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(X)

        # 将异常点标记为 -1，正常点标记为 1
        df['Abnormal'] = outliers
        df['Abnormal'] = df['Abnormal'].map({1: 'Normal', -1: 'Abnormal'})

        # 可视化异常点
        st.subheader("📊 Isolation Forest 异常检测")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=feature_cols[0], y=feature_cols[1], data=df, hue='Abnormal', style='Abnormal',
                        palette='coolwarm', ax=ax)
        ax.set_title("Isolation Forest Result")
        st.pyplot(fig)

else:
    st.info("📥 请上传CSV格式的数据文件")

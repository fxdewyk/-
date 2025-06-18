import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🌳 Decision Tree 回归", layout="wide")
st.title("🌳 Decision Tree 回归模型分析")

# 上传数据
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### ⚙️ 模型配置")
        target_col = st.selectbox("🎯 选择目标列", df.columns)
        feature_cols = st.multiselect("📊 选择特征列", [col for col in df.columns if col != target_col])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        max_depth = st.slider("📏 最大深度 (max_depth)", 1, 20, 5)
        min_samples_split = st.slider("最小样本分割数 (min_samples_split)", 2, 20, 10)

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # 如果数据中包含非数值类型特征，进行编码
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.factorize(X[col])[0]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 模型训练
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 模型评估
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("✅ 模型评估结果")
        st.markdown(f"**均方误差 (MSE)**：`{mse:.4f}`")
        st.markdown(f"**R² (决定系数)**：`{r2:.4f}`")

        st.subheader("📊 实际值与预测值对比图")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_xlabel("实际值")
        ax.set_ylabel("预测值")
        ax.set_title("实际值与预测值对比")
        st.pyplot(fig)

        st.subheader("🌳 决策树可视化")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=feature_cols, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("⚠️ 请至少选择一个特征列")
else:
    st.info("📥 请上传CSV格式的数据文件。")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="朴素贝叶斯分类原型软件", layout="wide")

st.title("🧠 数据挖掘算法原型：朴素贝叶斯分类器")

# 上传数据
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件（需包含特征列和标签列）", type=["csv"])

# 设置参数
st.sidebar.header("⚙️ 参数设置")
test_size_ratio = st.sidebar.slider("测试集比例", min_value=0.1, max_value=0.5, step=0.05, value=0.3)
model_type = st.sidebar.selectbox("选择朴素贝叶斯模型类型", ["GaussianNB", "MultinomialNB", "BernoulliNB"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 数据预览")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64", "int32"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("❗ 请上传包含至少两个数值列的CSV文件。")
    else:
        st.subheader("🔎 特征选择（X）和标签选择（y）")

        selected_features = st.multiselect("选择特征列（用于模型输入 X）：", numeric_cols, default=numeric_cols[:-1])
        target_column = st.selectbox("选择标签列（用于预测 y）：", all_cols)

        if selected_features and target_column:
            X = df[selected_features]
            y = df[target_column]

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size_ratio, random_state=42)

            # 选择模型
            if model_type == "GaussianNB":
                model = GaussianNB()
            elif model_type == "MultinomialNB":
                model = MultinomialNB()
            else:
                model = BernoulliNB()

            # 模型训练
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 结果展示
            st.subheader("🔍 预测结果展示")
            _, X_test_with_index, _, y_test_with_index = train_test_split(
                X, y, test_size=test_size_ratio, random_state=42
            )

            # 构建预测结果 DataFrame，包含真实标签、预测标签和原始字段
            result_df = X_test_with_index.copy()
            result_df["真实标签"] = y_test_with_index.values
            result_df["预测标签"] = y_pred

            st.dataframe(result_df.head())

            # 准确率
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("✅ 模型准确率")
            st.success(f"准确率：{accuracy:.4f}")

            # 分类报告
            st.subheader("📋 分类报告")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # 混淆矩阵
            st.subheader("📊 混淆矩阵")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("预测标签")
            ax.set_ylabel("真实标签")
            st.pyplot(fig)

            # 下载预测结果
            st.subheader("📥 下载预测结果")
            download_csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="点击下载预测结果CSV",
                data=download_csv,
                file_name="naive_bayes_prediction_result.csv",
                mime="text/csv"
            )

else:
    st.info("请上传一个包含数值特征的CSV文件。")

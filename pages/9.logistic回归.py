import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 页面设置
st.set_page_config(page_title="Logistic 回归分类原型软件", layout="wide")
st.title("🧠 数据挖掘算法原型：Logistic 回归")

# 上传数据
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件（需包含特征列和标签列）", type=["csv"])

# 参数设置
st.sidebar.header("⚙️ 参数设置")
test_size_ratio = st.sidebar.slider("测试集比例", min_value=0.1, max_value=0.5, step=0.05, value=0.3)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 数据预览")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
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

            # 如果y不是数字类型，进行编码
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 划分训练/测试集
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size_ratio, random_state=42)

            # 训练模型
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            st.subheader("✅ 模型准确率")
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"模型预测准确率：{accuracy:.4f}")

            st.subheader("📋 分类报告")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("📊 混淆矩阵")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("预测标签")
            ax_cm.set_ylabel("真实标签")
            st.pyplot(fig_cm)

            # ROC 曲线绘制
            if len(np.unique(y_test)) == 2:
                st.subheader("📈 ROC 曲线与 AUC")

                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange", lw=2)
                ax_roc.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
                ax_roc.set_title("ROC 曲线")
                ax_roc.set_xlabel("假阳性率 (FPR)")
                ax_roc.set_ylabel("真阳性率 (TPR)")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.info("⚠️ 当前不是二分类问题，无法绘制 ROC 曲线。")

            # 下载预测结果
            result_df = pd.DataFrame({"真实标签": y_test, "预测标签": y_pred})
            st.subheader("📥 下载预测结果")
            st.download_button(
                label="下载预测结果CSV",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="logistic_regression_prediction.csv",
                mime="text/csv"
            )
else:
    st.info("请上传一个包含数值特征的CSV文件。")

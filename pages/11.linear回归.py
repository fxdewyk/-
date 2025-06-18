import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="线性回归预测器", layout="wide")
st.title("📈 线性回归预测器")

# 文件上传
uploaded_file = st.file_uploader("请上传包含 RM, AGE, MEDV 三列的 CSV 数据集", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if {'RM', 'AGE', 'MEDV'}.issubset(data.columns):
            st.subheader("数据预览")
            st.dataframe(data.head())

            X = data[['RM', 'AGE']]
            y = data['MEDV']

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            st.subheader("📊 预测效果图")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y, predictions, alpha=0.6, color='steelblue')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
            ax.set_xlabel('Actual Price', fontsize=12)
            ax.set_ylabel('Predicted Price', fontsize=12)
            ax.set_title('Linear Regression Performance', fontsize=14)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.subheader("📌 回归结果")
            coef_rm, coef_age = model.coef_
            intercept = model.intercept_
            r2_score = model.score(X, y)

            st.markdown(f"**回归方程：** `MEDV = {coef_rm:.2f} * RM + {coef_age:.2f} * AGE + {intercept:.2f}`")
            st.markdown(f"**R² 分数：** `{r2_score:.2f}`")

        else:
            st.error("数据集中必须包含 RM、AGE 和 MEDV 三列。")

    except Exception as e:
        st.error(f"发生错误：{e}")

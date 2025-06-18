import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(page_title="FP-Growth 关联规则挖掘", layout="wide")

st.title("🧠 FP-Growth 关联规则挖掘工具")

# 文件上传
uploaded_file = st.file_uploader("请上传0-1编码的商品交易数据集 (CSV格式)", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("数据预览")
        st.dataframe(data.head())

        # 用户设置最小支持度
        min_support = st.slider("选择最小支持度 (min_support)", 0.01, 1.0, 0.05, 0.01)

        # 提交按钮
        if st.button("开始挖掘频繁项集与关联规则"):
            with st.spinner("正在挖掘，请稍候..."):
                frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)

                if frequent_itemsets.empty:
                    st.warning("未发现满足支持度要求的频繁项集。")
                else:
                    st.success(f"共挖掘到 {len(frequent_itemsets)} 个频繁项集")
                    st.subheader("频繁项集")
                    st.dataframe(frequent_itemsets)

                    # 提取规则
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

                    if rules.empty:
                        st.warning("未能提取到有效的关联规则。")
                    else:
                        sort_by = st.selectbox("选择排序指标", ["lift", "confidence", "support"])
                        top_rules = rules.sort_values(by=sort_by, ascending=False).head(10)
                        st.subheader("Top 关联规则")
                        st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    except Exception as e:
        st.error(f"发生错误：{e}")

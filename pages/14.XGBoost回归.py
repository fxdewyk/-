
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost.callback import TrainingCallback

st.set_page_config(page_title="🎯 XGBoost 高级回归分析", layout="wide")
st.title("🎯 XGBoost 高级回归分析")

# ==================== 修复后的回调类 ====================
class StreamlitProgress(TrainingCallback):
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.progress_bar = None
        self.status_text = None

    def before_training(self, model):
        self.progress_bar = st.progress(0.0)
        self.status_text = st.empty()
        return model

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        progress = current / self.total_rounds
        self.progress_bar.progress(progress)
        self.status_text.text(f"训练进度: {current}/{self.total_rounds} 轮")
        return False

    def after_training(self, model):
        self.progress_bar.empty()
        self.status_text.empty()
        return model

# ==================== 数据预处理模块 ====================
def preprocess_data(df):
    with st.expander("🔧 数据预处理设置", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            missing_strategy = st.selectbox(
                "缺失值处理",
                ["删除缺失行", "数值列填充中位数", "分类列填充众数"],
                index=1
            )

        with col2:
            scale_numeric = st.checkbox("标准化数值特征", True)
            encode_categorical = st.checkbox("编码分类特征", True)

    if missing_strategy == "删除缺失行":
        df = df.dropna()

    return df, {
        'missing_strategy': missing_strategy,
        'scale_numeric': scale_numeric,
        'encode_categorical': encode_categorical
    }

# ==================== 主程序流程 ====================
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed_df, preprocess_params = preprocess_data(df)

    with st.sidebar:
        st.markdown("## 🎯 目标变量设置")
        target_col = st.selectbox("选择目标列", processed_df.columns)

        st.markdown("## 📊 特征选择")
        feature_cols = st.multiselect(
            "选择预测特征",
            [col for col in processed_df.columns if col != target_col]
        )

        st.markdown("## ⚙ 高级参数设置")
        with st.expander("模型参数"):
            params = {
                'max_depth': st.slider("最大深度", 1, 12, 6),
                'learning_rate': st.slider("学习率", 0.001, 0.5, 0.1, 0.005),
                'n_estimators': st.slider("树的数量", 10, 2000, 500, 10),
                'gamma': st.slider("gamma", 0.0, 1.0, 0.0, 0.1),
                'subsample': st.slider("子采样率", 0.1, 1.0, 1.0, 0.05)
            }

        with st.expander("训练设置"):
            early_stop = st.number_input("早停轮数", 10, 100, 50)

    if feature_cols:
        # ==================== 数据预处理管道 ====================
        numeric_features = processed_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_features = list(set(feature_cols) - set(numeric_features))

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()) if preprocess_params['scale_numeric'] else ('passthrough', 'passthrough')
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')) if preprocess_params['encode_categorical'] else ('passthrough', 'passthrough')
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # ==================== 模型训练 ====================
        X = processed_df[feature_cols]
        y = processed_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        dtrain = xgb.DMatrix(preprocessor.fit_transform(X_train), label=y_train)
        dtest = xgb.DMatrix(preprocessor.transform(X_test), label=y_test)

        # 修复评估结果存储
        evals_result = {}  # 创建存储评估结果的字典

        progress_callback = StreamlitProgress(params['n_estimators'])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            early_stopping_rounds=early_stop,
            evals=[(dtest, "测试集")],
            evals_result=evals_result,  # 传入存储字典
            callbacks=[progress_callback],
            verbose_eval=False
        )

        # ==================== 模型评估 ====================
        st.subheader("📈 模型评估")
        y_pred = model.predict(dtest)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        with col2:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("最佳迭代轮次", model.best_iteration)

        # ==================== 可视化模块 ====================
        tabs = st.tabs(["📉 预测分析", "📊 特征重要性", "🔍 SHAP解释", "📈 学习曲线"])

        with tabs[0]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            ax1.scatter(y_test, y_pred, alpha=0.6)
            ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax1.set_title("实际值 vs 预测值")

            residuals = y_test - y_pred
            ax2.hist(residuals, bins=30)
            ax2.set_title("残差分布")
            st.pyplot(fig)

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(model, ax=ax)
            st.pyplot(fig)

        with tabs[2]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(dtest)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig)

        with tabs[3]:
            # 使用存储的评估结果
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(evals_result['测试集']['rmse'], label='测试集')  # 从字典获取数据
            ax.set_xlabel('迭代轮次')
            ax.set_ylabel('RMSE')
            ax.set_title('学习曲线')
            ax.legend()
            st.pyplot(fig)

        # ==================== 模型保存 ====================
        st.sidebar.markdown("## 💾 模型管理")
        if st.sidebar.button("保存当前模型"):
            model.save_model('xgboost_model.json')
            joblib.dump(preprocessor, 'preprocessor.pkl')
            st.sidebar.success("模型已保存为 xgboost_model.json 和 preprocessor.pkl")

else:
    st.info("📥 请上传CSV格式的数据文件")

import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# Set page config with custom icon
st.set_page_config(
    page_title="Osteoporosis Gene Analysis",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🦴 Osteoporosis Gene Mechanism Analysis")
st.markdown("""
    This tool uses transcriptome data to predict osteoporosis risk and provides mechanistic insights 
    using SHAP (SHapley Additive exPlanations) visualizations. Adjust gene expression levels in the sidebar 
    to explore their impact on bone metabolism regulation.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/osteoporosis_data.xlsx')  # 更新数据文件
    return df[['VAMP1', 'ATP10B', 'ABCC4', 'LCT', 'NTNG1', 
              'GLRA2', 'ELSPBP1', 'IL13', 'MLN', 'ZNF280A',
              'CA7', 'ELAVL4', 'SPDEF', 'SRPK3', 'FSCN3',
              'CLDN18', 'IL19', 'CD80', 'ZFYVE9', 'CARTPT']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/OSTEOPOROSIS_MODEL.h5')  # 更新模型文件

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for genes (示例值，需根据实际数据调整)
default_values = {
    'VAMP1': 7.9, 'ATP10B': 2.38, 'ABCC4': 33.93,
    'LCT': 7.18, 'NTNG1': 18.7, 'GLRA2': 3.17,
    'ELSPBP1': 2.83, 'IL13': 2.95, 'MLN': 9.78,
    'ZNF280A': 2.41, 'CA7': 5.53, 'ELAVL4': 5.66,
    'SPDEF': 27.8, 'SRPK3': 17.95, 'FSCN3': 16.1,
    'CLDN18': 11.1, 'IL19': 27.2, 'CD80': 31.9,
    'ZFYVE9': 8.37, 'CARTPT': 2.37
}


# Sidebar configuration
st.sidebar.header("🧬 Gene Expression Inputs")
st.sidebar.markdown("Adjust expression levels of osteoporosis-related genes:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# 动态生成三列布局以适应更多基因
gene_features = list(default_values.keys())
gene_values = {}
cols = st.sidebar.columns(3)  # 改为三列布局

for i, gene in enumerate(gene_features):
    with cols[i % 3]:
        gene_values[gene] = st.number_input(
            gene,
            min_value=float(background_data[gene].min()),
            max_value=float(background_data[gene].max()),
            value=default_values[gene],
            step=0.01,
            format="%.2f",
            key=gene
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([gene_values])

# Main analysis
if st.button("🔬 Analyze Gene Impacts", key="calculate"):
    input_df = prepare_input_data()
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    st.header("📈 Risk Prediction")
    st.metric("Osteoporosis Risk Score", f"{prediction:.4f}", 
             delta="High Risk" if prediction >= 0.5 else "Low Risk",
             delta_color="inverse")
    
    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Force Plot", "Waterfall Plot", "Decision Plot", "Mechanistic Insights"])
    
    with tab1:
        st.subheader("Feature Impact Visualization")
        explanation = shap.Explanation(
            values=shap_values, 
            base_values=base_value, 
            feature_names=input_df.columns,
            data=input_df.values
        )
        shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab4:  # 新增机制分析标签页
        st.subheader("Mechanistic Insights")
        st.markdown("""
        **Key Osteoporosis-related Pathways:**
        - VAMP1: 参与破骨细胞囊泡运输
        - IL13/IL19: 炎症调节因子
        - CLDN18: 骨细胞间连接调控
        - CARTPT: 神经内分泌调节
        """)
        importance_df = pd.DataFrame({'Gene': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))

# 更新说明文档
with st.expander("📚 About This Osteoporosis Analysis", expanded=True):
    st.markdown("""
    ### Model Overview
    This deep learning model analyzes 20 key genes involved in:
    - Osteoclast differentiation
    - Bone matrix remodeling
    - Calcium homeostasis
    - Inflammatory regulation
    
    ### SHAP Interpretation Guide
    1. **Force Plot**: 显示各基因对风险评分的推拉效应
    2. **Waterfall Plot**: 逐步展示特征贡献
    3. **Decision Plot**: 累积效应可视化
    4. **Mechanistic Insights**: 结合SHAP值与已知生物学机制的分析
    """)

# Footer
st.markdown("---")
st.markdown(f"Developed for Osteoporosis Research | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
"""
Toxicity Detection Web Interface
Built with Streamlit - Final Project Demo
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .toxic-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .toxic-yes {
        background-color: #ffe6e6;
        border-color: #ff4444;
    }
    .toxic-no {
        background-color: #e6ffe6;
        border-color: #44ff44;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load DistilBERT model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    tokenizer = DistilBertTokenizer.from_pretrained('./models/distilbert')
    model = DistilBertForSequenceClassification.from_pretrained('./models/distilbert')
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device

def predict_toxicity(text, tokenizer, model, device):
    """Get toxicity predictions for input text"""
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)
    
    label_names = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
    
    results = {
        'predictions': preds,
        'probabilities': probs,
        'labels': label_names,
        'is_toxic': int(preds[0])  # Main toxic label
    }
    
    return results

def create_probability_chart(labels, probs):
    """Create interactive bar chart of probabilities"""
    colors = ['#ff4444' if p > 0.5 else '#44ff44' for p in probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs * 100,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Toxicity Probabilities by Category',
        xaxis_title='Confidence (%)',
        yaxis_title='Category',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 110])
    )
    
    # Add threshold line
    fig.add_vline(x=50, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (50%)")
    
    return fig

def highlight_potential_toxic_words(text):
    """Highlight potentially toxic words (simple heuristic)"""
    toxic_keywords = [
        'fuck', 'shit', 'damn', 'ass', 'hell', 'stupid', 'idiot', 
        'hate', 'kill', 'die', 'moron', 'dumb', 'loser'
    ]
    
    highlighted = text
    for word in toxic_keywords:
        # Case-insensitive replacement with highlighting
        import re
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color: #ffcccc; padding: 2px 4px; border-radius: 3px;">{word}</span>',
            highlighted
        )
    
    return highlighted

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Toxicity Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered content moderation using DistilBERT</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This tool uses a fine-tuned DistilBERT model to detect toxic content 
        across 6 categories:
        - **Toxic**: General toxicity
        - **Severe Toxic**: Highly toxic
        - **Obscene**: Profanity
        - **Threat**: Threats of violence
        - **Insult**: Insulting language
        - **Identity Hate**: Hate speech
        """)
        
        st.header("⚙️ Model Info")
        st.write("""
        - **Model**: DistilBERT (66M params)
        - **Training Data**: 127K comments
        - **Macro F1**: ~0.79
        - **Improvement**: +13% over baseline
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This is a research prototype. It may:
        - Flag non-toxic content (false positives)
        - Miss subtle toxicity (false negatives)
        - Show bias across demographics
        
        **Not for production use without human review.**
        """)
        
        st.header("📊 Project Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Time", "~2 hours")
            st.metric("Languages", "English")
        with col2:
            st.metric("Accuracy", "92%")
            st.metric("Dataset Size", "159K")
    
    # Load model
    try:
        tokenizer, model, device = load_model()
        st.success(f"✅ Model loaded successfully on {device}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure the model is saved in './models/distilbert/' directory")
        st.stop()
    
    # Input section
    st.header("🔍 Analyze Text")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste a comment here...",
        height=150,
        help="Enter any text to check for toxicity. The model will analyze it across 6 categories."
    )
    
    # Example buttons
    st.write("**Quick Examples:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🟢 Clean Example"):
            text_input = "This is a great article, thank you for sharing!"
    with col2:
        if st.button("🟡 Borderline"):
            text_input = "This is fucking awesome work!"
    with col3:
        if st.button("🔴 Toxic Example"):
            text_input = "You're an idiot and don't know what you're talking about"
    with col4:
        if st.button("🔴 Severe Example"):
            text_input = "I hate you and people like you shouldn't exist"
    
    # Analyze button
    analyze_clicked = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
    
    # Analysis results
    if analyze_clicked and text_input:
        with st.spinner("Analyzing..."):
            results = predict_toxicity(text_input, tokenizer, model, device)
        
        # Main result
        is_toxic = results['is_toxic']
        toxic_prob = results['probabilities'][0]
        
        if is_toxic:
            st.markdown(f"""
            <div class="toxic-box toxic-yes">
                <h2>⚠️ POTENTIALLY TOXIC</h2>
                <p style="font-size: 1.2rem;">Confidence: <b>{toxic_prob*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="toxic-box toxic-no">
                <h2>✅ LIKELY NOT TOXIC</h2>
                <p style="font-size: 1.2rem;">Confidence: <b>{(1-toxic_prob)*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed breakdown
        st.header("📊 Detailed Analysis")
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Probability chart
            fig = create_probability_chart(results['labels'], results['probabilities'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Category Results")
            for label, pred, prob in zip(results['labels'], results['predictions'], results['probabilities']):
                status = "🔴 DETECTED" if pred else "🟢 Not detected"
                st.write(f"**{label}**: {status}")
                st.progress(float(prob))
                st.caption(f"Confidence: {prob*100:.1f}%")
                st.divider()
        
        # Highlighted text
        st.header("🔦 Highlighted Analysis")
        st.markdown(
            f'<div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; font-size: 1.1rem;">{highlight_potential_toxic_words(text_input)}</div>',
            unsafe_allow_html=True
        )
        st.caption("⚠️ Highlighted words are common toxic indicators (heuristic-based, not model output)")
        
        # Explanation
        st.header("💡 Interpretation")
        
        detected_categories = [label for label, pred in zip(results['labels'], results['predictions']) if pred]
        
        if detected_categories:
            st.warning(f"**Detected toxicity types:** {', '.join(detected_categories)}")
            
            if 'Obscene' in detected_categories and not is_toxic:
                st.info("ℹ️ This text contains profanity but may not be genuinely toxic (e.g., positive emphasis)")
            
            if 'Threat' in detected_categories or 'Severe Toxic' in detected_categories:
                st.error("🚨 This content contains serious threats or severe toxicity. Human review recommended.")
        else:
            st.success("✅ No toxicity detected across all categories")
        
        # Technical details (expandable)
        with st.expander("🔬 Technical Details"):
            st.write("**Model Configuration:**")
            st.code(f"""
Model: DistilBERT-base-uncased
Max Sequence Length: 128 tokens
Threshold: 0.5 (50% confidence)
Device: {device}
Inference Time: ~0.1 seconds
            """)
            
            st.write("**Raw Probabilities:**")
            prob_df = pd.DataFrame({
                'Category': results['labels'],
                'Probability': [f"{p*100:.2f}%" for p in results['probabilities']],
                'Prediction': ['Toxic' if p else 'Clean' for p in results['predictions']]
            })
            st.dataframe(prob_df, use_container_width=True)
    
    elif analyze_clicked:
        st.warning("⚠️ Please enter some text to analyze")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit and DistilBERT</p>
        <p>Final Project - NLP Course | December 2024</p>
        <p><i>⚠️ Research prototype - Not for production use</i></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
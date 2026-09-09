"""
Dashboard Page 4: Model Performance & Governance
Comprehensive ML metrics, confusion matrix, ROC & PR curves,
feature importance rankings, and threshold trade-off analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.models.predict import get_model_metadata

def render_model_performance():
    st.markdown("## 🧠 Machine Learning Model Governance & Performance")
    st.markdown(
        "Evaluating classification metrics for fraud detection under extreme class imbalance (0.17% positive rate). "
        "Standard accuracy is misleading; **Precision, Recall, and PR-AUC** dictate business viability."
    )

    metadata = get_model_metadata()
    baseline = metadata.get("baseline_metrics", {})
    main_rf = metadata.get("main_metrics", {})

    # Top Metric Cards (Main Random Forest Model)
    st.subheader("Production Model Metrics (Random Forest)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Accuracy", f"{main_rf.get('accuracy', 0.9995):.4f}")
    with c2:
        st.metric("Precision", f"{main_rf.get('precision', 0.9211):.4f}", help="Of transactions flagged as fraud, how many actually were fraud.")
    with c3:
        st.metric("Recall", f"{main_rf.get('recall', 0.7368):.4f}", help="Of all actual fraud transactions, how many were caught.")
    with c4:
        st.metric("F1 Score", f"{main_rf.get('f1', 0.8187):.4f}")
    with c5:
        st.metric("ROC-AUC", f"{main_rf.get('roc_auc', 0.9646):.4f}")
    with c6:
        st.metric("PR-AUC", f"{main_rf.get('pr_auc', 0.8034):.4f}", help="Precision-Recall Area Under Curve: Gold standard for severe imbalance.")

    st.markdown("---")

    # Baseline vs Random Forest Comparison Table
    st.subheader("Model Benchmark: Baseline vs. Production Model")
    comp_df = pd.DataFrame([
        {
            "Model": "Logistic Regression (Baseline with Balanced Weights)",
            "Accuracy": f"{baseline.get('accuracy', 0.9743):.4f}",
            "Precision": f"{baseline.get('precision', 0.0543):.4f}",
            "Recall": f"{baseline.get('recall', 0.8737):.4f}",
            "F1 Score": f"{baseline.get('f1', 0.1022):.4f}",
            "PR-AUC": f"{baseline.get('pr_auc', 0.6801):.4f}",
            "False Positives (Alarms)": f"{baseline.get('false_positives', 1446):,}",
            "False Negatives (Missed)": f"{baseline.get('false_negatives', 12):,}"
        },
        {
            "Model": "Random Forest Classifier (Subsample Balanced)",
            "Accuracy": f"{main_rf.get('accuracy', 0.9995):.4f}",
            "Precision": f"{main_rf.get('precision', 0.9211):.4f}",
            "Recall": f"{main_rf.get('recall', 0.7368):.4f}",
            "F1 Score": f"{main_rf.get('f1', 0.8187):.4f}",
            "PR-AUC": f"{main_rf.get('pr_auc', 0.8034):.4f}",
            "False Positives (Alarms)": f"{main_rf.get('false_positives', 6):,}",
            "False Negatives (Missed)": f"{main_rf.get('false_negatives', 25):,}"
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Key Business Trade-off callout
    st.warning(
        "**Key Business Trade-off Explanation**:\n"
        "- **Logistic Regression Baseline** achieves high Recall (87.4%), catching 83 of 95 frauds, but suffers from low Precision (5.4%), generating **1,446 false alarms** that would alienate legitimate customers.\n"
        "- **Random Forest Classifier** achieves 92.1% Precision with only **6 false alarms** across 56,651 legitimate test transactions, striking a superior balance for production risk management."
    )

    st.markdown("---")

    # Confusion Matrix & Feature Importance
    col_cm, col_imp = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix (Test Set: 56,746 Tx)")
        cm = main_rf.get("confusion_matrix", [[56645, 6], [25, 70]])
        cm_labels = [["True Negatives<br>(Legit Allowed)", "False Positives<br>(False Alarms)"],
                     ["False Negatives<br>(Fraud Missed)", "True Positives<br>(Fraud Caught)"]]
        
        annot_text = [
            [f"{cm_labels[0][0]}<br><b>{cm[0][0]:,}</b>", f"{cm_labels[0][1]}<br><b>{cm[0][1]:,}</b>"],
            [f"{cm_labels[1][0]}<br><b>{cm[1][0]:,}</b>", f"{cm_labels[1][1]}<br><b>{cm[1][1]:,}</b>"]
        ]
        
        fig_cm = ff_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=annot_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_imp:
        st.subheader("Top Predictive Feature Importances")
        feat_imp = metadata.get("feature_importances", [])[:10]
        if feat_imp:
            imp_df = pd.DataFrame(feat_imp)
            fig_feat = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                labels={"importance": "Relative Importance", "feature": "PCA Feature"}
            )
            fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=360, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")

    # Interactive Threshold Tuning Simulator
    st.subheader("🎛️ Decision Threshold Optimization & Sensitivity Analysis")
    st.markdown("Explore how shifting the probability decision threshold impacts false positive costs versus missed fraud losses.")

    thresh_data = metadata.get("threshold_analysis", [])
    if thresh_data:
        thresh_df = pd.DataFrame(thresh_data)
        
        selected_thresh = st.slider(
            "Select Classification Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(metadata.get("optimal_threshold", 0.30)),
            step=0.05
        )

        curr_row = thresh_df.iloc[(thresh_df["threshold"] - selected_thresh).abs().argsort()[:1]].iloc[0]

        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Precision", f"{curr_row['precision']:.2%}")
        with tc2:
            st.metric("Recall", f"{curr_row['recall']:.2%}")
        with tc3:
            st.metric("False Alarms (FP)", f"{int(curr_row['false_positives']):,}", delta="Customer Friction", delta_color="inverse")
        with tc4:
            st.metric("Missed Frauds (FN)", f"{int(curr_row['false_negatives']):,}", delta="Direct Loss", delta_color="inverse")

        # Threshold curve plot
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["precision"], name="Precision", line=dict(color="#2563EB", width=2)))
        fig_curve.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["recall"], name="Recall", line=dict(color="#DC2626", width=2)))
        fig_curve.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["f1"], name="F1 Score", line=dict(color="#10B981", width=2, dash="dash")))
        fig_curve.add_vline(x=selected_thresh, line_width=2, line_dash="dot", line_color="black", annotation_text=f"Selected ({selected_thresh})")
        fig_curve.update_layout(
            xaxis_title="Classification Threshold",
            yaxis_title="Metric Value",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

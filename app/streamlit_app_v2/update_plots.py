import os
import glob

# Mappings of old strings to new strings
REPLACEMENTS = {
    # Directory paths
    "V8_PLOTS": "V8_PLOTS",
    "'v8 plots'": "'v8 plots'",
    "\"v5_plots\"": "\"v8 plots\"",
    "get_v8_plot_path": "get_v8_plot_path",
    
    # Notebook plot folder usage
    "V8_PLOTS": "V8_PLOTS", # Some are referred as V8_PLOTS

    # Individual file name mappings
    "02_categorical_feature_distributions.png": "02_categorical_feature_distributions.png",
    "03_numerical_feature_distributions.png": "03_numerical_feature_distributions.png",
    "04_outlier_detection_boxplots.png": "04_outlier_detection_boxplots.png",
    "05_numerical_features_by_match_outcome.png": "05_numerical_features_by_match_outcome.png",
    "06_categorical_features_by_match_outcome.png": "06_categorical_features_by_match_outcome.png",
    "07_pearson_correlation_matrix.png": "07_pearson_correlation_matrix.png",
    "08_top_30_interest_tags.png": "08_top_30_interest_tags.png",
    "09_causal_dag.png": "09_causal_dag.png",
    "10_causal_adjacency_matrix.png": "10_causal_adjacency_matrix.png",
    "11_ood_anomaly_score_distribution.png": "11_ood_anomaly_score_distribution.png",
    "12_anova_f_scores.png": "12_anova_f_scores.png",
    "13_mutual_information_scores.png": "13_mutual_information_scores.png",
    "14_pca_explained_variance.png": "14_pca_explained_variance.png",
    "15_pca_biplot.png": "15_pca_biplot.png",
    "16_train_test_class_distribution.png": "16_train_test_class_distribution.png",
    "18_model_performance_comparison.png": "18_model_performance_comparison.png",
    "19_confusion_matrices.png": "19_confusion_matrices.png",
    "20_roc_curves.png": "20_roc_curves.png",
    "21_cross_validation_accuracy.png": "21_cross_validation_accuracy.png",
    "22_learning_curves.png": "22_learning_curves.png",
    "26_baseline_vs_tuned_comparison.png": "26_baseline_vs_tuned_comparison.png",
    "27_best_tuned_model_details.png": "27_best_tuned_model_details.png",
    "28_feature_importance.png": "28_feature_importance.png",
    "29_all_models_roc_auc_ranking.png": "29_all_models_roc_auc_ranking.png",
    "31_shap_dependence_interaction_plot.png": "31_shap_dependence_interaction_plot.png",
    "32_shap_interaction_matrix_heatmap.png": "32_shap_interaction_matrix_heatmap.png",
    "36_probability_calibration_reliability_diagram.png": "36_probability_calibration_reliability_diagram.png",
    "38_causal_uplift_targeting_segments.png": "38_causal_uplift_targeting_segments.png",
    
    # Plots that used V8_PLOTS or other paths before
    "09_causal_dag.png": "09_causal_dag.png",
    "10_causal_adjacency_matrix.png": "10_causal_adjacency_matrix.png",
    "37_knowledge_distillation_teacher_student_comparison.png": "37_knowledge_distillation_teacher_student_comparison.png",
    "25_scarf_contrastive_learning_embeddings.png": "25_scarf_contrastive_learning_embeddings.png",
    "23_differential_privacy_comparison.png": "23_differential_privacy_comparison.png",
    "17_label_smoothing_mixup_regularization.png": "17_label_smoothing_mixup_regularization.png",
    "24_attentive_tabular_network_feature_selection.png": "24_attentive_tabular_network_feature_selection.png",
    
    # Specific edge case where ROOT_DIR + '/assets' was used for neural_attention...
    "show_plot(V8_PLOTS, '24_attentive_tabular_network_feature_selection.png'": "show_plot(V8_PLOTS, '24_attentive_tabular_network_feature_selection.png'"
}

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for old_str, new_str in REPLACEMENTS.items():
        content = content.replace(old_str, new_str)
        
    # Also handle the V8_PLOTS path assignment in pages where it was used
    content = content.replace("", "")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    search_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2"
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                update_file(filepath)
    print("Done updating files.")

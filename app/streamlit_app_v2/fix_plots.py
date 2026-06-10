import os
import glob

# Mappings of old strings to new strings
REPLACEMENTS = {
    # Directory paths
    "V5_PLOTS": "V8_PLOTS",
    "'v5_plots'": "'v8 plots'",
    "\"v5_plots\"": "\"v8 plots\"",
    "get_v5_plot_path": "get_v8_plot_path",
    
    # Notebook plot folder usage
    "NOTEBOOK_PLOTS": "V8_PLOTS",

    # Individual file name mappings
    "02_2_4_categorical_feature_distributions.png": "02_categorical_feature_distributions.png",
    "03_2_5_numerical_feature_distributions.png": "03_numerical_feature_distributions.png",
    "04_2_6_numerical_features_outlier_detection_boxplots.png": "04_outlier_detection_boxplots.png",
    "05_2_7_feature_vs_target_numerical_features_by_outcom.png": "05_numerical_features_by_match_outcome.png",
    "06_2_8_feature_vs_target_categorical_features_by_outc.png": "06_categorical_features_by_match_outcome.png",
    "07_2_9_correlation_heatmap_numerical_features.png": "07_pearson_correlation_matrix.png",
    "08_2_10_interest_tags_analysis.png": "08_top_30_interest_tags.png",
    "09_causal_discovery_going_beyond_correlation.png": "09_causal_dag.png",
    "10_causal_discovery_going_beyond_correlation.png": "10_causal_adjacency_matrix.png",
    "11_flex_11_out_of_distribution_ood_rejection_guardrai.png": "11_ood_anomaly_score_distribution.png",
    "12_2_21_anova_f_score_feature_selection_selectkbest.png": "12_anova_f_scores.png",
    "13_2_22_mutual_information_feature_selection.png": "13_mutual_information_scores.png",
    "14_2_24_explained_variance_analysis.png": "14_pca_explained_variance.png",
    "15_2_26_pca_biplot_first_two_principal_components.png": "15_pca_biplot.png",
    "16_section_7_train_test_split.png": "16_train_test_class_distribution.png",
    "18_10_2_model_comparison_table.png": "18_model_performance_comparison.png",
    "19_confusion_matrix_10_3_confusion_matrices.png": "19_confusion_matrices.png",
    "20_10_4_roc_curves.png": "20_roc_curves.png",
    "21_10_6_cross_validation_scores_5_fold.png": "21_cross_validation_accuracy.png",
    "22_10_7_learning_curves_top_3_models.png": "22_learning_curves.png",
    "26_11_3_before_vs_after_tuning_comparison.png": "26_baseline_vs_tuned_comparison.png",
    "27_confusion_matrix_11_4_best_tuned_model_detailed_results.png": "27_best_tuned_model_details.png",
    "28_section_11_feature_importance_analysis.png": "28_feature_importance.png",
    "29_section_12_final_model_summary.png": "29_all_models_roc_auc_ranking.png",
    "31_flex_13_shap_interaction_values_attribution_of_syn.png": "31_shap_dependence_interaction_plot.png",
    "32_flex_13_shap_interaction_values_attribution_of_syn.png": "32_shap_interaction_matrix_heatmap.png",
    "36_flex_14_model_calibration_reliability_diagrams.png": "36_probability_calibration_reliability_diagram.png",
    "38_flex_16_causal_uplift_modeling_t_learner_meta_clas.png": "38_causal_uplift_targeting_segments.png",
    
    # Plots that used NOTEBOOK_PLOTS or other paths before
    "causal_dag.png": "09_causal_dag.png",
    "causal_adjacency.png": "10_causal_adjacency_matrix.png",
    "knowledge_distillation.png": "37_knowledge_distillation_teacher_student_comparison.png",
    "scarf_embeddings.png": "25_scarf_contrastive_learning_embeddings.png",
    "differential_privacy.png": "23_differential_privacy_comparison.png",
    "regularization_real_curves.png": "17_label_smoothing_mixup_regularization.png",
    "neural_attention_1779900028860.png": "24_attentive_tabular_network_feature_selection.png",
    
    # Specific edge case where ROOT_DIR + '/assets' was used for neural_attention...
    "show_plot(ROOT_DIR + '/assets', '24_attentive_tabular_network_feature_selection.png'": "show_plot(V8_PLOTS, '24_attentive_tabular_network_feature_selection.png'"
}

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for old_str, new_str in REPLACEMENTS.items():
        content = content.replace(old_str, new_str)
        
    # Also handle the NOTEBOOK_PLOTS path assignment in pages where it was used
    content = content.replace("NOTEBOOK_PLOTS = os.path.join(ROOT_DIR, 'assets', 'notebook_plots')\n", "")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    search_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2"
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py') and file not in ['update_plots.py', 'fix_plots.py']:
                filepath = os.path.join(root, file)
                update_file(filepath)

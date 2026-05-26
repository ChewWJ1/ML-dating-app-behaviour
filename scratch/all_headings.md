P 27 (Normal): 1.0 Team Organization and Management
P 28 (Normal): 1.1 Team Formation and Collaboration Mechanisms
P 32 (Normal): 1.2 Roles and Responsibilities
P 36 (Normal): 1.3 Project Timeline and Gantt Chart
P 42 (Normal): 2.0 Problem and Objective
P 43 (Normal): 2.1 Project Background and Relevance
P 45 (Normal): 2.2 Dataset Breakdown and Target Definition
P 70 (Normal): 3.0 Methodology and Model Explanation
P 71 (Normal): 3.1 Preprocessing Pipeline & Feature Engineering
P 73 (List Paragraph): 1. Column Filtering: Dropped redundant columns `app_usage_time_label` and `swipe_right_label`, as they are simple string binned versions of their numerical counterparts.
P 74 (List Paragraph): 2. Target Binarization: Mapped the multi-class column `match_outcome` to a binary target based on relationship outcome success.
P 75 (List Paragraph): 3. Ordinal Encoding: Consolidated 7 income brackets and 9 education levels into 3-tier ordinal variables (Low, Middle, High encoded as 0, 1, 2). Keyword matching was programmed to prevent parsing failures caused by curly apostrophes (e.g. Bachelor’s vs. Bachelor's).
P 76 (List Paragraph): 4. One-Hot Nominal Encoding: Expanded 7 categorical columns (gender, orientation, location, swipe time, body type, relationship intent, zodiac) into 43 binary indicator variables.
P 77 (List Paragraph): 5. Multi-Hot Tag Binarization: Extracted the 3 comma-separated user interests from the `interest_tags` column and passed them to a MultiLabelBinarizer, generating 49 sparse binary columns.
P 78 (List Paragraph): 6. Normalization: Applied a StandardScaler to all 12 numerical features (centering to mean=0 and scaling to unit variance). This is mathematically vital for distance-based estimators like KNN and support vector classifiers.
P 79 (List Paragraph): 7. Feature Interaction Engineering: Engineered domain-specific composite features capturing user behavioural psychology, such as popularity_density (likes received normalized by app usage duration), bio_message_interaction (interaction product of bio character count and message sent volume), and selective_emoji_swiper (interaction of low swipe-right ratios with high emoji usage rates). These features capture intuitive dating archetypes and user engagement patterns that standard individual features obscure.
P 83 (Normal): 3.2 Feature Selection and PCA Analysis
P 92 (Normal): 3.3 Model Selection and Theoretical Framework
P 111 (Normal): 4.0 Results and Visualization
P 112 (Normal): 4.1 Baseline Performance Evaluation
P 121 (Normal): 4.2 Cross-Validation and Generalization Analysis
P 126 (Normal): 4.3 Hyperparameter Tuning and Optimization
P 149 (Normal): 5.0 Insights and Interpretation
P 150 (Normal): 5.1 Scientific Evaluation of Feature Signal
P 152 (Normal): 5.2 Model Explainability and Feature Attribution
P 156 (Normal): 5.3 Demographic Parity and Fairness Analysis
P 158 (Normal): 5.4 AutoML Benchmarking
P 160 (Normal): 5.5 Jupyter Notebook Structure and Code Index
P 164 (Normal): 6.0 Implemented Enhancements, Performance Optimization & Excluded Techniques
P 165 (Normal): 6.1 Summary of Implemented Enhancements & Optimizations
P 171 (Normal): 6.2 Detailed Technical Specifications
P 173 (Normal): 1. Class Imbalance Mitigation
P 177 (Normal): 2. Statistical Significance Testing
P 181 (Normal): 3. Multi-Threaded Bagging SVM Ensemble
P 185 (Normal): 4. Smart Checkpointing & Instant Loading
P 189 (Normal): 5. Cross-Validation Parallel Thread Manager
P 194 (Normal): 6. Feature Interaction Engineering
P 199 (Normal): 7. In-Notebook Interactive Matchmaker Simulator
P 204 (Normal): 6.3 Summary of Evaluated and Excluded Techniques
P 208 (Normal): 7.0 Conclusion and Future Work
P 209 (Normal): 7.1 Key Findings Summary
P 212 (Normal): 7.2 Recommendations for Future Research
P 216 (Normal): 1.  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785
P 217 (Normal): 2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (Vol. 30, pp. 4765-4774). Curran Associates, Inc.
P 218 (Normal): 3. Nisar, K. (2026). Dating App Behavior Dataset.Kaggle.https://www.kaggle.com/datasets/
keyushnisar/dating-app-behavior-dataset
P 219 (Normal): 4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(85), 2825-2830.
P 220 (Normal): 5. Universiti Malaya. (2026). WIA1006/WID3006 Machine Learning Group Assignment Guidelines. Faculty of Computer Science and Information Technology, University of Malaya.
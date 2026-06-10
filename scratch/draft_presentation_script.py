# Let's draft the script and count the words for each speaker.

speakers = {
    "CHEW WEI JIAN": (
        "Welcome. In data science, we often seek patterns in human behavior, but can algorithmically binarize the human heart? "
        "Our project, 'Tying the Data Knot', tackles matchmaking prediction on a mobile dating app using the Kaggle Dating App Behavior dataset, "
        "consisting of fifty thousand records and twenty-five features. We frame this as a binary classification task to predict whether a user "
        "achieves a meaningful connection or not. The positive class combines mutual matches, instant matches, dates, and relationships, representing "
        "thirty-nine point seven percent of the data. The remaining sixty point three percent represents the negative class, spanning ghosted, blocked, "
        "catfished, or ignored chats. We began by architecting a production-grade machine learning pipeline, running CUDA and DirectML auto-detection "
        "for hardware acceleration. Our goal was simple: evaluate if typical demographic and engagement features hold any predictive signal for romantic "
        "success. I'll now pass it over to Ku Jian Cheng to discuss our exploratory data analysis and feature engineering."
    ),
    "KU JIAN CHENG": (
        "Thank you, Wei Jian. We initiated our process with a comprehensive ten-part exploratory data analysis. The data distributions were "
        "highly uniform, with no outlier anomalies or strong linear correlations between features and the target. To move beyond mere correlation, "
        "we mapped the data's causal structure using the PC Algorithm with a KCI conditional independence test. This revealed a sparse causal DAG with "
        "no direct pathways from standard app metrics to matchmaking outcomes. In preprocessing, we engineered interaction terms and selectivity ratios, "
        "collapsing income and education into ordinal scales, and multi-hot encoding forty-nine unique interest tags. To protect our models, we deployed "
        "an Isolation Forest out-of-distribution guardrail to reject anomalous inputs. Finally, we performed feature selection by taking the union of the "
        "top forty ANOVA F-score and Mutual Information features, resulting in sixty-seven optimized features. Next, Ng Jin Ru will present our model training."
    ),
    "NG JIN RU": (
        "Thank you, Jian Cheng. With our feature matrix optimized, we split the dataset into an eighty-twenty stratified partition and applied SMOTE class "
        "balancing natively on the training set. We trained sixteen distinct models, ranging from traditional classifiers like Logistic Regression, KNN, "
        "Decision Trees, and Support Vector Machines, to advanced boosting ensembles such as LightGBM, CatBoost, and XGBoost. We also integrated deep learning "
        "architectures, including Multi-Layer Perceptrons, Feature Tokenizer Transformers, and SAINT. To optimize performance, we conducted a GPU-accelerated "
        "Optuna search across one thousand trials, maximizing the Matthew's Correlation Coefficient to ensure robust class balancing. We also implemented fourteen "
        "V8 methodology patches, including isolating our RobustScaler post-split, resolving SMOTE cross-validation integrity, and preventing data leakage "
        "in our calibration and conformal prediction splits. I will now hand over to Ang Ying En to elaborate on our advanced models and causal inference."
    ),
    "ANG YING EN": (
        "Thank you, Jin Ru. To push our modeling limits, we implemented several state-of-the-art frameworks. We deployed a Graph Attention Network "
        "modeling users as nodes in a similarity graph, trained an Attentive Tabular Network to extract instance-wise feature masks, and utilized SCARF "
        "self-supervised contrastive learning to pre-train robust profile embeddings. We also evaluated TabPFN, a zero-shot tabular transformer, and trained "
        "our neural network with Opacus Differential Privacy at a strict epsilon value of eight point zero to guarantee user data privacy. To measure "
        "intervention impact, we used Double Machine Learning to isolate the Average Treatment Effect of profile photo count on match outcomes, which yielded a "
        "p-value greater than zero point six zero, proving the causal effect is statistically indistinguishable from zero. We also generated Microsoft DiCE "
        "counterfactual recourses to provide users with actionable feedback. I'll now pass it to Chaang Wai Chiu to discuss evaluation and our key findings."
    ),
    "CHAANG WAI CHIU": (
        "Thank you, Ying En. We evaluated our models using five-fold cross-validation and rigorous Friedman and Nemenyi post-hoc statistical tests. "
        "Our key scientific finding is that all sixteen models—including AutoML baselines from FLAML and PyCaret—converge exactly at the majority class "
        "baseline of sixty point three percent accuracy, with an ROC-AUC of zero point five zero. Rather than a failure, this null result is a valuable "
        "scientific finding: standard app interactions hold zero predictive signal in programmatic synthetic data. For deployment, we built the SwipeIQ V2 "
        "Streamlit dashboard, calibrating predictions via Isotonic Regression, and mapping conformal prediction uncertainty bands. We recommend that future dating "
        "algorithms move beyond structured metrics, leveraging NLP and LLMs to analyze user bios and active behavioral cues like response latency. In conclusion, "
        "while love remains unpredictable, our engineering pipeline stands as a robust, production-ready framework for real-world connection discovery."
    )
}

for name, script in speakers.items():
    words = script.split()
    print(f"{name}: {len(words)} words")

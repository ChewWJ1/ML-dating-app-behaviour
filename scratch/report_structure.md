# Docx File: reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx

Total Paragraphs: 221

## First 50 Paragraphs:

P 0 (Style: Normal): Faculty of Computer Science and Information Technology

P 1 (Style: Normal): University of Malaya

P 2 (Style: Normal): Semester 2, Session 2025/2026

P 3 (Style: Normal): WIA 1006 - Machine Learning

P 4 (Style: Normal): OCC 6

P 7 (Style: Normal): Group Assignment Report

P 8 (Style: Normal): Tying the (Data) Knot: Predicting Meaningful Connections

P 11 (Style: Normal): Group 3
Group name:

P 12 (Style: Normal): Group Members:

P 13 (Style: Normal): CHEW WEI JIAN 23118568/2

P 14 (Style: Normal): KU JIAN CHENG 23079373/2

P 15 (Style: Normal): NG JIN RU 23116192/2

P 16 (Style: Normal): ANG YING EN 23116738/2

P 17 (Style: Normal): CHAANG WAI CHIU 23104771/2

P 19 (Style: Normal): Executive Summary

P 20 (Style: Normal): This report presents the development, evaluation, and optimization of an end-to-end Machine Learning classification pipeline designed to predict meaningful relationship connections on a mobile dating application. Utilizing a 50,000-sample dataset, we binarized 10 relationship outcomes into a target connection variable (representing Mutual Match, Instant Match, Date Happened, and Relationship Formed) and preprocessed 25 variables through ordinal, one-hot, and multi-hot encodings. We conducted feature selection using a union of ANOVA F-scores and Mutual Information, selecting 67 features, and projected them down to 55 dimensions via Principal Component Analysis (PCA). Fourteen baseline and advanced classifiers—Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, XGBoost, a custom multi-threaded Bagging SVM ensemble, LightGBM, CatBoost, Multi-Layer Perceptron (MLP), Balanced Random Forest, Cosine KNN CF, FT-Transformer, SAINT, and NODE were trained and tuned using RandomizedSearchCV, after natively balancing the training split via SMOTE. Validation was conducted via 5-fold cross-validation, paired t-tests, SHAP explainability analyses, and demographic parity audits.

P 21 (Style: Normal): Our key finding indicates that while the pipeline runs with full engineering integrity, all models converge at the majority class baseline (60.30% test accuracy, ROC-AUC ≈ 0.50). This result is a valuable scientific finding, mathematically demonstrating the absence of predictive signal within the programmatic dataset. Features like zodiac sign or swipe ratio carry no genuine correlation with connection success. Based on these results, we recommend that future dating algorithms focus on natural language bio analysis (via NLP/LLMs) and active behavioral cues (such as response latency and chat length) to capture the true, non-linear signals of human connections.

P 24 (Style: Normal): Table of Contents

P 27 (Style: Normal): 1.0 Team Organization and Management

P 28 (Style: Normal): 1.1 Team Formation and Collaboration Mechanisms

P 29 (Style: Normal): Our team consists of five members from OCC 10 of FCSIT, Universiti Malaya. The team leader is Chew Wei Jian, and the core members are Ku Jian Cheng, Ng Jin Ru, Ang Ying En and Chaang Wai Chiu. We established this group based on a shared academic interest in applied machine learning pipelines and a joint goal of achieving excellence in the WIA1006/WID3006 course assignment. To manage work across our varied skills, we structured our collaboration using professional project management workflows.

P 30 (Style: Normal): Communication was maintained through regular weekly synchronization meetings held via Microsoft Teams and in-person lab sessions. A shared WhatsApp group served as our primary channel for rapid communication, debugging, and task coordination. For source code management and collaborative integration, we established a central GitHub repository. Teammates worked on separate features using localized Jupyter Notebook branches. To ensure quality, we adopted a peer-review protocol where preprocessing code cells, feature selections, and baseline model runs were validated by another member before merging into the master pipeline notebook (`ML_dating_app_behaviour.ipynb`).

P 31 (Style: Normal): We implemented a critical-path execution schedule, prioritizing data preprocessing and categorical encoding in the early weeks. This ensured that our modeling engineers had a clean, normalized feature matrix (`X_selected`) ready for baseline training and parameter optimization, preventing pipeline delays and ensuring we stayed on track.

P 32 (Style: Normal): 1.2 Roles and Responsibilities

P 33 (Style: Normal): The roles were allocated based on technical strengths. Chew Wei Jian managed the integration, parallelization, and caching; Ku Jian Cheng drove the preprocessing and encoding; Ng Jin Ru handled initial visual checks; Ang Ying En programmed training loops and RandomizedSearchCV; and Chaang Wai Chiu developed the SHAP explainability plots, demographic audits, and the interactive dashboard. Table 1 outlines our specific responsibilities:

P 34 (Style: Normal): Table 1: Roles and Task Contributions for OCC 10 Group Members

P 36 (Style: Normal): 1.3 Project Timeline and Gantt Chart

P 37 (Style: Normal): Our work followed a structured 7-week cycle, matching the steps of a standard data science pipeline. Table 2 outlines the timeline of activities, and Table 3 details the progress Gantt chart:

P 38 (Style: Normal): Table 2: Weekly Project Timeline and Completed Phases

P 40 (Style: Normal): Table 3: Gantt Chart of Project Progress

P 42 (Style: Normal): 2.0 Problem and Objective

P 43 (Style: Normal): 2.1 Project Background and Relevance

P 44 (Style: Normal): Modern dating applications utilize matching algorithms to connect individuals. However, matching is often superficial and leads to high ghosting rates or negative outcomes. A key challenge is predicting connection success based on behavioral data rather than simple profiles. This project framing attempts to solve a binary classification problem: predicting whether a user will achieve a meaningful connection (defined as target=1, representing outcomes like Mutual Match, Instant Match, Date Happened, and Relationship Formed) or experience a negative outcome (defined as target=0, representing Blocked, Catfished, Chat Ignored, Ghosted, No Action, and One-sided Likes).

P 45 (Style: Normal): 2.2 Dataset Breakdown and Target Definition

P 46 (Style: Normal): The objective is to train a machine learning classifier that utilizes demographic factors and in-app behavioral signals. Our analysis is executed on the extended version of the dating app dataset (`dating_app_behavior_dataset_extended1.csv`). While the original dataset provides 19 baseline features, this project utilizes the extended version incorporating 6 additional variables that provide critical signals for connection modeling:

P 47 (Style: List Paragraph): age:

P 48 (Style: List Paragraph): Numeric (18–59).

P 49 (Style: List Paragraph): Age differences serve as a core preference constraint in mating functions.


## All Headings in Document:

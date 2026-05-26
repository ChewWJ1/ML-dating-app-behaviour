notes_path = 'PROJECT_NOTES.md'
with open(notes_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Let's clean up line endings by stripping \r\n and then perform matching
for i, line in enumerate(lines):
    # Fix Step 9 text
    if "### Step 9: Advanced Model Training" in line:
        # Check if next line contains We train 15
        if "We train 15 distinct" in lines[i+1]:
            lines[i+1] = "We train 16 distinct baseline models, similarity recommenders, PyTorch deep learning architectures, and zero-shot transformers.\n"
            # Append TabNet line after the last item of Step 9
            # Let's find the next item (usually starting with - **[V5])
            for idx in range(i+2, i+10):
                if "- **[V5] Label Smoothing" in lines[idx]:
                    lines[idx] = lines[idx] + "- **[V5.1] TabNet-style Attentive Neural Network**: Implemented a PyTorch Attentive Tabular Network that outputs dynamic, instance-wise feature selection masks, visualizing individual column targeting choices in an explainable selection heatmap.\n"
                    print("Successfully updated Step 9 text.")
                    break
                    
    # Fix Index Training line
    if "| 10 — Advanced Model Training |" in line:
        lines[i] = "| 10 — Advanced Model Training | 16 baseline/PyTorch/zero-shot models, GNN node classification, SCARF contrastive learning, Differential Privacy, **[V5] Zero-Shot TabPFN**, **[V5] Mixup & Label Smoothing**, **[V5.1] Label Smoothing Loss Visualizer**, **[V5.1] TabNet-style Attentive Tabular Selection Network** |\n"
        print("Successfully updated Index Training row.")

with open(notes_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("CRLF fix run complete.")

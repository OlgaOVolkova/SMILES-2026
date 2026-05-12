# SMILES-HALLUCINATION-DETECTION - Solution Report

## Start

### Google Colab

```python
Downloading the project
!git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
%cd SMILES-HALLUCINATION-DETECTION
!pip install -r requirements.txt

Replacing executable files
!wget https://raw.githubusercontent.com/OlgaOVolkova/SMILES-2026/main/aggregation.py
!wget https://raw.githubusercontent.com/OlgaOVolkova/SMILES-2026/main/probe.py
!wget https://raw.githubusercontent.com/OlgaOVolkova/SMILES-2026/main/splitting.py
!wget https://raw.githubusercontent.com/OlgaOVolkova/SMILES-2026/main/solution.py

Results
!python solution.py
```

## Final solution description

### splitting.py:
    Modified:
        Introduced cross-validation (k-fold) to reduce dependency on random data splitting and to mitigate the impact of class imbalance (70% hallucinations / 30% truthful)

### aggregation.py:
    Modified:
        Changed aggregation method
        Introduced geometric / statistical feature
    
    Best Results:
        Aggregation was changed to multi-layer fusion strategy (last 4 layers), which increased test AUROC by 1%
        Geometric features using inter-layer cosine similarity increased validation AUROC by 4%
    
    What was tried but did not work:
        Token pooling (mean, max, weighted) did not show good results. On average, AUROC on validation and test sets was 6-19% lower. This is likely because these pooling methods average out token-level information, losing the critical signal that is concentrated in the last token. For short responses, the last token already contains most of the relevant information for hallucination detection.
        Geometric features based on layer-wise activation norms and sequence length performed worse than inter-layer cosine similarity by approximately 1%. This may be because:
            Sequence length alone is a weak signal (hallucinations occur in both short and long responses)
            Layer-wise norms capture activation magnitude but not directional changes between layers
            Cosine similarity specifically measures representation drift, which better captures uncertainty and semantic shifts associated with hallucinations

### probe.py:
    Modified:
        Classifier architecture parameters
        Optimizer settings
    
    Best Results:
        Classifier: Dropout increased from 0.3 → 0.4, hidden layer size reduced from 256 → 128
        Optimizer: learning rate decreased from 1e-3 → 5e-4
        This combination increased test AUROC by 1.5%

    What was tried but did not work:
        Multi-stage compression (e.g., 512 → 256 → 128 → 1) did not improve results (likely overfitting due to too many parameters for small dataset)
        Batch training (mini-batches) did not improve results (full-batch training worked better, possibly because the dataset is small and batch noise disrupted learning)
        Varying the number of epochs (100, 200, 300, 500) had no significant impact  (200 epochs was optimal)

### solution.py
    Modified:
        Enabled geometric features (USE_GEOMETRIC = True)

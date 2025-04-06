# Phosphorylation Site Prediction: Complete Documentation

## Overview

This documentation explains a codebase that predicts phosphorylation sites in proteins. Phosphorylation is a process where a phosphate group is added to certain amino acids (usually Serine, Threonine, or Tyrosine) in a protein, which changes how the protein works. This codebase processes protein data, extracts features, and builds a machine learning model to predict where phosphorylation will happen.

## Flow Diagram

```
┌─────────────────────────┐
│ Load Raw Sequence Data  │
│   (Sequence_data.txt)   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Extract Headers and    │
│       Sequences         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Load Labels Data     │
│     (labels.xlsx)       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Merge Sequence and    │
│      Labels Data        │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Clean and Preprocess   │
│         Data            │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Generate Negative      │
│  Samples for Each       │
│  Protein Sequence       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│Extract Feature Windows  │
│Around Phosphorylation   │
│       Sites             │
└───────────┬─────────────┘
            ▼
┌──────────────────────────────────────────────────────────┐
│                Feature Extraction                         │
├───────────┬───────────┬───────────┬──────────┬───────────┤
│    AAC    │    DPC    │    TPC    │ Binary   │ Physico-  │
│ Features  │ Features  │ Features  │ Encoding │ chemical  │
└───────────┴───────────┴───────────┴──────────┴───────────┘
            │
            ▼
┌─────────────────────────┐
│  Merge All Features     │
│    into Final Set       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Split Data into       │
│ Train/Validation/Test   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Train XGBoost Model    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Evaluate Model on     │
│      Test Data          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Analyze Feature        │
│    Importance           │
└─────────────────────────┘
```

## Function Descriptions

### 1. Loading and Processing Sequences

**What it does:**
- Opens a text file with protein sequences in FASTA format
- Extracts headers (protein IDs) and sequences
- Creates a table (DataFrame) with two columns: Header and Sequence

```python
# Open and read the file
with open("Sequence_data.txt", "r") as file:
    for line in file:
        if line.startswith(">"):
            # Process header line
            headers.append(middle_part_of_header)
        else:
            # Process sequence line
            current_seq += line

# Create a DataFrame
df = pd.DataFrame({
    "Header": headers,
    "Sequence": sequences
})
```

### 2. Loading Labels Data

**What it does:**
- Loads data about known phosphorylation sites from an Excel file
- This tells us which amino acids in which proteins are known to be phosphorylated

```python
df_labels = pd.read_excel("labels.xlsx")
```

### 3. Merging Sequence and Label Data

**What it does:**
- Combines the sequence data with the labels data
- Matches proteins by their ID
- Creates a "target" column where 1 means it's a known phosphorylation site

```python
df_merged = pd.merge(
    df,
    df_labels,
    left_on="Header",
    right_on="UniProt ID",
    how="left"
)

df_merged["target"] = df_merged["UniProt ID"].notnull().astype(int)
```

### 4. Data Cleaning

**What it does:**
- Removes rows with missing protein IDs
- Removes extremely long protein sequences to keep processing manageable

```python
# Drop rows with missing UniProt ID
df_merged.dropna(subset=["UniProt ID"], inplace=True)

# Drop sequences with length > 5000
df_merged = df_merged[df_merged["Sequence"].str.len() <= 5000]
```

### 5. Generating Negative Samples

**What it does:**
- For each protein sequence:
  - Identifies all known phosphorylation sites (positives)
  - Finds all potential phosphorylation sites (all S/T/Y amino acids)
  - Creates a set of negative examples by randomly sampling from S/T/Y sites that are not known to be phosphorylated
  - Balances the dataset by using the same number of negatives as positives where possible

```python
for header_value, group in df_merged.groupby("Header"):
    # Extract the amino-acid sequence
    seq = group["Sequence"].iloc[0]
    
    # Get positive positions from the DataFrame
    positive_positions = group["Position"].unique().tolist()  
    
    # Find all S/T/Y positions in the sequence
    st_y_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
    
    # Exclude the positives to get negative candidates
    negative_candidates = [pos for pos in st_y_positions if pos not in positive_positions]
    
    # Number of positives for this sequence
    n_pos = len(positive_positions)
    
    # Sample negative sites (same number as positives if possible)
    sampled_negatives = random.sample(negative_candidates, min(n_pos, len(negative_candidates)))
    
    # Create new rows for negative sites and combine with positives
```

### 6. Window Extraction Function

**What it does:**
- Takes a protein sequence and a position as input
- Extracts a window of amino acids centered on that position
- This window will be used for feature calculation

```python
def extract_window(sequence, position, window_size=5):
    """Extract a window of amino acids around a position"""
    pos_idx = position - 1
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    window = sequence[start:end]
    return window
```

### 7. Amino Acid Composition (AAC) Features

**What it does:**
- Calculates the frequency of each amino acid in the input sequence window
- Creates a dictionary with all 20 standard amino acids as keys
- Counts how many times each amino acid appears
- Converts raw counts to frequencies (percentage of the sequence)

```python
def extract_aac(sequence):
    # Initialize dictionary with zeros
    aac = {aa: 0 for aa in amino_acids}
    
    # Count amino acids
    seq_length = len(sequence)
    for aa in sequence:
        if aa in aac:
            aac[aa] += 1
    
    # Convert counts to frequencies
    for aa in aac:
        aac[aa] = aac[aa] / seq_length if seq_length > 0 else 0
        
    return aac
```

### 8. Dipeptide Composition (DPC) Features

**What it does:**
- Calculates the frequency of adjacent amino acid pairs in the sequence window
- Creates a dictionary with all 400 possible amino acid pairs (20×20)
- Counts each pair of consecutive amino acids (dipeptides)
- Converts counts to frequencies

```python
def extract_dpc(sequence):
    # Initialize dictionary with all possible dipeptides
    dpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dpc[aa1 + aa2] = 0
    
    # Count dipeptides
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in dpc:
            dpc[dipeptide] += 1
    
    # Convert counts to frequencies
    total_dipeptides = len(sequence) - 1
    for dipeptide in dpc:
        dpc[dipeptide] = dpc[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
        
    return dpc
```

### 9. Tripeptide Composition (TPC) Features

**What it does:**
- Calculates the frequency of three consecutive amino acids in the sequence window
- Creates a dictionary with all 8000 possible amino acid triplets (20×20×20)
- Counts each triplet of consecutive amino acids
- Converts counts to frequencies

```python
def extract_tpc(sequence):
    # Initialize dictionary with all possible tripeptides
    tpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                tpc[aa1 + aa2 + aa3] = 0
    
    # Count tripeptides
    for i in range(len(sequence) - 2):
        tripeptide = sequence[i:i+3]
        if tripeptide in tpc:
            tpc[tripeptide] += 1
    
    # Convert counts to frequencies
    total_tripeptides = len(sequence) - 2
    for tripeptide in tpc:
        tpc[tripeptide] = tpc[tripeptide] / total_tripeptides if total_tripeptides > 0 else 0
        
    return tpc
```

### 10. TPC Batch Processing

**What it does:**
- Processes tripeptide composition features in smaller batches to avoid memory issues
- Creates a folder to store the intermediate batch files
- For each batch of data, extracts sequence windows and calculates TPC features
- Saves each batch to a CSV file

```python
def process_tpc_in_batches(df, batch_size=500, window_size=5, output_dir="tpc_batches"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate number of batches
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Process each batch
    for batch_idx in range(n_batches):
        # Get batch data
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Extract windows
        batch_df['Window'] = batch_df.apply(
            lambda row: extract_window(row['Sequence'], row['Position'], window_size), 
            axis=1
        )
        
        # Process TPC features for this batch
        # Save batch to CSV
```

### 11. Combining TPC Batches

**What it does:**
- Takes the list of batch files from the previous function
- Combines all the TPC feature batch files into a single output file

```python
def combine_tpc_batches(batch_files, output_file="phosphorylation_tpc_features_window5.csv"):
    # Combine batch files
    combined_df = pd.concat([pd.read_csv(file) for file in batch_files])
    
    # Save combined file
    combined_df.to_csv(output_file, index=False)
    
    return output_file
```

### 12. Binary Encoding Features

**What it does:**
- Creates one-hot encoded vectors for amino acids in the window
- For a single amino acid: creates a vector of 20 zeros and sets the position corresponding to the amino acid to 1
- For the entire window: applies one-hot encoding to each position and combines them
- This creates a positional encoding where each position has 20 features

```python
def binary_encode_amino_acid(aa):
    # Initialize vector with zeros
    encoding = [0] * 20
    
    # Set the corresponding position to 1
    if aa in amino_acids:
        idx = amino_acids.index(aa)
        encoding[idx] = 1
    
    return encoding

def extract_binary_encoding(sequence, position, window_size=10):
    # Extract window and pad if necessary
    # Binary encode each amino acid in the window
    # Return combined feature vector
```

### 13. Physicochemical Properties Features

**What it does:**
- Loads a CSV file with physical and chemical properties for each amino acid
- Properties might include hydrophobicity, charge, size, etc.
- Extracts these properties for a window of amino acids around the target position

```python
def load_physicochemical_properties(file_path="physiochemical_property.csv"):
    # Load properties from CSV file
    
def extract_physicochemical_features(sequence, position, window_size=10, properties=None):
    # Extract window around target position
    # For each amino acid in the window, add its physicochemical properties
    # Return combined vector of properties
```

### 14. Batch Processing Functions

**What it does:**
- Similar to the TPC batch processing, but for binary encoding and other features
- Processes data in smaller batches to avoid memory issues
- Adds identifying columns (Header, Position) and target label to feature dictionaries

```python
def process_binary_encoding(df, window_size=10, batch_size=1000, output_file="phosphorylation_binary_encoding.csv"):
    # Process in batches
    for batch_idx in range(n_batches):
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Extract binary encoding features for each sample
        # Save to CSV
```

### 15. Feature Merging

**What it does:**
- Loads all the individual feature files (AAC, DPC, TPC, Binary Encoding, Physicochemical)
- Combines them all together by matching on Header (protein ID), Position, and target
- Creates a single dataset with all features

```python
def merge_all_features(output_file="phosphorylation_all_features.csv"):
    # Load all feature files
    aac_df = pd.read_csv("phosphorylation_aac_features_window5.csv")
    dpc_df = pd.read_csv("phosphorylation_dpc_features_window5.csv")
    tpc_df = pd.read_csv("phosphorylation_tpc_features_window5.csv")
    be_df = pd.read_csv("phosphorylation_binary_encoding.csv")
    pc_df = pd.read_csv("phosphorylation_physicochemical.csv")
    
    # Merge all features on Header and Position
    merged_df = aac_df.merge(dpc_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(tpc_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(be_df, on=['Header', 'Position', 'target'])
    merged_df = merged_df.merge(pc_df, on=['Header', 'Position', 'target'])
    
    # Save merged features
    merged_df.to_csv(output_file, index=False)
    
    return merged_df
```

## Model Training with XGBoost

### 1. Loading Training, Validation, and Test Data

**What it does:**
- Loads the previously split data files for training, validation, and testing
- Separates features (X) from the target variable (y)
- Converts data into XGBoost's optimized data structure called DMatrix

```python
# Load training data
print("Loading training data...")
train_data = pd.read_csv('split_data/train_data.csv')
X_train = train_data.drop(['Header', 'Position', 'target'], axis=1)
y_train = train_data['target']

# Load validation data
print("Loading validation data...")
val_data = pd.read_csv('split_data/val_data.csv')
X_val = val_data.drop(['Header', 'Position', 'target'], axis=1)
y_val = val_data['target']

# Convert to DMatrix (XGBoost's optimized data structure)
print("Converting to DMatrix...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
```

### 2. Setting XGBoost Parameters

**What it does:**
- Configures parameters for the XGBoost model
- Uses GPU acceleration for faster training
- Sets hyperparameters like learning rate, tree depth, etc.
- Defines evaluation metrics

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'eta': 0.1,  # Learning rate
    'max_depth': 6,
    'device': 'cuda',
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',  # Use GPU acceleration
    'max_bin': 256  # For GPU, this can speed up training
}
```

### 3. Training the XGBoost Model

**What it does:**
- Trains the XGBoost model using the prepared training data
- Evaluates the model performance on validation data after each iteration
- Uses early stopping to prevent overfitting
- Prints progress during training

```python
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=50
)
```

### 4. Saving the Trained Model

**What it does:**
- Saves the trained model to a file for future use without needing to retrain

```python
model.save_model('phosphorylation_xgb_model.json')
print("Model saved to phosphorylation_xgb_model.json")
```

### 5. Evaluating on Test Data

**What it does:**
- Loads the separate test dataset
- Makes predictions using the trained model
- Calculates various performance metrics:
  - Accuracy: Overall correctness
  - Precision: How many predicted positives are actually positive
  - Recall: How many actual positives are correctly identified
  - F1 Score: Balance between precision and recall
  - ROC AUC: Area under the ROC curve, measuring discrimination ability
  - Confusion Matrix: Shows true/false positives and negatives

```python
test_data = pd.read_csv('split_data/test_data.csv')
X_test = test_data.drop(['Header', 'Position', 'target'], axis=1)
y_test = test_data['target']

# Create DMatrix for test data
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
```

### 6. Analyzing Feature Importance

**What it does:**
- Extracts feature importance scores from the trained model
- Sorts features by their importance
- Prints the features and their scores
- Helps understand which features are most predictive of phosphorylation sites

```python
# Get feature importances based on gain
importance_dict = model.get_score(importance_type='gain')

# Sort by importance (descending)
sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

# Pretty print
for feature, score in sorted_importance.items():
    print(f"{feature}: {score:.4f}")
```

## Results and Interpretation

### Model Performance

The XGBoost model achieved the following performance on the test set:

- **Accuracy**: 75.92%
  - The model correctly predicted about 76% of all samples.

- **Precision**: 75.33%
  - When the model predicted a site as phosphorylated, it was correct about 75% of the time.

- **Recall**: 77.10%
  - The model correctly identified about 77% of all actual phosphorylation sites.

- **F1 Score**: 76.21%
  - The balanced measure of precision and recall shows good overall performance.

- **ROC AUC**: 0.8404
  - Shows strong ability to distinguish between phosphorylated and non-phosphorylated sites.

- **Confusion Matrix**:
  ```
  True Negatives (TN): 4640
  False Positives (FP): 1569
  False Negatives (FN): 1423
  True Positives (TP): 4792
  ```

### Key Feature Importance Findings

1. **Physicochemical Properties Dominate**:
   - **PC_83** (score: 420.65) is overwhelmingly the most important feature
   - **PC_81** (166.69) and **PC_82** (43.25) also rank highly
   - This suggests that specific physicochemical properties of amino acids strongly influence phosphorylation.

2. **Binary Encoding Features Are Critical**:
   - **BE_133** (178.84) is the second most important feature
   - This indicates that the exact position of specific amino acids relative to the phosphorylation site matters greatly.

3. **Important Amino Acid Patterns**:
   - **TP** (79.47) and **YP** (75.75) dipeptides are highly important
   - This aligns with biological knowledge that proline-directed kinases often target S/T-P motifs
   - **W** (54.09): Tryptophan appears to be an important predictor
   - **SP** (32.21): Another proline-directed motif, as expected

4. **Biological Interpretation**:
   - The importance of proline-containing dipeptides (TP, YP, SP) confirms the known preference of many kinases for proline-directed sites
   - The dominance of physicochemical properties suggests that structural factors around the site (like charge, hydrophobicity, flexibility) are crucial for kinase recognition
   - Position-specific information (binary encoding features) indicates that the exact context around a site is critical

### Conclusion

The XGBoost model successfully predicts phosphorylation sites with good accuracy and balanced performance. The feature importance analysis reveals that both the chemical properties and specific sequence patterns around potential sites are critical factors in determining phosphorylation. These insights align with biological understanding of how kinases recognize and phosphorylate their target sites.

The model could be further improved through feature selection (focusing on the most important features), hyperparameter tuning, or by developing separate models for different types of phosphorylation sites (e.g., based on the kinase family).

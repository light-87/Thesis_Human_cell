# Phosphorylation Site Prediction: Code Documentation

## Overview

This codebase processes protein sequence data to identify and analyze phosphorylation sites. Phosphorylation is a post-translational modification where a phosphate group is added to certain amino acids (usually Serine, Threonine, or Tyrosine), which is critical for many cellular processes. This code prepares data for machine learning models to predict phosphorylation sites in protein sequences.

## Data Processing Flow

```
┌────────────────────────┐
│ Load Raw Sequence Data │
│  (Sequence_data.txt)   │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│   Extract Headers and   │
│       Sequences        │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│    Load Labels Data    │
│     (labels.xlsx)      │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│   Merge Sequence and   │
│      Labels Data       │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  Clean and Preprocess  │
│         Data           │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ Generate Negative      │
│ Samples for Each       │
│ Protein Sequence       │
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│Extract Feature Windows │
│Around Phosphorylation  │
│       Sites            │
└──────────┬─────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│                Feature Extraction                         │
├───────────┬───────────┬───────────┬──────────┬───────────┤
│    AAC    │    DPC    │    TPC    │ Binary   │ Physico-  │
│ Features  │ Features  │ Features  │ Encoding │ chemical  │
└───────────┴───────────┴───────────┴──────────┴───────────┘
           │
           ▼
┌────────────────────────┐
│  Merge All Features    │
│    into Final Set      │
└────────────────────────┘
```

## Detailed Function Explanations

### 1. Loading and Processing Sequences

```python
# Lists to store the processed headers and sequences
headers = []
sequences = []
current_seq = ""

# Open and read the file
with open("Sequence_data.txt", "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            # If there is an existing sequence, append it before starting a new one
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
            # Remove the ">" and extract the middle part from the header
            full_header = line[1:]
            parts = full_header.split("|")
            # Use the middle part if available; otherwise, use the full header
            middle = parts[1] if len(parts) > 1 else full_header
            headers.append(middle)
        else:
            # Concatenate sequence lines
            current_seq += line
    # Append the last collected sequence
    if current_seq:
        sequences.append(current_seq)

# Create a DataFrame with the extracted header parts and sequences
df = pd.DataFrame({
    "Header": headers,
    "Sequence": sequences
})
```

**What it does:**
- Opens a FASTA format sequence file
- For each sequence in the file:
  - Extracts the header (identifier) line that starts with ">"
  - Captures the protein sequence itself
  - Specifically extracts the UniProt ID from the header
- Creates a DataFrame with two columns: Header (protein ID) and Sequence (amino acid sequence)

### 2. Loading Labels Data

```python
df_labels = pd.read_excel("labels.xlsx") 
```

**What it does:**
- Loads known phosphorylation site data from an Excel file
- This data tells us which amino acids in which proteins are known to be phosphorylated
- Contains columns for UniProt ID, amino acid type (S/T/Y), and position in the sequence

### 3. Merging Sequence and Label Data

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

**What it does:**
- Combines the sequence data with the labels data
- Matches proteins by their UniProt ID
- Creates a "target" column where:
  - 1 = this row represents a known phosphorylation site
  - 0 = this row is not a known phosphorylation site
- Uses a left join to keep all protein sequences, even those without known phosphorylation sites

### 4. Data Cleaning

```python
# Drop rows with missing UniProt ID
df_merged.dropna(subset=["UniProt ID"], inplace=True)

# Drop sequences with length > 5000
df_merged = df_merged[df_merged["Sequence"].str.len() <= 5000]
```

**What it does:**
- Removes any rows where UniProt ID is missing
- Filters out extremely long protein sequences (>5000 amino acids) to keep processing manageable

### 5. Generating Negative Samples

```python
for header_value, group in df_merged.groupby("Header"):
    # Extract the amino-acid sequence (assuming one sequence per Header)
    seq = group["Sequence"].iloc[0]
    
    # (A) Positive positions from the DataFrame
    positive_positions = group["Position"].unique().tolist()  
    
    # (B) Find all S/T/Y positions in the sequence
    st_y_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
    
    # (C) Exclude the positives → negative candidates
    negative_candidates = [pos for pos in st_y_positions if pos not in positive_positions]
    
    # (D) Number of positives for this sequence
    n_pos = len(positive_positions)
    
    # Sample negative sites (same number as positives if possible)
    if len(negative_candidates) >= n_pos:
        sampled_negatives = random.sample(negative_candidates, n_pos)
    else:
        sampled_negatives = negative_candidates
    
    # Create new rows for negative sites
    new_rows = []
    for neg_pos in sampled_negatives:
        new_rows.append({
            "Header": header_value,
            "Sequence": seq,
            "UniProt ID": group["UniProt ID"].iloc[0], 
            "AA": seq[neg_pos - 1],   
            "Position": neg_pos,
            "target": 0
        })
    
    # Mark positives with target=1
    group = group.copy()
    group["target"] = 1
    
    # Combine positives & negatives
    neg_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([group, neg_df], ignore_index=True)
    df_list.append(combined_df)

# Combine all groups into final DataFrame
df_final = pd.concat(df_list, ignore_index=True)
```

**What it does:**
- For each protein sequence:
  - Identifies all known phosphorylation sites (positives)
  - Finds all potential phosphorylation sites (S/T/Y amino acids)
  - Creates a set of negative examples by randomly sampling from S/T/Y sites that are not known to be phosphorylated
  - Tries to balance the dataset by using the same number of negatives as positives
  - Labels positive sites with target=1 and negative sites with target=0
  - Combines all data into a final dataset

### 6. Window Extraction Function

```python
def extract_window(sequence, position, window_size=5):
    """Extract a window of amino acids around a position"""
    pos_idx = position - 1
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    window = sequence[start:end]
    return window
```

**What it does:**
- Takes a protein sequence and a position as input
- Extracts a window of amino acids centered on that position
- The window_size parameter controls how many amino acids to include on each side
- Handles edge cases (start and end of sequences)
- Returns the extracted window for feature calculation

### 7. Amino Acid Composition (AAC) Features

```python
def extract_aac(sequence):
    """
    Extract Amino Acid Composition (AAC) from a protein sequence.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
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

**What it does:**
- Calculates the frequency of each amino acid in the input sequence window
- Creates a dictionary with all 20 standard amino acids as keys
- Counts how many times each amino acid appears
- Converts raw counts to frequencies (percentage of the sequence)
- Returns a dictionary with amino acid frequencies

### 8. Dipeptide Composition (DPC) Features

```python
def extract_dpc(sequence):
    """
    Extract Dipeptide Composition (DPC) from a protein sequence.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with all possible dipeptides
    dpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dpc[aa1 + aa2] = 0
    
    # Count dipeptides
    if len(sequence) < 2:
        return dpc
    
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

**What it does:**
- Calculates the frequency of adjacent amino acid pairs in the sequence window
- Creates a dictionary with all 400 possible amino acid pairs (20×20)
- Counts each pair of consecutive amino acids (dipeptides)
- Converts counts to frequencies
- Returns a dictionary with dipeptide frequencies

### 9. Tripeptide Composition (TPC) Features

```python
def extract_tpc(sequence):
    """
    Extract Tripeptide Composition (TPC) from a protein sequence.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize dictionary with all possible tripeptides
    tpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                tpc[aa1 + aa2 + aa3] = 0
    
    # Count tripeptides
    if len(sequence) < 3:
        return tpc
    
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

**What it does:**
- Calculates the frequency of three consecutive amino acids in the sequence window
- Creates a dictionary with all 8000 possible amino acid triplets (20×20×20)
- Counts each triplet of consecutive amino acids
- Converts counts to frequencies
- Returns a dictionary with tripeptide frequencies

### 10. TPC Batch Processing

```python
def process_tpc_in_batches(df, batch_size=500, window_size=5, output_dir="tpc_batches"):
    """
    Process TPC features in batches to avoid memory errors
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate number of batches
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batch_files = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        # Get batch data
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Extract windows if not already done
        if 'Window' not in batch_df.columns:
            batch_df['Window'] = batch_df.apply(
                lambda row: extract_window(row['Sequence'], row['Position'], window_size=window_size), 
                axis=1
            )
        
        # Process TPC features for this batch
        tpc_batch = []
        for idx, row in batch_df.iterrows():
            window = row['Window']
            tpc_dict = extract_tpc(window)
            # Add identifier columns and target
            tpc_dict['Header'] = row['Header']
            tpc_dict['Position'] = row['Position']
            tpc_dict['target'] = row['target']
            tpc_batch.append(tpc_dict)
        
        # Convert to DataFrame and save this batch
        batch_output_file = os.path.join(output_dir, f"tpc_features_batch_{batch_idx+1}.csv")
        tpc_batch_df = pd.DataFrame(tpc_batch)
        tpc_batch_df.to_csv(batch_output_file, index=False)
        
        # Add file to list of batch files
        batch_files.append(batch_output_file)
        
    return batch_files
```

**What it does:**
- Processes tripeptide composition features in smaller batches to avoid memory issues
- Creates a folder to store the intermediate batch files
- For each batch of data:
  - Extracts sequence windows around each position
  - Calculates TPC features for each window
  - Saves the batch to a CSV file
- Returns a list of all batch file paths

### 11. Combining TPC Batches

```python
def combine_tpc_batches(batch_files, output_file="phosphorylation_tpc_features_window5.csv"):
    """
    Combine all TPC batch files into a single file
    """
    # Use pandas to combine batch files
    combined_df = pd.concat([pd.read_csv(file) for file in batch_files])
    
    # Save combined file
    combined_df.to_csv(output_file, index=False)
    
    return output_file
```

**What it does:**
- Takes the list of batch files from the previous function
- Reads and combines all the TPC feature batch files
- Saves the combined data to a single output file
- Returns the path to the combined file

### 12. Binary Encoding Features

```python
def binary_encode_amino_acid(aa):
    """
    Binary encode a single amino acid into a 20-dimensional vector.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Initialize vector with zeros
    encoding = [0] * 20
    
    # Set the corresponding position to 1
    if aa in amino_acids:
        idx = amino_acids.index(aa)
        encoding[idx] = 1
    
    return encoding

def extract_binary_encoding(sequence, position, window_size=10):
    """
    Extract binary encoding features for a window around the phosphorylation site.
    """
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Binary encode each amino acid in the window
    binary_features = []
    for aa in padded_window:
        binary_features.extend(binary_encode_amino_acid(aa))
    
    return binary_features
```

**What it does:**
- `binary_encode_amino_acid`: Converts a single amino acid to a one-hot encoded vector
  - Creates a vector of 20 zeros
  - Sets the position corresponding to the amino acid to 1
- `extract_binary_encoding`: Creates binary features for an entire window
  - Extracts a window around the target position
  - Pads the window if it's at the start or end of the sequence
  - Applies one-hot encoding to each amino acid in the window
  - Combines all one-hot vectors into a single feature vector
  - This creates a positional encoding where each position in the window has 20 features

### 13. Physicochemical Properties Features

```python
def load_physicochemical_properties(file_path="physiochemical_property.csv"):
    """
    Load physicochemical properties from CSV file.
    """
    prop_df = pd.read_csv(file_path)
    
    # Assuming first column is amino acid and others are properties
    properties = {}
    for _, row in prop_df.iterrows():
        aa = row.iloc[0]  # First column is amino acid
        properties[aa] = row.iloc[1:].values.tolist()
    
    return properties

def extract_physicochemical_features(sequence, position, window_size=10, properties=None):
    """
    Extract physicochemical features for a window around the phosphorylation site.
    """
    if properties is None:
        properties = load_physicochemical_properties()
    
    # Convert position to 0-based index
    pos_idx = position - 1
    
    # Define window boundaries
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    # Extract window sequence
    window = sequence[start:end]
    
    # Pad with 'X' if necessary to reach the desired window length
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    # Get properties for each amino acid in the window
    physico_features = []
    for aa in padded_window:
        if aa in properties:
            physico_features.extend(properties[aa])
        else:
            # Use zeros for unknown amino acids
            num_props = len(next(iter(properties.values())))
            physico_features.extend([0] * num_props)
    
    return physico_features
```

**What it does:**
- `load_physicochemical_properties`: Loads a CSV file with physical and chemical properties for each amino acid
  - Properties might include hydrophobicity, charge, size, etc.
  - Creates a dictionary mapping amino acids to their properties
- `extract_physicochemical_features`: Extracts these properties for a window of amino acids
  - Extracts a window around the target position
  - Pads the window if needed
  - For each amino acid in the window, adds its physicochemical properties
  - Returns a combined vector of all properties for the window

### 14. Batch Processing Functions

```python
def process_binary_encoding(df, window_size=10, batch_size=1000, output_file="phosphorylation_binary_encoding.csv"):
    """
    Process binary encoding features for all samples in batches
    """
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    all_data = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        batch_data = []
        for idx, row in batch_df.iterrows():
            binary_features = extract_binary_encoding(row['Sequence'], row['Position'], window_size)
            feature_dict = {f"BE_{i+1}": binary_features[i] for i in range(len(binary_features))}
            feature_dict['Header'] = row['Header']
            feature_dict['Position'] = row['Position']
            feature_dict['target'] = row['target']
            batch_data.append(feature_dict)
        
        all_data.extend(batch_data)
    
    be_df = pd.DataFrame(all_data)
    be_df.to_csv(output_file, index=False)
```

**What it does:**
- Similar to the TPC batch processing, but for binary encoding features
- Processes data in smaller batches to avoid memory issues
- For each sample in the dataset:
  - Extracts binary encoding features
  - Organizes features with proper naming (BE_1, BE_2, etc.)
  - Adds identifying columns (Header, Position) and target label
- Combines all batches and saves to a CSV file

### 15. Feature Merging

```python
def merge_all_features(output_file="phosphorylation_all_features.csv"):
    """
    Merge all extracted features for machine learning.
    """
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

**What it does:**
- Loads all the individual feature files (AAC, DPC, TPC, Binary Encoding, Physicochemical)
- Combines them all together by matching on Header (protein ID), Position, and target
- Creates a single dataset with all features
- Saves the combined data to a CSV file
- Returns the merged DataFrame for further use

## Summary

This code processes protein sequence data and extracts features for phosphorylation site prediction. The key steps are:

1. **Data Loading**: Loading protein sequences and known phosphorylation sites
2. **Data Preprocessing**: Cleaning data and generating balanced negative samples
3. **Feature Extraction**: Creating multiple feature types from sequence windows:
   - Amino Acid Composition (AAC) - frequency of each amino acid
   - Dipeptide Composition (DPC) - frequency of amino acid pairs
   - Tripeptide Composition (TPC) - frequency of amino acid triplets
   - Binary Encoding - one-hot encoding of each position in the window
   - Physicochemical Properties - physical and chemical attributes of amino acids
4. **Feature Merging**: Combining all features into a comprehensive dataset

The final output is a large feature matrix (62,120 samples × 8,819 features) ready for machine learning model training to predict phosphorylation sites in proteins.

#  Data Loading and Preprocessing

# Paths
base_path = '/kaggle/input/soil-classification-part-2/soil_competition-2025'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')

# Load CSV labels
train_labels = pd.read_csv(os.path.join(base_path, 'train_labels.csv'))
test_ids = pd.read_csv(os.path.join(base_path, 'test_ids.csv'))

print(f"Training data shape: {train_labels.shape}")
print(f"Test data shape: {test_ids.shape}")
print(f"Training data columns: {train_labels.columns.tolist()}")

# Display data info
print("\n Training Data Info:")
print(train_labels.head())
print(f"\nClass distribution:")
if 'soil_type' in train_labels.columns:
    print(train_labels['soil_type'].value_counts())

print("\n Test Data Info:")
print(test_ids.head())

# Train-validation split
train_df, val_df = train_test_split(train_labels, test_size=0.2, random_state=42, stratify=train_labels.get('soil_type'))

print(f"\n Data Split:")
print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")

if 'soil_type' in train_df.columns:
    print(f"\nTraining set class distribution:")
    print(train_df['soil_type'].value_counts())
    print(f"\nValidation set class distribution:")
    print(val_df['soil_type'].value_counts())

print("\n Data preprocessing completed!")
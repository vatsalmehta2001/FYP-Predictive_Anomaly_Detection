# Raw Data Management with DVC

This guide explains how to add, track, and manage raw datasets using DVC (Data Version Control) while keeping them out of Git.

## Quick Start: Adding New Data

### 1. Place Data in Correct Directory
```bash
# Download/copy your dataset to the appropriate location:
# data/raw/ukdale/     - UK household consumption data
# data/raw/lcl/        - London Smart Meters data
# data/raw/ssen/       - SSEN distribution feeder data
```

### 2. Track with DVC
```bash
# Track the entire dataset directory
dvc add data/raw/<dataset_name>

# This creates a .dvc pointer file (e.g., data/raw/ukdale.dvc)
# The actual data is ignored by Git, only the pointer is tracked
```

### 3. Commit Pointers to Git
```bash
# Stage DVC metadata (pointers and lock file)
git add data/raw/*.dvc dvc.lock .gitignore

# Commit the metadata (NOT the data itself)
git commit -m "DVC: track <dataset_name> dataset via pointer"

# Push to remote repository
git push
```

### 4. Optional: Set Up Remote Storage
```bash
# Configure cloud storage for team sharing
dvc remote add -d myremote s3://my-bucket/fyp-energy-data/
# or: dvc remote add -d myremote azure://mystorageaccount/fyp-data/

# Push data to remote storage
dvc push

# Team members can then pull data with:
# dvc pull
```

## Complete Workflow Example

```bash
# 1. Download UK-DALE dataset to local machine
wget https://ukerc.ac.uk/.../ukdale.h5 -O data/raw/ukdale/ukdale.h5

# 2. Track with DVC
dvc add data/raw/ukdale

# 3. Commit pointer to Git
git add data/raw/ukdale.dvc dvc.lock
git commit -m "DVC: track UK-DALE household consumption data"
git push

# 4. (Optional) Push data to shared storage
dvc remote add -d s3remote s3://fyp-energy-bucket/datasets/
dvc push

# 5. Team members get the data with:
git pull                    # Get updated .dvc pointers
dvc pull                    # Download actual data files
```

## Understanding Git vs DVC

| Component | Tracked By | Purpose |
|-----------|------------|---------|
| **Raw data files** | DVC | Large datasets, model artifacts |
| **`.dvc` pointer files** | Git | Small metadata files pointing to data |
| **`dvc.lock`** | Git | Dependencies and checksums |
| **Code & docs** | Git | Source code, documentation, configs |

### What Each Tool Handles

**Git handles:**
- Code, documentation, configuration
- DVC pointer files (`.dvc`)
- DVC lock file (`dvc.lock`)
- Small sample files (`data/samples/`)

**DVC handles:**
- Large raw datasets (`data/raw/`)
- Processed data artifacts (`data/processed/`, `data/derived/`)
- Model checkpoints and experiment outputs
- Data pipeline dependencies

## Team Collaboration

### For Data Contributors
```bash
# Add new dataset
dvc add data/raw/new_dataset
git add data/raw/new_dataset.dvc dvc.lock
git commit -m "DVC: add new_dataset"
git push

# Update remote storage
dvc push
```

### For Data Consumers
```bash
# Get latest pointers
git pull

# Download data files
dvc pull

# Check what data is available
dvc list . data/raw/
```

## FAQ

### Q: Why not commit data directly to Git?
**A:** Git is optimized for text files and code. Large binary datasets would:
- Make the repository massive and slow
- Bloat history with every data change
- Exceed GitHub/GitLab file size limits
- Make cloning extremely slow

### Q: How do I update a dataset?
**A:** Replace files in place, then:
```bash
dvc add data/raw/dataset_name  # Re-track (updates checksums)
git add data/raw/dataset_name.dvc dvc.lock
git commit -m "DVC: update dataset_name with new version"
dvc push  # Update remote storage
```

### Q: What if I don't have the data yet?
**A:** Use the synthetic samples in `data/samples/` for development and CI:
```bash
# Samples are small and tracked by Git
ls data/samples/
# lcl_sample.csv, ukdale_sample.csv, ssen_sample.csv
```

### Q: How do I keep CI fast without large data?
**A:** Our CI uses `data/samples/` for testing and `dvc repro` uses placeholder commands that don't require real data. When you need full datasets locally:
```bash
# Development with real data
dvc pull

# CI testing with samples (automatic)
pytest tests/  # Uses data/samples/
```

### Q: Can I use DVC without remote storage?
**A:** Yes! DVC works locally. Remote storage is optional but recommended for:
- Team collaboration
- Backup and disaster recovery
- Sharing large datasets across machines

### Q: What if `dvc add` fails?
**A:** Common issues and solutions:

```bash
# Directory is empty
echo "placeholder" > data/raw/dataset/.keep
dvc add data/raw/dataset

# Permission issues
sudo chown -R $USER:$USER data/raw/dataset
dvc add data/raw/dataset

# Git ignores .dvc files (should be fixed in this repo)
git check-ignore data/raw/dataset.dvc  # Should print nothing
```

### Q: How do I remove a dataset?
**A:**
```bash
# Remove from DVC tracking
dvc remove data/raw/dataset.dvc

# Remove from Git
git rm data/raw/dataset.dvc
git commit -m "DVC: remove dataset"

# Optionally delete local files
rm -rf data/raw/dataset/
```

## Storage Recommendations

### Local Development
- Keep 1-2 datasets locally for development
- Use `data/samples/` for quick testing
- Store DVC cache in fast storage (SSD preferred)

### Remote Storage Options
- **S3**: Best for AWS-based workflows
- **Azure Blob**: Good for Azure environments
- **Google Cloud Storage**: For GCP deployments
- **SSH/NFS**: For shared filesystem access

### Cost Optimization
- Use lifecycle policies to archive old versions
- Compress datasets before adding to DVC
- Share remote storage across team members

## Security Notes

- Never commit credentials to Git (use environment variables)
- Use IAM roles/policies for cloud storage access
- Encrypt sensitive datasets at rest
- Review data sharing agreements before uploading to cloud

For dataset-specific download instructions, see [`docs/download_links.md`](../docs/download_links.md).

#!/usr/bin/env python3

import os
import json
from pathlib import Path

def load_dataset_config(dataset_name):
    """Load dataset configuration from the main datasets.json file."""
    config_path = "datasets.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Datasets configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_datasets = json.load(f)
    
    if dataset_name not in all_datasets['datasets']:
        available = list(all_datasets['datasets'].keys())
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    return all_datasets['datasets'][dataset_name]

def list_available_datasets():
    """List all available dataset configurations."""
    config_path = "datasets.json"
    
    if not os.path.exists(config_path):
        print("No datasets.json file found.")
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_datasets = json.load(f)
    
    return list(all_datasets['datasets'].keys())

def show_dataset_info(dataset_name):
    """Show detailed information about a specific dataset."""
    try:
        config = load_dataset_config(dataset_name)
        print(f"\n=== Dataset: {config['ID']} ===")
        print(f"Title: {config['dataset_title']}")
        print(f"Authors: {config['authors']}")
        print(f"Subjects: {config['n_subjects']}")
        print(f"Modalities: {config['modalities']}")
        print(f"Data Link: {config['data_link']}")
        print(f"SNIRF Path: {config['snirf_path']}")
        print(f"File exists: {os.path.exists(config['snirf_path'])}")
        print("="*50)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")

def main():
    print("Available datasets:")
    datasets = list_available_datasets()
    
    if not datasets:
        print("No datasets found. Create datasets.json file.")
        return
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    print(f"\nTotal: {len(datasets)} datasets")
    
    # Show details for each dataset
    for dataset in datasets:
        show_dataset_info(dataset)

if __name__ == "__main__":
    main()
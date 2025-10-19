import pandas as pd
import numpy as np
import io

# --- Configuration ---
CLEANED_THERMO_FILE = 'output/cleaned_thermo_data.csv'
STRUCTURAL_FEATURES_FILE = 'output/structural_features.txt'
FINAL_ML_DATA_FILE = 'output/final_ml_dataset.csv'

# Based on your simulation details ("256 atoms")
NUM_ATOMS = 256 

def combine_and_create_features():
    """Combines thermo data with structural data to create the final ML dataset."""
    
    # 1. Load Thermodynamic Data
    try:
        df_thermo = pd.read_csv(CLEANED_THERMO_FILE)
        print(f"Loaded thermodynamic data (Shape: {df_thermo.shape})")
    except FileNotFoundError:
        print(f"ERROR: Cleaned thermo file not found at {CLEANED_THERMO_FILE}. Check your path.")
        return

    # 2. Load Structural Data (CORRECTED MAPPING based on file structure)
    
    # Define columns based EXACTLY on your file header:
    STRUCT_COLS = [
        'N_bcc',      # CommonNeighborAnalysis.counts.BCC
        'N_fcc',      # CommonNeighborAnalysis.counts.FCC
        'N_hcp',      # CommonNeighborAnalysis.counts.HCP
        'N_other',    # CommonNeighborAnalysis.counts.OTHER
        'Frame',      # OVITO Frame index
        'Timestep'    # LAMMPS Step number (0, 1000, 2000...)
    ]
    
    try:
        # Read the file content first to strip the starting '#' from the header line
        with open(STRUCTURAL_FEATURES_FILE, 'r') as f:
            content = f.read()
        
        # Replace the initial '#' with a space so pandas reads the header correctly
        content = content.replace('#', '', 1) 
        
        # Read the cleaned content into a DataFrame
        df_struct = pd.read_csv(
            io.StringIO(content),             # Read from the string content
            sep=r'\s+',                       # Robustly handle multiple spaces as separator
            header=0,                         # Header is the first line
            names=STRUCT_COLS,                # Use our explicit column names
            index_col=False
        )
        print(f"Loaded structural data (Shape: {df_struct.shape})")
    
    except FileNotFoundError:
        print(f"ERROR: Structural features file not found at {STRUCTURAL_FEATURES_FILE}. Check your path.")
        return
    except Exception as e:
        print(f"Error loading structural data: {e}. Ensure the number of columns is correct.")
        return

    # 3. Calculate the key ML Label Feature
    # The sum of all ordered phases (FCC, HCP, BCC) gives the solid atom count.
    df_struct['N_solid'] = df_struct['N_fcc'] + df_struct['N_hcp'] + df_struct['N_bcc']
    
    # The primary Y-label for the ML model:
    df_struct['Fraction_Solid'] = df_struct['N_solid'] / NUM_ATOMS

    # 4. Merge the two DataFrames on the correct key (Step/Timestep)
    df_final = pd.merge(
        df_thermo, 
        df_struct[['Timestep', 'Fraction_Solid']], 
        left_on='Step',          # Column in thermo data
        right_on='Timestep',     # Column in structural data (the LAMMPS step number)
        how='inner'              # Use 'inner' to keep only matching rows
    )

    # Drop the redundant 'Timestep' column
    df_final = df_final.drop(columns=['Timestep'])

    # 5. Save the final ML dataset
    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(FINAL_ML_DATA_FILE), exist_ok=True)
    
    df_final.to_csv(FINAL_ML_DATA_FILE, index=False)
    print(f"\nSuccessfully created FINAL ML DATASET: {FINAL_ML_DATA_FILE} (Shape: {df_final.shape})")
    print("Columns available for ML: Step, Temp, PE_per_atom, Density, Fraction_Solid, etc.")
    
    return df_final

if __name__ == '__main__':
    combine_and_create_features()
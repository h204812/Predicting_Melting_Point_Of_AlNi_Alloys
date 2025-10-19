import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
THERMO_FILE = 'output/thermo_data.dat'
OUTPUT_DIR = 'output/'

# IMPORTANT: Based on your thermo_data.dat content ("Loop time... with 256 atoms")
NUM_ATOMS = 256 
THERMO_COLUMNS = ['Step', 'Temp', 'PotEng', 'TotEng', 'Press', 'Density']


def load_and_clean_thermo_data_robust(filepath):
    """
    Loads LAMMPS thermo data by robustly finding all data blocks based on the 'Step' header.
    """
    data_blocks = []
    
    # 1. Read the entire file and find the indices of all header lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    header_indices = [i for i, line in enumerate(lines) if line.strip().startswith('Step')]
    
    if not header_indices:
        raise ValueError("Could not find any 'Step' header lines in the thermo file.")

    # 2. Extract and process each data block
    for header_index in header_indices:
        
        # Determine the start of the data (line after the header)
        data_start_index = header_index + 1
        
        # Slice the lines corresponding to the current data block
        # We stop when the line is non-numeric (e.g., comments, performance report, or end of file)
        current_data_lines = []
        for i in range(data_start_index, len(lines)):
            line = lines[i].strip()
            # Check if the line is a numerical data line
            if line and line.split()[0].isdigit():
                current_data_lines.append(line)
            else:
                break # End of current data block

        if current_data_lines:
            # Join lines back into a single string block
            data_string = '\n'.join(current_data_lines)
            
            # Read the block into a temporary DataFrame
            temp_df = pd.read_csv(
                pd.io.common.StringIO(data_string), 
                sep=r'\s+',              # Robustly handle multiple spaces as separator
                names=THERMO_COLUMNS,    # Use predefined column names
                index_col=False
            )
            data_blocks.append(temp_df)

    if not data_blocks:
        raise ValueError("Found headers but could not extract any valid numerical data blocks.")

    # 3. Concatenate all blocks and finalize cleaning
    df = pd.concat(data_blocks, ignore_index=True)
    
    # Remove duplicates (which occur at the beginning of the heating stage)
    df = df.drop_duplicates(subset=['Step'], keep='first')
    
    # Add per-atom features (PotEng is Total Potential Energy)
    df['PE_per_atom'] = df['PotEng'] / NUM_ATOMS
    df['E_per_atom'] = df['TotEng'] / NUM_ATOMS
    
    return df

def plot_thermo_data(df):
    """Plots the primary thermodynamic features for melting point detection."""
    
    # Filter for the heating phase (Step 20000 onwards) for clearer view of melting
    heating_df = df[df['Step'] >= 20000].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # --- Plot 1: Potential Energy (PE) per atom vs. Temperature ---
    axes[0].plot(heating_df['Temp'], heating_df['PE_per_atom'], label='PE/atom', color='blue', marker='o', markersize=3)
    axes[0].set_ylabel('Potential Energy per Atom (eV/atom)')
    axes[0].set_title(f'1. Potential Energy vs. Temperature (Heating Phase) - {NUM_ATOMS} Atoms')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: Density vs. Temperature ---
    axes[1].plot(heating_df['Temp'], heating_df['Density'], label='Density', color='green', marker='o', markersize=3)
    axes[1].set_ylabel('Density ($g/cm^3$)')
    axes[1].set_title('2. Density vs. Temperature (Phase Change)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Pressure vs. Temperature ---
    axes[2].plot(heating_df['Temp'], heating_df['Press'], label='Pressure', color='orange', marker='o', markersize=3)
    axes[2].set_ylabel('Pressure (Bar)')
    axes[2].set_xlabel('Temperature (K)')
    axes[2].set_title('3. Pressure vs. Temperature')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'melting_curve_analysis.png')
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Loading data from {THERMO_FILE}...")
    
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        data_df = load_and_clean_thermo_data_robust(THERMO_FILE)
        
        # Save cleaned data 
        data_df.to_csv(OUTPUT_DIR + 'cleaned_thermo_data.csv', index=False)
        print(f"Data successfully cleaned and saved to cleaned_thermo_data.csv. Shape: {data_df.shape}")
        
        plot_thermo_data(data_df)
        print(f"Plots saved to {OUTPUT_DIR}melting_curve_analysis.png")
        
        print("\n--- Step 1.1 Complete ---")
        print("NEXT STEP: Use this cleaned data for ML-assisted melting point analysis.")
        
    except FileNotFoundError:
        print(f"ERROR: File not found at {THERMO_FILE}. Check your directory structure.")
    except Exception as e:
        print(f"A critical error occurred during processing: {e}")
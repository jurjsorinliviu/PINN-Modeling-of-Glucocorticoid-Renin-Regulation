"""
Data extraction from Latia (2020) dissertation
"""

import numpy as np
import pandas as pd

def get_latia_2020_data():
    """
    Extract experimental data from Latia, L. (2020). 
    "Regulation des Renin-Gens durch Dexamethason"
    Heinrich-Heine-Universität Düsseldorf.
    
    Data sources:
    - ELISA: Table 4, Figure 6 (page 20-21)
    - Luciferase: Table 1, Figure 4 (page 16)
    
    Returns:
        dict: Contains ELISA and luciferase data
    """
    
    # ELISA data - Renin secretion at 24h
    # From Wilcoxon tests in Tables 2-4
    # Median values with IQR converted to approximate SD
    elisa_data = pd.DataFrame({
        'dex_conc_mg_dl': [0.0, 0.3, 3.0, 30.0],
        'renin_median': [28.1, 25.7, 23.8, 25.7],  # ng/ml/24h (from tables)
        'renin_q1': [26.0, 24.65, 23.15, 19.725],  # Q1 from IQR
        'renin_q3': [28.8, 27.025, 24.775, 27.725], # Q3 from IQR
        'n': [9, 9, 9, 9],
        'source': ['Table 4'] * 4
    })
    
    # Convert IQR to approximate SD (IQR ≈ 1.35*SD for normal distribution)
    elisa_data['renin_sd'] = (elisa_data['renin_q3'] - elisa_data['renin_q1']) / 1.35
    
    # Normalize to control (0 mg/dl)
    control_value = elisa_data.loc[0, 'renin_median']
    elisa_data['renin_normalized'] = elisa_data['renin_median'] / control_value
    elisa_data['renin_normalized_sd'] = elisa_data['renin_sd'] / control_value
    
    # Luciferase data - Promoter activity
    # From Table 1 (page 16)
    luciferase_data = pd.DataFrame({
        'dex_conc_mg_dl': [0.0, 0.3, 3.0, 30.0],
        'activity_mean': [0.0627470, 0.0563908, 0.0561095, 0.0497481],
        'activity_sd': [0.03110821, 0.05301851, 0.04596246, 0.05333876],
        'n': [3, 3, 3, 3],
        'source': ['Table 1'] * 4
    })
    
    # Normalize to control
    control_activity = luciferase_data.loc[0, 'activity_mean']
    luciferase_data['activity_normalized'] = luciferase_data['activity_mean'] / control_activity
    luciferase_data['activity_normalized_sd'] = luciferase_data['activity_sd'] / control_activity
    
    # Statistical significance (from Wilcoxon tests)
    significance = pd.DataFrame({
        'comparison': ['0 vs 0.3', '0 vs 3.0', '0 vs 30.0'],
        'p_value': [0.0039, 0.0039, 0.0391],
        'significant': [True, True, True],
        'test': ['Wilcoxon'] * 3,
        'source': ['Table 2-4'] * 3
    })
    
    return {
        'elisa': elisa_data,
        'luciferase': luciferase_data,
        'statistics': significance,
        'metadata': {
            'citation': 'Latia, L. (2020). Regulation des Renin-Gens durch Dexamethason. Doctoral dissertation, Heinrich-Heine-Universität Düsseldorf.',
            'cell_line': 'As4.1 (mouse juxtaglomerular-like cells)',
            'measurement_time': '24 hours',
            'replicates': 'n=9 (ELISA), n=3 (Luciferase)',
            'temperature': '37°C',
            'conditions': '95% air, 5% CO2, water-saturated'
        }
    }

def prepare_training_data(dataset='elisa', use_log_scale=False):
    """
    Prepare data for PINN training
    
    Args:
        dataset: 'elisa' or 'luciferase'
        use_log_scale: Whether to log-transform dex concentrations
        
    Returns:
        dict: Training data arrays
    """
    data = get_latia_2020_data()
    
    if dataset == 'elisa':
        df = data['elisa']
        y = df['renin_normalized'].values
        y_std = df['renin_normalized_sd'].values
    else:
        df = data['luciferase']
        y = df['activity_normalized'].values
        y_std = df['activity_normalized_sd'].values
    
    # Input features
    t = np.full_like(y, 24.0)  # All measurements at 24h
    dex = df['dex_conc_mg_dl'].values
    
    # Handle dex=0 for log scale
    if use_log_scale:
        dex = np.where(dex == 0, 0.01, dex)  # Replace 0 with small value
    
    return {
        'time': t,
        'dex_concentration': dex,
        'renin_normalized': y,
        'renin_std': y_std,
        'n_samples': len(t),
        'metadata': data['metadata']
    }

def get_citation():
    """Return formatted citation"""
    return """
    Latia, L. (2020). Regulation des Renin-Gens durch Dexamethason 
    [Doctoral dissertation, Heinrich-Heine-Universität Düsseldorf].
    """

if __name__ == "__main__":
    # Test data loading
    data = get_latia_2020_data()
    print("Data successfully loaded from Latia (2020)")
    print("\nELISA data:")
    print(data['elisa'])
    print("\nLuciferase data:")
    print(data['luciferase'])
    print("\nCitation:")
    print(get_citation())
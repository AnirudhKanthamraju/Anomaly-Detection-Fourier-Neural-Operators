
import pandas as pd

from typing import Dict 
from data_loaders import load_dataset
import numpy as np 

def segment_loaded_data(DataSet: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Uses an existing column (SYS_CTL) to trace the occupied, setback, and unoccupied states.
    Segments the dataset into multiple dataframes representing its state space representation.
    
    Args: 
        DataSet (pd.DataFrame): The loaded dataset to be segmented.
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: 
            Outer keys: unique classes in DataSet['SYS_CTL']
            Inner keys: {'external_state' ,'control_system_state', 'system_state'}
            Inner values: pd.DataFrame containing the respective columns.
    
    Raises:
        ValueError: If any defined columns are missing or if there is a column count mismatch.
    """

    # 1. Define the state space representation
    external_state_cols = [
        'Day','Month' , 'Year', 'Hour', 'Minute', 'OA_CFM', 'OA_HUMD', 'OA_TEMP'
    ]

    control_system_cols = [
        'SYS_CTL', 
        'RMCLGSPT_W', 'RMHTGSPT_W', 'VAVCFM_C_DM_W', 'VAVCFM_H_DM_W',
        'RMCLGSPT_SB', 'RMHTGSPT_SB', 'VAVCFM_C_DM_SB', 'VAVCFM_H_DM_SB',
        'RMCLGSPT_SA', 'RMHTGSPT_SA', 'VAVCFM_C_DM_SA', 'VAVCFM_H_DM_SA',
        'RMCLGSPT_E', 'RMHTGSPT_E', 'VAVCFM_C_DM_E', 'VAVCFM_H_DM_E',
        'OA_DMPR_DM', 'RA_DMPR_DM', 'EA_DMPR_DM',
        'HSA_SPSPT', 'HSA_TEMPSPT', 'CSA_SPSPT', 'CSA_TEMPSPT',
        'HWC_VLV_DM', 'CHWC_VLV_DM'
    ]

    system_state_cols = [
        'RM_TEMP_W', 'VAV_DAT_W', 'VAV_SP_C_W', 'VAV_SP_H_W', 'VAV_DMPR_C_W', 'VAV_DMPR_H_W', 'VAVCFM_C_W', 'VAVCFM_H_W', 'VAVCFM_T_W', 'VAV_EAT_C_W', 'VAV_EAT_H_W',
        'RM_TEMP_SB', 'VAV_DAT_SB', 'VAV_SP_C_SB', 'VAV_SP_H_SB', 'VAV_DMPR_C_SB', 'VAV_DMPR_H_SB', 'VAVCFM_C_SB', 'VAVCFM_H_SB', 'VAVCFM_T_SB', 'VAV_EAT_C_SB', 'VAV_EAT_H_SB',
        'RM_TEMP_SA', 'VAV_DAT_SA', 'VAV_SP_C_SA', 'VAV_SP_H_SA', 'VAV_DMPR_C_SA', 'VAV_DMPR_H_SA', 'VAVCFM_C_SA', 'VAVCFM_H_SA', 'VAVCFM_T_SA', 'VAV_EAT_C_SA', 'VAV_EAT_H_SA',
        'RM_TEMP_E', 'VAV_DAT_E', 'VAV_SP_C_E', 'VAV_SP_H_E', 'VAV_DMPR_C_E', 'VAV_DMPR_H_E', 'VAVCFM_C_E', 'VAVCFM_H_E', 'VAVCFM_T_E', 'VAV_EAT_C_E', 'VAV_EAT_H_E',
        'OA_DMPR', 'MA_TEMP', 'RA_CFM', 'RA_DMPR', 'RA_HUMD', 'RA_TEMP', 'RF_DP', 'RF_SPD', 'RF_WAT', 'EA_DMPR', 
        'HSA_SP', 'HSA_HUMD', 'HSA_CFM', 'HSA_TEMP', 'HSF_CS', 'HSF_DP', 'HSF_SPD', 'HSF_WAT', 
        'CSA_SP', 'CSA_HUMD', 'CSA_CFM', 'CSA_TEMP', 'CSF_CS', 'CSF_DP', 'CSF_SPD', 'CSF_WAT', 
        'HWC_DAT', 'HWC_EWT', 'HWC_LWT', 'HWC_MWT', 'HWC_VLV', 'HWP_GPMC', 'HWP_GPMT', 
        'CHWC_DAT', 'CHWC_EAH', 'CHWC_EWT', 'CHWC_LWT', 'CHWC_MWT', 'CHWC_VLV', 'CHWP_GPMC', 'CHWP_GPMT'
    ]

    # 2. Check for missing columns
    all_cols_state_space = set(external_state_cols + control_system_cols + system_state_cols)
    if not all_cols_state_space.issubset(DataSet.columns):
        missing_cols = all_cols_state_space - set(DataSet.columns)
        raise ValueError(f"❌ Missing columns in dataset for segmentation: {missing_cols}")
    
    # 3. Check for total column count mismatch
    expected_num_cols = len(external_state_cols) + len(control_system_cols) + len(system_state_cols)
    if DataSet.shape[1] != expected_num_cols:
        raise ValueError(f"❌ Column count mismatch: Expected {expected_num_cols} columns based on state space definition, but found {DataSet.shape[1]} columns in the dataset.")
    
    # 4. Create the nested dictionary based on SYS_CTL unique values
    segmented_data = {}
    unique_states = DataSet['SYS_CTL'].unique()

    for state in unique_states:
        # Filter the dataset by the current state
        state_df = DataSet[DataSet['SYS_CTL'] == state]
        
        # Populate the inner dictionary
        segmented_data[str(state)] = {
            'external_state': state_df[external_state_cols],
            'control_system_state': state_df[control_system_cols],
            'system_state': state_df[system_state_cols]
        }

    return segmented_data

def kernalise_segment(
    state_space_representation: dict[str, pd.DataFrame], 
    kernal_type: str, 
    num_kernels: int,
    mean_kernel_size: int,
    st_dev_kernel_size: int
) -> list[dict[str, pd.DataFrame]]:
    """
    Generates a list of kernels with sizes following a normal distribution.
    Automatically trims kernels furthest from the mean if space is insufficient.

    Args:
        state_space_representation: Dict containing 'external_state', etc.
        kernal_type: 'sliding_window', 'listed_sampling', or 'random_sampling'.
        num_kernels: Target number of kernels to generate.
        mean_kernel_size: Average number of rows per kernel.
        st_dev_kernel_size: Standard deviation for kernel size distribution.

    Returns:
        list[dict[str, pd.DataFrame]]: List of feature dictionaries for kernel instantiation.
    """
    # --- 1. Validation & Initialization ---
    valid_types = ['sliding_window', 'listed_sampling', 'random_sampling']
    if kernal_type not in valid_types:
        raise ValueError(f"❌ Invalid kernal_type. Choose from: {valid_types}")

    first_key = next(iter(state_space_representation))
    total_rows = len(state_space_representation[first_key])
    output_list = []

    # --- 2. Generate Size Distribution ---
    # Generate sizes and ensure they are at least 1 row and don't exceed the whole segment
    kernel_sizes = np.random.normal(loc=mean_kernel_size, scale=st_dev_kernel_size, size=num_kernels)
    kernel_sizes = np.clip(kernel_sizes, a_min=1, a_max=total_rows).astype(int).tolist()

    # --- 3. Extremity Trimming (for Non-Overlapping methods) ---
    if kernal_type in ['listed_sampling', 'random_sampling']:
        # If total requested rows > available rows, remove sizes furthest from mean first
        while sum(kernel_sizes) > total_rows and len(kernel_sizes) > 0:
            extremity_idx = max(range(len(kernel_sizes)), key=lambda i: abs(kernel_sizes[i] - mean_kernel_size))
            kernel_sizes.pop(extremity_idx)
            
    # --- 4. Extraction Logic ---
    if kernal_type == 'sliding_window':
        # Overlapping: Random start points for all generated sizes
        for k_size in kernel_sizes:
            start_idx = np.random.randint(0, total_rows - k_size + 1)
            output_list.append({
                state: df.iloc[start_idx : start_idx + k_size].copy() 
                for state, df in state_space_representation.items()
            })

    elif kernal_type == 'listed_sampling':
        # Sequential non-overlapping
        curr = 0
        for k_size in kernel_sizes:
            output_list.append({
                state: df.iloc[curr : curr + k_size].copy() 
                for state, df in state_space_representation.items()
            })
            curr += k_size

    elif kernal_type == 'random_sampling':
        # Shuffled non-overlapping
        indices = np.random.permutation(total_rows)
        curr = 0
        for k_size in kernel_sizes:
            batch = indices[curr : curr + k_size]
            output_list.append({
                state: df.iloc[batch].copy() 
                for state, df in state_space_representation.items()
            })
            curr += k_size

    print(f"✅ Created {len(output_list)} kernels ('{kernal_type}'). Requested: {num_kernels}")
    return output_list

def apply_kernalisation(
    state_space_representation: dict[str, pd.DataFrame], 
    kernal_type: str, 
    num_kernels: int,
    mean_kernel_size: int,
    st_dev_kernel_size: int
) -> list[dict[str, pd.DataFrame]]:
    """
    Applies kernelization logic utilizing variable kernel sizes and extremity trimming.
    """
    first_key = next(iter(state_space_representation))
    total_rows = len(state_space_representation[first_key])
    output_list = []

    # 1. Generate the ideal distribution of kernel sizes
    kernel_sizes = np.random.normal(loc=mean_kernel_size, scale=st_dev_kernel_size, size=num_kernels)
    # Ensure minimum size of 1, and no single kernel exceeds total available rows
    kernel_sizes = np.clip(kernel_sizes, a_min=1, a_max=total_rows).astype(int).tolist()

    # 2. Filter logic for non-overlapping types (Listed and Random)
    if kernal_type in ['listed_sampling', 'random_sampling']:
        # If the sum of sizes exceeds our rows, we must trim extreme values to protect the mean
        while sum(kernel_sizes) > total_rows and len(kernel_sizes) > 0:
            # Find the index of the kernel size furthest from the mean (the extremity)
            extremity_idx = max(range(len(kernel_sizes)), key=lambda i: abs(kernel_sizes[i] - mean_kernel_size))
            # Remove it to make space
            kernel_sizes.pop(extremity_idx)

    # 3. Kernel Extraction
    if kernal_type == 'sliding_window':
        # Sliding window allows overlap, so we pick random start points for all valid sizes
        for k_size in kernel_sizes:
            start_idx = np.random.randint(0, total_rows - k_size + 1)
            snapshot = {
                state: df.iloc[start_idx : start_idx + k_size].copy() 
                for state, df in state_space_representation.items()
            }
            output_list.append(snapshot)

    elif kernal_type == 'listed_sampling':
        # Non-overlapping sequential
        current_idx = 0
        for k_size in kernel_sizes:
            snapshot = {
                state: df.iloc[current_idx : current_idx + k_size].copy() 
                for state, df in state_space_representation.items()
            }
            output_list.append(snapshot)
            current_idx += k_size

    elif kernal_type == 'random_sampling':
        # Non-overlapping shuffled
        indices = np.random.permutation(total_rows)
        current_idx = 0
        for k_size in kernel_sizes:
            batch_indices = indices[current_idx : current_idx + k_size]
            snapshot = {
                state: df.iloc[batch_indices].copy() 
                for state, df in state_space_representation.items()
            }
            output_list.append(snapshot)
            current_idx += k_size

    target_met = len(output_list) == num_kernels
    print(f"✅ Generated {len(output_list)} kernels using '{kernal_type}'. (Target of {num_kernels} met: {target_met})")
    
    return output_list


#### Tests ##### 


if __name__ == "__main__":
    # Test block to verify loader functionality
    try:
        """# 1. Test batch loading
        data = load_anomaly_datasets()
        print("\nSummary of loaded datasets:")
        for name, df in data.items():
            print(f" - {name}: {len(df)} rows, {len(df.columns)} columns")"""
        
        # 2. Test single file loading
        test_file = "DualDuct_FaultFree.csv"
        test_dataset = load_dataset(test_file)
        print(f"\n✅ Successfully loaded '{test_file}': {len(test_dataset)} rows)")
        state_space_rep = segment_loaded_data(test_dataset)
        print("\n✅ Successfully segmented dataset into state space representation.")
        # 3. Test kernalisation
        kernalised_data = kernalise_segment(state_space_representation=state_space_rep, 
                                            kernal_type='listed_sampling', 
                                            num_kernels=10,
                                            mean_kernel_size=50,
                                            st_dev_kernel_size=10)

    except Exception as error:
        print(f"Critical Failure: {error}")
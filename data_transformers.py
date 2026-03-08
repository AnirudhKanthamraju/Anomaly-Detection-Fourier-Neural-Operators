import os
import pandas as pd
import pickle
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def segment_loaded_data ( DataSet : pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Segments the loaded dataset from cache into multiple dataframes representing its state space represenation.
    This function uses a predefined segmentation logic , specific to data set used. 
    This function segments the dataset using a columnwise operation. 

    Args: 
        DataSet (pd.DataFrame): The loaded dataset to be segmented.
    Returns:
        dict[str, pd.DataFrame]: A dictionary containing the segmented dataframes. 
        Keys : {'external_state' ,'control_system_state', 'system_state'}.
        Values: pd.DataFrame containing the respective colums from loaded dataset.
    
    Checks:
    1. Checks if all the defined columns can be found in the loaded dataset.
    2. Checks if the number of columns in the input dataset matches the expected number of
       columns based on the defined state space representation.
    
    Raises:
        ValueError: If any of the defined columns are missing from the dataset or if there is a column count mismatch.
        ValueError: If there is a mismatch between the expected number of columns based on the state space definition 
        and the actual number of columns in the dataset.  
    """
    # --- CONSTANT WARNING ---
    warnings.warn(
        "\n🚧 INCOMPLETE COLUMN LISTS: The 'segment_loaded_data' function is using a partial "
        "list of columns. Please update external_state_cols, control_system_cols, and "
        "system_state_cols before final deployment. and remove this warning. 🚧\n", 
        UserWarning, 
        stacklevel=2
    )

    # 1. Define the state space representaion of the dataset by defining the following list of columns.
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
        # West Zone Feedback
        'RM_TEMP_W', 'VAV_DAT_W', 'VAV_SP_C_W', 'VAV_SP_H_W', 'VAV_DMPR_C_W', 'VAV_DMPR_H_W', 'VAVCFM_C_W', 'VAVCFM_H_W', 'VAVCFM_T_W', 'VAV_EAT_C_W', 'VAV_EAT_H_W',
        # South B Zone Feedback
        'RM_TEMP_SB', 'VAV_DAT_SB', 'VAV_SP_C_SB', 'VAV_SP_H_SB', 'VAV_DMPR_C_SB', 'VAV_DMPR_H_SB', 'VAVCFM_C_SB', 'VAVCFM_H_SB', 'VAVCFM_T_SB', 'VAV_EAT_C_SB', 'VAV_EAT_H_SB',
        # South A Zone Feedback
        'RM_TEMP_SA', 'VAV_DAT_SA', 'VAV_SP_C_SA', 'VAV_SP_H_SA', 'VAV_DMPR_C_SA', 'VAV_DMPR_H_SA', 'VAVCFM_C_SA', 'VAVCFM_H_SA', 'VAVCFM_T_SA', 'VAV_EAT_C_SA', 'VAV_EAT_H_SA',
        # East Zone Feedback
        'RM_TEMP_E', 'VAV_DAT_E', 'VAV_SP_C_E', 'VAV_SP_H_E', 'VAV_DMPR_C_E', 'VAV_DMPR_H_E', 'VAVCFM_C_E', 'VAVCFM_H_E', 'VAVCFM_T_E', 'VAV_EAT_C_E', 'VAV_EAT_H_E',
        # Central AHU & General Feedback
        'OA_DMPR', 'MA_TEMP', 'RA_CFM', 'RA_DMPR', 'RA_HUMD', 'RA_TEMP', 'RF_DP', 'RF_SPD', 'RF_WAT', 'EA_DMPR', 
        'HSA_SP', 'HSA_HUMD', 'HSA_CFM', 'HSA_TEMP', 'HSF_CS', 'HSF_DP', 'HSF_SPD', 'HSF_WAT', 
        'CSA_SP', 'CSA_HUMD', 'CSA_CFM', 'CSA_TEMP', 'CSF_CS', 'CSF_DP', 'CSF_SPD', 'CSF_WAT', 
        # Heating & Cooling Coils/Pumps Feedback
        'HWC_DAT', 'HWC_EWT', 'HWC_LWT', 'HWC_MWT', 'HWC_VLV', 'HWP_GPMC', 'HWP_GPMT', 
        'CHWC_DAT', 'CHWC_EAH', 'CHWC_EWT', 'CHWC_LWT', 'CHWC_MWT', 'CHWC_VLV', 'CHWP_GPMC', 'CHWP_GPMT'
    ]

    # 2. Check if all the defined columns can be found in the loaded dataset.
    all_cols_state_space = set(external_state_cols + control_system_cols + system_state_cols)
    if not all_cols_state_space.issubset(DataSet.columns):
        missing_cols = all_cols_state_space - set(DataSet.columns)
        raise ValueError(f"❌ Missing columns in dataset for segmentation: {missing_cols}")
    
    # 3. Check if the number of columns in the input dataset matches the expected number of columns based on the defined state space representation.
    expected_num_cols = len(external_state_cols) + len(control_system_cols) + len(system_state_cols)
    if DataSet.shape[1] != expected_num_cols:
        raise ValueError(f"❌ Column count mismatch: Expected {expected_num_cols} columns based on state space definition, but found {DataSet.shape[1]} columns in the dataset.")
    
    
    external_state_df = DataSet[external_state_cols]
    control_system_df = DataSet[control_system_cols]
    system_state_df = DataSet[system_state_cols]

    return {
        'external_state': external_state_df,
        'control_system_state': control_system_df,
        'system_state': system_state_df
    }


def kernalise_dataset (Dataset : pd.DataFrame, kernal_type : str, kernal_size : int) -> list[pd.DataFrame]:
    """
    Generates a list of dataframes of the defined kernal size from the input dataset based on the defined kernal type.

    Args:
        Dataset (pd.DataFrame): The input dataset to be kernalised.
        kernal_type (str): The type of kernalisation to be applied. Options: 'sliding_window', 'random_sampling', 'stratified_sampling'.
        kernal_size (int): The size of each kernal (number of rows in each output dataframe).
        name_dataset (str): The name of the dataset being loaded to kernalise (used for caching purposes). 
    Returns:
        list[pd.DataFrame]: A list of dataframes, each representing a kernalised segment of the original dataset.
    Checks:
        1. Checks if Dataset being loaded is from the pre generated cache or if it is being loaded for the first time. 
    Raises:
        ValueError: If an invalid kernal type is provided or if the kernal size is not a positive integer.
        ValueError: If kernal size is integrally divisible by kernal_size.
        ValueError: If the loaded dataset has been cached 

    """
    if kernal_type == 'sliding_window':
        # Implement sliding window kernalisation
        pass
    elif kernal_type == 'random_sampling':
        # Implement random sampling kernalisation
        pass
    elif kernal_type == 'stratified_sampling':
        # Implement stratified sampling kernalisation
        pass
    else:
        raise ValueError(f"❌ Invalid kernal type: '{kernal_type}'. Valid options are 'sliding_window', 'random_sampling', 'stratified_sampling'.")
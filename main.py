
from create_dataset import kernel
from dataset_models import hvac_dataset
 
from data_transformers import kernalise_segment




if __name__ == "__main__":
    fault_free_dataset = hvac_dataset('DualDuct_FaultFree.csv')
    #fault_free_kernals = fault_free_dataset.kernalise_this_dataset(k_type='listed_sampling', k_size=10, stratify=True, balance=False)
    

    ### Select a list of faulty datasets 
    faulty_datasets = [
        'DualDuct_CoolSeqUnstable.csv', 
        'DualDuct_SensorBias_HSP_-4inwg.csv', 
        'DualDuct_DMPRStruck_Hot.csv'
    ]
    fault_free_dataset = hvac_dataset('DualDuct_FaultFree.csv')
    normal_segments = fault_free_dataset.segments

    all_extracted_kernels = []

    for sys_op_mode, state_space_dict in normal_segments.items(): 
        print(f"\n--- Processing Mode: {sys_op_mode} ---")
        
        # We request 5000 kernels. If the segment is too small (e.g. Setback), 
        # the function will return the max possible without overlapping.
        kernel_snapshots = kernalise_segment(
            state_space_representation=state_space_dict,
            kernal_type='sliding_window',
            num_kernels=10,
            mean_kernel_size=5, 
            st_dev_kernel_size=2
        )
        
        # Instantiate your kernel objects
        for features in kernel_snapshots:
            new_kernel = kernel(
                features=features, 
                anomoly=0, 
                source=f"FaultFree_Mode_{sys_op_mode}"
            )
            all_extracted_kernels.append(new_kernel)

    print(f"\n🚀 Total kernels extracted: {len(all_extracted_kernels)}")


    print (all_extracted_kernels[0].features.keys())
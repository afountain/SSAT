# parameter_exploration.py
import torch
from train_module_ssat import preprocess, set_random_seed, load_and_preprocess_data, get_dataloaders, create_model, train_model
from fttmaelossv7 import CombinedLoss
import torch.optim as optim
import itertools
import pandas as pd
import os

def main():
    # random seed
    set_random_seed(randnum=42)

    # version
    version = "ssatv12.1_Asia"
    region = version.split("_")[1]

    # load & preprocess

    print(f'Loading \033[1;33m {region} \033[0m data....')
    train_path, valid_path = preprocess(region)
    # Only for test!!!!!!!! Should change back later   

    yaml_path = 'modis_bin_SFD.yaml'  #1109 Aft. added
    target_col = 'MODIS'
    batch_size = 512 #doesn't matter

    # receive category_maps and categories_sizes
    train_set, val_set, categorical_cols, numerical_cols, category_maps, categories_sizes, lat_idx, lon_idx, date_diff_idx = load_and_preprocess_data(
    train_path, valid_path, yaml_path, target_col)
    num_continuous = len(numerical_cols)
    # get categories
    categories = list(categories_sizes.values())

    print(f"Categories: {categories}")
    print(f"Number of continuous features: {num_continuous}")

    print(train_set)
    
    print()
    # get data
    train_loader, val_loader = get_dataloaders(
        train_set, val_set,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_col=target_col,
        category_maps=category_maps,
        batch_size=batch_size
    )

    # extract categories
    categories = list(categories_sizes.values())

    print(f"Categories: {categories}")
    print(f"Number of continuous features: {num_continuous}")

    # superparas
    print(f'Loading \033[1;32m parameters \033[0m data....')

    # 1117 retrain 30epoch/exp
    param_grid = {
        'attn_dropout': [0.09, 0.11],
        'ff_dropout': [ 0.23],
        'num_pos_freqs': [45],
        'depth': [6],
        'heads': [16],
        'dim': [64],
        'lr': [4e-4],
        'alpha': [0.5],
        'beta': [0.5],
        'weight_decay': [4e-4]

    }
    # combination
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    num_epochs = 150
        

    # save result
    results_df = pd.DataFrame(columns=[
        'ExperimentID', 'Epoch', 'Train Loss', 'Train MSE', 
        'Valid MSE', 'Best Val MSE','Best Epoch', 'Params'
    ])


    experiment_file = f'experim_{version}.csv'
    if os.path.exists(experiment_file):
        results_df = pd.read_csv(experiment_file)
    print(f'\033[1;31m Training  \033[0m ....')

    for idx, params in enumerate(param_combinations):
        print(f"\nRunning experiment {idx+1}/{len(param_combinations)} with parameters:")
        print(params)


        set_random_seed(randnum=42 + idx)


        model = create_model(categories_sizes, num_continuous, params, lat_idx, lon_idx)

        criterion = CombinedLoss(alpha=params['alpha'], beta=params['beta'])

        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])


        save_path = f'best_model_exp_{idx+1}.pth'
        
        # get epoch id with best evaluation mse

        experiment_logs = train_model(model, train_loader, val_loader, 
                criterion, optimizer, num_epochs, 
                device, save_path, 
                version, lat_idx, lon_idx, date_diff_idx, 
                )


        # save experiemtn data
        for log in experiment_logs:
            log['ExperimentID'] = idx + 1
            log['Params'] = params
            results_df = pd.concat([results_df, pd.DataFrame([log])], ignore_index=True)

        results_df.to_csv(experiment_file, index=False)
        print(f"Experiment {idx+1} completed and logged. The best val MSE is: {log['Best Val MSE']} at Epoch: {log['Best Epoch']}")

if __name__ == "__main__":
    main()

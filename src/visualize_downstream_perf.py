import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def parse_log(file_path):
    train_data = []
    dev_data = []
    test_data = []
    best_dev_steps = []

    with open(file_path, 'r') as f:
        for line in f:
            train_match = re.search(r'train at step (\d+):\s*([-\d.]+)', line)
            if train_match:
                train_data.append((int(train_match.group(1)), float(train_match.group(2))))
            
            dev_match = re.search(r'dev at step (\d+):\s*([-\d.]+)', line)
            if dev_match:
                dev_data.append((int(dev_match.group(1)), float(dev_match.group(2))))

            test_match = re.search(r'test at step (\d+):\s*([-\d.]+)', line)
            if test_match:
                test_data.append((int(test_match.group(1)), float(test_match.group(2))))
                
            best_match = re.search(r'New best on dev at step (\d+):', line)
            if best_match:
                best_dev_steps.append(int(best_match.group(1)))
                
    return {
        'train': pd.DataFrame(train_data, columns=['step', 'value']).drop_duplicates('step').sort_values('step'),
        'dev': pd.DataFrame(dev_data, columns=['step', 'value']).drop_duplicates('step').sort_values('step'),
        'test': pd.DataFrame(test_data, columns=['step', 'value']).drop_duplicates('step').sort_values('step'),
        'best_dev_steps': best_dev_steps
    }

def main():
    base_dir = Path(__file__).parent.resolve()
    finetune_dir = base_dir / 'finetune'
    output_dir = base_dir / 'metrics' / 'downstream_perf'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup plotting style
    sns.set_theme(style="whitegrid")
    
    # 1. Gather all logs
    log_files = glob.glob(str(finetune_dir / '*' / 'exp' / '**' / 'log.log'), recursive=True)
    
    tasks_data = {}
    summary_records = []
    
    for log_path in log_files:
        path_obj = Path(log_path)
        # e.g. finetune/voxceleb1/exp/sid_ssamba_baseline_1e-4/log.log
        # Relative to finetune: task/exp/.../log.log
        rel_path = path_obj.relative_to(finetune_dir)
        parts = rel_path.parts
        
        task_name = parts[0]
        # Use the first directory after 'exp' as the main experiment name
        # to avoid long suffixes like 'unfreeze_cross-valid-on-fold1'
        exp_name_parts = parts[2:-1]
        exp_name = exp_name_parts[0] if exp_name_parts else "default"
        
        # Strip common task-specific prefixes to shorten legend
        exp_name = re.sub(r'^(emotion|sid|sc|sc_v1|sc_v2|esc50|audioset|urban8k|voxceleb1|iemocap)_', '', exp_name)
        
        parsed = parse_log(log_path)
        
        if task_name not in tasks_data:
            tasks_data[task_name] = {}
        tasks_data[task_name][exp_name] = parsed
        
        # Calculate summary metrics
        best_dev_step = parsed['best_dev_steps'][-1] if parsed['best_dev_steps'] else None
        
        best_dev_val = None
        test_val_at_best_dev = None
        max_test_val = None
        
        if not parsed['dev'].empty and best_dev_step:
            dev_row = parsed['dev'][parsed['dev']['step'] == best_dev_step]
            if not dev_row.empty:
                best_dev_val = dev_row.iloc[0]['value']
                
        if not parsed['test'].empty:
            max_test_val = parsed['test']['value'].max()
            if best_dev_step:
                test_row = parsed['test'][parsed['test']['step'] == best_dev_step]
                if not test_row.empty:
                    test_val_at_best_dev = test_row.iloc[0]['value']

        summary_records.append({
            'Task': task_name,
            'Experiment': exp_name,
            'Best Dev Step': best_dev_step,
            'Best Dev Perf': best_dev_val,
            'Test Perf @ Best Dev': test_val_at_best_dev,
            'Max Test Perf': max_test_val
        })

    # Save summary table
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(output_dir / 'summary.csv', index=False)
        print(f"Saved summary to {output_dir / 'summary.csv'}")

    # 2. Generate plots per task
    for task_name, experiments in tasks_data.items():
        # Plot Test Performance over steps
        plt.figure(figsize=(10, 6))
        for exp_name, data in experiments.items():
            test_df = data['test'].copy()
            if not test_df.empty:
                # Use a centered rolling window to smooth without phase shift (lag) distortion
                test_df['smoothed'] = test_df['value'].rolling(window=5, min_periods=1, center=True).mean()
                p = sns.lineplot(data=test_df, x='step', y='smoothed', label=exp_name, linewidth=2)
                # raw data lightly in background
                sns.lineplot(data=test_df, x='step', y='value', color=p.lines[-1].get_color(), alpha=0.2, legend=False)
                
        plt.title(f'Test Performance Over Steps - {task_name.capitalize()}')
        plt.xlabel('Step')
        plt.ylabel('Test Metric')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_dir / f'{task_name}_test_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Dev Performance over steps
        plt.figure(figsize=(10, 6))
        for exp_name, data in experiments.items():
            dev_df = data['dev'].copy()
            if not dev_df.empty:
                # Use a centered rolling window to smooth without phase shift (lag) distortion
                dev_df['smoothed'] = dev_df['value'].rolling(window=5, min_periods=1, center=True).mean()
                p = sns.lineplot(data=dev_df, x='step', y='smoothed', label=exp_name, linewidth=2)
                sns.lineplot(data=dev_df, x='step', y='value', color=p.lines[-1].get_color(), alpha=0.2, legend=False)
                
        plt.title(f'Dev Performance Over Steps - {task_name.capitalize()}')
        plt.xlabel('Step')
        plt.ylabel('Dev Metric')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_dir / f'{task_name}_dev_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Train Performance over steps
        plt.figure(figsize=(10, 6))
        for exp_name, data in experiments.items():
            train_df = data['train'].copy()
            if not train_df.empty:
                # Train typically has more points, so we can use a slightly larger window
                train_df['smoothed'] = train_df['value'].rolling(window=11, min_periods=1, center=True).mean()
                p = sns.lineplot(data=train_df, x='step', y='smoothed', label=exp_name, linewidth=2)
                sns.lineplot(data=train_df, x='step', y='value', color=p.lines[-1].get_color(), alpha=0.2, legend=False)
                
        plt.title(f'Train Performance Over Steps - {task_name.capitalize()}')
        plt.xlabel('Step')
        plt.ylabel('Train Metric')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_dir / f'{task_name}_train_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Bar chart comparing Best Test Performance across tasks
    if summary_records:
        df_summary_clean = df_summary.dropna(subset=['Test Perf @ Best Dev'])
        if not df_summary_clean.empty:
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(
                data=df_summary_clean,
                x='Task',
                y='Test Perf @ Best Dev',
                hue='Experiment'
            )
            plt.title('Best Test Performance (at highest Dev step) by Task')
            plt.ylabel('Test Metric')
            plt.xlabel('Downstream Task')
            plt.xticks(rotation=45)
            plt.legend(title='Experiment')
            plt.tight_layout()
            plt.savefig(output_dir / 'all_tasks_best_test_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    print(f"All plots have been saved to {output_dir}")

if __name__ == '__main__':
    main()

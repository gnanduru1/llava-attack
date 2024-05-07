import pandas as pd
import matplotlib.pyplot as plt

# results_dir = 'results'
results_dir = 'oldresults'
print("Attack Comparison Results")
successes = [0]*4
dists = [0]*4
iters = [0]*4
for id in range(4):
    results = pd.read_csv(f'{results_dir}/compare_attacks-{id}.csv')
    print(f"Attack {id+1}")
    print("Success rate:", len(results.dropna())/len(results))
    print("Average distance:", results['distance'].mean())
    print("Average iterations:", results['iters'].mean())
    successes[id] = len(results.dropna())/len(results)
    dists[id] = str(results['distance'].mean())
    iters[id] = str(results['iters'].mean())
    # print(results.describe())
    print("\n")

summary = pd.DataFrame({'Attack \#': list(range(1,5)), 'Average Distance': dists, 'Average Iterations': iters})
latex_table = summary.to_latex(index=False, caption='Attack Comparison (alpha=100)', label='tab:attack-comparison')


results_dir = 'results'
print("Fine tuning results")
for id in range(2):
    print(f"Attack {id+3}")
    results = pd.read_csv(f'{results_dir}/tune_alpha-{id}.csv')
    for alpha, df in results.groupby('alpha'):
        print("Alpha:", alpha)
        print("Success rate:", len(df.dropna())/len(df))
        print("Average distance:", results['distance'].mean())
        print("Average iterations:", results['iters'].mean())
        # print(df.describe())
        print("\n")
    # grouped = results.groupby('alpha')[['iters','distance']].mean().reset_index()
    # plt.figure(figsize=(10,10))
    # plt.title(f'Attack {id} distance')
    # plt.plot(grouped['alpha'], grouped['distance'])

    grouped = results.groupby('alpha')[['distance', 'iters']].agg(['mean', 'std', 'count', 'size']).reset_index()
    plt.figure(figsize=(10,8))
    # plt.errorbar(grouped['alpha'], grouped['distance']['mean'], yerr=grouped['distance']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, marker='s', markersize=5, linestyle='--')#, label='\mu \pm \sigma')
    plt.errorbar(grouped['alpha'], grouped['distance']['mean'], yerr=grouped['distance']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, markersize=5, linestyle='--')#, label='\mu \pm \sigma')
    plt.title(f'Attack {id+3} distance vs. alpha')
    plt.savefig(f'{results_dir}/distance-vs-alpha-{id}')

    plt.figure(figsize=(10,8))
    # plt.errorbar(grouped['alpha'], grouped['iters']['mean'], yerr=grouped['iters']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, marker='s', markersize=5, linestyle='--')
    plt.errorbar(grouped['alpha'], grouped['iters']['mean'], yerr=grouped['iters']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, markersize=5, linestyle='--')
    plt.title(f'Attack {id+3} iterations vs. alpha')
    plt.savefig(f'{results_dir}/iters-vs-alpha-{id}')

    plt.figure(figsize=(10,8))
    plt.plot(grouped['alpha'], grouped['iters']['count']/grouped['iters']['size'])
    plt.title(f'Attack {id+3} success rate vs. alpha')
    plt.savefig(f'{results_dir}/success-vs-alpha-{id}')

print(latex_table)
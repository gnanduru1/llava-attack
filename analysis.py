import pandas as pd
import matplotlib.pyplot as plt

results_dir = 'results'
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
    dists[id] = results['distance'].mean()
    iters[id] = results['iters'].mean()
    # print(results.describe())
    print("\n")

summary = pd.DataFrame({'Attack \#': list(range(1,5)), 'Average Distance': dists, 'Average Iterations': iters, 'Success Rate': successes})
latex_table = summary.to_latex(index=False, caption='Attack Comparison ($\lambda$=100)', label='tab:attack-comparison', float_format=lambda x: str(x))


print("Fine tuning results")
for id in range(2):
    print(f"Attack {id+3}")
    results = pd.read_csv(f'{results_dir}/tune_alpha-{id}.csv')
    # for alpha, df in results.groupby('alpha'):
    #     print("Alpha:", alpha)
    #     print("Success rate:", len(df.dropna())/len(df))
    #     print("Average distance:", results['distance'].mean())
    #     print("Average iterations:", results['iters'].mean())
    #     print(df.describe())
    #     print("\n")

    grouped = results.groupby('alpha')['distance'].agg(['count','size']).reset_index()
    success_alpha = grouped['alpha']
    success_rates = grouped['count']/grouped['size']

    df_clean = results.dropna()
    grouped = df_clean.groupby('alpha')[['distance', 'iters']].agg(['mean', 'std', 'count', 'size']).reset_index()

    print("Success rate vs. alpha")
    print(success_rates)
    print("Iterations vs. alpha")
    print(grouped['iters']['mean'])
    print("Distance vs. alpha")
    print(grouped['distance']['mean'])

    plt.figure(figsize=(16,8))
    plt.plot(success_alpha, success_rates)
    plt.title(f'Attack {id+3} Success Rate vs. $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('Success Rate')
    plt.savefig(f'{results_dir}/success-vs-lambda-{id}')

    plt.figure(figsize=(16,8))
    plt.errorbar(grouped['alpha'], grouped['distance']['mean'], yerr=grouped['distance']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, markersize=5, linestyle='--')#, label='\mu \pm \sigma')
    plt.title(f'Attack {id+3} Distance vs. $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('Average Distance')
    plt.savefig(f'{results_dir}/distance-vs-lambda-{id}')

    plt.figure(figsize=(16,8))
    plt.errorbar(grouped['alpha'], grouped['iters']['mean'], yerr=grouped['iters']['std'], fmt='o', ecolor='red', capsize=5, elinewidth=2, markersize=5, linestyle='--')
    plt.title(f'Attack {id+3} iterations vs. $\lambda$')
    plt.xlabel('$\lambda$')
    plt.ylabel('Average Iterations')
    plt.savefig(f'{results_dir}/iters-vs-lambda-{id}')

print(latex_table)
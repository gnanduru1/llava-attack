import pandas as pd

print("Attack Comparison Results")
for id in range(4):
    results = pd.read_csv(f'results/compare_attacks-{id}.csv')
    print(f"Attack {id+1}")
    print("Success rate:", len(results.dropna())/len(results))
    print("Average distance:", results['distance'].mean())
    print("Average iterations:", results['iters'].mean())
    # print(results.describe())
    print("\n")

print("Fine tuning results")
for id in range(2):
    print(f"Attack {id+3}")
    results = pd.read_csv(f'results/tune_alpha-{id}.csv')
    for alpha, df in results.groupby('alpha'):
        print("Alpha:", alpha)
        print("Success rate:", len(df.dropna())/len(df))
        print("Average distance:", results['distance'].mean())
        print("Average iterations:", results['iters'].mean())
        # print(df.describe())
        print("\n")
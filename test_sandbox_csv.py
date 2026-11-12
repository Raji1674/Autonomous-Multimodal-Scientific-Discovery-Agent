from src.experiments.sandbox_csv_runner import run_sandbox_csv_experiment

result = run_sandbox_csv_experiment(
    file_path="data/downloaded/sample_experiment.csv",
    target_column="target",
    baseline_params={"n_estimators": 10},
    treatment_params={"n_estimators": 20},
    n_seeds=10,
    test_size=0.2,
)

print("\n===== SANDBOX CSV EXPERIMENT RESULT =====")
print(result)
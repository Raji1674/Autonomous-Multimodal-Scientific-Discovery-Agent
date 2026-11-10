from unittest.mock import Mock, patch

from src.agents.csv_experiment_agent import CSVExperimentAgent


def test_classification_uses_sandbox_runner(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    state = {
        "dataset_path": str(dataset),
        "target_column": "target",
        "task_override": "classification",
        "selected_hypothesis": {},
    }

    expected = {
        "success": True,
        "dataset": "data.csv",
        "target_column": "target",
        "experiment": {"result_label": "executed"},
        "critic": {
            "analysis": {"mean_difference": 0.1},
            "issues": [],
            "reasons": [],
            "statistical_quality": "acceptable",
            "verdict": "supported",
        },
    }

    with patch(
        "src.agents.csv_experiment_agent.run_sandbox_csv_experiment",
        return_value=expected,
    ) as runner:
        result = CSVExperimentAgent().run(state)

    runner.assert_called_once()
    assert result["detected_task"] == "classification"
    assert result["experiment_results"] == expected


def test_regression_uses_regression_trainer(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,10\n2,20\n")

    expected = {
        "success": True,
        "task": "regression",
        "models": [{"model": "LinearRegression", "rmse": 1.0}],
        "best_model": "LinearRegression",
    }

    with patch(
        "src.agents.csv_experiment_agent.train_regression_models",
        return_value=expected,
    ) as trainer:
        result = CSVExperimentAgent().run({
            "dataset_path": str(dataset),
            "target_column": "target",
            "task_override": "regression",
            "selected_hypothesis": {},
        })

    trainer.assert_called_once()
    assert result["detected_task"] == "regression"
    assert result["experiment_results"]["best_model"] == "LinearRegression"


def test_clustering_uses_clustering_trainer_without_target(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature_a,feature_b\n1,2\n2,3\n3,4\n")

    expected = {
        "success": True,
        "task": "clustering",
        "models": [{"model": "KMeans", "silhouette_score": 0.5}],
        "best_model": "KMeans",
    }

    with patch(
        "src.agents.csv_experiment_agent.train_clustering_models",
        return_value=expected,
    ) as trainer:
        result = CSVExperimentAgent().run({
            "dataset_path": str(dataset),
            "task_override": "clustering",
            "selected_hypothesis": {},
        })

    trainer.assert_called_once()
    assert result["detected_task"] == "clustering"
    assert result["experiment_results"]["best_model"] == "KMeans"


def test_missing_dataset_returns_failure():
    result = CSVExperimentAgent().run({
        "dataset_path": "missing.csv",
        "target_column": "target",
        "selected_hypothesis": {},
    })

    assert result["failure_state"] == "EXECUTION_FAILED"


def test_invalid_task_returns_failure(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    result = CSVExperimentAgent().run({
        "dataset_path": str(dataset),
        "target_column": "target",
        "task_override": "unknown",
        "selected_hypothesis": {},
    })

    assert result["failure_state"] == "EXECUTION_FAILED"


def test_supervised_task_requires_target(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    result = CSVExperimentAgent().run({
        "dataset_path": str(dataset),
        "task_override": "regression",
        "selected_hypothesis": {},
    })

    assert result["failure_state"] == "EXECUTION_FAILED"


def test_failed_regression_returns_failure_state(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,10\n2,20\n")

    with patch(
        "src.agents.csv_experiment_agent.train_regression_models",
        return_value={
            "success": False,
            "issues": ["Invalid target"],
        },
    ):
        result = CSVExperimentAgent().run({
            "dataset_path": str(dataset),
            "target_column": "target",
            "task_override": "regression",
            "selected_hypothesis": {},
        })

    assert result["failure_state"] == "EXECUTION_FAILED"


def test_csv_debugger_repairs_failed_execution_and_records_reward(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    debugger = Mock(return_value={
        "generated_code": {
            "code": "print('repaired experiment')",
            "experiment_name": "csv_random_forest_experiment",
        },
        "debug_policy_strategy": "minimal_fix",
    })

    policy = Mock()
    policy.get_stats.return_value = []

    failed_result = {
        "success": False,
        "stage": "sandbox",
        "issues": ["Execution failed"],
        "debug_feedback": "Execution failed",
        "generated_code": {
            "code": "print('original experiment')",
            "experiment_name": "csv_random_forest_experiment",
        },
    }

    successful_result = {
        "success": True,
        "dataset": "data.csv",
        "target_column": "target",
        "experiment": {"result_label": "executed"},
        "critic": {"verdict": "accept"},
    }

    with patch(
        "src.agents.csv_experiment_agent.detect_task",
        return_value={"success": True, "task": "classification"},
    ), patch(
        "src.agents.csv_experiment_agent.run_sandbox_csv_experiment",
        side_effect=[failed_result, successful_result],
    ) as runner:

        agent = CSVExperimentAgent(
            debugger=debugger,
            reward_policy=policy,
        )

        result = agent.run({
            "dataset_path": str(dataset),
            "target_column": "target",
            "selected_hypothesis": {},
        })

    assert result["experiment_results"]["success"] is True
    assert result["debug_attempts"] == 2

    assert runner.call_count == 2
    assert runner.call_args_list[1].kwargs["script_override"] == (
        "print('repaired experiment')"
    )

    debugger.assert_called_once()

    policy.update.assert_called_once_with(
        "minimal_fix",
        1.0,
    )


def test_csv_debugger_stops_after_maximum_attempts(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    debugger = Mock(side_effect=[
        {
            "generated_code": {"code": "print('repair 1')"},
            "debug_policy_strategy": "minimal_fix",
        },
        {
            "generated_code": {"code": "print('repair 2')"},
            "debug_policy_strategy": "type_error_fix",
        },
    ])

    policy = Mock()
    policy.get_stats.return_value = []

    failed_result = {
        "success": False,
        "stage": "sandbox",
        "issues": ["Docker execution failed"],
        "debug_feedback": "Docker execution failed",
        "generated_code": {
            "code": "print('failed experiment')",
        },
    }

    with patch(
        "src.agents.csv_experiment_agent.detect_task",
        return_value={"success": True, "task": "classification"},
    ), patch(
        "src.agents.csv_experiment_agent.run_sandbox_csv_experiment",
        return_value=failed_result,
    ) as runner:

        agent = CSVExperimentAgent(
            debugger=debugger,
            reward_policy=policy,
        )

        result = agent.run({
            "dataset_path": str(dataset),
            "target_column": "target",
            "selected_hypothesis": {},
        })

    assert runner.call_count == 3
    assert debugger.call_count == 2

    assert result["failure_state"] == "EXECUTION_FAILED"
    assert result["debug_attempts"] == 3

    assert policy.update.call_count == 2

    policy.update.assert_any_call("minimal_fix", -1.0)
    policy.update.assert_any_call("type_error_fix", -1.0)


def test_csv_validation_failure_does_not_call_debugger(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n")

    debugger = Mock()
    policy = Mock()
    policy.get_stats.return_value = []

    validation_failure = {
        "success": False,
        "stage": "validation",
        "issues": ["Invalid dataset"],
    }

    with patch(
        "src.agents.csv_experiment_agent.detect_task",
        return_value={"success": True, "task": "classification"},
    ), patch(
        "src.agents.csv_experiment_agent.run_sandbox_csv_experiment",
        return_value=validation_failure,
    ):

        agent = CSVExperimentAgent(
            debugger=debugger,
            reward_policy=policy,
        )

        result = agent.run({
            "dataset_path": str(dataset),
            "target_column": "target",
            "selected_hypothesis": {},
        })

    debugger.assert_not_called()
    policy.update.assert_not_called()

    assert result["failure_state"] == "EXECUTION_FAILED"


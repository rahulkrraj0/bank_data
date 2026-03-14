import mlflow
import sys
from mlflow.tracking import MlflowClient

client = MlflowClient()

def list_experiments():
    list_of_experiments = []
    for exp in client.search_experiments():
        list_of_experiments.append(exp.name)
    return list_of_experiments

def list_runs(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs[["tags.mlflow.runName", "run_id"]]

def delete_runs(model_id):
    client.delete_run(model_id)

def get_indices_from_input(prompt):
    while True:
        indices_str = input(prompt)
        if not indices_str.strip():
            print("Input cannot be blank.")
            continue

        if indices_str.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            sys.exit(0)

        try:
            indices = []
            parts = [p.strip() for p in indices_str.split(',')]
            for part in parts:
                if ':' in part:
                    start, end = part.split(':')
                    start = int(start.strip())
                    end = int(end.strip())
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))
            return indices
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers or ranges (e.g., 1,2,5:10).")


def delete_model():
    """
    input number as coma separated or range as 1:10 to select the model to delete.
    """



    list_of_experiments = list_experiments()
    print("Available experiments:")
    for i, exp_name in enumerate(list_of_experiments):
        print(f"  {i}: {exp_name}")

    #get input from user
    exp_indices = get_indices_from_input("Enter the experiment indices (e.g., 1,2,5:10, or 'exit' to quit): ")

    for exp_index in exp_indices:
        if 0 <= exp_index < len(list_of_experiments):
            experiment = list_of_experiments[exp_index]
            print(f"\nProcessing experiment: {experiment}")

            model_list = list_runs(experiment)
            if model_list.empty:
                print("  No runs found in this experiment.")
                continue
            
            print("Available runs:")
            print(model_list.to_string())

            model_indices = get_indices_from_input("Enter the model indices to delete for this experiment (e.g., 1,2,5:10, or 'exit' to quit): ")

            for model_index in model_indices:
                if 0 <= model_index < len(model_list):
                    model_id = model_list.iloc[model_index]["run_id"]
                    print(f"  Deleting model with run_id: {model_id}")
                    delete_runs(model_id)
                    print(f"  Successfully deleted run {model_id}.")
                else:
                    print(f"  Invalid model index: {model_index}")
        else:
            print(f"Invalid experiment index: {exp_index}")


if __name__ == "__main__":
    delete_model()

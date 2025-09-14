# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn  # ensure sklearn flavor is available
import os
import json

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering", args.model_name)

    # -----------  IMPLEMENTATION -----------
    # Step 1: Load the model from the specified path
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log the loaded model with the given name as artifact path
    # (artifact path is a folder name inside the run’s artifacts)
    artifact_path = "model"
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)

    # Step 3: Register the logged model and get its registered version
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    registered_version = result.version

    print(f"Registered model '{args.model_name}' as version {registered_version}")

    # Step 4: Write registration details to JSON
    out_path = Path(args.model_info_output_path)
    # If a directory is provided, write to <dir>/model_info.json
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "model_info.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": args.model_name,
        "model_version": registered_version,
        "model_uri": model_uri
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote model info to: {out_path}")

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]
    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

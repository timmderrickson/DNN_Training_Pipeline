from itertools import product
from main import batch_run  # Import batch_run from your main pipeline


def test_batch_run_permutations(test_inputs, test_permutations_limit=None):
    """
    Runs multiple permutations of batch_run using provided test inputs.

    Args:
        test_inputs (dict): Keys are parameter names, values are lists of possible values.
            Must include 'image_paths' and 'ground_truth_json_paths' as lists.
            Optional: 'prediction_json_paths'.
        test_permutations_limit (int or None): Max number of permutations to run.
    """

    # Separate out fixed and variable parameters
    variable_params = {}
    for key, value in test_inputs.items():
        # image_paths, ground_truth_json_paths, prediction_json_paths should NOT be varied
        if key in ["image_paths", "ground_truth_json_paths", "prediction_json_paths"]:
            continue
        variable_params[key] = value

    param_names = list(variable_params.keys())
    param_values = [variable_params[name] for name in param_names]

    all_permutations = list(product(*param_values))
    total_tests = len(all_permutations)
    print(f"\nTotal permutations to test: {total_tests}")

    test_num = 1

    for values in all_permutations:

        if test_permutations_limit and test_num > test_permutations_limit:
            print("\nTest permutations limit reached. Ending tests.")
            return

        param_combo = dict(zip(param_names, values))

        print("\n" + "="*60)
        print(f" TEST RUN #{test_num} ")
        print("="*60)

        arguments = {
            "image_paths": test_inputs["image_paths"],
            "ground_truth_json_paths": test_inputs["ground_truth_json_paths"],
            "prediction_json_paths": test_inputs.get("prediction_json_paths", None),
            "model": "cellpose",
            "model_type": "cyto",
            "diameter": param_combo.get("diameter"),
            "channels": [0, 0],
            "output_dir": "data/predictions",
            "batch_size": 2,
            "save_visuals": False,
            "pred_class": param_combo.get("pred_class"),
            "gt_class": param_combo.get("gt_class"),
            "compare": param_combo.get("compare"),
            "overwrite_inference": param_combo.get("overwrite_inference")
        }

        for k, v in arguments.items():
            print(f"{k}: {v}")

        try:
            batch_run(**arguments)
        except Exception as e:
            print(f"❌ Error during test #{test_num}: {e}")

        test_num += 1

if __name__ == "__main__":
    test_inputs = {
        "image_paths": [
            "data/images/Araceli_A6_s2_w1_z0_1020e47f-73ff-427f-b5aa-44d2915e9068.tiff",
            "data/images/Araceli_A7_s4_w1_z0_af3998a3-849c-47fe-9274-382f3879f87c.tiff"
        ],
        "ground_truth_json_paths": [
            "data/annotations/A6_s2.json",
            "data/annotations/A7_s4.json"
        ],
        "prediction_json_paths": [
            "data/predictions/A6_s2_cellpose.json",
            "data/predictions/A6_s2_cellpose.json"
        ],
        "diameter": [20, 30],
        "pred_class": [['0'], ['0', '1']],
        "gt_class": [['0'], ['0', '1']],
        "compare": [True, False],
        "overwrite_inference": [False, True]
    }

    # Run the test permutations (for example, limit to just 3 runs while testing)
    test_batch_run_permutations(
        test_inputs=test_inputs,
        test_permutations_limit=3  # Optional, can be None to run all permutations
    )

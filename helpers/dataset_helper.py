import os
import yaml

def create_yaml(
    dataset_dir,
    class_list,
    train_path="images/train",
    val_path="images/val",
):
    """Create an Ultralytics dataset config for separate train/val sets."""
    yaml_data = {
        "path": os.path.abspath(dataset_dir),
        "train": train_path,
        "val": val_path,
        "names": {idx: name for idx, name in enumerate(class_list)},
    }

    os.makedirs(dataset_dir, exist_ok=True)
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            yaml_data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return yaml_path

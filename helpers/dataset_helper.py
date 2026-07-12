import os

def create_yaml(dataset_dir, class_list):
    yaml_content = f"""
path: {dataset_dir}
train: images
val: images
names:
"""
    for idx, name in enumerate(class_list):
        yaml_content += f"  {idx}: {name}\n"

    with open(os.path.join(dataset_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

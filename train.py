import argparse

import yaml
import mlfoundry

from QloraTrainer import QloraTrainer

client = mlfoundry.get_client()


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)
    with client.create_run(ml_repo="llama2", run_name="qlora-uncensored") as run:
        trainer = QloraTrainer(config, run)

        print("Load base model")
        trainer.load_base_model()

        print("Start training")
        trainer.train()

        print("Merge model and save")
        trainer.merge_and_save()

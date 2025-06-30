import os
import h5py
from pathlib import Path
from datetime import datetime


class MATLABLogger():
    def __init__(self, script_name: str, file_name: str):
        script_path = Path(__file__).resolve()
        mppi_solver_dirs = [p for p in script_path.parents if p.name == 'mppi_solver']
        if not mppi_solver_dirs:
            raise FileNotFoundError("Directory 'mppi_solver' not found")
        project_root = mppi_solver_dirs[-1]

        self.log_dir = project_root / 'mppi_solver' / 'runs' / datetime.now().strftime("%Y%m%d")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # run_nums = sorted(int(p.name) for p in self.log_dir.iterdir() if p.is_dir() and p.name.isdigit())
        # if run_nums:
        #     if (self.log_dir / str(run_nums[-1]) / script_name).exists():
        #         self.log_dir = self.log_dir / str(run_nums[-1] + 1) / script_name
        #     else:
        #         self.log_dir = self.log_dir / str(run_nums[-1]) / script_name
        # else:
        #     self.log_dir = self.log_dir / '1' / script_name
        self.log_dir = self.log_dir / '1' / script_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = h5py.File(self.log_dir / (file_name + ".mat"), 'w')
        self.log_dataset = dict()
        self.log_idx = dict()


    def create_dataset(self, dataset_name: str, shape: int):
        self.log_dataset[dataset_name] = self.log_file.create_dataset(dataset_name, shape=(0,shape), maxshape=(None,shape), chunks=True)
        self.log_idx[dataset_name] = 0 


    def log(self, dataset_name: str, data):
        self.log_dataset[dataset_name].resize(self.log_idx[dataset_name]+1, axis=0)
        self.log_dataset[dataset_name][self.log_idx[dataset_name]] = data
        self.log_idx[dataset_name] += 1
        self.log_file.flush()


    def close(self):
        self.log_file.close()

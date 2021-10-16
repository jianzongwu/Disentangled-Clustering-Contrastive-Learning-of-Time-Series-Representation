# Time Series Representation -- Cross dataset

Folders:
* eval/: experimental results (on UCR archive)
* training/: trained models in experiments (corresponding to eval/)
* models/
* data/
* ./*.ipynb: experimental notebooks

## Requirements

* Python 3.8
* torch==1.8.1
* scipy==1.6.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.2

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

There are 128 UCR datasets and processed datasets in `data/processed` (most are several datasets in UCR concatenated).

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `data/` so that each data file can be located by `data/UCRArchive_2018/<dataset_name>/<dataset_name>_*.tsv`.

## Usage

To train and evaluate, run the following command:

```train & evaluate
python train.py <dataset_name> --run_name <run_name> --datapath <data_path> --iters <iter_num> --eval
```

note the evaluation is run on all UCR datasets.

The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| batch_size | The batch size (defaults to 10) |
| lr | The learning rate (defaults to 0.001) |
| K | The number of cluster encoders (defaults to 3)|
| sim_fun | The cluster similarity function(defaults to cosine)|
| cate_fun | The cluster function (defaults to softmax) |
| iters | The number of training iterations (defaults to 2000) |
| save_every | Save the checkpoint every <save_every> iterations/epochs(defaults to 100) |
| valid | Whether to save and valid model every <save_every> iters(defaults to False) |
| latest | Whether to save model in a latest model folder(defaults to False) |
| seed | The random seed(defaults to 0) |
| max-threads | The maximum allowed number of threads used by this process(defaults to 8) | 
| eval | Whether to perform evaluation after training |

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 

## Code Example

```python
from model import OursModel
import utils

# Load 6_data (formed from 6 dataset from UCR)
train_data, train_labels, test_data, test_labels = utils.load_UCR_dataset('./data/processed', '6_data')
# (Both train_data and test_data have a shape of n_instances x n_timestamps)

# Train a model
model = OursModel(
    input_dims=1,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Compute representations for test set
test_repr = model.encode(test_data)  # n_instances x output_dims

# other tests on representations
```

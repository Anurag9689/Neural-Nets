## Simple Multilayer perceptrons (Artificial Neural Networks)

### Run it
1. `python3 -m venv env`

2. `source env/bin/activate`

3. `pip install -r requirements.txt` (uncomment cupy-cuda12x if you don't have a GPU)

4. `mkdir mnist_csv`

5. `cd mnist_csv`

6. `wget https://neural-net-assets.s3.ap-south-1.amazonaws.com/mnist_csv/mnist_train.csv`

7. `wget https://neural-net-assets.s3.ap-south-1.amazonaws.com/mnist_csv/mnist_test.csv`

8. `cd ..`

9. `python3 neural_net.py` (for example)


### Reference for further study
1. https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory

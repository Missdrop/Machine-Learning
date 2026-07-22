# Transformer Model for English to Classical Chinese Translation

This implementation follows the original paper of Transformer, without using `torch.nn.Transformer`.

A pre-trained model is available at GitHub release. Download and put it to `Deep Learning/Translation` folder. Which use
the hyperparameters below:

- max_length = 128
- min_frequency = 2
- embedding_dim = 512
- head_count = 8
- layer_count = 6
- ff_dim = 2048

Please ensure these hyperparameters are the same as shown when using the pre-trained model. Otherwise, the model may not
work properly.

> This repository also provides `Evaluate.ipynb` notebook to run the model without training.

If you want to train the model by yourself, please tune the `batch_size` to fit your GPU memory, and `epochs` to control
the training time.

Also, you can modify the hyperparameters to improve the performance, and `min_frequency` for the vocabulary.

# Transformer Model for English to Classical Chinese Translation

This implementation follows the original paper of Transformer, without using `torch.nn.Transformer`.

A pre-trained model is available at github release.

If you want to train the model by yourself, please tune the `batch_size` to fit your GPU memory,
and `epochs` to control the training time.

Also, you can modify the hyperparameters to improve the performance, and `min_frequency` for the vocabulary.

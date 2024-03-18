'''
Heavily Based on:
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
'''
from tqdm import tqdm
from flax import linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import torchvision.datasets as tv_datasets
import optax

'''Network'''
class ConvNet(nn.Module):
    '''A simple ConvNet for MNIST Classification'''

    hidden_features: int = 32
    num_classes: int = 10

    def setup(self) -> None:
        # nn.Dense instead of nn.Linear
        self.layer1 = nn.Conv(features=self.hidden_features, kernel_size=(3, 3))
        self.layer2 = nn.Conv(features=self.hidden_features, kernel_size=(3, 3))
        self.logits_layer = nn.Dense(features=self.num_classes)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.logits_layer(x)
        return x


'''Data Loading Utils'''
def get_dataloader(dataset, batch_size: int=32, shuffle: bool = True):
    '''Get the dataloader'''
    from torch.utils.data import DataLoader

    def numpy_collate(batch):
        '''Collate but works over numpy images
        # This collate function is taken from the JAX tutorial with PyTorch Data Loading
        # https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
        '''
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=numpy_collate
    )


def image_to_numpy(img):
    '''Convert an image to a numpy array of the right shape'''
    img = np.array(img, dtype=np.int32)
    # flatten the image completely
    if len(img.shape) == 2:
        return np.expand_dims(img, 2)
    return img


def calculate_loss_acc(state, params, batch):
    x, y_true = batch
    # Obtain the logits and predictions of the model for the input data
    y_pred = state.apply_fn(params, x) # (batch_size, num_classes)
    
    # loss similar to nn.CrossEntropyLoss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        y_pred, y_true
    ).mean()
    pred_labels = jnp.argmax(y_pred, axis=-1) # (batch_size,)
    acc = (pred_labels == y_true).mean()
    return loss, acc


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss_acc,  # Function to calculate the loss
        argnums=1,  # gradient wrt to loss
        has_aux=True  # since we also return accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc

def eval_model(state, test_dl):
    acc_list = []
    batch_sizes = []
    for batch in test_dl:
        _, acc = calculate_loss_acc(state, state.params, batch)
        acc_list.append(acc)
        batch_sizes.append(batch[0].shape[0])
    return np.average(acc_list, weights=batch_sizes)
        

if __name__ == "__main__":
    # prepare data
    dataset = tv_datasets.MNIST(root='data/', download=True, transform=image_to_numpy)
    test_dataset = tv_datasets.MNIST(root='data/', download=True, train=False, transform=image_to_numpy)
    dl = get_dataloader(dataset, batch_size=32, shuffle=True)
    test_dl = get_dataloader(test_dataset, batch_size=32, shuffle=True)

    item = dataset[0][0]
    num_input_features = item.shape[0]

    # prepare model
    model = ConvNet(hidden_features=64, num_classes=10)

    # batch = next(iter(dl))
    # out, params = model.init_with_output(jax.random.PRNGKey(42), batch[0])
    
    # get params for a model
    rng = jax.random.PRNGKey(42)
    rng, init_key = jax.random.split(rng, num=2)
    inp = jnp.ones((10, 28, 28, 1)) # sample input
    variables = model.init(init_key, inp)

    # sample forward pass
    optimizer = optax.adam(learning_rate=1.e-3)

    # get model state
    from flax.training import train_state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer
    )

    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        for i, batch in enumerate(dl):
            state, loss, acc = train_step(state, batch)
            if i % 500 == 0:
                print (f'Training Step: {i}: Loss: {loss}')

        # evaluate on test set after every epoch
        eval_acc = eval_model(state, test_dl)
        print (f'Epoch: {epoch}, Eval Acc: {eval_acc}')

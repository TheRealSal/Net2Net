import torch
import numpy as np


# Helper methods
def is_conv_layer(mod):
    """
    Check if a given module is a convolutional layer.

    Parameters:
    mod (torch.nn.Module): The module to check.

    Returns:
    bool: True if the module is a convolutional layer, False otherwise.
    """
    return "Conv" in mod.__class__.__name__


def is_linear_layer(mod):
    """
    Check if a given module is a linear layer.

    Parameters:
    mod (torch.nn.Module): The module to check.

    Returns:
    bool: True if the module is a linear layer, False otherwise.
    """
    return "Linear" in mod.__class__.__name__


def is_2d_conv(mod):
    """
    Check if a given module is a 2D convolutional layer.

    Parameters:
    mod (torch.nn.Module): The module to check.

    Returns:
    bool: True if the module is a 2D convolutional layer, False otherwise.
    """
    return mod.weight.data.dim() == 4


def is_3d_conv(mod):
    """
    Check if a given module is a 3D convolutional layer.

    Parameters:
    mod (torch.nn.Module): The module to check.

    Returns:
    bool: True if the module is a 3D convolutional layer, False otherwise.
    """
    return mod.weight.data.dim() == 4


# Net2Net Methods
def wider(input_layer, next_layer, new_width, out_size=None, batch_norm=None, noise=True, random_init=False,
          device=torch.device('cuda')):
    """
    Expand the width of a given layer and the subsequent layer in the neural network.

    Parameters:
    input_layer (torch.nn.Module): The layer to be widened.
    next_layer (torch.nn.Module): The subsequent layer after the input layer.
    new_width (int): The new width of the input layer.
    out_size (tuple, optional): The output size for 3D convolutional layers.
    batch_norm (torch.nn.Module, optional): The batch normalization layer, if any.
    noise (bool, optional): Whether to add noise to the new neurons.
    random_init (bool, optional): Whether to randomly initialize new neurons.
    device (torch.device, optional): The device on which to perform the operations.

    Returns:
    tuple: The new weights of the input layer, the new weights of the next layer, and the new biases.
    """
    # TODO - Random initialization of new neurons

    # Get weights
    w1 = input_layer.weight.data
    w2 = next_layer.weight.data
    bias = input_layer.bias.data

    if is_conv_layer(input_layer) and is_linear_layer(next_layer):
        assert w2.shape[1] % w1.shape[0] == 0, "Linear units need to be multiple"
        if is_2d_conv(input_layer):
            factor = int(np.sqrt(w2.size(1) // w1.size(0)))
            w2 = w2.view(w2.size(0), w2.size(1) // factor ** 2, factor, factor)

        elif is_3d_conv(input_layer):
            assert out_size is not None, \
                "For conv3d -> linear, out_size is necessary"
            factor = out_size[0] * out_size[1] * out_size[2]
            w2 = w2.view(w2.size(0), w2.size(1) // factor, out_size[0],
                         out_size[1], out_size[2])

    assert new_width > w1.size(0), "New size should be larger"
    assert w1.shape[0] == w2.shape[1], "Module weights are not compatible"

    new_w1 = w1.clone()
    new_w2 = w2.clone()
    new_bias = bias.clone()

    w1_shape = list(new_w1.shape)
    w2_shape = list(new_w2.shape)

    old_width = w1_shape[0]

    size_difference = new_width - old_width

    if size_difference <= 0:
        return

    # Create tensors to store new neurons
    w1_shape[0] = size_difference
    w2_shape[1] = size_difference
    input_new_columns = torch.zeros(w1_shape).to(device)
    output_new_columns = torch.zeros(w2_shape).to(device)

    if bias is not None:
        bias_new_columns = torch.zeros(size_difference).to(device)

    if batch_norm is not None:
        nrunning_mean = batch_norm.running_mean.clone().resize_(new_width)
        nrunning_var = batch_norm.running_var.clone().resize_(new_width)
        if batch_norm.affine:
            nweight = batch_norm.weight.data.clone().resize_(new_width)
            nbias = batch_norm.bias.data.clone().resize_(new_width)
            nrunning_var.narrow(0, 0, old_width).copy_(batch_norm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(batch_norm.running_mean)
            if batch_norm.affine:
                nweight.narrow(0, 0, old_width).copy_(batch_norm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(batch_norm.bias.data)

    # Mapping Algorithm for new Neurons
    mapping = {}
    for index in range(size_difference):
        idx = np.random.randint(old_width)  # Neuron to be copied
        if idx not in mapping:
            mapping[idx] = []
        mapping[idx].append(index)

    for idx, indeces in mapping.items():
        for index in indeces:
            # Copy incoming weights and apply noise if enabled
            input_new_columns[index, :] = new_w1[idx, :]
            if noise:
                noise_tensor = torch.normal(mean=0.0, std=5e-2 * input_new_columns[index].std(),
                                            size=input_new_columns[index].size()).to(device)
                input_new_columns[index] += noise_tensor

            # Copy outgoing weights
            output_new_columns[:, index] = new_w2[:, idx]

            # Copy bias
            if bias is not None:
                bias_new_columns[index] = new_bias[idx]

            # Divide outgoing weights by replication factor
            output_new_columns[:, index] /= len(indeces)

            # Copy Batch Normalization Values
            if batch_norm is not None:
                nrunning_mean[index] = batch_norm.running_mean[idx]
                nrunning_var[index] = batch_norm.running_var[idx]
                if batch_norm.affine:
                    nweight[index] = batch_norm.weight.data[idx]
                    nbias[index] = batch_norm.bias.data[idx]
                batch_norm.num_features = new_width
        new_w2[:, idx] /= len(indeces)  # Divide original outgoing weight by replication factor

    # Insert new neurons to the hidden layers' weights
    new_w1 = torch.cat((new_w1, input_new_columns), dim=0)
    new_w2 = torch.cat((new_w2, output_new_columns), dim=1)
    if bias is not None:
        new_bias = torch.cat((new_bias, bias_new_columns), dim=0)

    # Update layers parameters: convolution channel numbers and in/out features for linear layers
    if is_conv_layer(input_layer):
        input_layer.out_channels = new_width

    if is_conv_layer(next_layer):
        next_layer.in_channels = new_width

    if is_conv_layer(input_layer) and is_linear_layer(next_layer):
        if is_2d_conv(input_layer):
            new_w2 = new_w2.view(w2.shape[0], new_width * factor ** 2)
            next_layer.in_features = new_width * factor ** 2
        elif is_3d_conv(input_layer):
            new_w2 = new_w2.view(w2.shape[0], new_width * factor)
            next_layer.in_features = new_width * factor

    # Set the layer's weights to the updated ones
    input_layer.weight.data = new_w1
    next_layer.weight.data = new_w2
    input_layer.bias.data = new_bias

    if batch_norm is not None:
        batch_norm.running_var = nrunning_var
        batch_norm.running_mean = nrunning_mean
        if batch_norm.affine:
            batch_norm.weight.data = nweight
            batch_norm.bias.data = nbias

    return new_w1, new_w2, new_bias

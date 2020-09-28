import argparse
import os
import h5py
import numpy as np
import plotly.graph_objects as go
from keras.models import model_from_json
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency, overlay
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from IPython.display import HTML
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def loaddata(data_dir, filename):

    with h5py.File(data_dir+filename, 'r') as f:
        # List all groups
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    X= np.asarray(data, 'float32')

    return X

def main():
    parser = argparse.ArgumentParser(
        description='test AD recognition')
    parser.add_argument('--input', type=str, required=True, help="path to test data")
    parser.add_argument('--model', type=str, required=True, help="path to pre-trained model")
    parser.add_argument('--id', type=int, required=True, help="data id")
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(os.getcwd()), args.model)
    data = loaddata(args.input, 'testa.h5')
    print('data_shape:{}'.format(data.shape))
    print("[INFO] loading pre-trained network...")
    json_file = open(model_dir+'AD_3dcnnmodel.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(model_dir+"AD_3dcnnmodel.hd5")
    model.summary()

    layer_idx = utils.find_layer_idx(model, 'activation_10')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=data[args.id], backprop_modifier='guided', keepdims=True)
    volume1 = np.squeeze(data[args.id], axis=0)
    volume1 = (volume1 - np.min(volume1)) / (np.max(volume1) - np.min(volume1))
    fig, axs = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    axs[1].imshow(volume1[39,:,:], cmap='gray')
    plt.show()
    import pdb
    pdb.set_trace()
    volume2 = np.squeeze(grads, axis=0)
    vol= overlay(volume1, volume2)
    axial_planes = []
    for i in range(vol.shape[0]):
        axial_planes.append(vol[i,:,:])

    # Matplotlib animate heart
    Hz = np.zeros([vol.shape[1], vol.shape[2]])
    im = ax.imshow(Hz)

    def init():
        im.set_data(np.zeros(Hz.shape))
        return [im]

    def animate(i):
        im.set_data(axial_planes[i])
        im.autoscale()

        return [im]

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=len(axial_planes),
                                   interval=100,
                                   blit=True)
    plt.show()


if __name__ == '__main__':
    main()

 grads = np.squeeze(grads, axis=0)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    extent1 = 0, grads.shape[1], 0, grads.shape[0]
    extent2 = 0, grads.shape[2], 0, grads.shape[1]
    extent3 = 0, grads.shape[2], 0, grads.shape[0]
    ax[0].imshow(data[9][0, :, :, 45], cmap='gray', interpolation='nearest', extent=extent1)
    ax[0].imshow(grads[:, :, 45], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent1)
    ax[0].set_title('A')
    ax[1].imshow(data[9][0, 39, :, :], cmap='gray', interpolation='nearest', extent=extent2)
    ax[1].imshow(grads[39, :, :], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent2)
    ax[1].set_title('C')
    ax[2].imshow(data[9][0, :, 60, :], cmap='gray', interpolation='nearest', extent=extent3)
    ax[2].imshow(grads[:, 60, :], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent3)
    ax[2].set_title('S');
    plt.show()
    import pdb
    pdb.set_trace()
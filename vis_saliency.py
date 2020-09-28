import argparse
import os
import h5py
import numpy as np
import plotly.graph_objects as go
from keras.models import model_from_json
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency
import matplotlib.pyplot as plt
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
    grads = np.squeeze(grads, axis=0)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    extent1 = 0, grads.shape[1], 0, grads.shape[0]
    extent2 = 0, grads.shape[2], 0, grads.shape[1]
    extent3 = 0, grads.shape[2], 0, grads.shape[0]
    ax[0].imshow(data[args.id][0, :, :, 45], cmap='gray', interpolation='nearest', extent=extent1)
    ax[0].imshow(grads[:, :, 45], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent1)
    ax[0].set_title('A')
    ax[1].imshow(data[args.id][0, 39, :, :], cmap='gray', interpolation='nearest', extent=extent2)
    ax[1].imshow(grads[39, :, :], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent2)
    ax[1].set_title('C')
    ax[2].imshow(data[args.id][0, :, 60, :], cmap='gray', interpolation='nearest', extent=extent3)
    ax[2].imshow(grads[:, 60, :], cmap='hot', alpha=.5, interpolation='bilinear', extent=extent3)
    ax[2].set_title('S');
    plt.show()
    import pdb
    pdb.set_trace()
    volume = np.squeeze(grads, axis=0)
    volume= np.swapaxes(volume, 0, 2)
    r, c = volume.shape[1], volume.shape[2]

    # Define frames
    nb_frames = volume.shape[0]

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames - 1) * 0.1 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
        cmin=0, cmax=1
    ),
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(nb_frames - 1) * 0.1 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1]),
        colorscale='hot',
        cmin=0, cmax=1,
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='Transverse View Saliency',
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[-0.1, nb_frames*0.1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    fig.show()


if __name__ == '__main__':
    main()
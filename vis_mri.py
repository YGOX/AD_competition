import plotly.graph_objects as go
import h5py
import numpy as np

filename = 'AD_data/testa.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = np.asarray(list(f[a_group_key]), 'float32')

volume = np.squeeze(data[9], axis=0)
#volume = np.swapaxes(np.squeeze(data[9], axis=0), 0, 1)
volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
r, c = volume.shape[1], volume.shape[2]

# Define frames
nb_frames = volume.shape[0]

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=((nb_frames-1)*0.1 - k * 0.1) * np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames-1-k]),
    cmin=0, cmax=1
    ),
    name=str(k) # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=(nb_frames-1)*0.1* np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames-1]),
    colorscale='gray',
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
         title='Sagittal Slices',
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[-0.1, 9.5], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
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
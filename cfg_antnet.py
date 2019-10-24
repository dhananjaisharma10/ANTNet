"""Configuration file for ANTNet
"""

# architecture settings
model = dict(
    in_channels=3,
    num_classes=1000,
    num_stages=7,
    outplanes=[32, 16, 24, 32, 64, 96, 160, 320, 1280],
    repetitions=[1, 2, 3, 4, 3, 3, 1],
    expansions=[1, 6, 6, 6, 6, 6, 6],
    reduction_ratios=[8, 8, 12, 16, 24, 32, 64],
    strides=[2, 1, 2, 2, 2, 1, 2, 1, 1],
    groups=[1, 1, 1, 1, 1, 1, 1],
)

# training settings
cfg = dict(
    optimizer=dict(
        name='SGD',
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        weight_decay=4e-5
    ),
    criterion=dict(
        name='CrossEntropyLoss'
    ),
    scheduler=dict(
        name='MultiStepLR',
        milestones=[200, 300]
    ),
    epochs=400,
    device='cuda',
    weights='./weights/antnet'
)

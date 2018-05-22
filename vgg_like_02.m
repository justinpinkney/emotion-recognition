rng('default');
dataRoot = '.';
images = imageDatastore(dataRoot, 'IncludeSubfolders', true, ...
                            'LabelSource', 'foldernames');
[train, val, test] = splitEachLabel(images, 0.7, 0.15, 0.15);

augmenter = imageDataAugmenter('RandRotation', [-10, 10], ...
                                 'RandXTranslation', [-2, 2], ...
                                 'RandYTranslation', [-2, 2], ...
                                 'RandXReflection', true, ...
                                 'RandXScale', [0.95, 1.05], ...
                                 'RandYScale', [0.95, 1.05]);

augmentedTrain = augmentedImageDatastore([48, 48, 1], train);

%% Make network
layers = [
    imageInputLayer([48, 48, 1]) ...
    ...
    convolution2dLayer([3, 3], 64, 'Padding', 'same') ...
    convolution2dLayer([3, 3], 64, 'Padding', 'same') ...
    maxPooling2dLayer([3, 3], 'Stride', [2, 2]) ...
    dropoutLayer(0.25) ...
    ...
    convolution2dLayer([3, 3], 128, 'Padding', 'same') ...
    convolution2dLayer([3, 3], 128, 'Padding', 'same') ...
    maxPooling2dLayer([3, 3], 'Stride', [2, 2]) ...
    dropoutLayer(0.25) ...
    ...
    convolution2dLayer([3, 3], 256, 'Padding', 'same') ...
    convolution2dLayer([3, 3], 256, 'Padding', 'same') ...
    convolution2dLayer([3, 3], 256, 'Padding', 'same') ...
    maxPooling2dLayer([3, 3], 'Stride', [2, 2]) ...
    dropoutLayer(0.25) ...
    ...
    fullyConnectedLayer(1024) ...
    reluLayer ...
    dropoutLayer(0.5) ...
    ...
    fullyConnectedLayer(numel(unique(images.Labels)))...
    softmaxLayer() ...
    classificationLayer()
];

options = trainingOptions('adam', ...
                        'ExecutionEnvironment', 'cpu', ...
                        'MiniBatchSize', 128, ...
                        'MaxEpochs', 15, ...
                        'InitialLearnRate', 0.001, ...
                        'LearnRateSchedule', 'piecewise', ...
                        'LearnRateDropPeriod', 5, ...
                        'LearnRateDropFactor', 0.3, ...
                        'ValidationData', val, ...
                        'ValidationPatience', Inf, ...
                        'Verbose', true);
                   
diary on
net = trainNetwork(augmentedTrain, layers, options);
diary off
save('fer.mat', 'net');

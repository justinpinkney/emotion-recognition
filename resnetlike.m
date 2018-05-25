% Make a resnet like network

inputSize = [48, 48];
colourChannels = 1;
bottlenecks = true;
nCategories = 10;
bottleneck = true;
stage1Channels = 32;

nStages = 4;
blocksPerStage = [3, 4, 6, 3];

initialLayers = [...
    imageInputLayer([inputSize, colourChannels], ...
                            'Name', 'input')
    convolution2dLayer([3, 3], 2*stage1Channels, ...
                        'Name', 'initial_conv')
    ];

layers = layerGraph ();
layers = layers.addLayers(initialLayers);
outName = 'initial_conv';

for iStage = 1:nStages
    for iBlock = 1:blocksPerStage(iStage)
        if iBlock == 1
            [layers, outName] = bottleNeckBlock(layers, ...
                                            outName, ...
                                            [num2str(iStage), '_', num2str(iBlock)], ...
                                            (2^iStage)*stage1Channels, ...
                                            (2^(iStage+2))*stage1Channels, ...
                                            3);
        else
            [layers, outName] = bottleNeckBlock(layers, ...
                                            outName, ...
                                            [num2str(iStage), '_', num2str(iBlock)], ...
                                            (2^(iStage+2))*stage1Channels, ...
                                            (2^(iStage+2))*stage1Channels, ...
                                            1);
        end
    end
end


finalLayers = [...
    batchNormalizationLayer('Name', 'output_BN')
    reluLayer('Name', 'output_relu')
    averagePooling2dLayer([1, 1], 'Name', 'output_avePool')
    fullyConnectedLayer(nCategories, 'Name', 'output_fc')
    softmaxLayer('Name', 'output_softmax')
    classificationLayer('Name', 'output_classes')
    ];

layers = layers.addLayers(finalLayers);
layers = layers.connectLayers(outName, 'output_BN');

analyzeNetwork(layers)

% Using full pre-activation

% function plainBlock()
%     plainBlock = [
%         batchNormalizationLayer
%         reluLayer
%         convolution2dLayer([3, 3], n)
%         batchNormalizationLayer
%         reluLayer
%         convolution2dLayer([3, 3], n)
%         ];
% end

function [block, outName] = bottleNeckBlock(block, inputName, blockName, inputChannels, outputChannels, downSample)
    if inputChannels == outputChannels
        assert(downSample == 1)
    end

    bottleNeckChannels = outputChannels/4;
    convolutionBranch = [
        batchNormalizationLayer('Name', [blockName, '_BN_2a'])
        reluLayer('Name', [blockName, '_relu_2a'])
        convolution2dLayer([1, 1], bottleNeckChannels, ...
                            'Stride', [downSample, downSample], ...
                            'Name', [blockName, '_conv1_2a'])
        ...
        batchNormalizationLayer('Name', [blockName, '_BN_2b'])
        reluLayer('Name', [blockName, '_relu_2b'])
        convolution2dLayer([3, 3], bottleNeckChannels, ...
                            'Padding', 'same', ...
                            'Name', [blockName, '_conv3_2b'])

        batchNormalizationLayer('Name', [blockName, '_BN_2c'])
        reluLayer('Name', [blockName, '_relu_2c'])
        convolution2dLayer([1, 1], outputChannels, ...
                            'Name', [blockName, '_conv1_2c'])
        ];
    
    block = block.addLayers(convolutionBranch);
    block = block.connectLayers(inputName, [blockName, '_BN_2a']);
    
    if inputChannels == outputChannels
        shortcutBranch.Name = inputName;
    else
        shortcutBranch = ...
            convolution2dLayer([1, 1], outputChannels, ...
                                'Stride', [downSample, downSample], ...
                                'Name', [blockName, '_conv1_1']);
        block = block.addLayers(shortcutBranch);
        block = block.connectLayers(inputName, shortcutBranch.Name);
    end
    
    merge = additionLayer(2, 'Name', [blockName, '_add']);
    block = block.addLayers(merge);
    
    block = block.connectLayers(shortcutBranch.Name, [merge.Name, '/in1']);
    block = block.connectLayers([blockName, '_conv1_2c'], [merge.Name, '/in2']);
    
    outName = merge.Name;
    
end
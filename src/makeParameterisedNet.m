function layers = makeParameterisedNet(imSize, nClasses, nStartFilters, dropOutFactor, blockSize)

    layers = imageInputLayer(imSize);
        
    for iBlock = 1:numel(blockSize)
        layers = [layers; ...
                    makeBlock(2^(iBlock - 1)*nStartFilters, blockSize(iBlock), dropOutFactor)];
    end
    layers = [layers; 
                fullyConnectedLayer(1024);
                reluLayer;
                dropoutLayer(dropOutFactor*2);
                fullyConnectedLayer(nClasses);
                softmaxLayer();
                classificationLayer();
            ];

end

function block = makeBlock(nFilters, blockLength, dropOutFactor)
    block = [];
    for iLayer = 1:blockLength
        block = [block; makeConv(nFilters)];
    end
    
    block = [
        block;
        maxPooling2dLayer([2, 2], 'Stride', [2, 2]);
        dropoutLayer(dropOutFactor);
        ];
end

function convLayer = makeConv(nFilters)
    convLayer =  [
        convolution2dLayer([3, 3], nFilters, 'Padding', 'same');
        reluLayer;
        ];
end
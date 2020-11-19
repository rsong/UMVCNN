function net = cnn_initialise(classNames, varargin)
opts.base = 'imagenet-matconvnet-vgg-m';
opts.restart = false;
opts.nViews = 20;
opts.weightInitMethod = 'xavierimproved';
opts.scale = 1;
opts.networkType = 'dagnn';
opts.addBiasSamples = 1;
opts.addLossSmooth  = 1;
opts = vl_argparse(opts, varargin);


% init_bias = 0.1;
nClass = length(classNames);

% Load model, try to download it if not readily available
if ~ischar(opts.base)
    net = opts.base;
else
    netFilePath = fullfile('data','models', [opts.base '.mat']);
    if ~exist(netFilePath,'file')
        fprintf('Downloading model (%s) ...', opts.base) ;
        vl_xmkdir(fullfile('data','models')) ;
        urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models/', ...
            [opts.base '.mat']), netFilePath) ;
        fprintf(' done!\n');
    end
    net = load(netFilePath);
end
assert(strcmp(net.layers{end}.type, 'softmax'), 'Wrong network format');
dataTyp = class(net.layers{end-1}.weights{1});

net.layers{end}.name='softmax';

% Initiate the fc layer with random weights
widthPrev = size(net.layers{end-1}.weights{1}, 3);
nClass0 = size(net.layers{end-1}.weights{1},4);
if nClass0 ~= nClass || opts.restart
    net.layers{end-1}.weights{1} = init_weight(opts, 1, 1, widthPrev, nClass, dataTyp);
    net.layers{end-1}.weights{2} = zeros(nClass, 1, dataTyp);
end

% Initiate another fc layer with random weights
sz = size(net.layers{end-3}.weights{1});
net.layers{end-3}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
net.layers{end-3}.weights{2} = zeros(sz(4), 1, dataTyp);


% Initiate other layers w/ random weights if training from scratch is desired
if opts.restart
    w_layers = find(cellfun(@(c) isfield(c,'weights'),net.layers));
    for i=w_layers(1:end-1)
        sz = size(net.layers{i}.weights{1});
        net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
        net.layers{i}.weights{2} = zeros(sz(4), 1, dataTyp);
    end
end

% update meta data
net.meta.classes.name = classNames;
net.meta.classes.description = classNames;

% speial case: when no class names specified, remove the last 2 layers
if nClass==0
    net.layers = net.layers(1:end-2);
end

fc9 = net.layers{end-1};
fc9.name='fc9';
net.layers=[net.layers fc9];

obp = net.layers{end-1};
obp.name='obp';
net.layers=[net.layers obp];

if opts.nViews>1
    % convert to dagnn 
    % add the vs, sp and the the new loss layers
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.removeLayer('softmax') ;
    net.setLayerOutputs('fc8', {'vFeat'}) ;
   
    net.setLayerInputs('fc9', {'xSapool'}) ;
    net.setLayerOutputs('fc9', {'prediction'}) ;
    net.setLayerInputs('obp', {'prediction'}) ;
    net.setLayerOutputs('obp', {'obprob'}) ;
    
    relu7 = (arrayfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);
    net.addLayer('viewsa', VS('vstride', opts.nViews),{net.layers(relu7).outputs{1},'viewind'},'xViewSa',{});
    net.addLayer('sapool', SP('vstride', opts.nViews),{net.layers(relu7).outputs{1},'xViewSa'},'xSapool',{});
    
    net.addLayer('loss', dagnn.Loss('loss', 'softmaxlogmulti'), {'vFeat','obprob'}, 'objective');
   
    clear fc9 pfc9 psapool relu7
end
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end


% -------------------------------------------------------------------------




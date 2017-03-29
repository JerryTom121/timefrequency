% digite para rodar na linha de comando:
% nohup matlab -nodesktop -nosplash -r run_cnn_CYBHi_long_spectogram_1beat_T1T2 > ./saida_CYBHi_long_1beat_T1T2_SPECTO_w80_o76_archB_augm.txt </dev/null &
%
% autor: Eduardo Luz
% eduluz@gmail.com

function run_cnn_CYBHi_long_spectogram_1beat_T1T2()

addpath ./dataload/ ;
addpath ./evaluation/ ;
addpath ./matconvnet ;
addpath ./matconvnet/examples ;
addpath ./matconvnet/matlab/
addpath ./matconvnet/matlab/simplenn
%addpath ../wfdb/wfdb-app-toolbox-0-9-9/mcode/

setup ; % inicializa matconvnet

HBsize = 800; % numero de amostras no batimentos

%% Carrega dados para o bootstrapping

savefilestr = strcat('./IMDBstruct-spectogram-CYBHilong-1beat-t1t2', '-sz-1x',num2str(HBsize), '.mat');
savefilestrT = strcat('./TESTstruct-spectogram-CYBHilong-1beat-t1t2', '-sz-1x',num2str(HBsize), '.mat');
savefilestr_nc = strcat('./IMDBstruct-numCLass-spectogram-CYBHilong-1beat-t1t2.mat');

[stat, ~]=fileattrib(savefilestr);

if stat == 99
     imdb = load(savefilestr); 
     teste = load(savefilestrT); 
     load(savefilestr_nc);
     
     %treino = imdb.images;
else   
    [imdb treino teste numClass] = CYBHi_longTerm_Spectogram_T1T2_beatSeg(HBsize, 3, 1);
    %[imdb treino teste numClass] = CYBHi_longTerm_Spectogram_T1T2_beatSeg_2(HBsize, 3, 0);
    
    save(savefilestr,'-struct', 'imdb', '-v7.3');
    save(savefilestrT,'-struct', 'teste', '-v7.3');
    save(savefilestr_nc,'numClass');

    %treino = imdb.images;
end
display('tamano dos dados de treino e teste antes')
size(teste.data)
size(treino.data)

%% unifica classes
c_test=[];
f_test=[];
ind=1;
for j=1:size(teste.label,2)
    if sum(teste.label(1,j) == unique(treino.label)) >= 1
        c_test = [c_test teste.label(1,j)];
        f_test(:,:,ind) = teste.data(:,:,j);
        ind = ind+1;
    end
end

teste.data = f_test;
teste.label = c_test;

c_train=[];
f_train=[];
ind=1;
for j=1:size(treino.label,2)
    if sum(treino.label(1,j) == unique(teste.label)) >= 1
        c_train = [c_train treino.label(1,j)];
        f_train(:,:,ind) = treino.data(:,:,j);
        ind = ind+1;
    end
end

treino.data = f_train;
treino.label = c_train;

t_test=[];
c_class=[];

size(unique(treino.label))
size(unique(teste.label))

display('tamano dos dados de treino e teste depois unificacao')
size(teste.data)
size(treino.data)

%% Inicia metodo CNN
numClass

%numClass = size(unique(imdb.images.label)); % provisorio

size(unique(imdb.images.label),2)

size(imdb.images.data)

size(teste.data)

% -------------------------------------------------------------------------
% Part 1: faz o bootstrapping
% -------------------------------------------------------------------------
display('*** inicializa ...');
net = initializeCNN_CYBHi_specto_128(numClass);
%net = initializeCNN_CYBHi_specto_128_archA(numClass);

vl_simplenn_display(net, 'inputSize', [128 128 1 48]);

% trina para o bootstrap
display('*** treina a rede para bootstrap...');
trainOpts.batchSize = 48 ;
trainOpts.numEpochs = 80;
trainOpts.continue = true ;
%trainOpts.learningRate = [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.005 0.005 0.005 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001] ;
trainOpts.learningRate = [ 0.01*ones(1,50) 0.001*ones(1,20) 0.0001*ones(1,10)];

%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w100_o95' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w240_o120' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w160_o152' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w80_o76' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w80_o76' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w70_o67_archB_700' ;
%trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w80_o76_archA';
trainOpts.expDir = 'results_SPECTOGRAM_CYBHi_long_1beat_T1T2_w80_o76_augm';

trainOpts.errorFunction = 'multiclass' ;

trainOpts.gpus = [1] ;
trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

net.imageMean = imageMean ;
net.layers(end) = [] ; % remove a camada de loss
net.layers(end) = [] ; % remove a camada dropout

train_.data = imdb.images.data;
train_.label = imdb.images.label;
imdb = [];

%% EVAL
%% EVAL
display('cria vetores de caracterisitcas')
for i=1:size(treino.data,3)
    
    im_ = treino.data(:,:,i);
    im_ = im_ - net.imageMean;
    im_ = 255 * reshape(im_, 128, 128, 1, []) ;
    res = vl_simplenn(net, im_) ;
    
    fv = gather(res(end-1).x) ;
    fv = squeeze(fv);
    feature_vector_train(i,:) = fv(:);

    %class_train(i)=train.label(i);          

end

for i=1:size(teste.data,3)
    
    im_ = teste.data(:,:,i);
    im_ = im_ - net.imageMean;
    im_ = 255 * reshape(im_, 128, 128, 1, []) ;
    res = vl_simplenn(net, im_) ;
    
    fv = gather(res(end-1).x) ;
    fv = squeeze(fv);
    feature_vector_test(i,:) = fv(:);

    %class_test(i)=test.label(i);          

end

display('avalia..');

%% Avalia nos dados de treino T1 - T1
c_test = []; f_test = [];
c_train = []; f_train = [];

size(feature_vector_train)

for i=1:63    
    %i
    
    idx = find(treino.label(1,:)==i);
    half = length(idx)/2;
    half = floor(half);
    %size(idx)
    
    if isempty(idx) == 0
        c_test = [c_test treino.label(1,idx(half+1:end))];
        f_test = [f_test;feature_vector_train(idx(half+1:end),:)];
        c_train = [c_train treino.label(1,idx(1:half))];
        f_train = [f_train;feature_vector_train(idx(1:half),:)];
    end
    
end

[DI, DE, tTime] = EVALUATION_VERIFICATION(f_train, f_test, c_train, c_test);
[ver_rate, miss_rate, rates] = produce_ROC_PhD(DI,DE,5000);

fprintf('\n Eucl. T1T1 HTER TREINO = : %.6f \n', rates.minHTER_er);
fprintf('Eucl. T1T1 EER TREINO = : %.6f \n\n', rates.EER_er);

FAR_CNN_BEAT_SPEC = 1-ver_rate;
FRR_CNN_BEAT_SPEC = miss_rate;

save(['./' trainOpts.expDir '/DET_CNN_beat_SPEC_T1T1.mat'], 'FAR_CNN_BEAT_SPEC', 'FRR_CNN_BEAT_SPEC', 'DI', 'DE');

DI=[];
DE=[];

%% AVALIACAO OFICIAL : T1 - T2
[DI, DE, tTime] = EVALUATION_VERIFICATION(feature_vector_train, feature_vector_test, treino.label, teste.label);
[ver_rate, miss_rate, rates] = produce_ROC_PhD(DI,DE,5000);

fprintf('\n *** T1-T2 ****\n Eucl. T1T2 HTER = : %.6f \n', rates.minHTER_er);
fprintf('Eucl. T1T2 EER = : %.6f \n', rates.EER_er);

FAR_CNN_BEAT_SPEC = 1-ver_rate;
FRR_CNN_BEAT_SPEC = miss_rate;

save(['./' trainOpts.expDir '/DET_CNN_beat_SPEC_TESTE.mat'], 'FAR_CNN_BEAT_SPEC', 'FRR_CNN_BEAT_SPEC', 'DI', 'DE');

% %% AVALIACAO OFICIAL : T1 - T2T2
% [DI, DE, tTime] = EVALUATION_VERIFICATION(feature_vector_test, feature_vector_test, teste.label, teste.label);
% [ver_rate, miss_rate, rates] = produce_ROC_PhD(DI,DE,5000);
% 
% fprintf('\n *** T1-T2 ****\n Eucl. T1-T2T2 HTER = : %.6f \n', rates.minHTER_er);
% fprintf('Eucl. T1-T2T2 EER = : %.6f \n', rates.EER_er);
% 
% FAR_CNN_BEAT_RAW = 1-ver_rate;
% FRR_CNN_BEAT_RAW = miss_rate;
% 
% save(['./' trainOpts.expDir '/DET_CNN_beat_raw_T2T2.mat'], 'FAR_CNN_BEAT_RAW', 'FRR_CNN_BEAT_RAW');

end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
im = 255 * reshape(im, 128, 128, 1, []) ;
labels = imdb.images.label(1,batch) ;
end









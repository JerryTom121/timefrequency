clc;

%Import the data
[~, ~, raw] = xlsread('C:\Users\alber\OneDrive\Documentos\Mestrado\PDI\microgrid\microgrid\Microgrid B.xlsx','Sheet1');
raw = raw(2:end,end);

%Create output variable
carga = reshape([raw{:}],size(raw));

%Clear temporary variables
clearvars raw;
dia = 1;
etiquetas = [];
descritores = [];
dias  = reshape(carga, [], 24);
[lin, col] = size(dias);
for i=1:lin
    coefs (:, :, i) = cwt(dias(i, :),1:24,'sym2');
    img = coefs (:, :, i);
    descritores = [descritores; img(:)];
    etiquetas = [etiquetas; dia];
    dia = dia +1;
    if dia==8 
        dia = 1;
    end
end

[train, test] = crossvalind('holdOut', etiquetas);
label_data = etiquetas(train, :);
train_data = descritores(train, :);
test_data = descritores(test, :);

predict = classify(test_data, train_data, label_data);




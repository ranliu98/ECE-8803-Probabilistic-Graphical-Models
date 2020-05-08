import brml.*
load('ChowLiuData.mat');

DAG = p3ChowLiu(X);
drawNet(DAG); title('ChowLiu')

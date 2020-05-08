import brml.*
load('EMprinter.mat');


p{1} = array(1,condp(rand(1,2))); % Fuse
p{2} = array(2,condp(rand(1,2))); % Drum
p{3} = array(3,condp(rand(1,2))); % Toner
p{4} = array(4,condp(rand(1,2))); % Paper
p{5} = array(5,condp(rand(1,2))); % Roller
p{6} = array([6 1],condp(rand(2,2),1)); % Burning
p{7} = array([7 2 3 4],condp(rand(2,2,2,2),1)); % Quality 
p{8} = array([8 1 4],condp(rand(2,2,2),1)); % Wrinkled 
p{9} = array([9 4 5],condp(rand(2,2,2),1)); % MultPages
p{10} = array([10 1 5],condp(rand(2,2,2),1)); % PaperJam


var.tol=0.0001; var.maxiterations=50; var.plotprogress=10;
[p loglikelihood]=EMbeliefnet(p,x,var);


jp = multpots(p);
disptable(condpot(setpot(jp,[8 6 7], [1 1 2]),2),'2');


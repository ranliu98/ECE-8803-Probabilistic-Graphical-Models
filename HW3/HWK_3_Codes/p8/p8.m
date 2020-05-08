function p8
import brml.*
load('pMRF.mat');
phi=str2cell(setpotclass(phi,'array'));

nstates=2;
[W X Y Z]=assign(nstates*ones(1,4)); % number of states of each var
[w x y z]=assign(1:4);
p = condpot(multpots(phi)); % normalization, similar as demo

%% question (a)

[LBP_s mess A]=LBP_self(phi);
fprintf('----------------LBP----------------\n');
for i=1:4
    fprintf('var-(%d) \n',i);
    disp(table(LBP_s{i}))
end

%% question (b)

% construct q...
qw = array(w,normp(rand([W 1]))); qx = array(x,normp(rand([X 1])));
qy = array(y,normp(rand([Y 1]))); qz = array(z,normp(rand([Z 1])));

for iteration=1:50 % tried 100 and 10
    uni_order=randperm(4);
    for u_o=uni_order
        switch u_o
            case 1
                qw = condpot(exppot(sumpot(multpots([logpot(p) qx qy qz]),w,0),1));
            case 2
                qx = condpot(exppot(sumpot(multpots([logpot(p) qw qy qz]),x,0),1));
            case 3
                qy = condpot(exppot(sumpot(multpots([logpot(p) qx qw qz]),y,0),1));
            case 4
                qz = condpot(exppot(sumpot(multpots([logpot(p) qx qy qw]),z,0),1));
        end
    end
end

[MF_s{1} MF_s{2} MF_s{3} MF_s{4}]=assign([qw qx qy qz]);
fprintf('----------------MF----------------\n');
for i=1:4
    fprintf('var-(%d) \n',i);
    disp(table(MF_s{i}))
end

%% question (c)

exact_s=multpots(phi);

fprintf('----------------Exact----------------\n');
for i=1:4
    fprintf('var-(%d) \n',i);
    disp(table(condpot(exact_s,i)))
end

%% question (d)

for i=1:4
    BPerror(i) = mean(abs(table(condpot(exact_s,i))-table(LBP_s{i})));
    MFerror(i) = mean(abs(table(condpot(exact_s,i))-table(MF_s{i})));
end

fprintf('average error BP = %g\n',mean(BPerror))
fprintf('average error MF = %g\n',mean(MFerror))





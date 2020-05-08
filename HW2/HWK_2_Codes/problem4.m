function Isingp4
import brml.*
N=10;

isinginit(1,1)=exp(1);
isinginit(1,2)=exp(0);
isinginit(2,1)=exp(0);
isinginit(2,2)=exp(1);

nodes = reshape(1:N*N,N,N);
c=0;
for numb1=1:N*N
    [col1 row1]=find(nodes==numb1);
    for numb2=numb1+1:N*N
        [col2 row2]=find(nodes==numb2);
        if (row1==row2)&(abs(col1-col2)==1) | (col1==col2)&(abs(row1-row2)==1)
            c=c+1;
            phi{c}=array([numb1 numb2],isinginit);
            potmap(col1,col2,row1,row2)=c;
            potmap(col2,col1,row2,row1)=c;
        end
    end
end


for i=1:N
    colphi{i}=const(1);
    if i ~= N
        for j=1:N-1
            colphi{i}=multpots([colphi{i} phi{potmap(j,j,i,i+1)}]);
            colphi{i}=multpots([colphi{i} phi{potmap(j,j+1,i,i)}]);
        end
        colphi{i}=multpots([colphi{i} phi{potmap(N,N,i,i+1)}]);
    else
        for j=1:N-1
            colphi{i-1}=multpots([colphi{i-1} phi{potmap(j,j+1,i,i)}]);
        end
    end
end


eli=sumpot(colphi{1},setdiff(colphi{1}.variables,intersect(colphi{1}.variables,colphi{2}.variables)));
bigeli=[];
for i=2:N-2
    sumover=setdiff(colphi{i}.variables,intersect(colphi{i}.variables,colphi{i+1}.variables));
    eli=sumpot(multpots([eli colphi{i}]),sumover);
    bigeli=[bigeli max(eli.table(:))];
    eli.table=eli.table./bigeli(end);
end
logZ = log(table(sumpot(multpots([eli colphi{N-1}]),[],0)))+ sum(log(bigeli));
logZ
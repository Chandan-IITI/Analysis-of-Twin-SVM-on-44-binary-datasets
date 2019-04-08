function [Predict_Y,err,A,B,x1,x2] = pinTSVM(TestX,DataTrain,FunPara)
%% Reference: Xu, Yitian, Zhiji Yang, and Xianli Pan. "A novel twin support-vector machine with pinball loss."
%%IEEE transactions on neural networks and learning systems, 28 (2017): 359-370.
% % K=load('fertility.txt');
% % DataTrain=K(1:70,:);
[no_input,no_col]=size(DataTrain);
 obs =DataTrain(:,no_col);    
 A = zeros(1,no_col-1);
 B = zeros(1,no_col-1);
tic;
for i = 1:no_input
    if(obs(i) == 1)
        A = [A;DataTrain(i,1:no_col-1)];
    else
        B = [B;DataTrain(i,1:no_col-1)];
    end
end
% % FunPara=struct('c1',0.5,'c2',0.5,'v1',0.05,'v2',0.05,'kerfPara',struct('type','rbf','pars',[2^1,2^4]));
c1 = FunPara.c1;
c2 = FunPara.c2;
v1 = FunPara.v1;
v2 = FunPara.v2;
t1 = FunPara.t1;
t2 = FunPara.t2;
kerfPara = FunPara.kerfPara;
m1=size(A,1);
m2=size(B,1);
% % tau1=0.5;
% % tau2=0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(kerfPara.type,'lin')
    H=A;
    G=B;
else
   X=[A;B];
    H=kernelfun(A,kerfPara,A);
    G=kernelfun(B,kerfPara,B);
    H2=kernelfun(A,kerfPara,X);
    G2=kernelfun(B,kerfPara,X);
end
e1=ones(size(H,1),1);
e2=ones(size(G,1),1);
% % e_1=eye(size(A,2));
% % e_2=eye(size(B,2));
% % phixp=kernelfun(A,kerfPara,e_1);
% % phixn=kernelfun(B,kerfPara,e_1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute (w1) and (w2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%DTWSVM1
H1=kernelfun(A,kerfPara,B);
G1=kernelfun(B,kerfPara,A);
alpha=quadprog(H,-v1/m2*G1'*e2,[],[],e1',v1,-c1/m1*t1*e1,[]); %Sor
w1=H2'*alpha-v1/m2*G2'*e2;	
%%%%DTWSVM2
gamma=quadprog(G,-v2/m1*H1'*e1,[],[],e2',v2, -c2/m2*t2*e2,[]); %Sor
w2=-G2'*gamma+v2/m1*H2'*e1;	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute (b1) and (b2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N1=0;
N2=0;
d1=0;
d2=0;
for i=1:size(alpha,1)
    if ((0<alpha(i,1))&(alpha(i,1)<c1/m1))
        N1=N1+1;
        d1=d1+H2(i,:)*w1;
    end
end
b1=d1/N1;
for i=1:size(gamma,1)
    if ((0<gamma(i,1))&(gamma(i,1)<c2/m2))
        N2=N2+1;
        d2=d2+G2(i,:)*w2;
    end
end
b2=-d2/N2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % TestX=K(71:100,:);
[no_test,m1]=size(TestX);
if strcmp(kerfPara.type,'lin')
    P_1=TestX(:,1:m1-1);
    y1=P_1*w1+b1;
    y2=P_1*w2+b2;    
else
    C=[A;B];
    P_1=kernelfun(TestX(:,1:m1-1),kerfPara,C);
    y1=P_1*w1+b1;
    y2=P_1*w2+b2;
end
Predict_Y=zeros(size(y1,1),1);
for i=1:size(y1,1)
    if (min(abs(y1(i)),abs(y2(i)))==abs(y1(i)))
        Predict_Y(i) = 1;
    else
        Predict_Y(i) =-1;
    end
end
[no_test,no_col] = size(TestX);
x1=[]; x2 =[];err = 0.;
Predict_Y = Predict_Y';
obs1 = TestX(:,no_col);
for i = 1:no_test
    if(sign(Predict_Y(1,i)) ~= sign(obs1(i)))
        err = err+1;
    end
end  
for i=1:no_test
    if Predict_Y(1,i) ==1
        x1 = [x1; TestX(i,1:no_col-1)];
    else 
        x2 = [x2; TestX(i,1:no_col-1)];
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classification performance
% ACC=Accuracy
% SEN=Sensitivity
% SPF=Specificity
%%%%%%%%%%%%%%%%%%%%%%%%%%      
  Z=Predict_Y;
  O=Z';
  for i=1:size(O,1)
      for j=1
          if O(i,1)==1
              O(i,1)=O(i,1);
          else
              O(i,1)=-1;
          end
      end
  end
 acc=((size(TestX,1)-(err))/(size(TestX,1)))*100;
 time1=toc;
 end
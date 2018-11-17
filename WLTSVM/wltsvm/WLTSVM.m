function Predict_Y = WLTSVM(TestX,DataTrain,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WLTSVM: An efficient weighted linear loss twin support vector machine for pattern classification
%
% Predict_Y = WLTSVM(TestX,DataTrain,FunPara)
% 
% Input:
%    TestX - Test Data matrix. Each row vector of fea is a data point.
%
%    DataTrain - Struct value in Matlab(Training data).
%                DataTrain.A: Positive input of Data matrix.
%                DataTrain.B: Negative input of Data matrix.
%
%    FunPara - Struct value in Matlab. The fields in options that can be set:
%              c1,c2,c3,c4: [0,inf] Paramter to tune the weight. 
%              kerfPara:Kernel parameters. See kernelfun.m.
%
% Output:
%    Predict_Y - Predict value of the TestX.
%
% Examples:
%    DataTrain.A = rand(50,10);
%    DataTrain.B = rand(60,10);
%    TestX=rand(20,10);
%    FunPara.c1=.1;
%    FunPara.c2=.1;
%    FunPara.c3=.1;
%    FunPara.c4=.1;
%    FunPara.kerfPara.type = 'lin';
%    Predict_Y =WLTSVM(TestX,DataTrain,FunPara);
% 
% Reference:
%    Yuan-Hai Shao, Wei-Jie Chen, Zhen Wang and Nai-Yang Deng, "WLTSVM: An efficient weighted 
%    linear loss twin support vector machine for pattern classification" 
%    Submitted 2013
%
%    Version 1.0 --May/2013 
%
%    Written by Yuan-Hai Shao (shaoyuanhai21@163.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
Xpos = DataTrain.A;
Xneg = DataTrain.B;
c1 = FunPara.c1;
c2 = FunPara.c2;
c3 = FunPara.c3;
c4 = FunPara.c4;
kerfPara = FunPara.kerfPara;
[m1,n] = size(Xpos);
[m2,n] = size(Xneg);
m=m1+m2;
e1 = ones(m1,1);
e2 = ones(m2,1);
%linear kernel
if strcmp(kerfPara.type,'lin')
    Xpos=[Xpos,e1];
    Xneg=[Xneg,e2];
else
    %nonlinear kernel
    if m>1000 %reduced kernel
        TempX=[Xpos;Xneg];
        X = TempX(crossvalind('Kfold',TempX(:,1),10)==1,:);
        clear TempX;
    else
        X=[Xpos;Xneg];
    end
    Xpos=[kernelfun(Xpos,kerfPara,X),e1];
    Xneg=[kernelfun(Xneg,kerfPara,X),e2];
    TestX=kernelfun(TestX,kerfPara,X);
    X=kernelfun(X,kerfPara);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(Xpos,1)>size(Xpos,2)
    %%%%%%%%%%%%%%%%% directly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w1=-c1*( (Xpos'*Xpos)+c3*eye(size(Xpos,2)) )\Xneg'*e2*(m1/(m1+m2));
    w2=c2*( (Xneg'*Xneg)+c4*eye(size(Xpos,2)) )\Xpos'*e1*(m2/(m1+m2));
else
    %%%%%%%%%%%%%%%%% SMW fomular(if necessary) %%%%%%%%%%%%%%%%%%%%%%%%%%
    tmpr=Xpos'/(eye(size(Xpos,1))+1/c3*(Xpos*Xpos'))*Xpos/c3;
    SumG = sum(Xneg,1)';SumH = sum(Xpos,1)';
    w1=-c1/c3*(SumG-tmpr*SumG);
    tmpr=Xneg'/(eye(size(Xneg,1))+1/c4*(Xneg*Xneg'))*Xneg/c4;
    w2=c2/c4*(SumH-tmpr*SumH);
end
    %%%%%%%%%%%%%%%%% weighted process %%%%%%%%%%%%%%%%%%%%%%%%%%
    xi2=Xneg*w1+e2;
    eta1=e1-Xpos*w2;    
    J1=mean(abs(xi2));
    J2=mean(abs(eta1)); 
    for i=1:m2
        if xi2(i)>J1
            v1(i)=.00001;
        else
            v1(i)=1;
        end
    end
    for i=1:m1
        if eta1(i)>J2
            v2(i)=.00001;
        else
            v2(i)=1;
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training again
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%% directly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        w1=-c1*( (Xpos'*Xpos)+c3*eye(size(Xpos,2)) )\Xneg'*v1';
        w2=c2*( (Xneg'*Xneg)+c4*eye(size(Xpos,2)) )\Xpos'*v2';
% %%%%%%%%%%%%%%%%% SMW fomular(if necessary)  %%%%%%%%%%
%         tmpr=Xpos'/(eye(size(Xpos,1))+1/c3*(Xpos*Xpos'))*Xpos/c3;
%         SumG = sum(Xneg,1)';SumH = sum(Xpos,1)';
%         w1=-c1/c3*(SumG-tmpr*SumG);
%         tmpr=Xneg'/(eye(size(Xneg,1))+1/c4*(Xneg*Xneg'))*Xneg/c4;
%         w2=c2/c4*(SumH-tmpr*SumH);
%     end   
toc;        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    m=size(TestX,1);m1 = size(w1,1);m2 = size(w2,1);
    e = ones(m,1);
    b1 = w1(m1); b2 = w2(m2);
    w1 = w1(1:m1-1); w2 = w2(1:m2-1);
    if strcmp(kerfPara.type,'lin')        
        w11=sqrt(w1'*w1);  w22=sqrt(w2'*w2);            
    else
        w11=sqrt(w1'*X*w1);  w22=sqrt(w2'*X*w2);
    end
    Y1=TestX*w1+b1*e;  Y2=TestX*w2+b2*e;
    Y1 = Y1/w11; Y2 = Y2/w22;    
    DarwY.Y1 = Y1;
    DarwY.Y2 = Y2;
    DarwY.Y3 = abs(Y2)-abs(Y1);
    Predict_Y=sign(DarwY.Y3);
end
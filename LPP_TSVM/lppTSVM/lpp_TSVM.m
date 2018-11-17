%    ___________________________________________________________________
%
%   This is twin SVM Linear programming problem  method for nonlinear case.
%   The twin SVM formulation is considered and
%   its penalty form in its dual is formulated in 1-norm and solved 
%   using Newton method 
%   date : Dec 20, 2011
%_____________________________________________________________________
%     The formulation is very sensitive to the parameter values. For sinc
%   function the following values of the parameters, in general, give 
%   better results when 200 samples for traiing and 800 samples for testing 
%   are considered. 
 
function [classifier err] = lpp_TSVM(C,test_data,mew,nu)

   [no_input,no_col] = size(C);
   % x1 = train(:,1:no_col-1);
    obs = C(:,no_col);  %Observed values  

    A = zeros(1,no_col-1);
    B = zeros(1,no_col-1);

for i = 1:no_input
    if(obs(i) == 1)
        A = [A;C(i,1:no_col-1)];
    else
        B = [B;C(i,1:no_col-1)];
    end;
end;

     [rowA,n] = size(A);
     A = A(2:rowA,:);
     [rowB,n] = size(B);
     B = B(2:rowB,:) ; 
     alpha = 1.0; % It works when alpha = 1.0   %
     neta = 10^-5;  
     ep = 0.1; %  penality parameter   %
     tol = 0.01;
    itmax = 100;    
    beta = 10^-3;  
    
    
%    m = no_input
%      n = no_col -1
%      m2 = 2 * m  

    [m1,n] = size(A);    
    e1 = ones(m1,1); 
    [m2,n] = size(B);
     e2 = ones(m2,1);
     m= m1 + m2;
     I = speye(m);
     C = [A ; B];      
    % [m,n] = size(C);
%       C = C(:,1:no_col-1);
    K=zeros(m1,m);
    for i=1:m1
        for j=1:m
            nom = norm( A(i,:)  - C(j,:)  );
            K(i,j) = exp( -mew * nom * nom );
        end
    end
       
     G = [K e1];
     size(G);
     GT = G';
     
    
     K=zeros(m2,m);
    for i=1:m2
        for j=1:m
            nom = norm( B(i,:)  - C(j,:)  );
            K(i,j) = exp( -mew * nom * nom );
        end
    end

    H = [K e2];
    size(H); 
    HT = H';    
   
%     y = y1;       
      
     em1 = m+1;
    e = ones(em1,1);
    iter = 0;
    u1 = ones(m2,1);
    v1 = ones(m1,1);
    delphi= 1;
%     delphi_2 = ones(m,1);
%     delta = zeros(m,1);
%     i = 0;
  while( iter < itmax && norm (delphi) > tol )     
        iter = iter + 1;
        del11 = max( -HT * u1 + GT * v1 - neta * e, 0 );
        del12 = max(  HT * u1 - GT * v1 - neta * e, 0 );
        del13 = max(  -v1 - e1, 0 );
        del14 = max(  v1 - e1, 0 );
        del15 = max( u1 - nu * e2, 0 );
         
         delu1 = max( -u1, 0 );
    initial_train_time=tic;    
    delphi_1 = -ep * e2 + H * ( - del11 + del12 ) + del15 - alpha * delu1;
    delphi_2 = - del13 + del14 + G * ( del11 -  del12 );
     
         xx = diag(sign(del11)) + diag(sign(del12));
        H12 = - H * ( xx ) * GT ;
        H21 = - G * ( xx ) * HT;
        H11 = H * xx * HT + diag(sign(del15)) +  alpha * diag(sign(delu1));
        H22 = G * xx * GT + diag(sign(del13)) + diag(sign(del14)) ;
   
     hessian = [ [H11 H12]; [H21 H22] ];
     delphi = [delphi_1;delphi_2] ;
     
     delta = ( hessian + beta * I ) \ delphi ;
    u1 = u1 - delta(1:m2);
    v1 = v1 - delta (m2+1:m);
    norm(delta);
    norm(delphi);
  end
  
iter;

       del11 = max( -HT * u1 + GT * v1 - neta * e, 0 );
       del12 = max( HT * u1 - GT * v1 - neta * e, 0 );

     w1 = ( del11 - del12 ); 
    
%   _______________________________________________________________________


    iter = 0;
    u2 = ones(m1,1);
    v2 = ones(m2,1);
    delphi= 1;
%     delphi_2 = ones(m,1);
%     delta = zeros(m2,1);
%     i = 0;

     while( iter < itmax && norm (delphi) > tol )     
        iter = iter + 1;
        del11 = max( GT * u2 + HT * v2- neta * e, 0 );
        del12 = max( - GT * u2 - HT * v2 - neta * e, 0 );
        del13 = max( - v2 - e2, 0 );
        del14 = max(   v2 - e2, 0 );
        del15 = max( u2 - nu * e1, 0 );
         
        delu2 = max( -u2, 0 );
         
    delphi_1 = - ep * e1 + G * (del11 - del12) + del15 - alpha * delu2;
    delphi_2 = - del13 + del14 + H * ( del11 -  del12 );
     
        xx = diag(sign(del11)) + diag(sign(del12));
        H12 = G * (xx) * HT ;
        H21 = H * (xx) * GT;
        H11 = G * xx * GT + diag(sign(del15)) +  alpha * diag(sign(delu2));
        H22 = diag(sign(del13)) + diag(sign(del14)) + H * xx * HT ;
   
     hessian = [ [H11 H12]; [H21 H22] ];
     delphi = [delphi_1;delphi_2] ;
     
     delta = ( hessian + beta * I ) \ delphi ;
    u2 = u2 - delta(1:m1);
    v2 = v2 - delta (m1+1:m);
    norm(delta);
    norm(delphi);
     end
iter;

       del11 = max( GT * u2 + HT *v2 - neta * e, 0 );
       del12 = max( - GT * u2 - HT * v2 - neta * e, 0 );

     w2 = ( del11 - del12 );   
     time = toc(initial_train_time);
    
%% ------------------ Testing Part ------------- %%     
[no_test,no_col] = size(test_data);

  Ker_row =zeros(no_test,no_input);
 for i=1:no_test
        for j=1:no_input
      nom = norm( test_data(i,1:no_col-1)  - C(j,:)  );
      Ker_row(i,j) = exp( -mew * nom * nom );
        end 
 end
K = [Ker_row ones(no_test,1)];
size(K);
 y1 = K * w1 / norm(w1);
 y2 = K * w2 / norm(w2);
 
 for i = 1 : no_test
    if abs(y1(i)) < abs(y2(i))
        classifier(i) = 1;
    else
        classifier(i) = -1;
    end;
end;
x1 =[]; x2 =[];
for i=1:no_test
    if classifier(i) == 1
        x1 = [x1; test_data(i,1:no_col-1)];
    else 
        x2 = [x2; test_data(i,1:no_col-1)];
    end
end
%-----------------------------
err = 0.;
classifier = classifier';
obs = test_data(:,no_col);

%[test_size,n] = size(classifier);
for i = 1:no_test
    if(classifier(i) ~= obs(i))
        err = err+1;
    end
end
test_data1 =[];
test_data2 =[];

for i=1:no_test
    if obs(i)== 1
     test_data1 = [test_data1; test_data(i,1:no_col-1)];
    else
     test_data2 = [test_data2; test_data(i,1:no_col-1)];
    end
end

%% To check the sparseness 
count=0;
index=0;
   for i= 1:size(w1,1)
       if(w1(i)>10^-2 && w2(i)>10^-2)
           count=count+1;
           index(count)=i;
       end
   end
% count = 0;
% for i=1:size(w1,1)
%     if w1(i) ~=0 & w2(i) ~=0
%         count = count+1
%     end
% end
% count
% 
% count1 = 0;
% for i=1:size(w1,1)
%     if w1(i) ~=0 
%         count1 = count1+1;
%     end
% end
% 
% count2 = 0;
% for i=1:size(w2,1)
%     if w2(i) ~=0
%         count2 = count2+1;
%     end
% end
    
    
     
     

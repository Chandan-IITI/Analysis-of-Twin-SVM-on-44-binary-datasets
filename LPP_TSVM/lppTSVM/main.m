    clc;
    clear all;
    close all;
    file1 = fopen('results_TSVM.txt','w+');
    file2 = fopen('results_c_mu.txt','w+');
   
%------------------------------------------------%
 for load_file = 1:1   
    switch load_file
        case 1
            file = 'tic',            
               mews=2^-3;
               cvs=10^1;
            test_start = 672; 
        case 2
            file = 'iono',           
               mews=2^-2;
               cvs=10^0;
            test_start = 247;     
         case 3
             file = 'bupa',             
               mews=2^2;
               cvs=10^2;
             test_start = 242;   
         case 4
            file= 'votes'
             mews=2^-7;
             cvs=10^1;
             test_start = 307;  
         case 5
            file= 'wpbc'           
              mews= 2^-3;
              cvs=10^-1;
            test_start = 138;  
       case 6
            file= 'pima'           
            mews= 2^1;
            cvs=10^1;            
            test_start = 538;     
       case 7
           file= 'splice'       
              mews=2^-6;
              cvs=10^-5;
           test_start = 501;  
      case 8
            file= 'cleve'
             mews= 2^-6;
             cvs=10^0;  
             test_start = 178;     
     case 9
            file= 'ger'          
            mews=2^-4;
            cvs=1;   
            test_start = 801;    
      case 10
             file= 'aus'          
            mews=2^-7;
            cvs=10^1; % 540:150 
           test_start = 541;  
     case 11
            file = 'haberman';
            cvs= 10^0;
            mews= 2^1;
             test_start = 201;  
     case 12
            file = 'transfusion';
            cvs= 10^2;
            mews= 2^-6;
             test_start = 601;    
     case 13
            file = 'wdbc';
            cvs= 10^1;
            mews= 2^-2;
             test_start = 501; 
     case 14
           file= 'Heart-c';       
            test_start = 178; 
             cvs= 10^0;
             mews= 2^0;       
     case 15
            file = 'monks-1'; % its a special dataset, cant change test_size
            cvs= 10^-3;
            mews= 2^-2;
             test_start = 125; 
     case 17
            file = 'monk2';
            cvs= 10^-1;
            mews= 2^-6;
            test_start = 170; % its a special dataset, cant change test_size   
     case 18
            file = 'monk3';
            cvs= 10^-4;
            mews= 2^-2;
             test_start = 123; % its a special dataset, cant change test_size   
     case 19
            file = 'heart-stat';
            cvs= 10^0;
            mews= 2^-2;
             test_start = 201; 
     case 20
           file= 'sonar'; 
           cvs= 10^-5;
           mews= 2^-1;
            test_start = 151; 
     case 21
           file= 'cmc';
           cvs= 10^1;
           mews= 2^0;
            test_start = 1001;     
     case 22
           file= 'Ripley';       
            test_start = 251; 
%      case 23
%            file= 'crossplane150' 
%             test_start = 81;
      
     case 23
           file= 'ndc500';  %Only NDC datasets, we normalize in standard manner (not scalling)        
           test_start = 501;  
    case 24
           file= 'ndc1k';          
           test_start = 1001; 
    case 25
           file= 'ndc2k';          
           test_start = 2001; 
    case 26
           file= 'ndc3k';          
           test_start = 3001; 
    case 27
           file= 'ndc5k';          
           test_start = 5001;        
    case 28
           file= 'ndc8k';          
           test_start = 8001;   
    case 29
           file= 'ndc10k';          
           test_start = 10001;
    case 30
           file= 'ndc50k';          
           test_start = 50001;      
     
         
              
              
      
              
              otherwise
            continue;
    end
%                mews=[2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10];
%                cvs=[10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
            cvs= 10^1;
            mews= 2^-4;

    A =  load(strcat('../dataset/',file,'.txt') );
    [m,n] = size(A);    
%      A(:,1:n-1) = normalize (A(:, 1:n-1));   
    test_data = A(test_start:m,:);
    if test_start > 1
    train_data = A(1:test_start-1,:);         
    end  
%     noise = normrnd(0,0.2,size(train_data,1),1);
%     train_data(:,1:n-1) = train_data(:,1:n-1) +noise;
    [m,n] = size(train_data);
    noise = (-0.1 + 0.2 * rand(m,n-1) );
    train_data(:,1:n-1) = train_data(1:m,1:n-1)+ noise;
     train_data(:,1:n-1) = scale (train_data(:, 1:n-1));
     test_data(:,1:n-1) = scale (test_data(:, 1:n-1));   
%      train_data(:,1:n-1) = normalize (train_data(:, 1:n-1));
%      test_data(:,1:n-1) = normalize (test_data(:, 1:n-1)); 
     
         t = cputime;   

% --------------------------------------------------------------------------     
%      no_part = 10.;
%      [m,n] = size(train_data);
%     % initialize minimum error variable and corresponding c
%     min_c=1.;
%     min_err=1000000000000000.;
%     min_mu=1.;
%      for mui=1:length(mews)
%         %for different values of mu
%         mu=mews(mui);
% 
%         for ci=1:length(cvs)
%                 c=cvs(ci);
%             %for different values of c
%     %             c=cvs(ci);
%             %training statement
%             block_size=m/(no_part*1.0);
%             part=0;
%             avgerr=0;
%             t_1=0;
%             t_2=0;
%             while ((part+1)* block_size) <= m
%                 t_1 = ceil(part*block_size);
%                 t_2 = ceil((part+1)*block_size);                
%                 Data_test= train_data(t_1+1: t_2,:); 
%                 Data =[train_data(1:t_1,:); train_data(t_2+1:m,:)];
%                               
%                 [err] = lpp_TSVM(Data,Data_test,mu,c); %call for training and testing
% %                 fprintf(file2, 'example file %s; err= %8.6g, part num= %8.6g, mu= %8.6g, c= %8.6g\n', file,err,part,mu,c);
%                 avgerr = avgerr + err;
%                 part=part+1;
%             end
%             %testing statement
%             %for particular c and for particular file
%              fprintf(file2, 'example no: %s\t avgerr: %g\t mu=%g\t c=%g\n',file, avgerr,mu,c);
%              if avgerr < min_err
%                  min_c=c;
%                  min_err=avgerr;
%                  min_mu=mu;
%              end
%        end %for c values
%     end %for mu values
%     %final training
% %     t=cputime;
   
                min_mu = mews;
                min_c = cvs;
                 
%   Replace comments by uncomments and vice-versa before this.
%   _______________________________________________________________________
  [err,x1,x2,test_data1,test_data2,A,B,w1,w2,count,time] = lpp_TSVM(train_data,test_data,min_mu,min_c);
   fprintf(file1,'example file: %s;err = %8.6g of %g,mu= %8.6g,c = %8.6g\n', file,err,length(test_data(:,1)),min_mu,min_c);
fclose(file1);
  file1=fopen('results_TSVM.txt','a+'); 

%  plot(test_data1(:,1),test_data1(:,2),'r-',test_data2(:,1),test_data2(:,2),'b-',A(:,1),A(:,2),'bo',B(:,1),B(:,2),'ko')
plot(A(:,1),A(:,2),'b*',B(:,1),B(:,2),'k*',test_data1(:,1),test_data1(:,2),'ro',test_data2(:,1),test_data2(:,2),'go')
 end   
    fclose(file1);
    fclose(file2);
%................complete code.............................%    
            

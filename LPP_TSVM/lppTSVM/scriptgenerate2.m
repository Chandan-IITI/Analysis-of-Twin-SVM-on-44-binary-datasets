%% To generate crossplane dataset
clc;
clear all;
close all;
class1 = 75;
class2 = 75;
for load_file = 1:1    %% initializing variables case level and loading file
    switch load_file
        case 1
           file = 'crossplane';
                  x = rand(class1,1);
                  p = rand(class2,1); 
%                   noise = normrnd(0,0.05,train_size,1);              
                  x1 = -2+x*4; %between -2 and 2
                  y1 = x1 + 1;               
                  a1 = [x1,y1,ones(class1,1)];
                  save crossplaneclass1.txt a1 -ASCII;                  
                  p1 = -2+p*4;                 
                  q1 = -p1 + 2;
                  b1 = [p1,q1,-ones(class2,1) ];
                  save crossplaneclass2.txt b1 -ASCII;
				  b = [a1;b1];
				  save crossplane.txt b -ASCII;   
                  
                  
                 % B= b(randperm(150),:) where b is the dataset  
                         
                  
                  
                  
                  
                  
             
      
    end
end

clear
clc

names = importdata('names.txt');

for datanum = 1:length(names)
    datanum
          if(~ismember(datanum, [4 10 25 57 59 65 67 90 94 111]))   %%% 44 Datasets remain after removing these time cosuming datasets 

name = names{datanum};
acc = 0;
try
acc = seperate_eval(name);
catch
    try
   acc = kfold_eval(name); 
    catch
        disp('data belongs to multi-class classification or dataset not found')
        continue
    end
end
  
if acc ~= 0
    xlRange1 = ['A' num2str(datanum)];
    xlswrite('all_results.xlsx', {name}, 1, xlRange1);
    xlRange2 = ['B' num2str(datanum)];
    xlswrite('all_results.xlsx', acc*100, 1, xlRange2);
end
    
end

end
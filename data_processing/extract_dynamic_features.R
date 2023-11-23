rm(list=ls())
args=(commandArgs(TRUE))
if(length(args)==0){
  print("No arguments supplied.")
  ##supply default values
}else{
  for(i1 in 1:length(args)){
    eval(parse(text=args[[i1]]))
  }
}

## A shell script was used to pass the following values to this R script
## For example, to extract dynamic features for a specific variable in the dataset in scenario_2/fold_0, use
## scenario = 1
## fold = 0 
## k = 1 # which variable to extract dynamic features for

path = "/gpfs/group/quc16/default/EPiC_2023/"
setwd(paste0(path,"feature_engineering"))
library(data.table)#faster way to read csv 
library(parallel)# parallel computation 
library(foreach)# parallel computation 
library(doParallel )# parallel computation 
library(tsfeatures)
library(rsample)
library(runner)
#library(nonlinearTseries)
library(zoo)
#library(DataExplorer)# Automate Data Exploration
library(dplyr)
library(lubridate)
#library(factoextra)
#library(e1071)
library(vars)#Information criteria and FPE for different VAR(p)
#library(caret)


################################################################################################################################################
#Generate Dynamic Feature 


Dynamic_feature<-function(df,variable){ 
  fea_name<-paste0(paste0(variable,"_"),c("entropy","max_level_shift","max_var_shift","max_kl_shift","hurst","spike",
              "arch_acf","garch_acf","ARCH.LM","std1st_der","mean"))
  
  tep<-c()
  ## window size: use 10s as the window size; time interval in the dataset = 50ms
  for (i in 1:1){
    tep1<-401
    tep<-c(tep,paste0(fea_name,"_",tep1[i]))
  }
  fea_name<-tep
  rm(tep1,tep)
  # n_lag_max<-0
  # for (i in 1:max(test$sleepPeriod)){
  #   
  #   tep<-c()
  #   cnt<-test$actactivity[test$sleepPeriod==i] %>% ts()
  #   
  #   #n_lag<-max(estimateEmbeddingDim(cnt),timeLag(cnt,technique="ami"),timeLag( cnt,technique="acf"),na.rm = T)
  #   #get the optimal lag 
  #   n_lag<-VARselect(cnt)  
  #   n_lag<-n_lag$selection %>% max()
  #   n_lag_max<-max(n_lag_max,n_lag)
  # }
  
  #n_lag=n_lag_max
  n_lag = 10
  
  fea_table_final<-data.frame(matrix(data = NA,ncol=1+length(fea_name)+n_lag,nrow=1))
  colnames(fea_table_final)<-c("Value",fea_name,paste0(paste0(variable,"_mean_lag",seq(20, 200, 200/n_lag))))
  
  
  for (vv in unique(df$VIDEO_FILENAME)){
    df_video<-df[df$VIDEO_FILENAME==vv,]
    fea_table<-data.frame(matrix(data = NA,ncol=1+length(fea_name)+n_lag,nrow=nrow(df_video)))
    colnames(fea_table)<-c("Value",fea_name,paste0(paste0(variable,"_mean_lag",seq(20, 200, 200/n_lag))))
    fea_table$Value<-df_video[,variable]
    
    
    #fill the table
    #entropy
    circle<-401
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%entropy() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_entropy_",circle[i]))]<-tep
    }
    
    #max_level_shift.
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%max_level_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_level_shift)%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_max_level_shift_",circle[i]))]<-tep
    }
    #max_var_shift
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%max_var_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_var_shift)%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_max_var_shift_",circle[i]))]<-tep
    }
    
    #max_kl_shift
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%max_kl_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_kl_shift)%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_max_kl_shift_",circle[i]))]<-tep
    }
    
    #hurst
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%hurst()%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_hurst_",circle[i]))]<-tep
    }
    
    #spike
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%stl_features()%>%t() %>% as.data.frame()%>%dplyr:: select(spike)%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_spike_",circle[i]))]<-tep
      
    }
    
    #arch_acf/garch_acf_/arch_r2_/garch_r2_
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) {if (var(x)>0){
                       ts(x)%>%heterogeneity()%>%t() %>% as.data.frame()%>%as.numeric() }else{NA}})
      
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_arch_acf_",circle[i]))]<-tep[,1]
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_garch_acf_",circle[i]))]<-tep[,2]
      
    }
    
    #ARCH.LM
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%arch_stat()%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_ARCH.LM_",circle[i]))]<-tep
    }
    
    #"std1st_der"
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) ts(x)%>%std1st_der()%>%as.numeric() )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_std1st_der_",circle[i]))]<-tep
    }
    
    
    
    
    #"actimove"
    
    #tep<-rollapply(fea_table$Value, width = 9,
    #               FUN = function(x) sum(x[1:2],x[8:9])/25+ sum(x[3:4],x[6:7])/5+2*x[5] )
    #fea_table[(9%/%2+1):((nrow(fea_table)-(9%/%2))),which(colnames(fea_table)=="actimove")]<-tep
    
    #"mean"
    for (i in 1:length(circle)){
      tep<-rollapply(fea_table$Value, width = circle[i],
                     FUN = function(x) mean(x) )
      fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0(variable,"_mean_",circle[i]))]<-tep
    }
  
    #lag
    for (ll in 1:n_lag){
      
      fea_table[,which(colnames(fea_table)==paste0(variable,"_mean_lag",20*ll))]<-lag(fea_table[,which(colnames(fea_table)==paste0(variable,"_mean_",circle[i]))],20*ll)
    }
    
    fea_table_final<-rbind(fea_table_final,fea_table)
  } 
  
  fea_table_final<-fea_table_final[-1,]
  #fea_table_final<-cbind(fea_table_final[,-ncol(fea_table_final)],df[,-1])
  fea_table_final<-cbind(fea_table_final[,-1],df[,c("time","unique_number")])
  return(fea_table_final)
}


############################################################################################################

feature = c("EDA_Tonic","EDA_Phasic","ECG_Rate","PPG_Rate","RSP_Rate","EMG_Amplitude")
dfall=fread(paste0(path,"train_data/train_scenario_",scenario,"_fold_",fold,".csv"))
dfall=data.frame(dfall)
ID = unique(dfall$ID) 

df = dfall[dfall$ID == ID[id],]
train = Dynamic_feature(df, feature[k])
print(ID[id])
#Output processed feature 
fwrite(train,paste0(path,"feature_engineering/scenario",scenario,"/fold",fold,"/train_scenario_",scenario,"_fold_",fold,"_sub_",id,"_",feature[k],".csv"))
  




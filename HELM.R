library(pracma)

zscore = function(x){
  return((x-mean(x))/sd(x))
}

obtain_label_matrix = function(labels, num_classes){
  label_matrix = matrix(0, nrow=length(labels), ncol=num_classes)
  for(i in 1:num_classes){
    label_matrix[which(labels==i), i] = 1
  }
  return(label_matrix)
}

mapminmax = function(x){
  return(2*(x-min(x))/(max(x)-min(x))-1)
}

sparse_elm_autoencoder = function(A, B, lam, itrs){
  AA = t(A) %*% A
  Lf = max(eigen(AA)$val)
  Li = 1 / Lf
  alp = lam * Li
  m = ncol(A)
  n = ncol(B)
  x = matrix(0, nrow=m, ncol=n)
  yk = x
  tk = 1
  L1 = 2 * Li * AA
  L2 = 2 * Li * t(A) %*% B
  for(i in 1:itrs){
    ck = yk - L1 %*% yk + L2
    dk = abs(ck) - alp
    dk[dk < 0] = 0
    x1 = dk * sign(ck)
    tk1 = 0.5 + 0.5 * sqrt(1 + 4 * tk^2)
    tt = (tk - 1) / tk1
    yk = x1 + tt * (x - x1)
    tk = tk1
    x = x1
  }
  return(x)
}

minmax_matrix = function(input){
  rows = nrow(input)
  cols = ncol(input)
  min_matrix = diag(apply(input, 2, min), cols, cols)
  max_matrix = diag(apply(input, 2, max), cols, cols)
  out_matrix = (input-matrix(1,nrow=rows,ncol=cols)%*%min_matrix)/(matrix(1,nrow=rows,ncol=cols)%*%max_matrix-matrix(1,nrow=rows,ncol=cols)%*%min_matrix)
  return(list(out_matrix, max_matrix, min_matrix))
}

tansig<-function(x){
  x_nrow<-nrow(x)
  x_ncol<-ncol(x)
  y<-matrix(nrow=x_nrow,ncol=x_ncol)
  for(i in 1:x_nrow){
    for(j in 1:x_ncol){
      y[i,j]<-2/(1+exp(-2*x[i,j]))-1
    }
  }
  return(y)
}

obtained_acc_G_mean<-function(x){
  the_sum<-0
  the_G_mean<-1
  for(i in 1:nrow(x)){
    the_sum<-the_sum+x[i,i]
    the_G_mean<-the_G_mean*(x[i,i]/sum(x[i,]))
  }
  the_acc<-the_sum/sum(x)
  the_G_mean<-the_G_mean^(1/nrow(x))
  return(list(the_acc*100,the_G_mean*100))
}

model = function(N1, N2, N3, C, S, num_classes, train_path, samples_number = 0, test_path = "src", single = FALSE){
  if (single) {
    total_data = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
  } else {
    data_train = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    data_test = read.table(test_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    total_data = rbind(data_train, data_test)
    samples_number = nrow(data_train)
  }
  variables_number = ncol(total_data) - 1
  total_data$label = as.numeric(total_data$label)
  total_data_normial = as.data.frame(lapply(total_data[,c(1:variables_number)], mapminmax))
  total_data_normial = as.data.frame(t(as.matrix(as.data.frame(lapply(as.data.frame(t(as.matrix(total_data_normial))), zscore)))))
  total_data = cbind(total_data_normial, total_data[variables_number+1])
  training_data = total_data[c(1:samples_number),]
  testing_data = total_data[-c(1:samples_number),]
  
  training_start_time = Sys.time()
  training_data_variables = as.matrix(training_data[, c(1:variables_number)])
  training_data_labels = obtain_label_matrix(training_data$label, num_classes)
  
  H1 = cbind(training_data_variables, 0.1)
  B1 = matrix(runif((variables_number+1)*N1,min=-1,max=1),nrow=variables_number+1,ncol=N1)
  A1 = t(apply(t(H1 %*% B1), 2, mapminmax))
  Beta1 = t(sparse_elm_autoencoder(A1, H1, 0.001, 50))
  T1 = H1 %*% Beta1
  results1 = minmax_matrix(T1)
  T1 = results1[[1]]
  max1 = results1[[2]]
  min1 = results1[[3]]
  
  H2 = cbind(T1, 0.1)
  B2 = matrix(runif((N1+1)*N2,min=-1,max=1),nrow=N1+1,ncol=N2)
  A2 = t(apply(t(H2 %*% B2), 2, mapminmax))
  Beta2 = t(sparse_elm_autoencoder(A2, H2, 0.001, 50))
  T2 = H2 %*% Beta2
  results2 = minmax_matrix(T2)
  T2 = results2[[1]]
  max2 = results2[[2]]
  min2 = results2[[3]]
  
  H3 = cbind(T2, 0.1)
  B3 = t(orth(t(matrix(runif((N2+1)*N3,min=-1,max=1),nrow=N2+1,ncol=N3))))
  T3 = H3 %*% B3
  L3 = S / max(T3)
  T3 = tansig(T3 * L3)
  
  Beta = solve(diag(x=C,ncol(T3),ncol(T3))+t(T3)%*%T3,tol=eps(0.002))%*%t(T3)%*%training_data_labels
  training_end_time = Sys.time()
  
  testing_start_time = Sys.time()
  testing_data_variables = as.matrix(testing_data[,c (1:variables_number)])
  num_test = nrow(testing_data_variables)
  HH1 = cbind(testing_data_variables, 0.1)
  TT1 = HH1 %*% Beta1
  TT1 = (TT1-matrix(1,nrow=num_test,ncol=N1)%*%min1)/(matrix(1,nrow=num_test,ncol=N1)%*%max1-matrix(1,nrow=num_test,ncol=N1)%*%min1)
  
  HH2 = cbind(TT1, 0.1)
  TT2 = HH2 %*% Beta2
  TT2 = (TT2-matrix(1,nrow=num_test,ncol=N2)%*%min2)/(matrix(1,nrow=num_test,ncol=N2)%*%max2-matrix(1,nrow=num_test,ncol=N2)%*%min2)
  
  HH3 = cbind(TT2, 0.1)
  TT3 = tansig(HH3 %*% B3 * L3)
  aim_result = as.data.frame(TT3 %*% Beta)
  aim_result$result<-0
  for(i in 1:nrow(aim_result)){
    aim_result[i,ncol(aim_result)]<-which.max(aim_result[i,c(1:(ncol(aim_result)-1))])
  }
  testing_end_time = Sys.time()
  
  table0<-table(testing_data$label,aim_result$result)
  final_result<-obtained_acc_G_mean(table0)
  Acc<-final_result[[1]]
  Gmean<-final_result[[2]]
  training_time = as.numeric(as.character(difftime(training_end_time, training_start_time, units="secs")))
  testing_time = as.numeric(as.character(difftime(testing_end_time, testing_start_time, units="secs")))
  print(c(Acc,Gmean,training_time,testing_time))
  comp<-data.frame(Acc,Gmean,training_time,testing_time)
  names(comp)<-c("Acc","Gmean","training_time","testing_time")
  saver<-read.table("D:/program/data.csv",header=TRUE,sep=",")
  saver<-rbind(saver,comp)
  write.csv(saver,"D:/program/data.csv",row.names=FALSE)
}

for (number in 1:5){
  model(200, 200, 2000, 10^-10, 1, 26, "D:/program/letter.csv", 16000, single = TRUE)
}


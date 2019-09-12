import numpy as np
import math as mt
#读取并处理数据的函数
def fun_read(file_name,p,q):#p是样本数是矩阵的列 q是特征数是矩阵的行
	file = open(file_name,'r') #只读模式打开file
	all_txt = file.read()  
	k=1
	sourceInLine=[]
	sourceInLine.insert(0,all_txt)
	for line in sourceInLine:
		temp1=line.strip('\n')
		temp2=temp1.replace(':1','')	
		temp3=temp2.split('\n')#对字符串操作分割得到的是list
	i=0
	temp4=[]
	X=np.mat(np.zeros((p,q)))
	Y=np.mat(np.zeros((p,1)))
	R=np.mat(np.zeros((i,i)))
	for line in temp3:
		temp3[i]=temp3[i].strip(' ')
		temp4=temp3[i].split(' ')
		#print('temp4',temp4)
		k=0
		for line2 in temp4:
			if k==0:
				if int(temp4[k])==1:
					Y[i,0]=int(temp4[k])
				else:
					Y[i,0]=int(temp4[k])+1
			else:
				X[i,int(temp4[k])-1]=1
			k=k+1
		i=i+1
	return X,Y

#求一阶导函数
def gradient_figure(r,X,Y,W):  
	U=1/(1+np.exp(-W.T*X))
	P=Y-U
	W_11=X*P.T-1/2*r*W
	return W_11,U,P  #返回一阶偏导矩阵以及U（用于计算二阶导）

#求hessian矩阵
def hessian_figure(r,i,X,U): 
	R=np.mat(np.zeros((i,i)))
	k=0
	while k<=i-1:
		R[k,k]=U[0,k]*(1-U[0,k])-r
		k=k+1
	H=-X*R*X.T
	return H #返回二阶偏导矩阵
if __name__ == "__main__":
	#X是矩阵，列数代表总的样本数，行代表特征个数，
	#其中 Xi是纵向量，代表第i个数据的特征数据
	#Y是横向量，列数代表总的样本数，一行代表类别
	#i代表总的样本数,t代表特征总数
	r=float(input('输入正则化的系数'))
	r2=float(input('输入收敛精度'))
	#训练
	i=16281
	t=123
	(X,Y)=fun_read('a9a.t',i,t)
	X=X.T
	Y=Y.T
	t=t+1
	b=np.mat(np.ones((1,i)))
	X=np.row_stack((X,b))
	W=np.mat(np.zeros((t,1)))
	W3=np.mat(np.ones((t,1)))
	iden=np.mat(np.eye((t)))
	P=np.mat(np.ones((1,i)))
	l=0#表示迭代次数，超过10次判定为发散
	while (abs(np.sum(P))>=r2) and (l<=10):
		(W_11,U,P)=gradient_figure(r,X,Y,W)
		H=hessian_figure(r,i,X,U)+0.001*iden
		W2=W
		W=W-H.I*W_11
		W3=abs(W-W2)
		print('第',l+1,'次迭代所有样本的预估与实际差值之和',np.sum(P))
		l=l+1
	if l==11:
		print('正则化系数过大不收敛，请调整系数大小或者收敛精度')
		exit()
	else:
	#training data
		p=0
		num_accur_training=0
		U1=1/(1+np.exp(-W.T*X))
		while p<=i-1:
			if (U1[0,p]>=0.5 and Y[0,p]) or (U1[0,p]<0.5 and abs((Y[0,p]-1))):
				num_accur_training=num_accur_training+1
			p=p+1
		accuracy_trainingdata=num_accur_training/i
		print('the accuracy of training data is',accuracy_trainingdata)


	#testing data
		i=32561
		t=123
		(X,Y)=fun_read('a9a',i,t)
		X=X.T
		Y=Y.T
		b=np.mat(np.ones((1,i)))
		X=np.row_stack((X,b))
		p=0
		num_accur_testing=0
		U1=1/(1+np.exp(-W.T*X))
		while p<=i-1:
			if (U1[0,p]>=0.5 and Y[0,p]) or (U1[0,p]<0.5 and abs((Y[0,p]-1))):
				num_accur_testing=num_accur_testing+1
			p=p+1
		accuracy_testingdata=num_accur_testing/i
		print('the accuracy of testing data is',accuracy_testingdata)


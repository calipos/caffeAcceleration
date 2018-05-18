import caffe
import numpy as np
from collections import OrderedDict
import copy

def deleteArray(A,nums=[],channels=[]):
	assert len(A.shape)==4
	B=np.array([])
	C=np.array([])
	D=np.array([])
	if len(nums)>0:
		for n in range(A.shape[0]):
			if n in nums:continue
			if B.shape[0]==0 : B=A[n].reshape(1,A[n].shape[0],A[n].shape[1],A[n].shape[2])
			else : B= np.vstack((B,A[n].reshape(1,A[n].shape[0],A[n].shape[1],A[n].shape[2])))
		C=B
	else:
		C=A
	if len(channels)>0:
		C=C.transpose(1,0,2,3)
		for c in range(C.shape[0]):
			if c in channels:continue
			if D.shape[0]==0 : D=C[c].reshape(1,C[c].shape[0],C[c].shape[1],C[c].shape[2])
			else : D= np.vstack((D,C[c].reshape(1,C[c].shape[0],C[c].shape[1],C[c].shape[2])))
		D=D.transpose(1,0,2,3)
		return D
	else :
		return C
	return A

def getNewWeight(net,thisConv,formConvRemainChannelsFlags=[]):
	#print thisConv
	#print formConvRemainChannelsFlags
	thisRemianChannels=[]
	if len(formConvRemainChannelsFlags)!=0:
		for i in range(len(formConvRemainChannelsFlags)):
			thisRemianChannels=thisRemianChannels+formConvRemainChannelsFlags[i]


	layer1_blobs=net._layer_by_name(thisConv)
	weights1_=layer1_blobs.blobs[0].data
	if len(thisRemianChannels)!=0 : assert len(thisRemianChannels) == weights1_.shape[1]
	
	delFlags=[]
	for i in range(len(thisRemianChannels)):
		if thisRemianChannels[i]<0 :delFlags.append(i)

	weights1=deleteArray(weights1_,[],delFlags)

	bias1=np.array([])
	dim1 = weights1.shape
	kernel_result_list=[]
	for i in range(dim1[0]):
		thisKernel=weights1[i].reshape(dim1[1]* dim1[2] * dim1[3])
		kernel_result_list.append( np.std(thisKernel,ddof=1))
	#print (kernel_result_list)
	#print weights1[1]
	sorted_list=sorted(kernel_result_list,reverse=True)
	#print sorted_list
	#exit(0)
	threshold = sorted_list[int(len(sorted_list)*0.8)]-1e-7
	#print threshold
	#exit(0)
	has_bias1=len(layer1_blobs.blobs)==2
	if has_bias1:
		bias1=layer1_blobs.blobs[1].data
	newWeight1=np.array([])
	newbias1=[]
	output_remain=[]
	for i in range(dim1[0]):
		if kernel_result_list[i]>threshold:
			output_remain.append(1)
			if newWeight1.shape[0]==0:
				newWeight1=weights1[i].reshape(1,weights1[i].shape[0],weights1[i].shape[1],weights1[i].shape[2])
			else:
				newWeight1= np.vstack((newWeight1,weights1[i].reshape(1,weights1[i].shape[0],weights1[i].shape[1],weights1[i].shape[2])))
			if has_bias1 :
				newbias1.append(bias1[i])
		else:
			output_remain.append(-1)
			continue
	newbias1=np.array(newbias1)
	
	return (newWeight1,newbias1,has_bias1,output_remain)


def getNewWeight_onlyFixChannels(net,thisConv,formConvRemainChannelsFlags=[]):
	#print thisConv
	#print formConvRemainChannelsFlags
	thisRemianChannels=[]
	if len(formConvRemainChannelsFlags)!=0:
		for i in range(len(formConvRemainChannelsFlags)):
			thisRemianChannels=thisRemianChannels+formConvRemainChannelsFlags[i]

	layer1_blobs=net._layer_by_name(thisConv)
	weights1_=layer1_blobs.blobs[0].data
	delFlags=[]
	for i in range(len(thisRemianChannels)):
		if thisRemianChannels[i]<0 :delFlags.append(i)

	weights1=deleteArray(weights1_,[],delFlags)
	bias1=np.array([])
	dim1 = weights1.shape

	has_bias1=len(layer1_blobs.blobs)==2
	if has_bias1:
		bias1=layer1_blobs.blobs[1].data
	return (weights1,bias1,has_bias1,[])


def getFixedLaterWeight(net,laterConv,output_remain):
	layer2_blobs=net._layer_by_name(laterConv)
	#print "newWeight1.shape = ",newWeight1.shape
	#print "newbias1.shape   = ",newbias1.shape
	#print newWeight1
	#print "output_remain.shape = ",output_remain
	weights2=layer2_blobs.blobs[0].data
	dim2 = weights2.shape
	#print "weight2.shape  = ",dim2
	newWeight2=np.array([])
	for i in range(dim2[0]):
		thisInput=weights2[i,output_remain[0]].reshape(1,1,dim2[2],dim2[3])
		#print thisInput.shape
		for j in range(len(output_remain)-1):
			#print thisInput.shape
			thisInput= np.concatenate((thisInput,weights2[i,output_remain[j+1]].reshape(1,1,dim2[2],dim2[3])),axis=1)
		if newWeight2.shape[0]==0:	newWeight2=thisInput
		else:newWeight2= np.vstack((newWeight2,thisInput))
	#print newWeight2.shape
	return (newWeight1,newbias1,has_bias1,newWeight2)

def modifiyProto(protoFile,changedConv,output,deployName):
	fileObj=open(protoFile,'r')
	lines=fileObj.readlines()
	fileObj.close()
	fileTempObj=open(deployName,'w')
	startLine=0
	endLine=0
	for i,line in enumerate(lines):
		if line.find("\""+changedConv+"\"")!=-1 and line.find("name")!=-1: 
			startLine=i
			break
	isModify=False
	for i,line in enumerate(lines):
		if i<startLine :
			fileTempObj.write(line)
			
		elif isModify==False and line.find("num_output")!=-1:
			fileTempObj.write("    num_output : %d\n"%output)
			isModify=True
		else:
			fileTempObj.write(line)
	fileTempObj.close

def setWeight(formerConv,has_bias,W1,b,caffeModelName,originalModel="",originalWeight=""):


	#for i in range(len(net._layer_names)):
	#	print net._layer_names[i]
	#	print len(net._layer_by_name(net._layer_names[i]).blobs)
	
	
	net2=caffe.Net("tempdeploy.prototxt",caffe.TEST)
	layer1_blobs=net2._layer_by_name(formerConv)
	#print len(layer1_blobs.blobs)
	#print layer1_blobs.blobs[0].shape[0],layer1_blobs.blobs[0].shape[1],layer1_blobs.blobs[0].shape[2],layer1_blobs.blobs[0].shape[3]

	#print layer1_blobs.blobs[0].data[0]
	#print W1[0]
	num=layer1_blobs.blobs[0].num
	channels=layer1_blobs.blobs[0].channels
	height=layer1_blobs.blobs[0].height
	width=layer1_blobs.blobs[0].width
	for n in range(num):
		for c in range(channels):
			for h in range(height):
				for w in range(width):
					layer1_blobs.blobs[0].data[n,c,h,w]=W1[n,c,h,w]
	#print layer1_blobs.blobs[0].data[0]
	if has_bias:
		num=layer1_blobs.blobs[1].num
		for n in range(num):
			layer1_blobs.blobs[1].data[n]=b[n]
			
			
			
	if originalModel!="" and originalWeight!="":
		net_original=caffe.Net(originalModel,originalWeight,caffe.TEST)
		for olayerIdx in range(len(net_original._layer_names)):
			originalLayerName =  net_original._layer_names[olayerIdx]
			for nlayerIdx in range(len(net2._layer_names)):
				newLayerName =  net2._layer_names[nlayerIdx]
				if originalLayerName==newLayerName :
					oriLayer = net_original._layer_by_name(originalLayerName)
					newLayer = net2._layer_by_name(originalLayerName)
					assert len(oriLayer.blobs)==len(newLayer.blobs)
					match=True
					for blobIdx in range(len(newLayer.blobs)):
						match=oriLayer.blobs[blobIdx].shape == newLayer.blobs[blobIdx].shape
					if match!=True:break
					else:
						print originalLayerName,"not changed so copy the original weights to it"
						for blobIdx in range(len(newLayer.blobs)):
							num=oriLayer.blobs[blobIdx].num
							channels=oriLayer.blobs[blobIdx].channels
							height=oriLayer.blobs[blobIdx].height
							width=oriLayer.blobs[blobIdx].width
							for n in range(num):
								for c in range(channels):
									for h in range(height):
										for w in range(width):
											newLayer.blobs[blobIdx].data[n,c,h,w]=oriLayer.blobs[blobIdx].data[n,c,h,w]
	net2.save(caffeModelName)

def updateDict(dict,layerName,output):
	print layerName
	print output
	for key in dict.keys():
		for idx in range(len(dict[key].fromName)):
			if layerName == dict[key].fromName[idx]:
				dict[key].fromOutput[idx]=output


class peleeUnit:
	idx=0
	def __init__(self,name,fromName,fromOutput,endFlag=0):
		self.name=name
		self.fromName=fromName
		self.fromOutput=fromOutput
		self.endFlag=endFlag
		self.idx=peleeUnit.idx
		#print self.idx
		peleeUnit.idx+=1
		assert len(self.fromName)==len(self.fromOutput)
	def __cmp__(self,other):
		if self.idx<other.idx:
			return -1
		elif self.idx==other.idx:
			return 0
		else:return 1

vgg_weight_pair=[["conv1_1","conv1_2"],
                ["conv1_2","conv2_1"],
                ["conv2_1","conv2_2"],
                ["conv2_2","conv3_1"],
                ["conv3_1","conv3_2"],
                ["conv3_2","conv3_3"],
                ["conv3_3","conv4_1"],
                ["conv4_1","conv4_2"],
                ["conv4_2","conv4_3"],
                ["conv4_3","conv5_1"],
                ["conv5_1","conv5_2"],
                ["conv5_2","conv5_3"],
                ["conv5_3","fc6"],
                ["fc6","fc7"],
				]


pelee_nobn_unit=OrderedDict()
pelee_nobn_unit["stem2a"]=peleeUnit("stem2a",["stem1"],[[]])
pelee_nobn_unit["stem2b"]=peleeUnit("stem2b",["stem2a"],[[]])
pelee_nobn_unit["stem2b"]=peleeUnit("stem2b",["stem2a"],[[]])
pelee_nobn_unit["stem3"]=peleeUnit("stem3",["stem1","stem2b"],[[],[]])

pelee_nobn_unit["stage1_1/branch1a"]=peleeUnit("stage1_1/branch1a",["stem3"],[[]])
pelee_nobn_unit["stage1_1/branch1b"]=peleeUnit("stage1_1/branch1b",["stage1_1/branch1a"],[[]])
pelee_nobn_unit["stage1_1/branch2a"]=peleeUnit("stage1_1/branch2a",["stem3"],[[]])
pelee_nobn_unit["stage1_1/branch2b"]=peleeUnit("stage1_1/branch2b",["stage1_1/branch2a"],[[]])
pelee_nobn_unit["stage1_1/branch2c"]=peleeUnit("stage1_1/branch2c",["stage1_1/branch2b"],[[]])

pelee_nobn_unit["stage1_2/branch1a"]=peleeUnit("stage1_2/branch1a",["stem3","stage1_1/branch1b","stage1_1/branch2c"],[[],[],[]])
pelee_nobn_unit["stage1_2/branch1b"]=peleeUnit("stage1_2/branch1b",["stage1_2/branch1a"],[[]])
pelee_nobn_unit["stage1_2/branch2a"]=peleeUnit("stage1_2/branch2a",["stem3","stage1_1/branch1b","stage1_1/branch2c"],[[],[],[]])
pelee_nobn_unit["stage1_2/branch2b"]=peleeUnit("stage1_2/branch2b",["stage1_2/branch2a"],[[]])
pelee_nobn_unit["stage1_2/branch2c"]=peleeUnit("stage1_2/branch2c",["stage1_2/branch2b"],[[]])

pelee_nobn_unit["stage1_3/branch1a"]=peleeUnit("stage1_3/branch1a",["stem3","stage1_1/branch1b","stage1_1/branch2c","stage1_2/branch1b","stage1_2/branch2c"],[[],[],[],[],[]])
pelee_nobn_unit["stage1_3/branch1b"]=peleeUnit("stage1_3/branch1b",["stage1_3/branch1a"],[[]])
pelee_nobn_unit["stage1_3/branch2a"]=peleeUnit("stage1_3/branch2a",["stem3","stage1_1/branch1b","stage1_1/branch2c","stage1_2/branch1b","stage1_2/branch2c"],[[],[],[],[],[]])
pelee_nobn_unit["stage1_3/branch2b"]=peleeUnit("stage1_3/branch2b",["stage1_3/branch2a"],[[]])
pelee_nobn_unit["stage1_3/branch2c"]=peleeUnit("stage1_3/branch2c",["stage1_3/branch2b"],[[]])

pelee_nobn_unit["stage1_tb"]=peleeUnit("stage1_tb",["stem3","stage1_1/branch1b","stage1_1/branch2c","stage1_2/branch1b","stage1_2/branch2c","stage1_3/branch1b","stage1_3/branch2c"],[[],[],[],[],[],[],[]],1)

if __name__ == '__main__':


	caffe.set_mode_gpu()
	caffe.set_device(0)
	
	deployprototxt="/media/hdd/lbl_trainData/dataBase/ssdpelee/pelee_nobn.prototxt"
	caffemodel="/media/hdd/lbl_trainData/dataBase/ssdpelee/pelee_nobn.caffemodel"
	
	trainprototxt="/media/hdd/lbl_trainData/dataBase/ssdpelee/pelee_nobn_train.prototxt"
	testprototxt="/media/hdd/lbl_trainData/dataBase/ssdpelee/pelee_nobn_test.prototxt"
	
	
	net=caffe.Net(deployprototxt,caffemodel,caffe.TEST)

	
	W1,b,has_bias,remain_out=getNewWeight(net,"stem1")
	modifiyProto(deployprototxt,"stem1",len([x for x in remain_out if x>0]),"tempdeploy.prototxt")
	modifiyProto(trainprototxt,"stem1",len([x for x in remain_out if x>0]),"temptrain.prototxt")
	modifiyProto(testprototxt,"stem1",len([x for x in remain_out if x>0]),"temptest.prototxt")
	
	setWeight("stem1",has_bias,W1,b,"tempWeight.caffemodel")
	#pelee_nobn_unit["stem2a"].fromOutput[0]+=remain_out
	updateDict(pelee_nobn_unit,"stem1",remain_out)
	
	#exit(0)
	
	for key in pelee_nobn_unit.keys():
		print "********************pruning this layer : ",key
		net2=caffe.Net("tempdeploy.prototxt","tempWeight.caffemodel",caffe.TEST)
		isEnd=pelee_nobn_unit[key].endFlag==1
		if isEnd==False:
			W1,b,has_bias,remain_out=getNewWeight(net,pelee_nobn_unit[key].name,pelee_nobn_unit[key].fromOutput)
			modifiyProto("tempdeploy.prototxt",pelee_nobn_unit[key].name,len([x for x in remain_out if x>0]),"tempdeploy.prototxt")
			modifiyProto("temptrain.prototxt",pelee_nobn_unit[key].name,len([x for x in remain_out if x>0]),"temptrain.prototxt")
			modifiyProto("temptest.prototxt",pelee_nobn_unit[key].name,len([x for x in remain_out if x>0]),"temptest.prototxt")
			setWeight(pelee_nobn_unit[key].name,has_bias,W1,b,"tempWeight.caffemodel")
			updateDict(pelee_nobn_unit,pelee_nobn_unit[key].name,remain_out)
			for key in pelee_nobn_unit.keys():print "##",pelee_nobn_unit[key].name," ",pelee_nobn_unit[key].fromName," ",pelee_nobn_unit[key].fromOutput
		else:
			W1,b,has_bias,remain_out=getNewWeight_onlyFixChannels(net,pelee_nobn_unit[key].name,pelee_nobn_unit[key].fromOutput)
			#modifiyProto("tempPrototxt.prototxt",pelee_nobn_unit[key].name,len([x for x in remain_out if x>0]),"tempPrototxt.prototxt")
			setWeight(pelee_nobn_unit[key].name,has_bias,W1,b,"tempWeight.caffemodel",    deployprototxt,caffemodel)
			updateDict(pelee_nobn_unit,pelee_nobn_unit[key].name,remain_out)
			for key in pelee_nobn_unit.keys():print "##",pelee_nobn_unit[key].name," ",pelee_nobn_unit[key].fromName," ",pelee_nobn_unit[key].fromOutput
			print "finished"
			exit(0)
	
	
	

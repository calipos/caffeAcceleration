# 1 merge_bn 
移动 到caffe的相关位置，在makefile中，开启c++11编译。 
其中 bn前一层的conv只能含有weights，不能含有bias。 
示例 参数全都写死在了代码里。 
merge bn的时候，注意看清原prototxt中bn参数的eps！
 
# 2 svd分解 
把一个卷积核通过svd分解成两个小的卷积核，并且通过sigma阀值来保留其前80%的能力。 
分解 的卷积核非1x1，并且只分解bn前的卷积，利用其没有偏值得特点，使分解简单。 
分解后 需要finetune，这里我把分解的第一个小卷积核固定，不让其随loss更新，而第二个小卷积核更新，因为它后面紧跟bn。

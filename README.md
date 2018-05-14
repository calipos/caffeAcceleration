# merge_bn

 
移动 到caffe的相关位置，在makefile中，开启c++11编译。
 
其中 bn前一层的conv只能含有weights，不能含有bias。
 
示例 参数全都写死在了代码里。
 
merge bn的时候，注意看清原prototxt中bn参数的eps！
 

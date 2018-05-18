在 python文件中，要调用一个_layer_by_name 函数，所以需要在caffe_root/python/caffe/_caffe.cpp中，在net类中再导出一个方法：


      .def("_layer_by_name", &Net<Dtype>::layer_by_name)

from tensorboardX import SummaryWriter
with SummaryWriter() as w:
    # https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/mobilenetv2-int8-sparse-v1-tf-0001/FP32/mobilenetv2-int8-sparse-v1-tf-0001.xml
    w.add_openvino_graph('examples/mobilenetv2.xml')


import caffe
import scipy.io
    
class DebugLayer(caffe.Layer):
    """ """
    def setup(self, bottom, top): pass
    def reshape(self, bottom, top): pass 

    def backward(self, top, propagate_down, bottom): 
        """
        capture backward diffs
        """
        import ipdb ; ipdb.set_trace()
    
    def forward(self, bottom, top):

        var_list = ["images",
        "image_ids",
        'labels',
        'cues',
        "conv1_1",
        "conv1_2",
        "pool1",
        "conv2_1",
        "conv2_2",
        "pool2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "pool3",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "pool4",
        "conv5_1",
        "conv5_2",
        "conv5_3",
        "pool5",
        "pool5a",
        "fc6",
        "fc7",
        "fc8-SEC",
        'fc8-SEC-Softmax',
        'fc8-SEC-CRF-log',
        "loss-Expand",
        "loss-Seed",
        "loss-Constrain",
         ]
        dest = '/tmp/store.mat'
        store = {}
        for ii, var in enumerate(var_list):
            new_name = var.replace('-', '_')
            store[new_name] = bottom[ii].data
        scipy.io.savemat(dest, store)
        print('saving blobs to {}'.format(dest))

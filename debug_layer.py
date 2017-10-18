import caffe
import json
    
class DebugLayer(caffe.Layer):
    """ """
    
    def setup(self, bottom, top):
        pass
        # assert len(bottom) == 2,    'requires two layer.bottoms'
        # assert len(top) == 1,       'requires a single layer.top'
    
        # if hasattr(self, 'param_str') and self.param_str:
            # params = json.loads(self.param_str)
        # else:
            # params = {}
    
        # self.top_k = params.get('top_k', 1)
    
    def reshape(self, bottom, top):
        pass 
    
    def forward(self, bottom, top):
        import ipdb ; ipdb.set_trace()
        # Renaming for clarity
        predictions = bottom[0].data
        ground_truth = bottom[1].data
    
        num_correct = 0.0
    
        # NumPy magic - get top K predictions for each datum
        top_predictions = (-predictions).argsort()[:, :self.top_k]
        for batch_index, predictions in enumerate(top_predictions):
            if ground_truth[batch_index] in predictions:
                num_correct += 1
    
        # Accuracy is averaged over the batch
        top[0].data[0] = num_correct / len(ground_truth)
    
    def backward(self, top, propagate_down, bottom):
        pass

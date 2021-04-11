import tensorflow as tf

# Creating Custom Metric
class CustomIOU(tf.keras.metrics.Metric):
    def __init__(self, n_classes, threshold = None, smooth= 1.0, average='macro', weight = None, name = 'weighted_iou', **kwargs):
        super().__init__(name = name, **kwargs)
        self.n_classes = n_classes
        self.smooth = smooth
        self.threshold = threshold
        self.average = average
        if weight is None:
            self.weight = tf.ones((n_classes,), tf.float32)
        else:
            self.weight = tf.convert_to_tensor(weight, tf.float32)
            
        self.sum = self.add_weight(name = 'sum', initializer = 'zeros', shape = (self.n_classes,))
        self.total = self.add_weight(name = 'total', initializer = 'zeros')
        self.h_iou = self.add_weight(name = 'h_iou', initializer = 'zeros')
    
    def update_state(self, y_true, y_pred, sample_weight = None): 
        y_true = tf.cast(tf.cast(y_true, tf.int32), tf.float32)
        if self.threshold is not None:
            y_pred = tf.cast(tf.cast(y_pred > self.threshold, tf.int32), tf.float32)
        else:
            y_pred = tf.cast(tf.one_hot( tf.argmax(y_pred, axis = -1), self.n_classes  ), tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2])
        union = tf.reduce_sum(y_true, [1,2]) + tf.reduce_sum(y_pred, [1,2]) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        self.sum.assign_add(tf.reduce_sum(iou, axis = 0))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], "float32"))
            
        if self.average == 'macro':
            self.h_iou.assign(tf.math.divide_no_nan(tf.reduce_sum(self.weight), tf.reduce_sum(tf.math.divide_no_nan( self.weight, tf.math.divide_no_nan(self.sum, self.total) ))))
        elif self.average == 'micro':
            self.h_iou.assign( tf.reduce_mean(self.weight * tf.math.divide_no_nan(self.sum, self.total)) )
        else:
            raise NameError
        
    def result(self):
        return self.h_iou
    
    def reset_states(self):
        self.sum.assign(self.sum * 0.0)
        self.total.assign(0.0)
        self.h_iou.assign(0.0)


if __name__ == "__main__":
    from unet import get_unet_128
    import numpy as np
    model = get_unet_128(num_classes = 3)
    x = np.random.uniform(0,1, (10,128,128,3))
    target = np.random.randint(0,2, (10,128,128,3))
    prediction = model(x, training = False)

    m = CustomIOU(3)
    m.update_state(target, prediction)
    print(m.result())

import tensorflow as tf

class CustomWeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, epsilon = 1e-9, name = 'wce'):
        super().__init__(name=name)
        self.class_weights = tf.convert_to_tensor(class_weights, tf.float32)
        self.epsilon = epsilon
    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        return -tf.reduce_mean(tf.reduce_sum(self.class_weights* y_true* tf.math.log(y_pred + self.epsilon), axis = [-1]))


class CustomMyDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1.0 ,name = 'my_dice_loss'):
        super().__init__(name = name)
        self.smooth = smooth

    def compute_dice_coef(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
        union = tf.reduce_sum(y_true, [1,2]) + tf.reduce_sum(y_pred, [1,2])
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return tf.reduce_mean(score, axis = -1)

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        score  = self.compute_dice_coef(y_true, y_pred)
        return 1 - score

class CustomDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1.0 ,name = 'dice_loss'):
        super().__init__(name = name)
        self.smooth = smooth

    def compute_dice_coef(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true, [1,2,3]) + tf.reduce_sum(y_pred, [1,2,3])
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return score

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        score  = self.compute_dice_coef(y_true, y_pred)
        return 1 - score



class CustomTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha = 0.7, smooth = 1.0 ,name = 'tversky_loss'):
        super().__init__(name = name)
        self.smooth = smooth
        self.alpha = alpha

    def compute_tversky_index(self, y_true, y_pred):
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)

        ti = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return ti

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        score  = self.compute_tversky_index(y_true, y_pred)
        return 1 - score
        

class CustomFocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha = 0.7, gamma = 0.75, smooth = 1.0 ,name = 'focal_tversky'):
        super().__init__(name = name)
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def compute_tversky_index(self, y_true, y_pred):
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)

        ti = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return ti

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        score  = self.compute_tversky_index(y_true, y_pred)
        return tf.pow(1-score, self.gamma)



class CustomLogDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1.0 ,name = 'log_dice_loss'):
        super().__init__(name = name)
        self.smooth = smooth

    def compute_dice_coef(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true, [1,2,3]) + tf.reduce_sum(y_pred, [1,2,3])
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return score

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        score  = self.compute_dice_coef(y_true, y_pred)
        return tf.math.log((tf.exp(1-score) + tf.exp(-(1-score))) / 2.0)
        

        



if __name__ == "__main__":
    from unet import get_unet_128
    import numpy as np
    model = get_unet_128(num_classes = 3)
    x = np.random.uniform(0,1, (10,128,128,3))
    target = np.random.randint(0,2, (10,128,128,3))
    prediction = model(x, training = True)


    l = CustomWeightedCategoricalCrossentropy([1,1,1])
    print(l(target, prediction))

    l = tf.keras.losses.CategoricalCrossentropy()
    print(l(target, prediction))

    l = CustomDiceLoss()
    print(l(target, prediction))

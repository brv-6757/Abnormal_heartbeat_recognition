import numpy as np
target_samples = 188
original_samples = 100
original_indices = np.linspace(0, 1, original_samples)
target_indices = np.linspace(0, 1, target_samples)
def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

extra = samples[idx].reshape(1,100)
linear_interpolated = np.zeros((extra.shape[0], target_samples))
for i in range(extra.shape[0]):
    for j in range(target_samples):
        left_index = np.searchsorted(original_indices, target_indices[j]) - 1
        right_index = left_index + 1
        
        if left_index < 0:
            left_index = 0
            right_index = 1
        if right_index >= original_samples:
            right_index = original_samples - 1
            left_index = right_index - 1
        
        t = (target_indices[j] - original_indices[left_index]) / (original_indices[right_index] - original_indices[left_index])
        interpolated_value = (1 - t) * extra[i, left_index] + t * extra[i, right_index]
        
        linear_interpolated[i, j] = interpolated_value * 3.3 / 4095
    linear_interpolated[i] = normalize(linear_interpolated[i])

class AutoEncoder:
	def __init__(self):
		self.input_dim = 188
		self.latent_dim = 4
		self.l1 = 4*2
        
        self.conv1_weights = np.load("wts/en_conv_1_wts.npy")  
        self.conv1_bias = np.load("wts/en_conv_1_b.npy")  
        self.conv2_weights = np.load("wts/en_conv_2_wts.npy")  
        self.conv2_bias = np.load("wts/en_conv_2_b.npy")
        
        self.conv3_weights = np.load("wts/de_conv_1_wts.npy")  
        self.conv3_bias = np.load("wts/de_conv_1_b.npy")
        self.conv4_weights = np.load("wts/de_conv_2_wts.npy")  
        self.conv4_bias = np.load("wts/de_conv_2_b.npy")

        self.bn1 = np.load("wts/en_bn1.npy")
        self.bn2 = np.load("wts/en_bn2.npy")
        self.bn3 = np.load("wts/de_bn1.npy")
        self.bn4 = np.load("wts/de_bn2.npy")

        self.dense_weights = np.load("wts/de_dense_wts.npy")
        self.dense_bias = np.load("wts/de_dense_b.npy")

    def relu(self, x):
        return np.maximum(0, x) 

    def batch_normalization(self,x, gamma, beta, moving_mean, moving_variance, epsilon=1e-5):
        gamma = gamma.reshape(1, 1, -1)
        beta = beta.reshape(1, 1, -1)
        moving_mean = moving_mean.reshape(1, 1, -1)
        moving_variance = moving_variance.reshape(1, 1, -1)
        
        normalized = (x - moving_mean) / np.sqrt(moving_variance + epsilon)
        output = gamma * normalized + beta
        return output

    def conv1d_manual(self, x, weights,bias, strides=1, padding='same'):
        kernel_size = weights.shape[0]
        if padding == 'same':
            pad_size = kernel_size // 2
            x = np.pad(x, ((0, 0), (pad_size, pad_size), (0, 0)), 'constant')

        output_length = (x.shape[1] - kernel_size) // strides + 1
        output_channels = weights.shape[2]
        conv_output = np.zeros((x.shape[0], output_length, output_channels))

        for i in range(output_length):
            start = i * strides
            end = start + kernel_size
            for j in range(output_channels):
                conv_output[:, i, j] = np.sum(x[:, start:end, :] * weights[:, :, j], axis=(1, 2))+bias[j]
                
        return conv_output

    def max_pooling1d(self, x, pool_size=2, padding='same'):
        if padding == 'same' and x.shape[1] % pool_size != 0:
            x = np.pad(x, ((0, 0), (0, 1), (0, 0)), 'constant')

        output_length = x.shape[1] // pool_size
        pooled_output = np.zeros((x.shape[0], output_length, x.shape[2]))

        for i in range(output_length):
            start = i * pool_size
            end = start + pool_size
            pooled_output[:, i, :] = np.max(x[:, start:end, :], axis=1)
        
        return pooled_output

    def up_sampling1d(self, x, size=2):
        return np.repeat(x, size, axis=1)
    
    def dense(self, x, weights,bias):
        return x @ weights + bias
    
    def encode(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        X = self.conv1d_manual(X, self.conv1_weights,self.conv1_bias)
        X = self.relu(X)
        X = self.batch_normalization(X, self.bn1[0], self.bn1[1],self.bn1[2],self.bn1[3])  # Pass gamma and beta for scaling and shifting
        X = self.max_pooling1d(X)

        X = self.conv1d_manual(X, self.conv2_weights,self.conv2_bias)
        X = self.relu(X)
        X = self.batch_normalization(X, self.bn2[0], self.bn2[1],self.bn2[2],self.bn2[3])
        X = self.max_pooling1d(X)
        return X

    def decode(self, X):
        X = self.conv1d_manual(X, self.conv3_weights,self.conv3_bias)
        X = self.relu(X)
        X = self.up_sampling1d(X)
        X = self.batch_normalization(X, self.bn3[0], self.bn3[1],self.bn3[2],self.bn3[3])
        X = self.conv1d_manual(X, self.conv4_weights,self.conv4_bias)
        X = self.relu(X)
        X = self.up_sampling1d(X)
        X = self.batch_normalization(X, self.bn4[0], self.bn4[1],self.bn4[2],self.bn4[3])
        X = X.reshape(X.shape[0], -1)
        X = self.dense(X, self.dense_weights,self.dense_bias)
        
        return X
	
	def call(self, X):
		encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded

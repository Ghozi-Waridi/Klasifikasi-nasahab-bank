import importlib.util

import numpy as np


if importlib.util.find_spec("tqdm.auto") is not None:
    tqdm = __import__("tqdm.auto", fromlist=["tqdm"]).tqdm
elif importlib.util.find_spec("tqdm") is not None:
    tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
else:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class NeuralNetwork:

    
    def __init__(self, layer_dims, random_seed=42):
 
        np.random.seed(random_seed)
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1 
        self.params = {}
        self.cache = {}

        self._initialize_parameters()


        self.adam_state = self._initialize_adam()
        self.t = 0  
    
    def _initialize_parameters(self):
        """He Initialization untuk weights dan zeros untuk bias."""
        for l in range(1, self.L + 1):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]
            
            
            self.params[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            self.params[f'b{l}'] = np.zeros((n_out, 1))
    
    def _initialize_adam(self):
        """Inisialisasi momen pertama (m) dan kedua (v) untuk Adam."""
        adam_state = {}
        for l in range(1, self.L + 1):
            adam_state[f'mW{l}'] = np.zeros_like(self.params[f'W{l}'])
            adam_state[f'mb{l}'] = np.zeros_like(self.params[f'b{l}'])
            adam_state[f'vW{l}'] = np.zeros_like(self.params[f'W{l}'])
            adam_state[f'vb{l}'] = np.zeros_like(self.params[f'b{l}'])
        return adam_state
    
    @staticmethod
    def relu(Z):

        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z):

        return (Z > 0).astype(float)
    
    @staticmethod
    def sigmoid(Z):

        Z = np.clip(Z, -500, 500) 
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_derivative(Z):

        s = NeuralNetwork.sigmoid(Z)
        return s * (1 - s)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):

        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def forward_propagation(self, X):

        cache = {}
        A = X.T 
        
        for l in range(1, self.L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            Z = W @ A + b 
            
            cache[f'Z{l}'] = Z
            cache[f'A{l-1}'] = A
            
            if l < self.L:
                A = self.relu(Z) 
            else:
                A = self.sigmoid(Z)  
        
        AL = A.flatten()  
        return AL, cache
    
    def backward_propagation(self, y_true, AL, cache):

        grads = {}
        m = len(y_true)  
        eps = 1e-8

        AL_clipped = np.clip(AL, eps, 1 - eps)
        dAL = -(y_true / AL_clipped - (1 - y_true) / (1 - AL_clipped)) / m

        Z_L = cache[f'Z{self.L}']
        dZ = dAL * self.sigmoid_derivative(Z_L.flatten())
        dZ = dZ.reshape(1, -1)

        for l in range(self.L, 0, -1):
            A_prev = cache[f'A{l-1}']
            W = self.params[f'W{l}']
            
            grads[f'dW{l}'] = dZ @ A_prev.T  
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)  
            
            if l > 1:
                
                dA_prev = W.T @ dZ
                Z_prev = cache[f'Z{l-1}']
                dZ = dA_prev * self.relu_derivative(Z_prev)
        
        return grads
    
    def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.t += 1
        beta1_t = beta1 ** self.t
        beta2_t = beta2 ** self.t
        
        for l in range(1, self.L + 1):
            for p in ['W', 'b']:
                g = grads[f'd{p}{l}']
                m_ = self.adam_state[f'm{p}{l}']
                v_ = self.adam_state[f'v{p}{l}']
                
                
                m_new = beta1 * m_ + (1 - beta1) * g
                v_new = beta2 * v_ + (1 - beta2) * (g ** 2)
                self.adam_state[f'm{p}{l}'] = m_new
                self.adam_state[f'v{p}{l}'] = v_new
                
                
                m_hat = m_new / (1 - beta1_t)
                v_hat = v_new / (1 - beta2_t)
                
                
                self.params[f'{p}{l}'] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def predict(self, X):

        y_prob, _ = self.forward_propagation(X)
        return y_prob
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=64, lr=0.001, 
              patience=10, verbose=True):
      
        
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        w_pos = (n_neg + n_pos) / (2 * n_pos)
        w_neg = (n_neg + n_pos) / (2 * n_neg)
        
        best_val_loss = np.inf
        patience_cnt = 0
        best_params = None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': []
        }
        
        n_batches = int(np.ceil(len(y_train) / batch_size))
        
        epoch_iterator = tqdm(range(1, epochs + 1), desc="Training", unit="epoch", leave=True, disable=not verbose)

        for epoch in epoch_iterator:
            
            idx = np.random.permutation(len(y_train))
            X_shuf, y_shuf = X_train[idx], y_train[idx]
            
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                Xb, yb = X_shuf[start:end], y_shuf[start:end]
                
                
                AL, cache = self.forward_propagation(Xb)
                
                
                eps_ = 1e-8
                AL_c = np.clip(AL, eps_, 1 - eps_)
                w_vec = np.where(yb == 1, w_pos, w_neg)
                
                
                grads = self.backward_propagation(yb, AL, cache)
                
                
                self.update_adam(grads, lr=lr)
            
            
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            
            train_loss = self.binary_cross_entropy(y_train, train_pred)
            val_loss = self.binary_cross_entropy(y_val, val_pred)
            train_acc = np.mean((train_pred >= 0.5) == y_train)
            val_acc = np.mean((val_pred >= 0.5) == y_val)
            
            
            order = np.argsort(val_pred)[::-1]
            y_sorted = y_val[order]
            n_pos_v = y_sorted.sum()
            n_neg_v = len(y_sorted) - n_pos_v
            val_auc = (np.cumsum(1 - y_sorted)[y_sorted == 1].sum() /
                      (n_pos_v * n_neg_v + 1e-8)) if n_pos_v > 0 and n_neg_v > 0 else 0.5
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)

            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}"
            )
            
            
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_params = {k: v.copy() for k, v in self.params.items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:>3}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            
            if patience_cnt >= patience:
                if verbose:
                    print(f"\n⏹Early stopping pada epoch {epoch} "
                          f"(val_loss tidak membaik selama {patience} epoch)")
                break
        
        
        if best_params is not None:
            self.params = best_params
        
        return history
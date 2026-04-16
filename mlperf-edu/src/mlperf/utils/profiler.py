import numpy as np
from typing import Tuple, Dict, Any

class FLOPCounter:
    """
    Calculates theoretical FLOPs and Memory R/W bytes for ML layers.
    Specifically tuned for TinyTorch educational components.
    """
    
    @staticmethod
    def count_linear(in_features: int, out_features: int, batch_size: int = 1) -> Dict[str, int]:
        """
        Linear Layer: y = xW + b
        FLOPs: 2 * batch * in * out (Multiply-Accumulate)
        Bytes: batch*in (read x) + in*out (read W) + out (read b) + batch*out (write y)
        """
        # 2 FLOPs per multiply-add
        flops = 2 * batch_size * in_features * out_features
        
        # Assume float32 (4 bytes)
        bytes_read = (batch_size * in_features + in_features * out_features + out_features) * 4
        bytes_written = (batch_size * out_features) * 4
        
        return {
            "flops": flops,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "total_bytes": bytes_read + bytes_written
        }

    @staticmethod
    def count_conv2d(in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
                     input_size: Tuple[int, int], output_size: Tuple[int, int], 
                     batch_size: int = 1, bias: bool = True) -> Dict[str, int]:
        """
        Conv2d Layer FLOPs and Memory.
        FLOPs: 2 * batch * out_c * out_h * out_w * (in_c * k_h * k_w)
        """
        kh, kw = kernel_size
        oh, ow = output_size
        ih, iw = input_size
        
        # 2 FLOPs per multiply-add in the convolution dot product
        flops = 2 * batch_size * out_channels * oh * ow * (in_channels * kh * kw)
        
        # Memory
        bytes_read = (batch_size * in_channels * ih * iw + # input
                      out_channels * in_channels * kh * kw + # weights
                      (out_channels if bias else 0)) * 4 # bias
        bytes_written = (batch_size * out_channels * oh * ow) * 4 # output
        
        return {
            "flops": flops,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "total_bytes": bytes_read + bytes_written
        }

    @staticmethod
    def get_op_metrics(op_name: str, input_shapes: Any, output_shape: Any, dtype: str = "float32") -> Dict[str, int]:
        """
        Calculates FLOPs and Bytes for a single tensor operation.
        """
        # Assume float32 (4 bytes)
        bytes_per_elem = 4
        
        # Calculate elements in input and output
        # input_shapes is a list of tuples
        input_elements = [int(np.prod(shape)) for shape in input_shapes]
        output_elements = int(np.prod(output_shape))
        
        flops = 0
        if op_name == "matmul":
            # (M, K) @ (K, N) -> (M, N)
            # FLOPs = 2 * M * N * K
            m_shape, k_shape = input_shapes[0], input_shapes[1]
            if len(m_shape) >= 2 and len(k_shape) >= 2:
                M, K = m_shape[-2], m_shape[-1]
                N = k_shape[-1]
                # Account for batch dimensions if any
                batch = int(np.prod(m_shape[:-2])) if len(m_shape) > 2 else 1
                flops = 2 * batch * M * N * K
            else:
                # 1D dot product or other cases
                flops = 2 * max(input_elements)
        elif op_name in ["add", "sub", "mul", "div", "add_scalar", "sub_scalar", "mul_scalar", "div_scalar"]:
            flops = output_elements
        elif op_name in ["sum", "mean"]:
            flops = int(np.prod(input_shapes[0]))
        elif op_name == "max":
            flops = int(np.prod(input_shapes[0]))
        elif op_name == "conv2d":
            # (B, Cin, Hi, Wi) and (Cout, Cin, Kh, Kw) -> (B, Cout, Ho, Wo)
            x_shape, w_shape = input_shapes[0], input_shapes[1]
            batch = x_shape[0]
            in_c = x_shape[1]
            out_c = w_shape[0]
            kh, kw = w_shape[2], w_shape[3]
            oh, ow = output_shape[2], output_shape[3]
            # 2 FLOPs per multiply-add
            flops = 2 * batch * out_c * oh * ow * (in_c * kh * kw)
        
        # Memory estimation
        bytes_read = sum(input_elements) * bytes_per_elem
        bytes_written = output_elements * bytes_per_elem
        
        return {
            "flops": flops,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "total_bytes": bytes_read + bytes_written
        }

    @staticmethod
    def get_layer_metrics(layer: Any, x_shape: Tuple[int, ...], y_shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Introspects a TinyTorch layer to calculate its performance metrics.
        """
        layer_name = layer.__class__.__name__
        batch_size = x_shape[0]
        
        if layer_name == "Linear":
            in_f = layer.in_features if hasattr(layer, 'in_features') else x_shape[-1]
            out_f = layer.out_features if hasattr(layer, 'out_features') else y_shape[-1]
            return FLOPCounter.count_linear(in_f, out_f, batch_size)
            
        elif layer_name == "Conv2d":
            in_c = layer.in_channels
            out_c = layer.out_channels
            k_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            
            # TinyTorch shapes are typically (B, C, H, W)
            input_size = (x_shape[2], x_shape[3])
            output_size = (y_shape[2], y_shape[3])
            
            return FLOPCounter.count_conv2d(in_c, out_c, k_size, input_size, output_size, batch_size, bias=(layer.bias is not None))
            
        # Default for activations etc (approximation: 1 FLOP per element)
        elements = int(np.prod(y_shape))
        return {
            "flops": elements,
            "bytes_read": elements * 4,
            "bytes_written": elements * 4,
            "total_bytes": elements * 8
        }

Quantizer config can be modified from 
'quantizer_config'           : {
                                        'kernel_quantizer':  'quantized_bits(6,1,alpha=1)',  // for QConv2DBatch and QDenseBatch/ QDense
                                        'bias_quantizer':  'quantized_bits(6,1,alpha=1)',    // for QConv2DBatch and QDenseBatch/ QDense
                                        'q_activation': 'quantized_relu(bits=6, integer=1)', // for usual relu activation
                                        'activation_bits': 'quantized_bits(14,4,1,alpha=1)', // for quantizing input layers and other layers that would affect the precision 
                                        'final_q_activation': 'quantized_relu(bits=14, integer=6)' // final relu
                                        }
in exp_electorns/param_conf.py.

Remember to set use_hls4ml=True to apply the quantizer config.

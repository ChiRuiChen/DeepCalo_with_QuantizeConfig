def get_params():
    """
    Returns a dictionary containing parameters to be passed to the model
    container.

    Please see the README for documentation.
    """

    params = {
        
        # Quantize
        'use_hls4ml'                 : True,  
        'quantizer_config'           : {
                                        'kernel_quantizer':  'quantized_bits(6,1,alpha=1)', # ap8,3
                                        'bias_quantizer':  'quantized_bits(6,1,alpha=1)', # ap8,3
                                        'q_activation': 'quantized_relu(bits=6, integer=1)',
                                        'activation_bits': 'quantized_bits(14,4,1,alpha=1)', # ap16,6
                                        'final_q_activation': 'quantized_relu(bits=14, integer=6)' #ap 16,8
                                        },
                                        
        # Training
        'epochs'                     : 300,
        'batch_size'                 : 1024,
        'loss'                       : 'logcosh',
        'metrics'                    : ['mae'],
        'optimizer'                  : 'Nadam',
        'lr_finder'                  : {'use':True,
                                      'scan_range':[1e-5, 1e-2],
                                      'epochs':1,
                                      'prompt_for_input':False},
        'lr_schedule'                : {'name':'CLR',
                                      'range':[5e-4, 7e-2],
                                      'step_size_factor':3,
                                      'kwargs':{}},
        'auto_lr'                    : True,

        # Misc.
        'use_earlystopping'          : False,
        'restore_best_weights'       : True,
        'pretrained_model'           : {'use':False,
                                      'weights_path':'path/to/weights',
                                      'params_path':None,
                                      'layers_to_load':['top', 'cnn', 'FiLM_generator', 'scalar_net'],
                                      'freeze_loaded_layers':False},
        'upsampling'                 : {'use':True,
                                      'wanted_size':(56,55)},

        # Submodels
        'top'                        : {'activation':'leakyrelu',
                                      'normalization':'batch',
                                      'units':[256,256,1],
                                      'final_activation':'relu'},
        'cnn'                        : {'activation':'leakyrelu',
                                      'normalization':'batch',
                                      'block_depths':[1,2,2,2,2],
                                      'n_init_filters':16,
                                      'downsampling':'maxpool',
                                      'min_size_for_downsampling':6},
        'scalar_net'                 : {'activation':'leakyrelu',
                                      'normalization':'batch',
                                      'units':[256],
                                      'connect_to':['FiLM_gen']},
        'track_net'                  : {'activation':'leakyrelu',
                                      'normalization':'batch',
                                      'phi_units':[128,128],
                                      'rho_units':[128,128],
                                      'apply_mask': False,
                                      'connect_to':['FiLM_gen']},
        'FiLM_gen'                   : {'activation':'leakyrelu',
                                      'normalization':'batch',
                                      'use':True,
                                      'units':[512,1024]},
          }

    return params





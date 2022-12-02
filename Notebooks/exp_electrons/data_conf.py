def get_params():
    """
    Returns a dictionary containing parameters to be passed to the loading
    function.
    """

    params = {
              'target_factor'          : 1,           #scale down the targets
              'target_name'            : 'p_truth_E',
              'img_names'              : ['em_endcap'], # NOTE: Should actually be the added barrel and endcap layers!
              'gate_img_prefix'        : None,
              'scalar_names'           : [ # Set to None if you only want to use the images
                                          'p_eAccCluster',
                                          'p_cellIndexCluster',
                                          'p_f0Cluster',
                                          'p_R12',
                                          'p_pt_track',
                                          'p_nTracks',
                                          'p_eta',
                                          'p_deltaPhiRescaled2',
                                          'p_etaModCalo',
                                          'p_deltaEta2',
                                          'NvtxReco',
                                          'averageInteractionsPerCrossing',
                                          'p_poscs2',
                                          'p_dPhiTH3',
                                          'p_fTG3',
                                          ],
              'track_names'            : [ # Set to None if you don't want to use tracks.
                                         'tracks_pt', # It is recommended to also add p/q and d0/sigma_d0
                                         'tracks_eta',
                                         'tracks_z0',
                                         'tracks_d0',
                                         'tracks_dR',
                                         'tracks_theta'
                                         ],
              'max_tracks'             : None,
              'multiply_output_name'   : None,
              'sample_weight_name'     : None,
              }

    return params

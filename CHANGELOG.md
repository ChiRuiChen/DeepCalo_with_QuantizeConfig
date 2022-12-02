# New in version ...

## 0.2.2
* Deleted keras_contrib dependecy as it was a frequent source of installation problems, and was only used peripherally.
* Deteled scikit-optimize and tqdm as dependencies, as they were only used in the `demos/hp_search.py` script.
* Added `demos/atlas_specific_usecases/use_trained_model/inference.py` which provides a single function for doing inference with a trained model.
* Added both [the master's thesis](#https://gitlab.com/ffaye/deepcalo/blob/master/demos/atlas_specific_usecases/train_recommended_models/thesis.pdf) that this package was made in conjunction with, along with [a short overview](#https://gitlab.com/ffaye/deepcalo/blob/master/demos/atlas_specific_usecases/train_recommended_models/recipe.pdf) of the knowledge that came out of that thesis. 
* Added strides as an available hyperparameter.
* Saved images of models as .png, as .pdf was causing some trouble on some platforms.
* Made minor convenient changes to `utils` functions.

## 0.2.1
* Updated demos:
  - Included a more realistic use case of training a model.
  - Included a demo showing how to load and use a trained model.
  - Made `hp_search.py` more memory efficient in that different processes don't have their own copy of the data.
* Added functionality for multiplying the model output with a scalar variable (see the `multiply_output_name` argument of `deepcalo/utils/load_atlas_data`).
* Added bias correction classes, which use a (1D or 2D) spline to fit the median error of a model.

## 0.2.0
* Added network_in_network model
* Changed name of time_net to gate_net, as multiple types of cell data can be processed using this - this breaks backward compatibility!
* Added possibility to scan learning rates logarithmically in `LRFinder` ([#1](https://gitlab.com/ffaye/deepcalo/issues/1))
* Made it possible to divide target by a scalar variable (e.g. the total accordion energy, when doing ER) in load_atlas_data
* Changed naming convention of ECAL layers in load_atlas_data to fit the new data
* Bugfix when trying to load gate_net weights into TimeDistributed
* Bugfix when trying to plot the FiLM generator
* Bugfix when giving tracks to self.cnn_with_upsampling in model_container.py
* Bugfix in merge_dicts

## 0.1.5
* Added 1Cycle learning rate schedule and improved docs for learning rate schedules in general
* Bugfix in `SGDR_lr_schedule.py` (missing imports)
* Bugfix in `load_atlas_data` (targets were divided by 1000)
* Changed the way GPUs were counted in the demos, as the old code was wrong if more than 9 GPUs were used

## 0.1.4
* Updated `load_data` to `load_atlas_data`, which now works with the newly uploaded data
* Added custom model checkpoint callback that allows models to be saved as jsons
* Bugfix in get_track_net
* Bugfix in datagenerator

## 0.1.3
* Bugfix in utils.set_auto_lr
* Save models instead of just weights

## 0.1.2
* Deleted unneeded Python path insertion

## 0.1.1
* Deleted `deepcalo.utils.apply_preprocessing`, such that there is no dependency on `scikit-learn`

## 0.1.0
* Initial release

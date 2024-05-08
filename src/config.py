train_model_config_params = {
    'version': 1,
    'epochs': 3,
    'learning_rate': 0.001,
    'pretrained': True,
    'fine_tune': True,
    'use_scheduler': False,
    'models_dir': 'models/',
    'load_model': False,
}

aggregate_models_config_params = {
    'local_models_dir': 'models/',
    'central_model_dir': 'central_models/',
    'scenario': 'static_aggregate',  # 'static_aggregate' or 'dynamic_aggregate'
    'model_names': ['model_v1', 'model_v2']
}

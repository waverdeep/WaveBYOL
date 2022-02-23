import src.models.model_wavebyol_combine as model_wavebyol_combine
import src.models.model_wavebyol_reconst as model_wavebyol_reconst
import torch


def load_model(config, model_name, checkpoint_path=None):
    model = None
    if model_name == "WaveBYOLCombine":
        model = model_wavebyol_combine.WaveBYOLCombine(
            config=config,
            encoder_input_dim=config['encoder_input_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            encoder_filter_size=config['encoder_filter_size'],
            encoder_stride=config['encoder_stride'],
            encoder_padding=config['encoder_padding'],
            mlp_input_dim=config['mlp_input_dim'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            mlp_output_dim=config['mlp_output_dim'],
            combine_model_name=config['combine_model_name']
        )
    elif model_name == "WaveBYOLReconst":
        model = model_wavebyol_reconst.WaveBYOLCombine(
            config=config,
            encoder_input_dim=config['encoder_input_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            encoder_filter_size=config['encoder_filter_size'],
            encoder_stride=config['encoder_stride'],
            encoder_padding=config['encoder_padding'],
            decoder_output_padding=config['decoder_output_padding'],
            mlp_input_dim=config['mlp_input_dim'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            mlp_output_dim=config['mlp_output_dim'],
        )

    if checkpoint_path is not None:
        print("load checkpoint...")
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model

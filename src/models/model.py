import src.models.model_wavebyol_original as model_wavebyol_original
import src.models.model_wavebyol as model_wavebyol
import src.models.model_downstream as model_downstream
import torch


def load_model(config, model_name, checkpoint_path=None):
    model = None
    if model_name == "WaveBYOL":
        model = model_wavebyol.WaveBYOL(
            config=config,
            encoder_input_dim=config['encoder_input_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            encoder_filter_size=config['encoder_filter_size'],
            encoder_stride=config['encoder_stride'],
            encoder_padding=config['encoder_padding'],
            mlp_input_dim=config['mlp_input_dim'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            mlp_output_dim=config['mlp_output_dim'],
            feature_extractor_model=config['feature_extractor_model'],
            pretrain=config['feature_extractor_model_pretrain']
        )
    elif model_name == "DownstreamClassification":
        model = model_downstream.DownstreamClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )

    if checkpoint_path is not None:
        print("load checkpoint...")
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model

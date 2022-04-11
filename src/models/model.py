import src.models.model_wavebyol_original as model_wavebyol_original
import src.models.model_wavebyol as model_wavebyol
import src.models.model_wavebyol_verific as model_wavebyol_verific
import src.models.model_wavebyol_u2 as model_wavebyol_u2
import src.models.model_wavebyol_u3 as model_wavebyol_u3
import src.models.model_wavebyol_u4 as model_wavebyol_u4
import src.models.model_wavebyol_u5 as model_wavebyol_u5
import src.models.model_wavebyol_hydr as model_wavebyol_hydr
import src.models.model_wavebyol_upda as model_wavebyol_upda
import src.models.model_wavebyol_gua as model_wavebyol_gua
import src.models.model_wavebyol_oldfsh as model_wavebyol_oldfsh
import src.models.model_downstream as model_downstream
import torch


def load_model(config, model_name, checkpoint_path=None, pretext_model=None):
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
    elif model_name == "WaveBYOL_Verific":
        model = model_wavebyol_verific.WaveBYOL(
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
    elif model_name == 'WaveBYOLU2':
        model = model_wavebyol_u2.WaveBYOL(
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
    elif model_name == "WaveBYOLU3":
        model = model_wavebyol_u3.WaveBYOL(
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
    elif model_name == "WaveBYOLU4":
        model = model_wavebyol_u4.WaveBYOL(
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
    elif model_name == "WaveBYOLU5":
        model = model_wavebyol_u5.WaveBYOL(
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
    elif model_name == 'WaveBYOL_HYDR':
        model = model_wavebyol_hydr.WaveBYOL(
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
    elif model_name == 'WaveBYOL_UPDA':
        model = model_wavebyol_upda.WaveBYOL(
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
    elif model_name == 'WaveBYOL_GUA':
        model = model_wavebyol_gua.WaveBYOL(
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
    elif model_name == 'WaveBYOL_OLDFSH':
        model = model_wavebyol_oldfsh.WaveBYOL(
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
    elif model_name == "DownstreamEarlyClassification":
        model = model_downstream.DownstreamEarlyClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name == "DownstreamFlatClassification":
        model = model_downstream.DownstreamFlatClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name == "DownstreamFullFlatClassification":
        model = model_downstream.DownstreamFullFlatClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name == "DownstreamU3Classification":
        model = model_downstream.DownstreamU3Classification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name =='DownstreamFlatTransferClassification':
        model = model_downstream.DownstreamFlatTransferClassification(
            pretext_model=pretext_model,
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name =='DownstreamFlatEmbeddingTransferClassification':
        model = model_downstream.DownstreamFlatEmbeddingTransferClassification(
            pretext_model=pretext_model,
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

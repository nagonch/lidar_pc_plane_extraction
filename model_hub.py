from ds_net.dataset import build_dataloader as build_dsnet_dataloader
from ds_net.semantic import build_model as build_dsnet_semantic

def get_model(
    model_name,
    train_filepaths,
    val_filepaths,
    scene_size,
    batch_size,
    device_name,
    model_state_path,
    n_classes,
):
    if model_name == 'dsnet':
        train_loader = build_dsnet_dataloader(
            train_filepaths,
            scene_size,
            batch_size=batch_size,
            n_classes=n_classes,
        )
        val_loader = build_dsnet_dataloader(
            val_filepaths,
            scene_size,
            batch_size=batch_size,
            n_classes=n_classes,
        )
        model = build_dsnet_semantic(device_name, model_state_path, n_classes)
        return model, train_loader, val_loader
    else:
        raise NotImplementedError(f'Model with name {model_name} not implemented')


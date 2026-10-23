from lungai.config import load_config
from lungai.train import train


def test_train_smoke(fake_dataset, tmp_path):
    config = load_config()
    config["data"].update(root=str(fake_dataset), image_size=[64, 64], batch_size=8)
    config["model"]["name"] = "custom_cnn"
    config["train"].update(epochs=1, output_dir=str(tmp_path / "models"))

    model, history = train(config, weights=None)

    assert "val_loss" in history
    assert (tmp_path / "models" / "custom_cnn.keras").exists()
    assert (tmp_path / "models" / "custom_cnn_history.csv").exists()

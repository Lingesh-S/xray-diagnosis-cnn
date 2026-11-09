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


def test_fine_tune_from_saved_model(fake_dataset, tmp_path):
    config = load_config()
    config["data"].update(root=str(fake_dataset), image_size=[64, 64], batch_size=8)
    config["model"].update(name="mobilenetv2", fine_tune_layers=5)
    config["train"].update(epochs=1, output_dir=str(tmp_path / "models"))
    base, _ = train(config, weights=None, run_name="base")

    config["train"]["learning_rate"] = 1e-5
    tuned, _ = train(config, weights=None, init_from=tmp_path / "models" / "base.keras", run_name="ft")

    assert (tmp_path / "models" / "ft.keras").exists()
    trainable = [w for w in tuned.trainable_weights if "pneumonia_prob" not in w.path]
    assert trainable, "backbone layers should be unfrozen for fine-tuning"

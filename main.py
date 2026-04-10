import os
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Фиксируем генераторы случайных чисел, чтобы результаты были воспроизводимыми.
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


@dataclass
class TrainingConfig:
    """Параметры обучения модели."""

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.005
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    test_size: float = 0.2
    val_size: float = 0.2
    use_cuda: bool = True


class JobStatusClassifier(nn.Module):
    """Многослойная сеть для многоклассовой классификации."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim_1: int, hidden_dim_2: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_one_hot_encoder() -> OneHotEncoder:
    """Создает совместимый с разными версиями sklearn one-hot encoder."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_and_prepare_data(csv_path: str, config: TrainingConfig):
    """Читает датасет, кодирует признаки и готовит train/test выборки."""

    df = pd.read_csv(csv_path)

    # Идентификатор сотрудника не несет полезной информации для прогноза.
    df = df.drop(columns=["Employee_ID"])

    target_column = "Job_Status"
    x = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()

    # Числовые признаки стандартизируем, категориальные кодируем one-hot схемой.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", build_one_hot_encoder(), categorical_columns),
        ]
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=config.test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)

    # Переводим данные в тензоры PyTorch.
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    use_pin_memory = config.use_cuda and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "input_dim": x_train_tensor.shape[1],
        "output_dim": len(label_encoder.classes_),
        "class_names": label_encoder.classes_,
    }


def train_model(model: nn.Module, train_loader: DataLoader, config: TrainingConfig, device: torch.device) -> dict[str, list[float]]:
    """Обучает модель методом градиентного спуска."""

    criterion = nn.CrossEntropyLoss()

    # Используем Adam для более быстрой и стабильной сходимости.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            # Обнуляем накопленные градиенты перед новым шагом оптимизации.
            optimizer.zero_grad()

            logits = model(batch_features)
            loss = criterion(logits, batch_targets)

            # Вычисляем градиенты функции потерь по параметрам сети.
            loss.backward()

            # Делаем шаг градиентного спуска.
            optimizer.step()

            total_loss += loss.item() * batch_features.size(0)
            predictions = logits.argmax(dim=1)
            correct_predictions += (predictions == batch_targets).sum().item()
            total_examples += batch_targets.size(0)

        epoch_loss = total_loss / total_examples
        epoch_accuracy = correct_predictions / total_examples
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"Эпоха {epoch:3d}/{config.epochs}: "
                f"loss = {epoch_loss:.4f}, accuracy = {epoch_accuracy:.4f}"
            )

    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, class_names, device: torch.device) -> dict[str, object]:
    """Оценивает модель на тестовой выборке."""

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            logits = model(batch_features)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_targets.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_predictions).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nТочность на тестовой выборке: {accuracy:.4f}\n")
    print("Подробный отчет по классам:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Примеры предсказаний модели:")
    for index in range(min(10, len(y_true))):
        true_label = class_names[y_true[index]]
        predicted_label = class_names[y_pred[index]]
        print(f"{index + 1:2d}. Истинный класс: {true_label:10s} | Предсказание: {predicted_label}")

    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_training_history_plot(history: dict[str, list[float]], output_path: str) -> None:
    """Сохраняет графики loss и accuracy по эпохам с помощью matplotlib."""

    sns.set_theme(style="whitegrid")

    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(x=list(epochs), y=history["loss"], ax=axes[0], color="#2563eb", linewidth=2.5)
    axes[0].set_title("Функция потерь по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")

    sns.lineplot(x=list(epochs), y=history["accuracy"], ax=axes[1], color="#16a34a", linewidth=2.5)
    axes[1].set_title("Точность на обучении")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)

if __name__ == "__main__":
    os.environ["TCL_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
    os.environ["TK_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6"
    
    config = TrainingConfig()
    data = load_and_prepare_data("ai_job_impact.csv", config)

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Устройство для обучения: {device}")

    model = JobStatusClassifier(
        input_dim=data["input_dim"],
        output_dim=data["output_dim"],
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
    ).to(device)

    history = train_model(model, data["train_loader"], config, device)
    metrics = evaluate_model(model, data["test_loader"], data["class_names"], device)

    history_plot_path = "training_history.png"
    confusion_plot_path = "confusion_matrix.png"
    save_training_history_plot(history, history_plot_path)

    print(f"\nГрафик обучения сохранен в: {os.path.abspath(history_plot_path)}")
    print(f"Матрица ошибок сохранена в: {os.path.abspath(confusion_plot_path)}")

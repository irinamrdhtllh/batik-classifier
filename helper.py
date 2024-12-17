import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split


def split_and_transform(dataset, train_transform, eval_transform):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_data.dataset.transform = train_transform
    val_data.dataset.transform = eval_transform
    test_data.dataset.transform = eval_transform

    train_data = DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = DataLoader(val_data, batch_size=32, shuffle=False)
    test_data = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_data, val_data, test_data


def evaluate(model, eval_data, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in eval_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def train(
    model,
    train_data,
    val_data,
    criterion,
    optimizer,
    device,
    num_epoch=1000,
    early_stopping=False,
):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    model.train()

    for epoch in range(num_epoch):
        epsilon = 1e-2
        running_loss = 0.0

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        val_accuracy = evaluate(model, val_data, device)

        print(
            f"Epoch {epoch+1}/{num_epoch}, Loss: {running_loss/len(train_data):.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        if early_stopping and (
            running_loss / len(train_data) <= epsilon and val_accuracy >= 90
        ):
            break


def predict(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted

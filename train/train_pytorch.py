import torch
from torch import nn
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Démarrage de train_pytorch.py")  

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, save=False, plot=False):
        self.model.train()
        self.train_acc = []
        self.train_loss = []
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)

            for batch in progress_bar:
                input_datas, labels = batch
                input_datas, labels = input_datas.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, preds = outputs.max(1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                total_correct += correct
                total_samples += total
                total_loss += loss.item()

                batch_accuracy = 100.0 * correct / total
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / total_samples

                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
            print(f"Epoch {epoch + 1}/{self.epochs} - Avg Acc: {average_accuracy:.2f}%, Avg Loss: {average_loss:.4f}")
        if save:
            torch.save(self.model.state_dict(), "amy_model.torch")
        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=True):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.show()

def run_pytorch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du dispositif : {device}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)  
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Chargement des données...")
    try:
        train_data = datasets.ImageFolder('data/training', transform=transform)
        test_data = datasets.ImageFolder('data/testing', transform=transform)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    trainer = Trainer(model, train_loader, test_loader, lr=0.001, wd=0.01, epochs=5, device=device)
    trainer.train(save=True, plot=True)
    trainer.evaluate()

if __name__ == '__main__':
    run_pytorch()
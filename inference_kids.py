import torch
from torchvision import transforms
import PIL

image_path = "/home/tyler/Downloads/tyler.jpg"
model_path = "./models/kids.pkl"

classes = ['aiden', 'cheryl', 'clara', 'emery', 'tyler'] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model = torch.load(model_path).to(device)
predictions = model(trans(PIL.Image.open(image_path)).unsqueeze(0).to(device))
print(predictions)
most_likely = classes[torch.argmax(predictions)]

print(most_likely)
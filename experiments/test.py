import transformers
from transformers.models.vit import ViTModel

model = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")

for name, modules in model.named_modules():
    print(name, modules.__class__.__name__)

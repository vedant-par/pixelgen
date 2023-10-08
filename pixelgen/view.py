import torch
from architecture import Transformer
from PIL import Image

device = 'cuda'



model = Transformer()
model.load_state_dict(torch.load(f'models/20,1.6334.pth', map_location=device))
model.to(device)   
for i in range(10):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    lst = ((model.generate(context, max_new_tokens=400)[0].tolist()))
    print(len(lst))
    lst.pop(0)


    root = int((len(lst))**0.5)
    print(root)
    print(len(lst))
    print(lst[:10])
    im = Image.new("L", (root, root))
    im.putdata(lst)
    im.save(f'generation/generation{i}.png')

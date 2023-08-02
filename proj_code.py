import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available else 128 

def image_loader(path):
    img = Image.open(path)
    img = transforms.Resize((imsize,imsize))(img)
    img = transforms.ToTensor()(img)
    
    return img.to(dev, torch.float)

im_style = image_loader(r"D:\Sai_Mayura\pics_\style1.jpg")
im_cont = image_loader(r"D:\Sai_Mayura\pics_\face.jpg")
im_out = im_cont.clone().requires_grad_(True)

def disp_image(im, name):
    plt.figure()    
    image = im.clone() 
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.title(name)
    plt.show()
    plt.pause(1)

disp_image(im_style, "style image")
disp_image(im_cont, "content image")


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)#works well for vgg networks
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)#works well for vgg networks
        #-1,1,1, because we need to work with 
    def forward(self, img):
        return (img - self.mean) / self.std    

normalization = Normalization()

# just in order to have an iterable access to or list of content/style
# losses


# assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
# to put in modules that are supposed to be activated sequentially

model = nn.Sequential(normalization)
    
vgg=models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features 

class VGG(nn.Module):
    def __init__(self,chosen_layers):
        super(VGG, self).__init__()
        self.f_layers=chosen_layers
        for i, layer in enumerate(vgg):
           model.add_module(str(i+2),layer)
                #print(i, layer, layer.shape, end="\n")
        self.model=model    

    def forward(self, inp):
        features = []
        for i, layer in enumerate(self.model):
            inp = layer(inp)
            if i in self.f_layers:
                #print(i, layer, layer.shape, end="\n")
                features.append(inp)

        return features
    

def gram_matrix(input_layer):
    ch, h, w = input_layer.shape
    new = input_layer.view(ch, h*w)
    gram_m = torch.mm(new, new.t())
    return gram_m.div(ch*h*w)

sty_layers=[0,5,10,19,28]
cont_layers=[19]

model_style=VGG(sty_layers).to(dev).eval()
model_cont=VGG(cont_layers).to(dev).eval()

def Style_loss():
    #print("hi\n",im_style.shape,end="\n")
    style_loss = 0
    
    style_features = model_style(im_style)
    for i in style_features:
        i.detach()
    output_features = model_style(im_out)
    for style_feat, output_feat in zip(style_features, output_features):
        style_gram = gram_matrix(style_feat).detach()
        out_gram = gram_matrix(output_feat)
        style_loss += torch.mean((style_gram - out_gram)**2)

    return style_loss

def Content_loss():
    cont_loss = 0
    content_features = model_cont(im_cont)
    for i in content_features:
        i.detach()
    output_features = model_cont(im_out)
    for content_feat, output_feat in zip(content_features, output_features):
        cont_loss += torch.mean((content_feat - output_feat).detach()**2)
    return cont_loss


iterations=1000
beta=1
alpha=1000000
optimizer=opt.Adam([im_out],lr=0.003)
    

for i in range(iterations):
      print(i)
      optimizer.zero_grad()
      style_loss = Style_loss()
      content_loss = Content_loss()
      total_loss = alpha * style_loss + beta * content_loss
      total_loss.backward()
      optimizer.step()
      
      if i%200==0:
       save_image(im_out,"output_twoppl{}.png".format(i))   
      
  
  
      with torch.no_grad():
        im_style.clamp_(0, 1)   
     
    
save_image(im_out,"final.png")    
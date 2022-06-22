import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import skorch
import torch
from torch.optim import Adam
from torch.nn import ReLU, Sigmoid, Tanh, Softmax, CrossEntropyLoss
from torch.nn import Module, Conv2d, MaxPool2d, Linear


list_kelas = {0 : 'Safe driving',
              1 : 'Texting - right',
              2 : 'Talking on the phone - right',
              3 : 'Texting - left',
              4 : 'Talking on the phone - left'}

input_size = [64, 64]


class ConvNeuralNet(Module):
    def __init__(self, hidden, act='relu'):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=4,
                            kernel_size=(3, 3),
                            bias=False)
        
        self.conv2 = Conv2d(in_channels=4,
                        out_channels=3,
                        kernel_size=(3, 3),
                        bias=False)
        
        self.pool = MaxPool2d(2, 2)
        
        self.fc1 = Linear(in_features=3 * 30 * 30,
                          out_features=hidden,
                          bias=False)
        
        self.fc2 = Linear(in_features=hidden,
                          out_features=5,
                          bias=False)
        
        self.softmax = Softmax(dim=1)
        
        
        if act=='sigmoid':
            self.act = Sigmoid()
        elif act=='tanh':
            self.act = Tanh()
        else:
            self.act = ReLU()
    def forward(self, x_input):
        x = self.act(self.conv1(x_input))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.softmax(self.fc2(x))
        return x


def predict(model, img_path:str):
  start_time = time.time()
  image = Image.open(img_path)
  image = image.resize(tuple(input_size))
  inp = np.array(image).astype('float32') / 255
  inp = inp.reshape(1, 3, input_size[0], input_size[1]).astype(np.float32)

  hasil = model.predict_proba(inp)
  kelas = list_kelas[hasil.argmax()]

  exec_time = time.time() - start_time
  
  return hasil, kelas, exec_time


@st.cache(allow_output_mutation=True)
# def load_model(path: str = 'model/safe_driving_wo_tuning_2022-03-03_07-43-39.pkl') -> ConvNeuralNet:
def load_model(path: str = 'model/safe_driving_demo_2022-04-08_04-54-40.pkl') -> ConvNeuralNet:
    with open(path, 'rb') as f:
      model = pickle.load(f)

    return model


if __name__ == "__main__":
  st.set_page_config(page_title="Safe Driving Detection")

  model = load_model()
  
  st.title("Safe Driving Detection")
  st.write("")

  file_up = st.file_uploader("Upload an image", type='jpg')

  if file_up is not None:
    img = Image.open(file_up)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    img = np.array(img).astype('float32') / 255.
    hasil, kelas, exec_time = predict(model, file_up)
    
    st.subheader(f"Label : {kelas}\n")
    st.write("")

    formatted_pred = []
    for pred_class, pred_prob in zip(list_kelas.values(), hasil[0]):
      pred_prob = pred_prob * 100
      formatted_pred.append((pred_class, f"{pred_prob:.3f}%"))
    
    df = pd.DataFrame(data=formatted_pred,
                      columns=['Label', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
    st.write("")
    st.write(f"Execution time : {exec_time:.4f} seconds")    

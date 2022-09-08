import torch
import gradio as gr

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process(image):
  image.thumbnail((1024, 1024))
  results = model(image, augment=True)
  results.save()
  return results.ims[0]

gr.Interface(process, "pil", "image").launch(share=True)


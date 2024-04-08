import streamlit as st
from PIL import Image

from transformers import pipeline
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

st.set_page_config(layout='wide',
                   page_title='Model for Diabetic Retinopathy Detection APP'
                   )

def main():
    
    st.title("Diabetic Retinopathy Detection App")
    st.markdown("## Overview")   
    st.markdown("### Backgroud")
    st.markdown("Welcome to our Diabetic Retinopathy Detection App! This app utilizes deep learning models to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.")
  
    st.markdown("Stages of Diabetic Retinopathy: ")  

  
    st.markdown("1. Normal - a normal condition of an eye.")  
    st.markdown("2. Mild - The initial stage of DR. Micro aneurysms(small swelling) appear that may cause leak of fluid into the retina. ")  
    st.markdown("3. Moderate - The progressive stage that cause swell and distortion of blood vessels that are connected to retina(for blood and nourishment.")  
    st.markdown("4. Severe - The severe condition of DR where many blood vessels blocked making less supply of blood to retina(new blood vessels appear).")  
    st.markdown("5. Proliferative - The advanced stage of DR caused by blockage of tiny blood vessels(that grow inside). It damage the retina.")  

  
    # st.markdown("### Dataset")  
    # st.markdown("The Diabetic Retinopathy dataset is from Kaggle. There are totally 36450 pictures in this dataset. And this model is an image classification model for this dataset. There are 5 classes for this dataset, which are Normal(), mild, moderate (400), Sever(584), Proliferative(472.")  
    # st.markdown("### Model")  
    # st.markdown("The model is based on the [ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) model, which is short for the Vision Transformer. It was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), which was introduced in June 2021 by a team of researchers at Google Brain. And first released in [this repository](https://github.com/rwightman/pytorch-image-models). I trained this model with PyTorch. I think the most different thing between using the transformer to train on an image and on a text is in the tokenizing step. ")  
    # st.markdown("There are 3 steps to tokenize the image:")  
    # st.markdown("1. Split an image into a grid of sub-image patches")  
    # st.markdown("2. Embed each patch with a linear projection")  
    # st.markdown("3. Each embedded patch becomes a token, and the resulting sequence of embedded patches is the sequence you pass to the model.")  
    
    st.header("Try it out!")

    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    
    if uploaded_file!=None:

        img=Image.open(uploaded_file)

        extractor = AutoFeatureExtractor.from_pretrained("bhimrazy/diabetic-retinopathy-detection")
        model = AutoModelForImageClassification.from_pretrained("bhimrazy/diabetic-retinopathy-detection")

        inputs = extractor(img,return_tensors="pt")
        outputs = model(**inputs)
        label_num=outputs.logits.softmax(1).argmax(1)
        label_num=label_num.item()

        st.write("The prediction class is:")

        if label_num==0:
            st.write("Normal")
        elif label_num==1:
            st.write("Mild")
        elif label_num==2:
            st.write("Modelrate")
        elif label_num==3:
            st.write("Severe")
        else:
            st.write("Proliferative")

        st.image(img)


if __name__ == '__main__':
    main()
